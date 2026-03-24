import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# ─── Настройки страницы ───────────────────────────────────────
st.set_page_config(
    page_title="Parking Detection",
    page_icon="🅿️",
    layout="wide"
)

# ─── Кастомный CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* Заголовок */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #a8d4f5;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }

    /* Карточки метрик */
    .metric-card {
        background: #1e2535;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #2d3748;
    }
    .metric-card .value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-card .label {
        font-size: 0.9rem;
        color: #8892a4;
        margin: 0.3rem 0 0 0;
    }
    .metric-free { border-top: 3px solid #48bb78; }
    .metric-free .value { color: #48bb78; }
    .metric-occupied { border-top: 3px solid #fc8181; }
    .metric-occupied .value { color: #fc8181; }
    .metric-total { border-top: 3px solid #63b3ed; }
    .metric-total .value { color: #63b3ed; }

    /* Инструкция */
    .instruction-box {
        background: #1a2332;
        border-left: 3px solid #2d6a9f;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .instruction-box p {
        margin: 0;
        color: #a8d4f5;
        font-size: 0.95rem;
    }

    /* Прогресс бар */
    .progress-bar-container {
        background: #2d3748;
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.3s ease;
    }

    /* Убираем padding у main */
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── Загрузка модели ──────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("../notebooks/runs/models/parking_finetuned/weights/best.pt")

model = load_model()

@st.cache_resource
def load_cnn():
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = models.mobilenet_v2(weights=None)
    cnn.classifier[1] = torch.nn.Linear(1280, 2)
    
    # Путь относительно этого файла
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", "cnn_classifier.pth")
    
    cnn.load_state_dict(torch.load(
        model_path,
        map_location=device,
        weights_only=True
    ))
    cnn.eval()
    cnn = cnn.to(device)
    return cnn, device

cnn_model, cnn_device = load_cnn()

cnn_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─── Вспомогательные функции ──────────────────────────────────
def process_frame_yolo(image, conf_threshold):
    results = model.predict(source=image, conf=conf_threshold, device=0, verbose=False)
    result = results[0]
    empty = int((result.boxes.cls == 0).sum())
    occupied = int((result.boxes.cls == 1).sum())
    canvas = image.copy()
    for box in result.boxes:
        cls = int(box.cls.item())
        conf = box.conf.item()
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = (0, 255, 0) if cls == 0 else (255, 0, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        cv2.putText(canvas, f"{'E' if cls == 0 else 'O'} {conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas, empty, occupied

def extract_zone(image, points):
    pts = np.array(points, dtype=np.float32)
    width, height = 200, 300
    dst = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (width, height))

def is_occupied(image, points, threshold=45):
    """Использует CNN если доступна, иначе простой классификатор"""
    patch = extract_zone(image, points)
    
    try:
        # CNN классификация
        from PIL import Image as PILImage
        patch_pil = PILImage.fromarray(patch).convert("RGB")
        tensor = cnn_transform(patch_pil).unsqueeze(0).to(cnn_device)
        
        with torch.no_grad():
            output = cnn_model(tensor)
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(1).item()
            confidence = prob[0][pred].item()
        
        # 0=empty, 1=occupied
        return pred == 1, confidence
    
    except Exception:
        # Fallback на простой метод
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        score = np.std(hsv[:,:,2]) * 0.7 + np.mean(hsv[:,:,1]) * 0.3
        return score > threshold, score

def draw_zones(image, zones, results=None):
    canvas = image.copy()
    for i, zone in enumerate(zones):
        pts = np.array(zone, np.int32)
        color = (255, 0, 0) if (results and results[i]) else (0, 255, 0)
        if not results:
            color = (0, 255, 255)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color)
        canvas = cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0)
        cv2.polylines(canvas, [pts], True, color, 2)
        cx = int(np.mean([p[0] for p in zone]))
        cy = int(np.mean([p[1] for p in zone]))
        cv2.putText(canvas, str(i+1), (cx-5, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return canvas

def show_metrics(empty, occupied):
    total = empty + occupied
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card metric-free">
            <p class="value">{empty}</p>
            <p class="label">🟢 Свободно</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-occupied">
            <p class="value">{occupied}</p>
            <p class="label">🔴 Занято</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card metric-total">
            <p class="value">{total}</p>
            <p class="label">📊 Всего мест</p>
        </div>""", unsafe_allow_html=True)

    if total > 0:
        pct_free = empty / total
        pct_occ = occupied / total
        color = "#48bb78" if pct_free > 0.5 else "#fc8181"
        st.markdown(f"""
        <div style="margin-top: 1rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8892a4; font-size:0.85rem;">Заполненность парковки</span>
                <span style="color:white; font-size:0.85rem; font-weight:600;">{round(pct_occ*100)}% занято</span>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width:{pct_occ*100}%; background:{color};"></div>
            </div>
        </div>""", unsafe_allow_html=True)

# ─── Заголовок ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🅿️ Parking Detection</h1>
    <p>Умная система мониторинга парковочных мест</p>
</div>
""", unsafe_allow_html=True)

# ─── Сайдбар ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Настройки")
    st.markdown("---")

    mode = st.radio(
        "Режим работы",
        ["🤖 Auto (YOLO)", "✏️ Manual zones"],
    )

    st.markdown("---")

    if mode == "🤖 Auto (YOLO)":
        st.markdown("**Параметры модели**")
        conf_threshold = st.slider("Порог уверенности", 0.1, 0.9, 0.2, 0.05)
        st.markdown("""
        <div class="instruction-box">
        <p>💡 Подходит для фото с камеры под углом. Модель автоматически находит и классифицирует места.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="instruction-box">
        <p>💡 Подходит для любого угла. Размечай места вручную — CNN определит занятость.</p>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Настройки видео**")
        check_interval_ms = st.slider("Интервал проверки (мс)", 20, 500, 100, 20)
        stability_frames = st.slider("Стабильность (кадров)", 1, 10, 3, 1,
                                    help="Сколько кадров подряд нужно для смены статуса")

    st.markdown("---")
    st.markdown("**Обозначения:**")
    st.markdown("🟢 Зелёный = свободно")
    st.markdown("🔴 Красный = занято")
    st.markdown("---")
    st.markdown("<p style='color:#8892a4; font-size:0.8rem;'>Parking Detection v1.0</p>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# РЕЖИМ 1 — AUTO YOLO
# ══════════════════════════════════════════════════════════════
if mode == "🤖 Auto (YOLO)":
    uploaded_file = st.file_uploader(
        "Загрузи фото или видео парковки",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
    )

    if uploaded_file:
        file_type = uploaded_file.type

        if file_type.startswith("image"):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Оригинал**")
                st.image(image_rgb, use_column_width=True)

            with st.spinner("Анализируем..."):
                canvas, empty, occupied = process_frame_yolo(image_rgb, conf_threshold)

            with col2:
                st.markdown("**Результат**")
                st.image(canvas, use_column_width=True)

            st.markdown("---")
            show_metrics(empty, occupied)

        elif file_type.startswith("video"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.flush()
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            st.info(f"📹 Видео: {total_frames} кадров, {fps:.1f} FPS")

            progress = st.progress(0)
            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()

            frame_num = 0
            process_every = max(1, int(fps / 2))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_num % process_every == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result_frame, empty, occupied = process_frame_yolo(frame_rgb, conf_threshold)
                    frame_placeholder.image(result_frame, use_column_width=True)
                    with metrics_placeholder.container():
                        show_metrics(empty, occupied)
                    progress.progress(min(frame_num / total_frames, 1.0))
                frame_num += 1

            cap.release()
            try:
                os.unlink(tfile.name)
            except:
                pass
            progress.progress(1.0)
            st.success("✅ Обработка завершена!")
    else:
        st.markdown("""
        <div class="instruction-box">
        <p>👆 Загрузи фото или видео парковки чтобы начать анализ</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# РЕЖИМ 2 — MANUAL ZONES
# ══════════════════════════════════════════════════════════════
else:
    tab1, tab2 = st.tabs(["🖼️ Фото", "🎬 Видео"])

    # ════════════════════════════════
    # ВКЛАДКА 1 — ФОТО
    # ════════════════════════════════
    with tab1:
        st.markdown("""
        <div class="instruction-box">
        <p>📋 Загрузи фото → Выбери инструмент → Размечай места → Нажми <strong>Анализировать</strong></p>
        </div>""", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Фото парковки", type=["jpg", "jpeg", "png"], key="photo_upload")

        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            zoom = st.slider("🔍 Приближение", 0.5, 2.0, 1.0, 0.1, key="photo_zoom")
            max_width = 800
            h, w = image_rgb.shape[:2]
            scale = min(max_width / w, 600 / h) * zoom
            canvas_w = int(w * scale)
            canvas_h = int(h * scale)
            pil_resized = pil_image.resize((canvas_w, canvas_h))

            if "zones" not in st.session_state:
                st.session_state.zones = []
            if "poly_points" not in st.session_state:
                st.session_state.poly_points = []

            col_tool1, col_tool2 = st.columns(2)
            with col_tool1:
                drawing_mode = st.radio(
                    "Инструмент разметки",
                    ["⬜ Прямоугольник", "📍 4 точки"],
                    horizontal=True
                )

            if drawing_mode == "⬜ Прямоугольник":
                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.2)",
                    stroke_width=2,
                    stroke_color="#00FF00",
                    background_image=pil_resized,
                    update_streamlit=True,
                    height=canvas_h,
                    width=canvas_w,
                    drawing_mode="rect",
                    key="canvas_rect"
                )
                if canvas_result.json_data:
                    objects = canvas_result.json_data.get("objects", [])
                    # Обновляем session_state из canvas
                    new_zones = []
                    for obj in objects:
                        x = obj["left"] / scale
                        y = obj["top"] / scale
                        bw = obj["width"] / scale
                        bh = obj["height"] / scale
                        new_zones.append([[x,y],[x+bw,y],[x+bw,y+bh],[x,y+bh]])
                    st.session_state.zones = new_zones
                active_zones = st.session_state.zones

            else:
                # Рисуем превью с сохранёнными зонами
                preview_small = cv2.resize(image_rgb.copy(), (canvas_w, canvas_h))
                for zone in st.session_state.zones:
                    pts = np.array([[int(p[0]*scale), int(p[1]*scale)] for p in zone], np.int32)
                    overlay = preview_small.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    preview_small = cv2.addWeighted(overlay, 0.3, preview_small, 0.7, 0)
                    cv2.polylines(preview_small, [pts], True, (0, 255, 0), 2)
                    cx = int(np.mean([p[0]*scale for p in zone]))
                    cy = int(np.mean([p[1]*scale for p in zone]))
                    cv2.putText(preview_small, str(st.session_state.zones.index(zone)+1),
                                (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                for pt in st.session_state.poly_points:
                    cv2.circle(preview_small, (int(pt[0]*scale), int(pt[1]*scale)), 6, (255,255,0), -1)
                if len(st.session_state.poly_points) > 1:
                    for i in range(len(st.session_state.poly_points) - 1):
                        p1 = (int(st.session_state.poly_points[i][0]*scale),
                            int(st.session_state.poly_points[i][1]*scale))
                        p2 = (int(st.session_state.poly_points[i+1][0]*scale),
                            int(st.session_state.poly_points[i+1][1]*scale))
                        cv2.line(preview_small, p1, p2, (255,255,0), 2)

                canvas_result = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=1,
                    stroke_color="#FFFF00",
                    background_image=Image.fromarray(preview_small),
                    update_streamlit=True,
                    height=canvas_h,
                    width=canvas_w,
                    drawing_mode="point",
                    point_display_radius=5,
                    key="canvas_poly"
                )
                if canvas_result.json_data:
                    new_pts = [o for o in canvas_result.json_data.get("objects", []) if o["type"] == "circle"]
                    if len(new_pts) > len(st.session_state.poly_points):
                        latest = new_pts[-1]
                        st.session_state.poly_points.append([latest["left"]/scale, latest["top"]/scale])
                        if len(st.session_state.poly_points) == 4:
                            st.session_state.zones.append(st.session_state.poly_points.copy())
                            st.session_state.poly_points = []
                            st.rerun()
                pts_left = 4 - len(st.session_state.poly_points)
                if pts_left < 4:
                    st.info(f"Точек: {len(st.session_state.poly_points)}/4")
                active_zones = st.session_state.zones

            zones_count = len(active_zones)
            col_s, col_a = st.columns([3, 1])
            with col_s:
                if zones_count > 0:
                    st.success(f"✅ Размечено мест: {zones_count}")
                else:
                    st.warning("Нарисуй хотя бы одно место")
            with col_a:
                analyze = st.button("🔍 Анализировать", type="primary", disabled=zones_count == 0)

            if analyze and zones_count > 0:
                results_occupied = [is_occupied(image_rgb, z)[0] for z in active_zones]
                result_img = draw_zones(image_rgb, active_zones, results_occupied)
                st.image(result_img, use_column_width=True)
                st.markdown("---")
                show_metrics(results_occupied.count(False), results_occupied.count(True))
        else:
            st.session_state.zones = []
            st.session_state.poly_points = []
            st.markdown("""<div class="instruction-box"><p>👆 Загрузи фото парковки</p></div>""",
                        unsafe_allow_html=True)

    # ════════════════════════════════
    # ВКЛАДКА 2 — ВИДЕО
    # ════════════════════════════════
    with tab2:
        st.markdown("""
        <div class="instruction-box">
        <p>📋 Загрузи видео → Размечай зоны на первом кадре → Нажми <strong>Запустить</strong></p>
        </div>""", unsafe_allow_html=True)

        uploaded_video = st.file_uploader(
            "Видео парковки", type=["mp4", "avi", "mov"], key="video_upload"
        )

        if uploaded_video:
            # Сохраняем видео во временный файл
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.flush()
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Читаем первый кадр
            ret, first_frame = cap.read()
            cap.release()

            if ret:
                first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                pil_first = Image.fromarray(first_rgb)

                zoom_v = st.slider("🔍 Приближение", 0.5, 2.0, 1.0, 0.1, key="video_zoom")
                max_width = 800
                h, w = first_rgb.shape[:2]
                scale = min(max_width / w, 600 / h) * zoom_v
                canvas_w = int(w * scale)
                canvas_h = int(h * scale)
                pil_resized = pil_first.resize((canvas_w, canvas_h))

                st.markdown(f"**Видео:** {total_frames} кадров, {fps:.1f} FPS")
                st.markdown("**Размечай зоны на первом кадре:**")

                if "video_zones" not in st.session_state:
                    st.session_state.video_zones = []
                if "video_poly_points" not in st.session_state:
                    st.session_state.video_poly_points = []

                video_draw_mode = st.radio(
                    "Инструмент",
                    ["⬜ Прямоугольник", "📍 4 точки"],
                    horizontal=True,
                    key="video_draw_mode"
                )

                if video_draw_mode == "⬜ Прямоугольник":
                    canvas_result = st_canvas(
                        fill_color="rgba(0, 255, 0, 0.2)",
                        stroke_width=2,
                        stroke_color="#00FF00",
                        background_image=pil_resized,
                        update_streamlit=True,
                        height=canvas_h,
                        width=canvas_w,
                        drawing_mode="rect",
                        key="canvas_video"
                    )
                    video_zones = []
                    if canvas_result.json_data:
                        for obj in canvas_result.json_data.get("objects", []):
                            x = obj["left"] / scale
                            y = obj["top"] / scale
                            bw = obj["width"] / scale
                            bh = obj["height"] / scale
                            video_zones.append([[x,y],[x+bw,y],[x+bw,y+bh],[x,y+bh]])

                else:
                    # 4 точки — рисуем превью
                    preview_small = cv2.resize(first_rgb.copy(), (canvas_w, canvas_h))
                    for zone in st.session_state.video_zones:
                        pts = np.array([[int(p[0]*scale), int(p[1]*scale)] for p in zone], np.int32)
                        overlay = preview_small.copy()
                        cv2.fillPoly(overlay, [pts], (0, 255, 0))
                        preview_small = cv2.addWeighted(overlay, 0.3, preview_small, 0.7, 0)
                        cv2.polylines(preview_small, [pts], True, (0, 255, 0), 2)
                    for pt in st.session_state.video_poly_points:
                        cv2.circle(preview_small, (int(pt[0]*scale), int(pt[1]*scale)), 6, (255,255,0), -1)
                    if len(st.session_state.video_poly_points) > 1:
                        for i in range(len(st.session_state.video_poly_points) - 1):
                            p1 = (int(st.session_state.video_poly_points[i][0]*scale),
                                int(st.session_state.video_poly_points[i][1]*scale))
                            p2 = (int(st.session_state.video_poly_points[i+1][0]*scale),
                                int(st.session_state.video_poly_points[i+1][1]*scale))
                            cv2.line(preview_small, p1, p2, (255,255,0), 2)

                    canvas_result = st_canvas(
                        fill_color="rgba(0,0,0,0)",
                        stroke_width=1,
                        stroke_color="#FFFF00",
                        background_image=Image.fromarray(preview_small),
                        update_streamlit=True,
                        height=canvas_h,
                        width=canvas_w,
                        drawing_mode="point",
                        point_display_radius=5,
                        key="canvas_video_poly"
                    )
                    if canvas_result.json_data:
                        new_pts = [o for o in canvas_result.json_data.get("objects", []) if o["type"] == "circle"]
                        if len(new_pts) > len(st.session_state.video_poly_points):
                            latest = new_pts[-1]
                            st.session_state.video_poly_points.append([latest["left"]/scale, latest["top"]/scale])
                            if len(st.session_state.video_poly_points) == 4:
                                st.session_state.video_zones.append(st.session_state.video_poly_points.copy())
                                st.session_state.video_poly_points = []
                                st.rerun()
                    pts_left = 4 - len(st.session_state.video_poly_points)
                    if pts_left < 4:
                        st.info(f"Точек: {len(st.session_state.video_poly_points)}/4")

                    video_zones = st.session_state.video_zones

                zones_count = len(video_zones)
                if zones_count > 0:
                    st.success(f"✅ Размечено зон: {zones_count}")
                else:
                    st.warning("Нарисуй хотя бы одну зону")

                if st.button("▶️ Запустить", type="primary", disabled=zones_count == 0):
                    status_history = [[] for _ in range(zones_count)]
                    stable_status = [False] * zones_count
                    prev_stable_status = [None] * zones_count  # для отслеживания изменений
                    events_log = []  # история событий

                    cap2 = cv2.VideoCapture(tfile.name)
                    frame_placeholder = st.empty()
                    metrics_placeholder = st.empty()
                    progress = st.progress(0)

                    # Плейсхолдер для истории
                    st.markdown("---")
                    st.markdown("### 📋 История событий")
                    log_placeholder = st.empty()

                    interval_frames = max(1, int(fps * check_interval_ms / 1000))
                    frame_num = 0

                    while cap2.isOpened():
                        ret, frame = cap2.read()
                        if not ret:
                            break

                        if frame_num % interval_frames == 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            current_time = frame_num / fps  # секунды

                            for i, zone in enumerate(video_zones):
                                occ, _ = is_occupied(frame_rgb, zone)
                                status_history[i].append(occ)

                                if len(status_history[i]) > stability_frames:
                                    status_history[i].pop(0)

                                if len(status_history[i]) == stability_frames:
                                    if all(status_history[i]):
                                        new_status = True
                                    elif not any(status_history[i]):
                                        new_status = False
                                    else:
                                        new_status = stable_status[i]

                                    # Фиксируем событие если статус изменился
                                    if prev_stable_status[i] is not None and new_status != stable_status[i]:
                                        minutes = int(current_time // 60)
                                        seconds = int(current_time % 60)
                                        time_str = f"{minutes:02d}:{seconds:02d}"
                                        action = "🔴 занято" if new_status else "🟢 освободилось"
                                        events_log.append({
                                            "time": time_str,
                                            "zone": i + 1,
                                            "action": action,
                                            "frame": frame_num
                                        })

                                    prev_stable_status[i] = stable_status[i]
                                    stable_status[i] = new_status

                            # Рисуем результат
                            result_frame = draw_zones(frame_rgb, video_zones, stable_status)
                            empty = stable_status.count(False)
                            occupied = stable_status.count(True)

                            cv2.putText(result_frame,
                                        f"Free: {empty} | Occupied: {occupied}",
                                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                        (255, 255, 255), 2)

                            # Время на кадре
                            minutes = int(current_time // 60)
                            seconds = int(current_time % 60)
                            cv2.putText(result_frame,
                                        f"{minutes:02d}:{seconds:02d}",
                                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (200, 200, 200), 2)

                            frame_placeholder.image(result_frame, use_column_width=True)
                            with metrics_placeholder.container():
                                show_metrics(empty, occupied)
                            progress.progress(min(frame_num / total_frames, 1.0))

                            # Обновляем лог событий
                            if events_log:
                                import pandas as pd
                                df_log = pd.DataFrame([
                                    {
                                        "Время": e["time"],
                                        "Зона": f"Место #{e['zone']}",
                                        "Событие": e["action"]
                                    }
                                    for e in reversed(events_log[-20:])
                                ])
                                log_placeholder.dataframe(
                                    df_log,
                                    use_container_width=True,
                                    hide_index=True
                                )
                            else:
                                log_placeholder.info("Событий пока нет — изменений статуса не обнаружено")

                        frame_num += 1

                    cap2.release()
                    try:
                        os.unlink(tfile.name)
                    except:
                        pass
                    progress.progress(1.0)
                    st.success(f"✅ Обработка завершена! Всего событий: {len(events_log)}")
        else:
            st.markdown("""<div class="instruction-box"><p>👆 Загрузи видео парковки</p></div>""",
                        unsafe_allow_html=True)