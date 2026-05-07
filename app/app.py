"""Parking Detection — main Streamlit entry point."""

import os
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier import is_occupied, load_cnn
from drawing import zone_drawing_ui
from ui import draw_zones, info_box, inject_css, render_header, show_metrics
import video_processor

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Parking Detection", layout="wide")
inject_css()

# ── Model ─────────────────────────────────────────────────────
@st.cache_resource
def _load():
    return load_cnn()

cnn_model, cnn_device, cnn_err = _load()

# ── Header ────────────────────────────────────────────────────
render_header()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["Manual zones", "Auto detection (in development)"],
        index=0,
    )

    st.markdown("---")

    if mode == "Manual zones":
        if cnn_err:
            st.warning(f"CNN not loaded — HSV fallback active.\n\n{cnn_err}")
        info_box("Mark parking spaces manually. CNN classifies each space.")

        st.markdown("---")
        st.markdown("**Classifier**")
        conf_threshold = st.slider(
            "Confidence threshold",
            0.50, 0.99, 0.60, 0.01,
            help="Predictions below this confidence are shown in yellow (uncertain).",
        )
    else:
        st.info("Auto detection is under development.")
        conf_threshold = 0.60

    st.markdown("---")
    st.markdown("Green  = free")
    st.markdown("Red    = occupied")
    st.markdown("Yellow = uncertain")
    st.markdown("---")
    st.markdown(
        "<p style='color:#8892a4;font-size:0.8rem;'>Parking Detection v1.0</p>",
        unsafe_allow_html=True,
    )

# ── Auto mode placeholder ─────────────────────────────────────
if mode == "Auto detection (in development)":
    st.info(
        "Automatic parking space detection is under development. "
        "Please use Manual zones mode."
    )
    st.stop()

# ══════════════════════════════════════════════════════════════
# MANUAL ZONES
# ══════════════════════════════════════════════════════════════


def _canvas_size(image_rgb: np.ndarray):
    h, w  = image_rgb.shape[:2]
    scale = min(800 / w, 600 / h)
    return scale, int(w * scale), int(h * scale)


def _read_frame(vpath: str, frame_idx: int):
    cap = cv2.VideoCapture(vpath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return ret, frame


tab_photo, tab_video = st.tabs(["Photo", "Video"])

# ══════════════════════════════════════════════════════════════
# PHOTO TAB
# ══════════════════════════════════════════════════════════════
with tab_photo:
    info_box("Upload photo  →  Choose tool  →  Mark spaces  →  Click Analyze")

    uploaded = st.file_uploader(
        "Parking photo", type=["jpg", "jpeg", "png"], key="photo_upload"
    )

    if uploaded:
        raw       = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image_rgb = cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        scale, cw, ch = _canvas_size(image_rgb)

        zones = zone_drawing_ui(
            image_rgb  = image_rgb,
            scale      = scale,
            canvas_w   = cw,
            canvas_h   = ch,
            zones_key  = "photo_zones",
            poly_key   = "photo_poly_pts",
            canvas_key = "photo",
        )

        col_status, col_btn = st.columns([3, 1])
        with col_status:
            if zones:
                st.success(f"Marked spaces: {len(zones)}")
            else:
                st.warning("Draw at least one space")
        with col_btn:
            analyze = st.button("Analyze", type="primary", disabled=not zones)

        if analyze and zones:
            pairs     = [is_occupied(image_rgb, z, cnn_model, cnn_device) for z in zones]
            results   = [p[0] for p in pairs]
            confs     = [p[1] for p in pairs]
            annotated = draw_zones(image_rgb, zones, results, confs, conf_threshold)
            st.image(annotated, use_column_width=True)
            st.markdown("---")
            show_metrics(results.count(False), results.count(True))
    else:
        st.session_state["photo_zones"]    = []
        st.session_state["photo_poly_pts"] = []
        info_box("Upload a parking photo to begin.")


# ══════════════════════════════════════════════════════════════
# VIDEO TAB
# ══════════════════════════════════════════════════════════════
with tab_video:
    info_box("Upload video  →  Mark zones on first frame  →  Inspect frames or process")

    uploaded_video = st.file_uploader(
        "Parking video", type=["mp4", "avi", "mov"], key="video_upload"
    )

    if uploaded_video:
        # Cache temp file so seeking doesn't re-write on every rerun
        if (
            "video_tfile" not in st.session_state
            or st.session_state.get("video_fname") != uploaded_video.name
        ):
            old = st.session_state.get("video_tfile")
            if old:
                try:
                    os.unlink(old)
                except OSError:
                    pass
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.flush()
            tfile.close()
            st.session_state["video_tfile"]      = tfile.name
            st.session_state["video_fname"]      = uploaded_video.name
            st.session_state["video_seek_frame"] = 0
            st.session_state.pop("video_job_id", None)

        vpath = st.session_state["video_tfile"]

        cap          = cv2.VideoCapture(vpath)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first   = cap.read()
        cap.release()

        if not ret:
            st.error("Could not read the video file.")
            st.stop()

        first_rgb     = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
        scale, cw, ch = _canvas_size(first_rgb)
        duration      = total_frames / fps

        st.markdown(
            f"**Video:** {total_frames} frames  |  {fps:.1f} FPS  |  "
            f"{int(duration // 60):02d}:{int(duration % 60):02d}"
        )

        # ── Step 1: Mark zones ────────────────────────────────
        st.markdown("**Step 1: Mark zones on the first frame**")
        video_zones = zone_drawing_ui(
            image_rgb  = first_rgb,
            scale      = scale,
            canvas_w   = cw,
            canvas_h   = ch,
            zones_key  = "video_zones",
            poly_key   = "video_poly_pts",
            canvas_key = "video",
        )

        if not video_zones:
            st.warning("Draw at least one zone to continue.")
            st.stop()

        st.success(f"Marked zones: {len(video_zones)}")
        st.markdown("---")

        # ── Step 2: Frame inspector ───────────────────────────
        st.markdown("**Step 2: Inspect any frame**")

        if "video_seek_frame" not in st.session_state:
            st.session_state["video_seek_frame"] = 0

        # Navigation buttons
        nav = st.columns([1, 1, 1, 3, 1, 1, 1])
        for col, (step, label) in zip(
            [nav[0], nav[1], nav[2], nav[4], nav[5], nav[6]],
            [(-100, "-100"), (-10, "-10"), (-1, "-1"), (1, "+1"), (10, "+10"), (100, "+100")],
        ):
            with col:
                if st.button(label, key=f"nav{step}"):
                    st.session_state["video_seek_frame"] = int(
                        np.clip(st.session_state["video_seek_frame"] + step, 0, total_frames - 1)
                    )
                    st.rerun()

        seek = st.slider(
            "Frame",
            0, total_frames - 1,
            value=int(st.session_state["video_seek_frame"]),
            key="video_frame_slider",
        )
        st.session_state["video_seek_frame"] = seek

        t_seek = seek / fps
        st.caption(
            f"Frame {seek} / {total_frames - 1}  —  "
            f"{int(t_seek // 60):02d}:{int(t_seek % 60):02d}"
        )

        ret_s, frame_s = _read_frame(vpath, seek)
        if ret_s:
            frame_rgb = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)
            pairs     = [is_occupied(frame_rgb, z, cnn_model, cnn_device) for z in video_zones]
            results   = [p[0] for p in pairs]
            confs     = [p[1] for p in pairs]
            annotated = draw_zones(frame_rgb, video_zones, results, confs, conf_threshold)
            st.image(annotated, use_column_width=True)
            show_metrics(results.count(False), results.count(True))

        # ── Step 3: Batch processing ──────────────────────────
        st.markdown("---")
        st.markdown("**Step 3: Process entire video**")

        col_int, col_stab = st.columns(2)
        with col_int:
            check_interval_ms = st.slider(
                "Check interval (ms)", 20, 500, 100, 20, key="v_interval"
            )
        with col_stab:
            stability_frames = st.slider(
                "Stability (frames)", 1, 10, 3, 1, key="v_stability",
                help="Consecutive frames required before status change is committed",
            )

        interval = max(1, int(fps * check_interval_ms / 1000))
        job_id   = id(st.session_state)   # unique per browser session

        col_run, col_cancel = st.columns([2, 1])
        with col_run:
            if st.button("Process full video", type="primary"):
                video_processor.start_job(
                    job_id          = job_id,
                    vpath           = vpath,
                    zones           = video_zones,
                    fps             = fps,
                    total_frames    = total_frames,
                    interval_frames = interval,
                    stability_frames= stability_frames,
                    cnn_model       = cnn_model,
                    cnn_device      = cnn_device,
                    conf_threshold  = conf_threshold,
                )
                st.session_state["video_job_id"] = job_id
                st.rerun()

        with col_cancel:
            if st.session_state.get("video_job_id") == job_id:
                job_snap = video_processor.get_job(job_id)
                if job_snap and not job_snap["done"]:
                    if st.button("Cancel"):
                        video_processor.cancel_job(job_id)
                        st.rerun()

        # ── Live progress panel (auto-refreshes every 0.5 s) ─
        @st.fragment(run_every=0.5)
        def _progress_panel():
            jid = st.session_state.get("video_job_id")
            if jid is None:
                return
            job = video_processor.get_job(jid)
            if job is None:
                return

            if job.get("error"):
                st.error(f"Processing error: {job['error']}")
                return

            if not job["done"]:
                pct  = job["progress"]
                text = f"{pct * 100:.0f}%  —  processing…"
                st.progress(pct, text=text)
                if job["latest_frame"] is not None:
                    st.image(job["latest_frame"], use_column_width=True)
                    show_metrics(job["free"], job["occupied"])
                evts = job.get("events", [])
                if evts:
                    st.dataframe(
                        pd.DataFrame(list(reversed(evts[-20:]))),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No status changes detected yet.")
            else:
                evts = job.get("events", [])
                st.success(f"Processing complete. Total events: {len(evts)}")
                if job["latest_frame"] is not None:
                    st.image(job["latest_frame"], use_column_width=True)
                    show_metrics(job["free"], job["occupied"])
                if evts:
                    st.dataframe(
                        pd.DataFrame(list(reversed(evts[-20:]))),
                        use_container_width=True,
                        hide_index=True,
                    )

        _progress_panel()

    else:
        for k in ("video_tfile", "video_fname", "video_seek_frame",
                  "video_zones", "video_poly_pts", "video_job_id"):
            st.session_state.pop(k, None)
        info_box("Upload a parking video to begin.")
