"""Shared UI components: CSS, header, metrics panel, zone overlay."""

import cv2
import numpy as np
import streamlit as st


CSS = """
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: white; font-size: 2.5rem; margin: 0; font-weight: 700; }
    .main-header p  { color: #a8d4f5; margin: 0.5rem 0 0 0; font-size: 1.1rem; }

    .metric-card {
        background: #1e2535;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #2d3748;
    }
    .metric-card .value { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .metric-card .label { font-size: 0.9rem; color: #8892a4; margin: 0.3rem 0 0 0; }
    .metric-free     { border-top: 3px solid #48bb78; }
    .metric-free     .value { color: #48bb78; }
    .metric-occupied { border-top: 3px solid #fc8181; }
    .metric-occupied .value { color: #fc8181; }
    .metric-total    { border-top: 3px solid #63b3ed; }
    .metric-total    .value { color: #63b3ed; }

    .info-box {
        background: #1a2332;
        border-left: 3px solid #2d6a9f;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .info-box p { margin: 0; color: #a8d4f5; font-size: 0.95rem; }

    .progress-bar-container {
        background: #2d3748; border-radius: 8px; height: 12px;
        overflow: hidden; margin: 0.5rem 0;
    }
    .progress-bar-fill { height: 100%; border-radius: 8px; }

    .block-container { padding-top: 1.5rem; }
</style>
"""

_FREE_COLOR      = (0,  200,   0)
_OCCUPIED_COLOR  = (220,  50,  50)
_PENDING_COLOR   = (0,  220, 220)
_UNCERTAIN_COLOR = (220, 160,   0)   # yellow-orange for low-confidence predictions


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>Parking Detection</h1>
        <p>Smart parking space monitoring</p>
    </div>
    """, unsafe_allow_html=True)


def info_box(text: str):
    st.markdown(f'<div class="info-box"><p>{text}</p></div>', unsafe_allow_html=True)


def show_metrics(empty: int, occupied: int):
    total = empty + occupied
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card metric-free">'
            f'<p class="value">{empty}</p>'
            f'<p class="label">Free</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card metric-occupied">'
            f'<p class="value">{occupied}</p>'
            f'<p class="label">Occupied</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card metric-total">'
            f'<p class="value">{total}</p>'
            f'<p class="label">Total spaces</p></div>',
            unsafe_allow_html=True,
        )

    if total > 0:
        pct_occ = occupied / total
        bar_color = "#48bb78" if pct_occ < 0.5 else "#fc8181"
        st.markdown(f"""
        <div style="margin-top:1rem;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="color:#8892a4;font-size:0.85rem;">Occupancy</span>
                <span style="color:white;font-size:0.85rem;font-weight:600;">
                    {round(pct_occ * 100)}% occupied</span>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar-fill"
                     style="width:{pct_occ*100}%;background:{bar_color};"></div>
            </div>
        </div>""", unsafe_allow_html=True)


def draw_zones(
    image_rgb: np.ndarray,
    zones: list,
    results=None,
    confidences: list | None = None,
    conf_threshold: float = 0.6,
) -> np.ndarray:
    """
    Overlay zone polygons on image_rgb.

    results[i]=True  → occupied (red)
    results[i]=False → free (green)
    confidence < conf_threshold → uncertain (yellow-orange), regardless of prediction
    results=None → pending / not yet analyzed (cyan)
    """
    canvas = image_rgb.copy()
    for i, zone in enumerate(zones):
        pts = np.array(zone, np.int32)
        if results is None:
            color = _PENDING_COLOR
        elif confidences is not None and confidences[i] < conf_threshold:
            color = _UNCERTAIN_COLOR
        else:
            color = _OCCUPIED_COLOR if results[i] else _FREE_COLOR

        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color)
        canvas = cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0)
        cv2.polylines(canvas, [pts], True, color, 2)

        cx = int(np.mean([p[0] for p in zone]))
        cy = int(np.mean([p[1] for p in zone]))

        cv2.putText(canvas, str(i + 1), (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return canvas
