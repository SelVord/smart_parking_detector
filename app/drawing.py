"""Zone drawing widget.

Both modes work the same way:
- Canvas stays live for the entire drawing session (no resets between zones).
- All shapes accumulate on the canvas.
- Zones are derived from canvas data on every rerun.
- "Remove last" is the only operation that resets the canvas (saves N-1 to background).

Rectangle mode: draw rects freely; each complete rect = 1 zone.
4-point mode:   place dots freely; every 4 consecutive dots = 1 zone.
"""

import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


def _init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


def _outline_bg(image_rgb: np.ndarray, scale: float,
                canvas_w: int, canvas_h: int, zones: list) -> Image.Image:
    """Scaled image with simple numbered zone outlines (no fill, no analysis colour)."""
    small = cv2.resize(image_rgb, (canvas_w, canvas_h))
    for i, zone in enumerate(zones):
        pts = np.array([[int(p[0] * scale), int(p[1] * scale)] for p in zone], np.int32)
        cv2.polylines(small, [pts], True, (30, 144, 255), 2)
        cx = int(np.mean([p[0] * scale for p in zone]))
        cy = int(np.mean([p[1] * scale for p in zone]))
        cv2.putText(small, str(i + 1), (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 144, 255), 2)
    return Image.fromarray(small)


def _parse_rects(json_data, scale: float) -> list:
    zones = []
    for obj in (json_data or {}).get("objects", []):
        if obj.get("type") != "rect":
            continue
        w, h = obj.get("width", 0), obj.get("height", 0)
        if w < 2 or h < 2:
            continue
        x, y = obj["left"] / scale, obj["top"] / scale
        bw, bh = w / scale, h / scale
        zones.append([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]])
    return zones


def _parse_dots(json_data, scale: float) -> list:
    """Return list of [x, y] for every dot on the canvas, in placement order."""
    return [
        [o["left"] / scale, o["top"] / scale]
        for o in (json_data or {}).get("objects", [])
        if o.get("type") == "circle"
    ]


def zone_drawing_ui(
    image_rgb: np.ndarray,
    scale: float,
    canvas_w: int,
    canvas_h: int,
    zones_key: str,
    poly_key: str,    # unused — kept for API compatibility
    canvas_key: str,
) -> list:
    """
    Returns the full list of current zones in original-image coordinates.
    """
    _init(zones_key, [])          # zones committed to background (from previous canvas sessions)
    _init(f"{canvas_key}_reset", 0)

    # ── toolbar ──────────────────────────────────────────────────
    col_mode, col_clear = st.columns([3, 1])
    with col_mode:
        mode = st.radio(
            "Marking tool",
            ["Rectangle", "4 points"],
            horizontal=True,
            key=f"tool_{canvas_key}",
        )
    with col_clear:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Clear all", key=f"clear_{canvas_key}"):
            st.session_state[zones_key] = []
            st.session_state[f"{canvas_key}_reset"] += 1
            st.rerun()

    reset_n = st.session_state[f"{canvas_key}_reset"]
    bg      = _outline_bg(image_rgb, scale, canvas_w, canvas_h,
                          st.session_state[zones_key])

    # ════════════════════════════════
    # RECTANGLE MODE
    # ════════════════════════════════
    if mode == "Rectangle":
        result = st_canvas(
            fill_color="rgba(0,160,0,0.15)",
            stroke_width=2,
            stroke_color="#00CC00",
            background_image=bg,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="rect",
            key=f"{canvas_key}_rect_{reset_n}",
        )

        canvas_zones = _parse_rects(result.json_data, scale) if result.json_data else []
        all_zones    = st.session_state[zones_key] + canvas_zones

        if st.button("Remove last", key=f"rm_{canvas_key}_rect",
                     disabled=not all_zones):
            st.session_state[zones_key] = all_zones[:-1]
            st.session_state[f"{canvas_key}_reset"] += 1
            st.rerun()

        _zone_io(all_zones, zones_key, canvas_key)
        return all_zones

    # ════════════════════════════════
    # 4-POINT MODE  (mirrors rect mode: no resets between zones)
    # ════════════════════════════════
    else:
        result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=1,
            stroke_color="#FFDD00",
            background_image=bg,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="point",
            point_display_radius=5,
            key=f"{canvas_key}_poly_{reset_n}",
        )

        # Group every 4 consecutive dots into a zone — no canvas resets
        all_dots      = _parse_dots(result.json_data, scale) if result.json_data else []
        n_complete    = len(all_dots) // 4
        canvas_zones  = [
            [all_dots[i * 4 + j] for j in range(4)]
            for i in range(n_complete)
        ]
        in_progress   = len(all_dots) % 4
        all_zones     = st.session_state[zones_key] + canvas_zones

        if in_progress > 0:
            st.caption(f"Points placed: {in_progress} / 4  —  {4 - in_progress} more for next zone")
        elif all_zones:
            total_on_canvas = len(all_dots)
            st.caption(f"{len(all_zones)} zone(s) marked  ({total_on_canvas} dots total)")

        can_remove = bool(all_zones) or (in_progress > 0)
        if st.button("Remove last", key=f"rm_{canvas_key}_poly",
                     disabled=not can_remove):
            if in_progress > 0:
                # Drop in-progress dots: commit complete canvas zones to background, reset canvas
                st.session_state[zones_key] = all_zones
            else:
                # Drop last zone
                st.session_state[zones_key] = all_zones[:-1]
            st.session_state[f"{canvas_key}_reset"] += 1
            st.rerun()

        _zone_io(all_zones, zones_key, canvas_key)
        return all_zones


# ── Zone save / load (shared by both modes) ───────────────────

def _zone_io(all_zones: list, zones_key: str, canvas_key: str):
    col_save, col_load = st.columns(2)

    with col_save:
        if all_zones:
            st.download_button(
                "Save zones",
                data=json.dumps({"zones": all_zones}, indent=2),
                file_name="parking_zones.json",
                mime="application/json",
                key=f"save_{canvas_key}",
            )

    with col_load:
        uploaded = st.file_uploader(
            "Load zones",
            type=["json"],
            key=f"load_{canvas_key}",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            try:
                loaded = json.load(uploaded).get("zones", [])
                if loaded:
                    st.session_state[zones_key]               = loaded
                    st.session_state[f"{canvas_key}_reset"]  += 1
                    st.rerun()
                else:
                    st.warning("JSON contains no zones.")
            except Exception as exc:
                st.error(f"Could not load zones: {exc}")
