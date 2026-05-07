"""
Background video processing via daemon threads.

State is stored in a module-level dict (not session_state) so the background
thread can write to it safely.  The Streamlit UI reads from it via get_job().
"""

import os
import sys
import threading

import cv2
import numpy as np

_jobs: dict = {}
_lock = threading.Lock()


def get_job(job_id: int) -> dict | None:
    """Return a snapshot of the job state, or None if not found."""
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def cancel_job(job_id: int):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["cancelled"] = True


def start_job(
    job_id: int,
    vpath: str,
    zones: list,
    fps: float,
    total_frames: int,
    interval_frames: int,
    stability_frames: int,
    cnn_model,
    cnn_device,
    conf_threshold: float,
):
    """Start a background processing thread for job_id."""
    with _lock:
        # Cancel any previous job for this session
        if job_id in _jobs:
            _jobs[job_id]["cancelled"] = True
        _jobs[job_id] = {
            "progress":     0.0,
            "events":       [],
            "latest_frame": None,
            "free":         0,
            "occupied":     0,
            "done":         False,
            "cancelled":    False,
            "error":        None,
        }

    t = threading.Thread(
        target=_run,
        args=(job_id, vpath, zones, fps, total_frames,
              interval_frames, stability_frames, cnn_model, cnn_device, conf_threshold),
        daemon=True,
    )
    t.start()


def _run(job_id, vpath, zones, fps, total_frames,
         interval_frames, stability_frames, cnn_model, cnn_device, conf_threshold):
    # Imports here so the thread sees the app/ directory on sys.path
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    from classifier import is_occupied
    from ui import draw_zones

    n              = len(zones)
    status_history = [[] for _ in range(n)]
    stable         = [False] * n
    prev           = [None]  * n
    events         = []

    try:
        cap = cv2.VideoCapture(vpath)

        for frame_num in range(total_frames):
            with _lock:
                if _jobs[job_id].get("cancelled"):
                    break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % interval_frames != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t         = frame_num / fps

            confs = []
            for i, zone in enumerate(zones):
                occ, conf = is_occupied(frame_rgb, zone, cnn_model, cnn_device)
                confs.append(conf)
                status_history[i].append(occ)
                if len(status_history[i]) > stability_frames:
                    status_history[i].pop(0)

                if len(status_history[i]) == stability_frames:
                    if all(status_history[i]):
                        new = True
                    elif not any(status_history[i]):
                        new = False
                    else:
                        new = stable[i]

                    if prev[i] is not None and new != stable[i]:
                        mm, ss = int(t // 60), int(t % 60)
                        events.append({
                            "Time":  f"{mm:02d}:{ss:02d}",
                            "Space": f"#{i + 1}",
                            "Event": "occupied" if new else "freed",
                        })

                    prev[i]   = stable[i]
                    stable[i] = new

            annotated = draw_zones(frame_rgb, zones, stable, confs, conf_threshold)
            free_n    = stable.count(False)
            occ_n     = stable.count(True)
            mm, ss    = int(t // 60), int(t % 60)
            cv2.putText(annotated, f"Free: {free_n}  Occupied: {occ_n}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(annotated, f"{mm:02d}:{ss:02d}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            with _lock:
                _jobs[job_id]["progress"]     = frame_num / total_frames
                _jobs[job_id]["latest_frame"] = annotated
                _jobs[job_id]["free"]         = free_n
                _jobs[job_id]["occupied"]     = occ_n
                _jobs[job_id]["events"]       = list(events)

        cap.release()

    except Exception as exc:
        with _lock:
            _jobs[job_id]["error"] = str(exc)
    finally:
        with _lock:
            _jobs[job_id]["progress"] = 1.0
            _jobs[job_id]["done"]     = True
