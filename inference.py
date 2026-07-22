"""
The SAFEX AI pipeline: one uploaded frame in, one JSON verdict out.

This is a direct port of the `while True:` loop that used to live in main.py.
The decision-making is identical — what changed is that frames arrive from a
browser instead of cv2.VideoCapture, and results leave as JSON instead of being
painted onto a cv2 window.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

import config
from building_map import create_grid, mark_fire
from fire_detection import detect_fire
from pathfinding import find_safe_path
from people_detection import detect_people
from session import Session
from smoke_detection import detect_smoke
from utils import map_to_grid, remove_duplicate_people

log = logging.getLogger("safex.inference")

MAX_PATHS = 8  # cap route solving so a crowded frame can't stall the worker


class FrameDecodeError(ValueError):
    """Raised when the uploaded bytes aren't a decodable image."""


# --------------------------------------------------------------------- exits
def choose_best_exit(global_people, exits, grid):
    """
    Pick the least-loaded exit that isn't next to fire.

    Unchanged from the original: count people within Manhattan distance 3 of
    each exit, and disqualify (cost 999) any exit with fire in its 5x5
    neighbourhood.
    """
    exit_load = {}

    for ex in exits:
        count = 0

        for person in global_people:
            pr, pc = person
            er, ec = ex

            if abs(pr - er) + abs(pc - ec) < 3:
                count += 1

        # avoid exits near fire
        danger = False
        for r in range(-2, 3):
            for c in range(-2, 3):
                nr, nc = ex[0] + r, ex[1] + c
                if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                    if grid[nr][nc] == 2:
                        danger = True

        if danger:
            exit_load[ex] = 999
        else:
            exit_load[ex] = count

    best_exit = min(exit_load, key=exit_load.get)
    return best_exit, exit_load


# ------------------------------------------------------------------- decode
def decode_frame(payload: bytes):
    """JPEG/PNG bytes -> BGR ndarray at the configured working resolution."""
    if not payload:
        raise FrameDecodeError("empty frame payload")

    if len(payload) > config.MAX_FRAME_BYTES:
        raise FrameDecodeError("frame too large")

    buf = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if frame is None:
        raise FrameDecodeError("could not decode frame")

    if (frame.shape[1], frame.shape[0]) != (config.FRAME_WIDTH, config.FRAME_HEIGHT):
        frame = cv2.resize(
            frame,
            (config.FRAME_WIDTH, config.FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    return frame


# ----------------------------------------------------------------- pipeline
def process_frame(session: Session, cam_id: int, payload: bytes) -> dict:
    """
    Analyse one frame for one camera and return the whole building's verdict.

    Safe to call concurrently for different sessions; the session lock keeps a
    single session's cameras consistent with each other.
    """
    started = time.perf_counter()
    frame = decode_frame(payload)
    h, w = frame.shape[:2]

    with session.lock:
        cam = session.camera(cam_id)
        cam.frame_count += 1
        cam.last_seen = time.time()
        run_detection = cam.should_run_detection()

    # ---------------------------------------------------------- fire + smoke
    fire_detected, fire_center, fire_size, fire_boxes = detect_fire(frame)

    fire_grid_positions = []
    for box in fire_boxes:
        cx, cy = box["center"]
        fire_grid_positions.append(map_to_grid(cx, cy, w, h))

    smoke_detected, smoke_pixels, smoke_limit = detect_smoke(frame)

    # -------------------------------------------------------------- people
    # Detect on a small copy — the filters in people_detection.py are tuned for
    # exactly this size, so the resize is part of the algorithm, not an optim.
    small = cv2.resize(
        frame,
        (config.PEOPLE_INPUT_WIDTH, config.PEOPLE_INPUT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    if run_detection:
        detected = detect_people(small)
        with session.lock:
            people = cam.remember_people(detected)
    else:
        with session.lock:
            people = list(cam.people_memory)

    scale_x = w / config.PEOPLE_INPUT_WIDTH
    scale_y = h / config.PEOPLE_INPUT_HEIGHT

    cam_people_grid = []
    cam_people_boxes = []

    for person in people:
        (x1, y1, x2, y2) = person["box"]
        (px, py) = person["center"]

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        px = int(px * scale_x)
        py = int(py * scale_y)

        cam_people_boxes.append(
            {
                "box": [x1, y1, x2, y2],
                "center": [px, py],
                "conf": person.get("conf", 0.0),
            }
        )
        cam_people_grid.append((map_to_grid(px, py, w, h), (px, py)))

    # ------------------------------------------- publish + fuse all cameras
    with session.lock:
        cam.people_grid = cam_people_grid
        cam.people_boxes = cam_people_boxes
        cam.fire_grid = fire_grid_positions
        cam.fire_boxes = fire_boxes
        cam.smoke_detected = smoke_detected

        # Publish this frame for the other viewers in the session. The old
        # bytes are dropped on reassignment, so exactly one frame per camera is
        # ever resident.
        if config.REMOTE_VIEW_ENABLED:
            cam.last_jpeg = payload
            cam.last_jpeg_at = time.time()

        global_people = []
        global_fire_positions = []
        fire_detected_global = False
        smoke_detected_global = False
        active_cams = sorted(session.cameras.keys())

        for other in session.cameras.values():
            global_people.extend(other.people_grid)
            global_fire_positions.extend(other.fire_grid)
            fire_detected_global = fire_detected_global or bool(other.fire_boxes)
            smoke_detected_global = smoke_detected_global or other.smoke_detected

    global_people = remove_duplicate_people(global_people)
    total_people = len(global_people)
    cam_people_count = len(cam_people_boxes)

    # --------------------------------------------------------------- grid
    grid = create_grid()
    for fire_pos in global_fire_positions:
        grid = mark_fire(grid, fire_pos)

    grid_people_only = [p[0] for p in global_people]
    best_exit, exit_load = choose_best_exit(grid_people_only, config.EXITS, grid)

    # Crowd override, straight from the original: a busy building always goes
    # to Exit B regardless of what the load balancer preferred.
    override = total_people >= config.CROWD_OVERRIDE_THRESHOLD
    if override:
        best_exit = config.EXITS[1]

    if best_exit == config.EXITS[0]:
        exit_text = "USE EXIT A"
        exit_name = "A"
        direction = "right"
    else:
        exit_text = "USE EXIT B"
        exit_name = "B"
        direction = "left"

    # ------------------------------------------------------------- arrows
    # Only this camera's people get arrows; each browser draws its own view.
    er, ec = best_exit
    arrows = []
    for (grid_pos, (px, py)) in cam_people_grid:
        pr, pc = grid_pos
        dx = ec - pc
        dy = er - pr
        arrows.append({"from": [px, py], "to": [px + int(dx * 20), py + int(dy * 20)]})

    # -------------------------------------------------------------- routes
    routes = []
    for (grid_pos, _) in global_people[:MAX_PATHS]:
        path = find_safe_path(grid, grid_pos, [best_exit])
        if path:
            routes.append([[r, c] for (r, c) in path])

    # ------------------------------------------------------------ announce
    announce = None
    if fire_detected_global:
        with session.lock:
            if session.may_announce():
                announce = f"Move to Exit {exit_name}"

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return {
        "ok": True,
        "session_id": session.session_id,
        "cam_id": cam_id,
        "seq": cam.frame_count,
        "frame": {"w": w, "h": h},
        "ran_detection": run_detection,
        "detections": {
            "fire": fire_boxes,
            "people": cam_people_boxes,
        },
        "smoke": {
            "detected": smoke_detected,
            "pixels": smoke_pixels,
            "threshold": smoke_limit,
            "any_camera": smoke_detected_global,
        },
        "counts": {
            "camera_people": cam_people_count,
            "total_people": total_people,
            "cameras": len(active_cams),
        },
        "fire": {
            "detected": fire_detected,
            "any_camera": fire_detected_global,
            "center": list(fire_center) if fire_center else None,
            "size": fire_size,
        },
        "exit": {
            "chosen": list(best_exit),
            "name": exit_name,
            "text": exit_text,
            "direction": direction,
            "override": override,
            "load": {f"{r},{c}": v for (r, c), v in exit_load.items()},
            "all": [list(e) for e in config.EXITS],
        },
        "arrows": arrows,
        "grid": grid,
        "grid_people": [[p[0][0], p[0][1]] for p in global_people],
        "routes": routes,
        "announce": announce,
        "timing_ms": round(elapsed_ms, 1),
    }
