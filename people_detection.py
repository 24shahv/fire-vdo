"""
People detection with yolov8n (COCO class 0 = person).

Every filter below is carried over from the original file unchanged. They are
tuned for the 224x168 downscaled frame that inference.py feeds in — if you
change PEOPLE_INPUT_WIDTH/HEIGHT in config.py, retune these thresholds too.
"""

from __future__ import annotations

import torch

import config
import models_loader

# Minimum plausible person, in the 224x168 detection frame.
MIN_HEIGHT = 40
MIN_WIDTH = 20
MIN_AREA = 2000

# A standing person is taller than wide, but not a lamp post.
MIN_ASPECT = 0.8
MAX_ASPECT = 3.5

# Ignore detections hugging the frame edge (hanging clothes, door frames...).
EDGE_MARGIN = 0.1


def detect_people(frame):
    """
    Args:
        frame: BGR ndarray, expected at PEOPLE_INPUT_WIDTH x PEOPLE_INPUT_HEIGHT.

    Returns:
        list of {"box": (x1,y1,x2,y2), "center": (cx,cy), "conf": float}
        in the coordinate space of `frame`.
    """
    model = models_loader.get_people_model()

    with models_loader.people_lock, torch.inference_mode():
        results = model.predict(
            frame,
            imgsz=config.PEOPLE_IMGSZ,
            conf=config.PEOPLE_CONF,
            classes=[0],  # ask YOLO for people only — cheaper than filtering after
            verbose=False,
        )

    people = []
    h_frame, w_frame = frame.shape[:2]

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls != 0:
                continue

            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

            width = x2 - x1
            height = y2 - y1
            area = width * height

            # 1. minimum size
            if height < MIN_HEIGHT or width < MIN_WIDTH:
                continue

            # 2. minimum area
            if area < MIN_AREA:
                continue

            # 3. aspect ratio
            aspect_ratio = height / max(width, 1)
            if aspect_ratio < MIN_ASPECT or aspect_ratio > MAX_ASPECT:
                continue

            # 4. drop edge junk
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if cx < EDGE_MARGIN * w_frame or cx > (1 - EDGE_MARGIN) * w_frame:
                continue
            if cy < EDGE_MARGIN * h_frame or cy > (1 - EDGE_MARGIN) * h_frame:
                continue

            people.append(
                {
                    "box": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "conf": round(conf, 3),
                }
            )

    return people
