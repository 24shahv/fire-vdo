"""
Fire / smoke detection with the custom-trained `best.pt` (classes: fire, smoke).

Detection logic is unchanged. What changed: this used to draw rectangles and
labels straight onto the frame with cv2. Drawing now happens in the browser, so
the function returns plain data instead.
"""

from __future__ import annotations

import torch

import config
import models_loader

# Labels from the trained model that count as a fire event.
FIRE_LABELS = {"fire", "smoke"}


def detect_fire(frame):
    """
    Run fire detection on a full-size BGR frame.

    Returns (fire_detected, fire_center, fire_size, boxes) where:
      fire_detected : bool
      fire_center   : (cx, cy) of the LARGEST detection, or None
      fire_size     : area in px of that largest detection
      boxes         : list of every accepted detection, as dicts:
                      {"box": [x1,y1,x2,y2], "label": str, "conf": float,
                       "center": [cx,cy], "area": int}

    The original returned only the *last* box the loop happened to see. Picking
    the largest is both deterministic and a better proxy for the seat of the
    fire; `boxes` exposes all of them so nothing is lost.
    """
    model = models_loader.get_fire_model()

    with models_loader.fire_lock, torch.inference_mode():
        results = model.predict(
            frame,
            imgsz=config.FIRE_IMGSZ,
            conf=config.FIRE_CONF,
            verbose=False,
        )

    boxes = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf < config.FIRE_CONF:
                continue

            label = model.names.get(cls, str(cls))
            if label.lower() not in FIRE_LABELS:
                continue

            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            area = max(0, x2 - x1) * max(0, y2 - y1)

            boxes.append(
                {
                    "box": [x1, y1, x2, y2],
                    "label": label,
                    "conf": round(conf, 3),
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                    "area": area,
                }
            )

    if not boxes:
        return False, None, 0, []

    largest = max(boxes, key=lambda b: b["area"])
    return True, tuple(largest["center"]), largest["area"], boxes
