"""
Fire / smoke detection with the custom-trained `best.pt` (classes: fire, smoke).

Detection logic is unchanged. What changed: this used to draw rectangles and
labels straight onto the frame with cv2. Drawing now happens in the browser, so
the function returns plain data instead.
"""

from __future__ import annotations

import torch

import config
import fire_fusion
import models_loader

# Labels from the trained model that count as a fire event.
FIRE_LABELS = {"fire", "smoke"}


def _looks_like_skin(frame, det, evidence) -> bool:
    """
    True when a network detection is better explained as skin than as flame.

    Two conditions must hold together:
      * the box is substantially skin-coloured, and
      * the scene is not changing shape the way a flame does.

    Both are required. Skin alone is not enough to veto — a person standing in
    front of a real fire would trip it — so the flicker term is what keeps a
    genuine flame safe.
    """
    x1, y1, x2, y2 = det["box"]
    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    skin = fire_fusion.skin_fraction(crop)

    return (
        skin >= config.FIRE_SKIN_VETO_RATIO
        and evidence.get("flicker", 0.0) < config.FIRE_SKIN_VETO_FLICKER
    )


def detect_fire(frame, colour_history=None):
    """
    Run fire detection on a full-size BGR frame.

    Args:
        frame:          BGR ndarray.
        colour_history: recent fire-colour ratios from this camera, for flicker
                        scoring. None disables the classical channel entirely.

    Returns (fire_detected, fire_center, fire_size, boxes, evidence) where:
      fire_detected : bool
      fire_center   : (cx, cy) of the LARGEST detection, or None
      fire_size     : area in px of that largest detection
      boxes         : list of every accepted detection, as dicts:
                      {"box": [x1,y1,x2,y2], "label": str, "conf": float,
                       "center": [cx,cy], "area": int, "source": str}
      evidence      : classical colour/flicker measurement (see fire_fusion)

    Detections are now graded rather than simply thresholded. A high-confidence
    network hit stands on its own. A weak hit is kept only when classical
    evidence corroborates it. And when the network returns nothing at all but
    the frame shows a large, flickering, saturated fire-coloured region, a
    synthetic box is raised from that region — this is the path that covers the
    frames the 712-image corpus does not represent.
    """
    model = models_loader.get_fire_model()

    with models_loader.fire_lock, torch.inference_mode():
        results = model.predict(
            frame,
            imgsz=config.FIRE_IMGSZ,
            conf=config.FIRE_CONF,
            verbose=False,
        )

    # ------------------------------------------------- classical evidence
    if config.FIRE_FUSION_ENABLED:
        evidence = fire_fusion.colour_evidence(frame, colour_history)
    else:
        evidence = {
            "ratio": 0.0, "flicker": 0.0, "blobs": 0,
            "box": None, "center": None, "area": 0, "strength": 0.0,
        }

    corroborated = config.FIRE_FUSION_ENABLED and fire_fusion.is_corroborating(evidence)

    # --------------------------------------------------- network detections
    raw = []

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

            raw.append(
                {
                    "box": [x1, y1, x2, y2],
                    "label": label,
                    "conf": round(conf, 3),
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                    "area": area,
                }
            )

    boxes = []

    for b in raw:
        # A box that is mostly skin and is not changing shape is a face, not a
        # flame. This is checked before confidence, because the network itself
        # is what produced the false positive — deferring to its score here is
        # exactly the mistake that put FIRE on a presenter's face.
        if config.FIRE_FUSION_ENABLED and _looks_like_skin(frame, b, evidence):
            continue

        if b["conf"] >= config.FIRE_STRONG_CONF:
            # Only a genuinely confident hit stands alone.
            b["source"] = "model"
            boxes.append(b)
        elif corroborated:
            # The network suspected fire and the frame physically agrees. This
            # is the difference between catching a small flame and missing it.
            b["source"] = "model+colour"
            boxes.append(b)
        # else: mid or weak hit with no physical support — dropped.

    # ------------------------------------------------- standalone fallback
    if not boxes and config.FIRE_FUSION_ENABLED and fire_fusion.is_standalone(evidence):
        boxes.append(
            {
                "box": evidence["box"],
                "label": "fire",
                # Reported as a real but lower confidence so the UI and the
                # severity score can tell this apart from a network detection.
                "conf": round(0.35 + 0.4 * evidence["strength"], 3),
                "center": evidence["center"],
                "area": evidence["area"],
                "source": "colour",
            }
        )

    if not boxes:
        return False, None, 0, [], evidence

    largest = max(boxes, key=lambda b: b["area"])
    return True, tuple(largest["center"]), largest["area"], boxes, evidence
