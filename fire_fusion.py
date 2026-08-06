"""
Classical fire evidence — colour signature plus flicker, fused with YOLO.

Why this exists
---------------
`best.pt` is trained on 712 images. On clean validation imagery it is good, but
recall drops on frames the training set does not represent well: a small flame
close to the lens, harsh indoor lighting, a phone screen playing fire footage.
A missed frame is not a small problem for a safety demo — the badge flickers and
the operator loses trust.

Fire, however, has two physical signatures a network is not needed to see:

  1. Colour. Combustion sits in a narrow, highly saturated red-orange-yellow
     band at high brightness, usually with a near-white hot core.
  2. Flicker. A flame's area changes frame to frame. A red jacket, a poster or a
     traffic cone does not. This is what separates fire from fire-coloured
     objects, and it is the reason colour alone is not enough.

This module measures both and returns *evidence*, not a verdict. The verdict is
formed in fire_detection.detect_fire(), which combines this with the network.
Cost is roughly 2-3 ms per frame — a fraction of one forward pass.
"""

from __future__ import annotations

import cv2
import numpy as np

import config

# ---------------------------------------------------------------- colour bands
# OpenCV hue is 0-179. Fire spans the wrap-around point, so it needs two ranges.
_LOW_HUE = ((0, 95, 165), (32, 255, 255))       # red -> orange -> yellow
_HIGH_HUE = ((162, 95, 165), (179, 255, 255))   # deep red on the far side

# The hot core: bright and desaturated (approaching white). On its own this is
# also a lamp or a window, so it only counts where it touches fire-coloured
# pixels — see below.
_CORE = ((0, 0, 242), (179, 70, 255))

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _fire_mask(hsv):
    """Binary mask of fire-coloured pixels, morphologically cleaned."""
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, _LOW_HUE[0], _LOW_HUE[1]),
        cv2.inRange(hsv, _HIGH_HUE[0], _HIGH_HUE[1]),
    )

    # A hot core only counts as fire when it is adjacent to fire colour,
    # which is what stops ceiling lights and windows registering.
    core = cv2.inRange(hsv, _CORE[0], _CORE[1])
    near_fire = cv2.dilate(mask, _KERNEL, iterations=2)
    mask = cv2.bitwise_or(mask, cv2.bitwise_and(core, near_fire))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
    return mask


def _flicker_score(history) -> float:
    """
    How much the fire-coloured area is moving, 0..1.

    Uses the coefficient of variation of recent mask ratios. A real flame is
    turbulent, so its area wanders; a static orange object holds a near-constant
    ratio and scores close to zero. Returning 0 when there is not enough history
    means a brand-new detection leans on the network until evidence accumulates.
    """
    if len(history) < 4:
        return 0.0

    arr = np.asarray(history, dtype=np.float32)
    mean = float(arr.mean())

    if mean <= 1e-6:
        return 0.0

    cv_ = float(arr.std() / mean)

    # cv ~0.08 is a lively flame; anything above 0.25 is saturated for scoring.
    return float(min(1.0, cv_ / 0.25))


def colour_evidence(frame, history=None) -> dict:
    """
    Measure classical fire evidence on a BGR frame.

    Args:
        frame:   BGR ndarray.
        history: iterable of recent `ratio` values from this camera, oldest
                 first. Pass None to skip flicker scoring.

    Returns a dict:
        ratio      float  fraction of the frame that is fire-coloured, 0..1
        flicker    float  temporal variability of that ratio, 0..1
        blobs      int    number of contiguous regions above the area floor
        box        [x1,y1,x2,y2] | None   largest region
        center     [cx,cy] | None
        area       int    pixel area of the largest region
        strength   float  0..1 combined colour+flicker confidence
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _fire_mask(hsv)

    h, w = mask.shape[:2]
    frame_area = max(1, h * w)
    ratio = float(cv2.countNonZero(mask)) / frame_area

    # Ignore specks: a region must cover at least this share of the frame.
    min_area = int(frame_area * config.FIRE_COLOUR_MIN_BLOB_RATIO)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = [c for c in contours if cv2.contourArea(c) >= min_area]

    box = None
    center = None
    area = 0

    if regions:
        largest = max(regions, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        box = [int(x), int(y), int(x + bw), int(y + bh)]
        center = [int(x + bw // 2), int(y + bh // 2)]
        area = int(bw * bh)

    flicker = _flicker_score(list(history)) if history is not None else 0.0

    # Colour term saturates at the configured ratio; a frame that is 6% fire
    # colour is already unambiguous, and more does not add certainty.
    colour_term = min(1.0, ratio / max(config.FIRE_COLOUR_FULL_RATIO, 1e-6))

    # Colour carries the evidence, flicker corroborates it. Flicker alone must
    # never be able to raise strength — a shaking camera would qualify.
    strength = colour_term * (0.65 + 0.35 * flicker)

    return {
        "ratio": round(ratio, 5),
        "flicker": round(flicker, 3),
        "blobs": len(regions),
        "box": box,
        "center": center,
        "area": area,
        "strength": round(float(min(1.0, strength)), 3),
    }


def is_corroborating(evidence: dict) -> bool:
    """
    True when classical evidence is strong enough to confirm a weak network hit.

    Deliberately not a standalone fire test. This lowers the bar for a detection
    the network already suspects; it does not create one from nothing.
    """
    return (
        evidence["ratio"] >= config.FIRE_COLOUR_MIN_RATIO
        and evidence["blobs"] > 0
    )


def is_standalone(evidence: dict) -> bool:
    """
    True when classical evidence alone justifies raising fire.

    The bar here is much higher: a large, saturated, *flickering* region. This
    is the path that catches the frames the network misses entirely, and it is
    why the flicker term is mandatory rather than a bonus.
    """
    return (
        evidence["ratio"] >= config.FIRE_COLOUR_STANDALONE_RATIO
        and evidence["flicker"] >= config.FIRE_COLOUR_STANDALONE_FLICKER
        and evidence["blobs"] > 0
    )
