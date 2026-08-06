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

# ------------------------------------------------------------------ skin bands
# Skin and flame overlap badly in HSV hue: lit skin sits at hue 5-25, which is
# squarely inside the fire band above. Hue alone therefore cannot separate a
# face from a flame, and on a warm indoor white-balance it will not even try.
#
# YCrCb does separate them. Skin occupies a tight, well-documented chroma
# cluster; flame sits outside it because combustion pushes Cr far higher and
# drives luma to the top of the range.
_SKIN_YCRCB = ((0, 133, 77), (255, 173, 127))

# A flame nearly always carries a small near-white core inside the coloured
# region. Skin never does. This is the second, independent separator.
_CORE_INSIDE_EROSION = 3

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _skin_mask(frame):
    """Binary mask of skin-coloured pixels, from BGR."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, _SKIN_YCRCB[0], _SKIN_YCRCB[1])
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)


def skin_fraction(frame) -> float:
    """Share of the frame (or crop) that is skin-coloured, 0..1."""
    if frame is None or frame.size == 0:
        return 0.0
    h, w = frame.shape[:2]
    return float(cv2.countNonZero(_skin_mask(frame))) / max(1, h * w)


def _fire_mask(hsv, bgr=None):
    """Binary mask of fire-coloured pixels, morphologically cleaned."""
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, _LOW_HUE[0], _LOW_HUE[1]),
        cv2.inRange(hsv, _HIGH_HUE[0], _HIGH_HUE[1]),
    )

    # A hot core only counts as fire when it sits INSIDE the coloured region.
    # The previous version accepted a core merely adjacent to fire colour, which
    # let a white wall behind a warm-lit face qualify — the wall is bright and
    # desaturated, and dilation reached it. Eroding the colour mask first means
    # the core has to be genuinely surrounded by flame colour.
    core = cv2.inRange(hsv, _CORE[0], _CORE[1])
    interior = cv2.erode(mask, _KERNEL, iterations=_CORE_INSIDE_EROSION)
    mask = cv2.bitwise_or(mask, cv2.bitwise_and(core, interior))

    # Remove skin. Lit skin is inside the fire hue band, so without this a face
    # in warm light is indistinguishable from flame on colour alone.
    if bgr is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(_skin_mask(bgr)))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
    return mask


def mask_signature(mask):
    """
    Downscale a fire mask to a small binary grid for temporal comparison.

    Kept tiny (32x24 = 768 bits) because a deque of these lives per camera and
    the whole point is that history is cheap.
    """
    small = cv2.resize(mask, (32, 24), interpolation=cv2.INTER_AREA)
    return (small > 96).astype(np.uint8)


def _flicker_score(history) -> float:
    """
    How much the fire region is *changing shape*, 0..1.

    This used to measure the variance of the fire-coloured AREA. That was wrong,
    and it is exactly why a phone screen playing fire footage scored zero: the
    flames churn violently but the total lit area barely moves, so the variance
    stayed flat and the standalone path never fired.

    What actually separates flame from a fire-coloured object is that the region
    changes SHAPE. Comparing consecutive mask signatures catches that — a flame
    rewrites its own outline every frame, a face or a traffic cone does not.

    Returns 0 when there is too little history, so a brand-new detection leans on
    the network until evidence accumulates.
    """
    sigs = [h for h in history if h is not None and not np.isscalar(h)]

    if len(sigs) < 3:
        return 0.0

    changes = []
    for a, b in zip(sigs, sigs[1:]):
        if a.shape != b.shape:
            continue
        union = int(np.count_nonzero(a | b))
        if union < 6:                       # nothing meaningful lit in either
            continue
        changed = int(np.count_nonzero(a ^ b))
        changes.append(changed / union)

    if not changes:
        return 0.0

    # A lively flame rewrites roughly a third of its own outline frame to frame.
    # Saturate there so a merely turbulent fire already scores 1.0.
    return float(min(1.0, (sum(changes) / len(changes)) / 0.33))


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
    # Work on a downscaled copy. Every quantity this function returns is either
    # a ratio (scale-invariant) or a geometry that can be scaled back up, so the
    # only thing lost is a few pixels of box precision — while the mask and
    # morphology work, which is the whole cost here, drops with the area.
    full_h, full_w = frame.shape[:2]
    scale = 1.0

    if config.COLOUR_EVIDENCE_MAX_WIDTH and full_w > config.COLOUR_EVIDENCE_MAX_WIDTH:
        scale = config.COLOUR_EVIDENCE_MAX_WIDTH / float(full_w)
        frame = cv2.resize(
            frame,
            (config.COLOUR_EVIDENCE_MAX_WIDTH, max(1, int(round(full_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _fire_mask(hsv, frame)

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

        # Back to full-frame coordinates. Callers — the synthetic fire box, the
        # grid projection — all work in the uploaded frame's space, so anything
        # geometric has to be undone here rather than left at working scale.
        inv = 1.0 / scale if scale else 1.0
        x, y, bw, bh = (int(round(v * inv)) for v in (x, y, bw, bh))

        box = [x, y, x + bw, y + bh]
        center = [x + bw // 2, y + bh // 2]
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
        # Stored by the caller and fed back in as history next frame.
        "signature": mask_signature(mask),
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

    Two tiers, because flicker needs history and history needs time. On a slow
    link the frame rate can fall to ~1 fps, and the flicker window then takes
    several seconds to fill — during which a frame that is visibly *full* of
    fire would raise nothing at all. That is the wrong failure for a safety
    system, so overwhelming colour is allowed to raise on a reduced flicker bar.

    Tier 1 — normal:       a large, saturated, clearly flickering region.
    Tier 2 — overwhelming: the frame is mostly fire colour (many times the
                           normal gate) and shows *some* movement. The ratio bar
                           here is far above anything skin, clothing or timber
                           produces after the skin mask, which is what keeps it
                           from becoming a false-positive path.

    Flicker is never dropped entirely. A perfectly static fire-coloured wall
    must not raise an alarm at any ratio.
    """
    if evidence["blobs"] <= 0:
        return False

    ratio = evidence["ratio"]
    flicker = evidence["flicker"]

    # Tier 1 — the standard path.
    if (ratio >= config.FIRE_COLOUR_STANDALONE_RATIO
            and flicker >= config.FIRE_COLOUR_STANDALONE_FLICKER):
        return True

    # Tier 2 — overwhelming colour, reduced flicker requirement.
    if (ratio >= config.FIRE_COLOUR_OVERWHELMING_RATIO
            and flicker >= config.FIRE_COLOUR_OVERWHELMING_FLICKER):
        return True

    return False
