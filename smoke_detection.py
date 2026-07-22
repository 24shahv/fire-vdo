"""
Classical (non-neural) smoke detector: high-frequency texture loss.

Same algorithm as before. Two changes:
  * it no longer returns the threshold image (an ndarray can't cross a JSON
    boundary, and nothing consumed it);
  * the pixel threshold now scales with frame area, so the 35000 figure that was
    tuned for 640x480 keeps meaning the same thing if the frame size changes.

This ran nowhere in the original main.py. It is now wired into the pipeline.
"""

from __future__ import annotations

import cv2

import config

_REFERENCE_AREA = 640 * 480


def detect_smoke(frame):
    """
    Args:
        frame: BGR ndarray.

    Returns:
        (smoke_detected: bool, smoke_pixels: int, threshold: int)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    diff = cv2.absdiff(gray, blur)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    smoke_pixels = int(cv2.countNonZero(thresh))

    h, w = gray.shape[:2]
    scale = (w * h) / _REFERENCE_AREA
    limit = int(config.SMOKE_PIXEL_THRESHOLD * scale)

    return smoke_pixels > limit, smoke_pixels, limit
