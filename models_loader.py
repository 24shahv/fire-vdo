"""
Loads the two YOLO networks exactly once per process.

The original code created a `YOLO(...)` at module import time inside both
fire_detection.py and people_detection.py. That is fine for a script that runs
once, but on a web server it means the weights are re-read whenever a module is
re-imported, and it makes startup order unpredictable. Everything now goes
through this module.
"""

from __future__ import annotations

import logging
import os
import threading

import torch
from ultralytics import YOLO

import config

log = logging.getLogger("safex.models")

# Keep the CPU maths single-threaded-ish; see config.TORCH_THREADS.
torch.set_num_threads(max(1, config.TORCH_THREADS))
torch.set_grad_enabled(False)

_lock = threading.Lock()
_fire_model: YOLO | None = None
_people_model: YOLO | None = None

# Ultralytics/torch are not safe to call re-entrantly from many threads on the
# same model object. One lock per model keeps inference correct while still
# letting fire and people detection overlap.
fire_lock = threading.Lock()
people_lock = threading.Lock()


def _load(path: str, what: str) -> YOLO:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{what} weights not found at {path!r}. "
            "Commit the .pt file to the repo or set FIRE_MODEL_PATH / "
            "PEOPLE_MODEL_PATH."
        )
    log.info("Loading %s model from %s", what, path)
    model = YOLO(path)
    model.fuse()  # small but free speedup for inference
    return model


def get_fire_model() -> YOLO:
    global _fire_model
    if _fire_model is None:
        with _lock:
            if _fire_model is None:
                _fire_model = _load(config.FIRE_MODEL_PATH, "fire")
    return _fire_model


def get_people_model() -> YOLO:
    global _people_model
    if _people_model is None:
        with _lock:
            if _people_model is None:
                _people_model = _load(config.PEOPLE_MODEL_PATH, "people")
    return _people_model


def warmup() -> None:
    """Run one throwaway inference per model so the first real frame is fast."""
    import numpy as np

    blank = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype="uint8")
    small = np.zeros(
        (config.PEOPLE_INPUT_HEIGHT, config.PEOPLE_INPUT_WIDTH, 3), dtype="uint8"
    )

    with fire_lock:
        get_fire_model().predict(
            blank, imgsz=config.FIRE_IMGSZ, conf=config.FIRE_CONF, verbose=False
        )
    with people_lock:
        get_people_model().predict(
            small, imgsz=config.PEOPLE_IMGSZ, conf=config.PEOPLE_CONF, verbose=False
        )
    log.info("Model warmup complete")
