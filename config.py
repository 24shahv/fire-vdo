"""
SAFEX AI — central configuration.

Everything here can be overridden with environment variables so the same code
runs locally and on Render without edits.

NOTE: `CAMERA_SOURCES` (the old cv2.VideoCapture index list) has been removed.
Cameras are now supplied by each visitor's browser, so the server never owns a
capture device.
"""

import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- building map
GRID_ROWS = _int("GRID_ROWS", 10)
GRID_COLS = _int("GRID_COLS", 10)

# Exit A = index 0, Exit B = index 1 (same as the original project).
EXITS = [(0, 9), (9, 9)]

# Horizontal interior wall: row 5, columns 3..6 (unchanged).
WALL_ROW = _int("WALL_ROW", 5)
WALL_COL_START = _int("WALL_COL_START", 3)
WALL_COL_END = _int("WALL_COL_END", 7)  # exclusive

# ------------------------------------------------------------------- pipeline
# The frame size the browser uploads and that every pixel coordinate refers to.
FRAME_WIDTH = _int("FRAME_WIDTH", 640)
FRAME_HEIGHT = _int("FRAME_HEIGHT", 480)

# People detection runs on this downscaled copy — these exact numbers matter:
# the size/area filters in people_detection.py are tuned for this resolution.
PEOPLE_INPUT_WIDTH = _int("PEOPLE_INPUT_WIDTH", 224)
PEOPLE_INPUT_HEIGHT = _int("PEOPLE_INPUT_HEIGHT", 168)

# YOLO letterbox sizes. Benchmarked on a 1-vCPU box:
#   fire  @416 ~55 ms   (vs ~127 ms @640)
#   people@320 ~43 ms
FIRE_IMGSZ = _int("FIRE_IMGSZ", 416)
PEOPLE_IMGSZ = _int("PEOPLE_IMGSZ", 320)

FIRE_CONF = _float("FIRE_CONF", 0.25)
PEOPLE_CONF = _float("PEOPLE_CONF", 0.30)

# Run people detection every Nth frame and reuse the cached result in between
# (this is the `frame_count % 2 == 0` trick from the original main.py).
DETECT_STRIDE = _int("DETECT_STRIDE", 2)

# How many frames a stale people-detection result stays alive.
PEOPLE_MEMORY_FRAMES = _int("PEOPLE_MEMORY_FRAMES", 5)

# Smoke threshold — tuned for a 640x480 frame, so it moves with the frame area.
SMOKE_PIXEL_THRESHOLD = _int("SMOKE_PIXEL_THRESHOLD", 35000)

# Merge two detections into one person when their centres are closer than this.
DUPLICATE_DISTANCE = _int("DUPLICATE_DISTANCE", 50)

# Force Exit B once the building holds at least this many people.
CROWD_OVERRIDE_THRESHOLD = _int("CROWD_OVERRIDE_THRESHOLD", 5)

# Seconds between spoken evacuation announcements.
ALERT_COOLDOWN = _float("ALERT_COOLDOWN", 5.0)

# ------------------------------------------------------------------- runtime
MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
)
FIRE_MODEL_PATH = os.environ.get("FIRE_MODEL_PATH", os.path.join(MODEL_DIR, "best.pt"))
PEOPLE_MODEL_PATH = os.environ.get(
    "PEOPLE_MODEL_PATH", os.path.join(MODEL_DIR, "yolov8n.pt")
)

# Torch threads. Render Starter gives ~0.5 vCPU, Standard ~1. Oversubscribing
# threads on a small instance makes inference slower, not faster.
TORCH_THREADS = _int("TORCH_THREADS", 1)

# Inference is CPU bound; cap how many frames can be in the model at once so a
# burst of visitors queues instead of thrashing the box.
MAX_CONCURRENT_INFERENCE = _int("MAX_CONCURRENT_INFERENCE", 2)

# Drop a session's state after this many seconds of silence (memory hygiene).
SESSION_TTL_SECONDS = _float("SESSION_TTL_SECONDS", 90.0)
SESSION_SWEEP_SECONDS = _float("SESSION_SWEEP_SECONDS", 30.0)

# Largest JPEG the server will accept from a client (bytes).
MAX_FRAME_BYTES = _int("MAX_FRAME_BYTES", 4 * 1024 * 1024)

# Warm the models at startup so the first visitor doesn't eat the lazy-load cost.
WARMUP_ON_START = os.environ.get("WARMUP_ON_START", "1") not in ("0", "false", "False")
