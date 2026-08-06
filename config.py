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

# ------------------------------------------------------- classical fire fusion
# The network is trained on 712 images and misses frames the corpus does not
# represent. These control the classical colour+flicker channel that runs
# alongside it. See fire_fusion.py for why colour alone is not sufficient.
FIRE_FUSION_ENABLED = os.environ.get("FIRE_FUSION_ENABLED", "1") not in (
    "0", "false", "False",
)

# A network hit at or above this confidence is accepted on its own. This bar is
# deliberately high: the model produced a 0.47 false positive on a human face in
# warm indoor light, so anything in the middle of the range must be corroborated
# by physical evidence before it is allowed to raise an alarm.
FIRE_STRONG_CONF = _float("FIRE_STRONG_CONF", 0.72)

# Retained for compatibility and for anyone tuning the old way. No longer used
# as a bypass — see FIRE_STRONG_CONF above.
FIRE_WEAK_CONF = _float("FIRE_WEAK_CONF", 0.45)

# Skin veto. A detection whose box is at least this skin-coloured, in a scene
# that is not changing shape, is treated as a face rather than a flame.
FIRE_SKIN_VETO_RATIO = _float("FIRE_SKIN_VETO_RATIO", 0.22)
FIRE_SKIN_VETO_FLICKER = _float("FIRE_SKIN_VETO_FLICKER", 0.30)

# Fraction of the frame that must be fire-coloured to corroborate a weak hit.
FIRE_COLOUR_MIN_RATIO = _float("FIRE_COLOUR_MIN_RATIO", 0.004)

# Fire colour ratio at which the colour term is considered fully saturated.
FIRE_COLOUR_FULL_RATIO = _float("FIRE_COLOUR_FULL_RATIO", 0.06)

# A contiguous region must cover at least this share of the frame to count.
FIRE_COLOUR_MIN_BLOB_RATIO = _float("FIRE_COLOUR_MIN_BLOB_RATIO", 0.0012)

# Standalone classical detection — the path that catches frames the network
# misses entirely. Flicker is mandatory and now measures shape change rather
# than area variance, so a screen playing fire footage qualifies.
FIRE_COLOUR_STANDALONE_RATIO = _float("FIRE_COLOUR_STANDALONE_RATIO", 0.015)
FIRE_COLOUR_STANDALONE_FLICKER = _float("FIRE_COLOUR_STANDALONE_FLICKER", 0.25)

# Overwhelming-evidence tier. When this much of the frame is fire-coloured
# (after skin removal) the flicker bar drops, because waiting for the flicker
# window to fill can take several seconds at a low frame rate — and a frame this
# saturated should not be silent while that happens.
#
# 0.09 is roughly six times the normal gate. Measured for reference: a face in
# warm light with a timber door in shot reaches 0.005 after skin removal; a
# phone screen playing fire footage reaches 0.184. The margin between those two
# is what makes this tier safe.
FIRE_COLOUR_OVERWHELMING_RATIO = _float("FIRE_COLOUR_OVERWHELMING_RATIO", 0.09)
FIRE_COLOUR_OVERWHELMING_FLICKER = _float("FIRE_COLOUR_OVERWHELMING_FLICKER", 0.08)

# How many recent frames feed the flicker measurement. Kept short because the
# service can drop to ~1 fps on a cold free-tier instance, and a long window
# would then take ten seconds to fill before fire could ever be raised.
FIRE_COLOUR_HISTORY = _int("FIRE_COLOUR_HISTORY", 6)

# ------------------------------------------------- temporal confirmation
# Frames of evidence required before an alarm raises, and frames it holds after
# evidence stops. Raising is cautious; clearing is slow. See hazard_state.py.
FIRE_CONFIRM_FRAMES = _int("FIRE_CONFIRM_FRAMES", 2)
FIRE_HOLD_FRAMES = _int("FIRE_HOLD_FRAMES", 12)
SMOKE_CONFIRM_FRAMES = _int("SMOKE_CONFIRM_FRAMES", 3)
SMOKE_HOLD_FRAMES = _int("SMOKE_HOLD_FRAMES", 8)

# ------------------------------------------------------------ overlay polish
# Ease detection boxes between frames so the overlay tracks instead of jitters.
BOX_SMOOTHING = os.environ.get("BOX_SMOOTHING", "1") not in ("0", "false", "False")
BOX_SMOOTHING_ALPHA = _float("BOX_SMOOTHING_ALPHA", 0.45)

# ---------------------------------------------------------- severity scoring
# Fire area ratio at which the extent term saturates.
SEVERITY_FULL_FIRE_RATIO = _float("SEVERITY_FULL_FIRE_RATIO", 0.08)
# Occupancy at which the people-at-risk term saturates.
SEVERITY_FULL_OCCUPANCY = _int("SEVERITY_FULL_OCCUPANCY", 8)

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

# ------------------------------------------------------------- remote viewing
# When on, each camera's latest frame is held in memory so other clients in the
# SAME session can watch it — this is what makes every location see every
# camera. Cost is one JPEG (~40 KB) per active camera.
#
# PRIVACY: the session id is the only thing protecting these feeds. Anyone who
# knows it can watch every camera in that session. The client generates a long
# random id by default; do not replace it with something guessable like "demo"
# unless you intend the feeds to be public. Set REMOTE_VIEW_ENABLED=0 to turn
# the feature off entirely and go back to counts-only fusion.
REMOTE_VIEW_ENABLED = os.environ.get("REMOTE_VIEW_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)

# A remote feed older than this is shown as stale rather than as live video.
REMOTE_STALE_SECONDS = _float("REMOTE_STALE_SECONDS", 6.0)

# Warm the models at startup so the first visitor doesn't eat the lazy-load cost.
WARMUP_ON_START = os.environ.get("WARMUP_ON_START", "1") not in ("0", "false", "False")
