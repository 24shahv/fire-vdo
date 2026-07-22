#!/usr/bin/env bash
# ============================================================================
# Render build step for SAFEX AI.
# ============================================================================
set -o errexit
set -o pipefail

echo "──> Python: $(python --version)"

pip install --upgrade pip
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Ultralytics declares `opencv-python>=4.6.0` as a hard dependency, so pip
# installs the GUI build no matter what requirements.txt says. That build links
# against libGL.so.1, which does not exist on Render's Python runtime, and the
# import fails with:
#
#   ImportError: libGL.so.1: cannot open shared object file
#
# Both packages ship the exact same `cv2` module, so the fix is to drop the GUI
# copy and reinstall the headless one last, guaranteeing it owns the files.
# ---------------------------------------------------------------------------
echo "──> Removing GUI OpenCV in favour of the headless build"
pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true
pip install --force-reinstall --no-deps opencv-python-headless==4.13.0.92

# Fail the build here rather than at 3am on the first request.
echo "──> Verifying runtime imports"
python - <<'PY'
import cv2, torch, ultralytics
print(f"  cv2         {cv2.__version__}")
print(f"  torch       {torch.__version__}")
print(f"  ultralytics {ultralytics.__version__}")
assert "+cu" not in torch.__version__, (
    "CUDA build of torch was installed — check the --extra-index-url line "
    "in requirements.txt"
)

from ultralytics import YOLO
import os
for name in ("best.pt", "yolov8n.pt"):
    path = os.path.join("models", name)
    assert os.path.exists(path), f"missing weights: {path}"
    YOLO(path)
    print(f"  loaded      {name}")
PY

echo "──> Build complete"
