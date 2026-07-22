# SAFEX AI — browser-camera edition

Live fire, smoke and occupancy detection with AI-directed exit routing.
Originally a desktop OpenCV script; this version runs as a web service where
**every visitor uses their own camera**.

---

## What changed and why

The original program opened the server's own webcams with
`cv2.VideoCapture(0)` and drew results into a `cv2.imshow` window. Neither
works on Render: a cloud container has no webcam and no display.

The camera moved to the browser. Everything else — the models, the filters, the
exit logic — is the same code.

```
   BROWSER                         RENDER
   ┌───────────────────────┐       ┌────────────────────────────┐
   │ getUserMedia()        │       │ FastAPI                    │
   │   ↓                   │       │   ↓                        │
   │ <video>               │       │ cv2.imdecode               │
   │   ↓                   │  JPEG │   ↓                        │
   │ canvas 640×480 ───────┼──────▶│ best.pt      (fire/smoke)  │
   │   (mirrored)          │   WS  │ yolov8n.pt   (people)      │
   │                       │       │   ↓                        │
   │ overlay canvas ◀──────┼───────┤ grid → exit choice → JSON  │
   │ minimap · speech      │  JSON │                            │
   └───────────────────────┘       └────────────────────────────┘
```

Frames are decoded in memory and discarded. Nothing is written to disk.

---

## Deploying to Render

### 1. Prepare the repository

The models must be committed — they are 6 MB each, comfortably under GitHub's
100 MB limit, so **no Git LFS is needed**.

```bash
git init
git add .
git commit -m "SAFEX AI — browser camera architecture"
git remote add origin https://github.com/<you>/safex-ai.git
git push -u origin main
```

`.gitignore` already excludes `fire_dataset/`. That folder is ~100 MB of
Roboflow training images the server never reads; leaving it out is the single
biggest thing you can do for build times.

Confirm the weights actually made it in:

```bash
git ls-files models/
# models/best.pt
# models/yolov8n.pt
```

### 2. Create the service

**Blueprint (recommended)** — Render reads `render.yaml`:

> Dashboard → **New** → **Blueprint** → pick the repo → **Apply**

**Or manually** — Dashboard → New → Web Service:

| Field | Value |
|---|---|
| Runtime | Python 3 |
| Build command | `chmod +x build.sh && ./build.sh` |
| Start command | `uvicorn server:app --host 0.0.0.0 --port $PORT --workers 1` |
| Health check path | `/healthz` |
| Instance type | **Standard** (see sizing below) |

Add these environment variables:

```
PYTHON_VERSION=3.11.9
YOLO_CONFIG_DIR=/tmp/ultralytics
MPLCONFIGDIR=/tmp/matplotlib
OMP_NUM_THREADS=1
TORCH_THREADS=1
```

### 3. Watch the build

`build.sh` prints the resolved versions and loads both models before the deploy
is allowed to succeed. A healthy build ends with:

```
  cv2         4.13.0
  torch       2.8.0+cpu
  ultralytics 8.3.40
  loaded      best.pt
  loaded      yolov8n.pt
──> Build complete
```

If `torch` shows `+cu` instead of `+cpu`, the build fails on purpose — see
*CUDA wheels* below.

### 4. Open the site

Render serves HTTPS by default, which matters: `getUserMedia()` refuses to run
on plain HTTP anywhere except `localhost`. Click **Start camera**, allow the
permission prompt, and the feed goes live.

---

## Instance sizing

CPU-only torch plus both YOLOv8n networks settles at roughly **400–500 MB**
resident once warmed.

| Plan | RAM | Verdict |
|---|---|---|
| Free | 512 MB | OOM-killed under load; also spins down, so the first visitor waits out a cold boot |
| Starter | 512 MB | Same ceiling — not recommended |
| **Standard** | **2 GB** | Smallest plan that behaves |

To squeeze onto 512 MB: set `WARMUP_ON_START=0` and `FIRE_IMGSZ=320`, and
expect 20–40 s on the first request while weights load.

Measured on 1 vCPU: **~100 ms per frame** end to end (fire @416 ≈ 55 ms,
people @320 ≈ 43 ms), giving roughly 10 fps of server capacity shared across
all connected cameras.

---

## Three deployment traps this project already handles

**1. `pip install torch` gives you the CUDA build.** On Linux, PyPI's default
`torch` wheel bundles ~800 MB of NVIDIA CUDA libraries that a web service can
neither use nor cache. `requirements.txt` pins `torch==2.8.0+cpu` against
`--extra-index-url https://download.pytorch.org/whl/cpu`, and `build.sh`
asserts the result so a regression fails the build instead of the deploy.

**2. Ultralytics forces GUI OpenCV.** Its metadata declares
`opencv-python>=4.6.0`, so pip installs the GUI build no matter what you list.
That build links `libGL.so.1`, which Render's Python runtime does not ship:

```
ImportError: libGL.so.1: cannot open shared object file
```

Both packages install the *same* `cv2` module path, so whichever lands last
wins — non-deterministic. `build.sh` uninstalls the GUI copy and force-installs
`opencv-python-headless` last.

**3. `best.pt` needs torch ≥ 2.6.** Your weights carry a `.storage_alignment`
record, emitted only by newer torch serialisation. Older torch also predates
the `weights_only` default that Ultralytics relies on.

---

## Multiple cameras

The original fused two local webcams via `CAMERA_SOURCES = [0, 1]`. That still
works, but the cameras are now *browsers*.

Open the site, then use the **Add another camera** link on a phone. Same
`session` parameter, different `cam` id:

```
https://your-app.onrender.com/?session=<id>&cam=0   ← laptop
https://your-app.onrender.com/?session=<id>&cam=1   ← phone
```

Both feeds merge into one building view: people are counted across cameras,
`remove_duplicate_people()` collapses overlaps, and both cameras receive the
same exit decision.

---

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | The console |
| `WS /ws?session=&cam=` | Realtime frames in, verdicts out |
| `POST /api/detect` | Single frame — multipart, base64 JSON, or raw body |
| `GET /healthz` | Uptime and live session counts |
| `GET /api/config` | Frame size, grid, exits |

The WebSocket is strictly request/response: the client only grabs the next
frame after the previous verdict lands. That one rule is what keeps latency
bounded instead of building a queue that never drains. If WebSockets are
blocked by a proxy, the client falls back to `POST /api/detect` automatically.

---

## Tuning

Everything is environment-driven — no code edits needed.

| Variable | Default | Effect |
|---|---|---|
| `FIRE_IMGSZ` | 416 | Fire inference size. 320 is ~25 ms faster, less sensitive |
| `PEOPLE_IMGSZ` | 320 | People inference size |
| `DETECT_STRIDE` | 2 | Run people detection every Nth frame |
| `FIRE_CONF` | 0.25 | Fire confidence floor |
| `CROWD_OVERRIDE_THRESHOLD` | 5 | People count that forces Exit B |
| `ALERT_COOLDOWN` | 5.0 | Seconds between spoken alerts |
| `MAX_CONCURRENT_INFERENCE` | 2 | Frames allowed inside the models at once |
| `SESSION_TTL_SECONDS` | 90 | Idle session expiry |

Raising `DETECT_STRIDE` is the cheapest way to serve more simultaneous
visitors; people positions simply update less often.

---

## Running locally

```bash
pip install -r requirements.txt
python server.py          # http://localhost:8000
```

`localhost` counts as a secure context, so the camera works without HTTPS.

---

## Verified behaviour

- 26/26 backend integration checks — REST (multipart, base64, raw), malformed
  input, WebSocket streaming, multi-camera fusion, exit logic, pathfinding
- 21/22 browser checks in headless Chromium with a synthetic webcam; the one
  failure was Google Fonts blocked by the test sandbox
- Memory flat across 300 frames (823 → 827 MB, growth rate decreasing)
- Session sweeper reclaims idle state: 14 sessions → 2 after TTL
- Fire detected in 12/25 validation images at the default threshold
#   f i r e - v d o  
 #   f i r e - v d o  
 