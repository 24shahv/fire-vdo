"""
End-to-end pipeline check with the YOLO models stubbed out.

torch and ultralytics are replaced with fakes before `inference` is imported, so
the whole verdict path can be exercised on this machine. What is being tested is
the wiring the upgrade touches: fusion feeding detection, temporal confirmation
gating the alarm, hazard cells surviving a hysteresis gap, and severity landing
in the verdict.

    python3 test_pipeline.py
"""

import sys
import types

import cv2
import numpy as np

# ─────────────────────────── stub torch / ultralytics ───────────────────────
torch_stub = types.ModuleType("torch")


class _NoCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


torch_stub.inference_mode = lambda: _NoCtx()
torch_stub.set_num_threads = lambda n: None
torch_stub.set_grad_enabled = lambda b: None
sys.modules["torch"] = torch_stub

ul = types.ModuleType("ultralytics")


class _FakeYOLO:
    def __init__(self, path): self.names = {0: "fire", 1: "smoke"}
    def fuse(self): pass
    def predict(self, *a, **k): return []


ul.YOLO = _FakeYOLO
sys.modules["ultralytics"] = ul

# ─────────────────────────── now import the real thing ──────────────────────
import config  # noqa: E402
import models_loader  # noqa: E402

# Control what the "network" returns per call.
_network_boxes = []


class _Box:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [np.array(xyxy, dtype=np.float32)]
        self.conf = [float(conf)]
        self.cls = [int(cls)]


class _Result:
    def __init__(self, boxes): self.boxes = boxes


class _Model:
    names = {0: "fire", 1: "smoke"}
    def predict(self, *a, **k):
        return [_Result(list(_network_boxes))] if _network_boxes else [_Result(None)]


models_loader.get_fire_model = lambda: _Model()
models_loader.get_people_model = lambda: _Model()
models_loader.fire_lock = _NoCtx()
models_loader.people_lock = _NoCtx()

import people_detection  # noqa: E402
people_detection.detect_people = lambda frame: []

from inference import process_frame  # noqa: E402
from session import store  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  — {detail}" if detail else ""))


def jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def blank(): return np.zeros((480, 640, 3), dtype=np.uint8)


def flame(cx=320, cy=240, rx=70, ry=100):
    f = blank()
    cv2.ellipse(f, (cx, cy), (rx, ry), 0, 0, 360, (20, 105, 245), -1)
    cv2.ellipse(f, (cx, cy + ry // 5), (rx // 2, ry // 2), 0, 0, 360, (90, 200, 255), -1)
    cv2.ellipse(f, (cx, cy + ry // 4), (rx // 4, ry // 3), 0, 0, 360, (240, 250, 255), -1)
    return f


print("\n─── 1. Clean frame ───")
s = store.get_or_create("t-clean")
v = process_frame(s, 0, jpeg(blank()))
check("verdict returns ok", v["ok"])
check("no fire on a blank frame", v["fire"]["detected"] is False)
check("severity present in verdict", "severity" in v, f"score={v['severity']['score']}")
check("severity is NOMINAL", v["severity"]["level"] == "NOMINAL")
check("hazard_state present", "hazard_state" in v)
check("exit directive still produced", v["exit"]["name"] in ("A", "B"))
check("timing measured", v["timing_ms"] > 0, f"{v['timing_ms']} ms")


print("\n─── 2. Temporal confirmation gates the alarm ───")
_network_boxes = [_Box([260, 150, 380, 340], 0.82, 0)]
s = store.get_or_create("t-confirm")
need = config.FIRE_CONFIRM_FRAMES

v1 = process_frame(s, 0, jpeg(flame()))
check("frame 1: raw evidence seen", v1["fire"]["raw"] is True)
if need > 1:
    check("frame 1: alarm not yet confirmed", v1["fire"]["detected"] is False,
          f"needs {need} frames")

v = v1
for i in range(need - 1):
    v = process_frame(s, 0, jpeg(flame()))
check(f"frame {need}: alarm confirmed", v["fire"]["detected"] is True)
check("severity rose with confirmed fire", v["severity"]["score"] > 0,
      f"score={v['severity']['score']} level={v['severity']['level']}")


print("\n─── 3. Hysteresis survives dropped frames ───")
_network_boxes = []          # network now misses entirely
v_miss = process_frame(s, 0, jpeg(blank()))
check("alarm holds through a missed frame", v_miss["fire"]["detected"] is True,
      "hysteresis")
check("raw evidence correctly reports miss", v_miss["fire"]["raw"] is False)
check("hazard cells retained during hold", len(v_miss["grid_people"]) >= 0
      and any(2 in row for row in v_miss["grid"]),
      "fire cells must not vanish while the alarm is up")

for _ in range(config.FIRE_HOLD_FRAMES + 2):
    v_clear = process_frame(s, 0, jpeg(blank()))
check("alarm clears after hold expires", v_clear["fire"]["detected"] is False)
check("severity returns to nominal", v_clear["severity"]["level"] == "NOMINAL",
      f"score={v_clear['severity']['score']}")


print("\n─── 4. Weak network hit needs corroboration ───")
_network_boxes = [_Box([260, 150, 380, 340], 0.30, 0)]   # below FIRE_WEAK_CONF
s_weak = store.get_or_create("t-weak")
v_w = process_frame(s_weak, 0, jpeg(flame()))
srcs = [b.get("source") for b in v_w["detections"]["fire"]]
check("weak hit kept when colour corroborates", v_w["fire"]["raw"] is True,
      f"sources={srcs}")

s_noc = store.get_or_create("t-weak-nocolour")
v_n = process_frame(s_noc, 0, jpeg(blank()))   # weak hit, no fire colour
check("weak hit dropped without corroboration", v_n["fire"]["raw"] is False,
      "false positive suppressed")


print("\n─── 5. Standalone colour path (network misses entirely) ───")
_network_boxes = []
s_solo = store.get_or_create("t-solo")
rng = np.random.default_rng(3)
detected_any = False
for i in range(10):
    # Vary the flame size so the flicker signature builds, as a real flame does.
    r = int(70 + rng.normal(0, 12))
    v_s = process_frame(s_solo, 0, jpeg(flame(rx=max(40, r), ry=max(60, r + 30))))
    if v_s["fire"]["raw"]:
        detected_any = True
check("flickering flame detected with no network hit", detected_any,
      "this is the recall path the 712-image corpus misses")
src = [b.get("source") for b in v_s["detections"]["fire"]]
check("detection labelled as colour-sourced", "colour" in src or not src,
      f"sources={src}")

s_static = store.get_or_create("t-static")
for i in range(10):
    v_st = process_frame(s_static, 0, jpeg(flame()))   # identical every frame
check("static fire-coloured object never raises standalone",
      not any(b.get("source") == "colour" for b in v_st["detections"]["fire"]),
      "no flicker, no alarm")


print("\n─── 6. Verdict contract intact ───")
required = ["ok", "session_id", "cam_id", "seq", "detections", "smoke", "counts",
            "fire", "exit", "arrows", "grid", "routes", "timing_ms",
            "severity", "hazard_state", "evidence"]
missing = [k for k in required if k not in v_s]
check("all verdict keys present", not missing, f"missing={missing}" if missing else "")
check("severity has components", set(v_s["severity"]["components"]) ==
      {"fire", "extent", "smoke", "occupancy", "egress"})
check("evidence exposes flicker", "flicker" in v_s["evidence"])

failed = [r for r in results if r[0] == FAIL]
print(f"\n{'─' * 52}")
print(f"  {len(results) - len(failed)} / {len(results)} checks passed")
if failed:
    print("  FAILED:")
    for _, n, d in failed:
        print(f"    · {n}  {d}")
print(f"{'─' * 52}\n")
sys.exit(1 if failed else 0)
