"""
Regression tests for the two failures seen on stage rehearsal:

  1. A human face in warm indoor light was reported as FIRE at 0.47.
  2. A phone screen playing fire footage was NOT detected at all.

Both traced to the same two design errors:
  * a mid-confidence network hit bypassed all corroboration, and
  * flicker measured AREA variance, which is flat for a full-screen fire video.

Run:  python3 test_regression.py        (no torch needed)
"""

import sys
import types

import cv2
import numpy as np

# ── stub torch / ultralytics so this runs without the ML stack installed ──
_t = types.ModuleType("torch")


class _NoCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


_t.inference_mode = lambda: _NoCtx()
_t.set_num_threads = lambda n: None
_t.set_grad_enabled = lambda b: None
sys.modules["torch"] = _t

_ul = types.ModuleType("ultralytics")


class _YOLO:
    def __init__(self, *a, **k): self.names = {0: "fire", 1: "smoke"}
    def predict(self, *a, **k): return []
    def fuse(self): return self


_ul.YOLO = _YOLO
sys.modules["ultralytics"] = _ul

import config
import fire_detection
import fire_fusion

PASS = FAIL = 0


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {label}" + (f"  — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {label}" + (f"  — {detail}" if detail else ""))


# ────────────────────────────────────────────────────────── synthetic scenes

def face_frame(w=640, h=480):
    """Warm-lit skin tone against a white wall — the false positive scene."""
    img = np.full((h, w, 3), 232, np.uint8)          # white wall
    # Skin: BGR roughly (150, 175, 205) is a warm mid brown-orange.
    cv2.ellipse(img, (w // 3, h // 2), (110, 150), 0, 0, 360, (150, 175, 205), -1)
    cv2.ellipse(img, (w // 3, h // 2 + 190), (150, 90), 0, 0, 360, (150, 175, 205), -1)
    return img


def fire_frame(w=640, h=480, phase=0, area_locked=True):
    """
    Fire footage on a phone screen: a dark rectangle full of saturated flame,
    where the flames churn but the lit AREA stays essentially constant.
    """
    img = np.full((h, w, 3), 235, np.uint8)          # room behind
    x1, y1, x2, y2 = 110, 130, 530, 360
    cv2.rectangle(img, (x1, y1), (x2, y2), (18, 18, 18), -1)   # dark screen

    rng = np.random.default_rng(1000 + phase)
    # Fixed number of flame tongues => near-constant lit area, moving positions.
    for i in range(26):
        cx = x1 + 20 + int(((i * 37 + phase * 29) % (x2 - x1 - 40)))
        base = y2 - 12
        top = base - rng.integers(60, 190)
        wdt = rng.integers(8, 20)
        colour = (0, int(rng.integers(90, 190)), int(rng.integers(225, 256)))
        pts = np.array([[cx - wdt, base], [cx, top], [cx + wdt, base]], np.int32)
        cv2.fillPoly(img, [pts], colour)

    # Hot core near the base.
    cv2.ellipse(img, ((x1 + x2) // 2, y2 - 26), (140, 20), 0, 0, 360, (205, 240, 252), -1)
    return img


def static_orange_frame(w=640, h=480):
    """A traffic cone / hi-vis jacket: fire-coloured, but never changes."""
    img = np.full((h, w, 3), 225, np.uint8)
    cv2.rectangle(img, (200, 150), (430, 380), (0, 120, 245), -1)
    return img


def history_for(make_frame, n=6):
    """Build a signature history the way the running service would."""
    hist = []
    for i in range(n):
        ev = fire_fusion.colour_evidence(make_frame(phase=i), hist)
        hist.append(ev["signature"])
    return hist


# ───────────────────────────────────────────────────────────────── the tests

print("\n─── 1. Face must not read as fire ───")

face = face_frame()
face_hist = []
for _ in range(6):
    ev_face = fire_fusion.colour_evidence(face, face_hist)
    face_hist.append(ev_face["signature"])

check(fire_fusion.skin_fraction(face) > 0.15,
      "scene is recognised as substantially skin",
      f"skin={fire_fusion.skin_fraction(face):.3f}")

check(not fire_fusion.is_standalone(ev_face),
      "static face never raises standalone fire",
      f"ratio={ev_face['ratio']:.4f} flicker={ev_face['flicker']:.2f}")

check(ev_face["flicker"] < config.FIRE_SKIN_VETO_FLICKER,
      "a motionless face scores low flicker",
      f"flicker={ev_face['flicker']:.2f}")

# The exact stage failure: network says 0.47 on a face.
det = {"box": [180, 120, 460, 420], "label": "fire", "conf": 0.47,
       "center": [320, 270], "area": 84000}

check(fire_detection._looks_like_skin(face, det, ev_face),
      "0.47 network hit on a face is vetoed",
      "this is the exact stage false positive")

check(0.47 < config.FIRE_STRONG_CONF,
      "0.47 no longer clears the stand-alone bar",
      f"FIRE_STRONG_CONF={config.FIRE_STRONG_CONF}")


print("\n─── 2. Screen fire must be detected ───")

hist = history_for(fire_frame, n=6)
ev_fire = fire_fusion.colour_evidence(fire_frame(phase=6), hist)

check(ev_fire["ratio"] >= config.FIRE_COLOUR_STANDALONE_RATIO,
      "enough of the frame is fire-coloured",
      f"ratio={ev_fire['ratio']:.4f} need≥{config.FIRE_COLOUR_STANDALONE_RATIO}")

check(ev_fire["flicker"] >= config.FIRE_COLOUR_STANDALONE_FLICKER,
      "churning flames score high flicker",
      f"flicker={ev_fire['flicker']:.2f} need≥{config.FIRE_COLOUR_STANDALONE_FLICKER}")

check(fire_fusion.is_standalone(ev_fire),
      "screen fire raises with no network hit at all",
      "this is the frame the model missed on stage")

# The old metric's blind spot, stated explicitly.
areas = []
for i in range(6):
    areas.append(fire_fusion.colour_evidence(fire_frame(phase=i), None)["ratio"])
arr = np.asarray(areas)
cv_ = float(arr.std() / max(arr.mean(), 1e-6))
check(cv_ < 0.25,
      "area variance alone would have MISSED this fire",
      f"old-metric cv={cv_:.3f} — under the 0.25 bar, scored ~0")


print("\n─── 3. Fire-coloured but static must stay rejected ───")

cone = static_orange_frame()
cone_hist = []
for _ in range(6):
    ev_cone = fire_fusion.colour_evidence(cone, cone_hist)
    cone_hist.append(ev_cone["signature"])

check(ev_cone["ratio"] >= config.FIRE_COLOUR_STANDALONE_RATIO,
      "cone is large and fire-coloured",
      f"ratio={ev_cone['ratio']:.4f}")

check(not fire_fusion.is_standalone(ev_cone),
      "but never raises, because it does not change shape",
      f"flicker={ev_cone['flicker']:.2f}")


print("\n─── 4. Person in front of real fire is still detected ───")

# Skin present AND the scene is churning — the veto must not fire here.
mixed = fire_frame(phase=3)
cv2.ellipse(mixed, (95, 300), (55, 85), 0, 0, 360, (150, 175, 205), -1)
mixed_hist = history_for(fire_frame, n=6)
ev_mixed = fire_fusion.colour_evidence(mixed, mixed_hist)

det_mixed = {"box": [110, 130, 530, 360], "label": "fire", "conf": 0.47,
             "center": [320, 245], "area": 96600}

check(not fire_detection._looks_like_skin(mixed, det_mixed, ev_mixed),
      "veto does NOT fire when the scene is churning",
      f"flicker={ev_mixed['flicker']:.2f} — a bystander cannot mask a real fire")


print("\n─── 5. Signature plumbing ───")

check(ev_fire["signature"] is not None and ev_fire["signature"].shape == (24, 32),
      "signature is emitted at the expected size",
      f"shape={ev_fire['signature'].shape}")

check(ev_fire["signature"].nbytes < 1024,
      "signature is small enough to keep per camera",
      f"{ev_fire['signature'].nbytes} bytes x {config.FIRE_COLOUR_HISTORY} frames")

wire = {k: v for k, v in ev_fire.items() if k != "signature"}
import json  # noqa: E402
try:
    json.dumps(wire)
    ok = True
except TypeError:
    ok = False
check(ok, "wire payload stays JSON serialisable once signature is stripped")


print("\n" + "─" * 52)
print(f"  {PASS} / {PASS + FAIL} checks passed")
print("─" * 52 + "\n")
sys.exit(1 if FAIL else 0)
