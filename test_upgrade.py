"""
Offline checks for the fusion + temporal upgrade.

Runs without torch: it exercises fire_fusion and hazard_state directly, which
are the two modules the upgrade actually adds. Synthetic frames stand in for
camera input so the thresholds can be checked deterministically.

    python3 test_upgrade.py
"""

import sys

import cv2
import numpy as np

import config
import fire_fusion
from hazard_state import CameraHazard, HazardTracker, compute_severity

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, condition, detail=""):
    results.append((PASS if condition else FAIL, name, detail))
    print(f"  [{PASS if condition else FAIL}] {name}" + (f"  — {detail}" if detail else ""))


def blank(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def flame(frame, cx, cy, rx, ry, bright=1.0):
    """Paint a plausible flame: orange body with a hot yellow-white core."""
    cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360,
                (int(20 * bright), int(105 * bright), int(245 * bright)), -1)
    cv2.ellipse(frame, (cx, cy + ry // 5), (rx // 2, ry // 2), 0, 0, 360,
                (int(90 * bright), int(200 * bright), int(255 * bright)), -1)
    cv2.ellipse(frame, (cx, cy + ry // 4), (rx // 4, ry // 3), 0, 0, 360,
                (240, 250, 255), -1)
    return frame


print("\n─── 1. Colour evidence ───")

f = flame(blank(), 320, 240, 60, 90)
ev = fire_fusion.colour_evidence(f, None)
check("flame produces fire-colour ratio", ev["ratio"] > config.FIRE_COLOUR_MIN_RATIO,
      f"ratio={ev['ratio']}")
check("flame produces a bounding region", ev["box"] is not None and ev["blobs"] > 0,
      f"blobs={ev['blobs']}")
check("flame corroborates a weak network hit", fire_fusion.is_corroborating(ev))

empty = blank()
ev_empty = fire_fusion.colour_evidence(empty, None)
check("empty frame gives no evidence", ev_empty["ratio"] < 0.001 and ev_empty["blobs"] == 0,
      f"ratio={ev_empty['ratio']}")
check("empty frame does not corroborate", not fire_fusion.is_corroborating(ev_empty))

# A dim room with a grey wall must not read as fire.
room = np.full((480, 640, 3), 90, dtype=np.uint8)
ev_room = fire_fusion.colour_evidence(room, None)
check("grey scene does not corroborate", not fire_fusion.is_corroborating(ev_room),
      f"ratio={ev_room['ratio']}")

# A bright white lamp on its own must not read as fire.
lamp = blank()
cv2.circle(lamp, (320, 200), 70, (255, 255, 255), -1)
ev_lamp = fire_fusion.colour_evidence(lamp, None)
check("white lamp alone does not corroborate", not fire_fusion.is_corroborating(ev_lamp),
      f"ratio={ev_lamp['ratio']}")


print("\n─── 2. Flicker discrimination ───")

# Static orange object: constant area across frames.
static_hist = [0.030] * 12
static_ev = fire_fusion.colour_evidence(flame(blank(), 320, 240, 70, 100), static_hist)
check("static object scores low flicker", static_ev["flicker"] < 0.25,
      f"flicker={static_ev['flicker']}")
check("static object rejected as standalone", not fire_fusion.is_standalone(static_ev),
      "colour alone must not raise fire")

# Real flame: area wanders frame to frame.
rng = np.random.default_rng(7)
live_hist = list(0.030 + rng.normal(0, 0.011, 12))
live_ev = fire_fusion.colour_evidence(flame(blank(), 320, 240, 70, 100), live_hist)
check("flickering flame scores high flicker", live_ev["flicker"] >= 0.28,
      f"flicker={live_ev['flicker']}")
check("flickering flame accepted as standalone", fire_fusion.is_standalone(live_ev),
      "this is the path that covers network misses")

check("strength ordering is correct", live_ev["strength"] > static_ev["strength"],
      f"{live_ev['strength']} > {static_ev['strength']}")


print("\n─── 3. Temporal confirmation ───")

t = HazardTracker(confirm_frames=3, hold_frames=5)
check("single frame does not raise alarm", t.update(True) is False, "streak 1 of 3")
check("second frame does not raise alarm", t.update(True) is False, "streak 2 of 3")
check("third frame raises alarm", t.update(True) is True, "confirmed")
check("alarm reports as active", t.active)

# One dropped frame must not clear it.
check("single miss holds the alarm", t.update(False) is True, "hysteresis")
check("alarm marked as holding", t.holding)
for _ in range(4):
    t.update(False)
check("alarm clears after hold expires", t.update(False) is False)

# A lone false positive must never raise.
t2 = HazardTracker(confirm_frames=3, hold_frames=5)
t2.update(True); t2.update(False); t2.update(True); t2.update(False)
check("alternating noise never raises", not t2.active, "false positive suppressed")


print("\n─── 4. Box smoothing ───")

hz = CameraHazard()
b1 = [{"box": [100, 100, 200, 200], "center": [150, 150], "area": 10000}]
b2 = [{"box": [120, 100, 220, 200], "center": [170, 150], "area": 10000}]
hz.smooth_boxes(b1)
sm = hz.smooth_boxes(b2)
x1 = sm[0]["box"][0]
check("smoothed box eases toward new position", 100 < x1 < 120, f"x1={x1} (raw jump 100→120)")
check("smoothing cache does not grow", len(hz._smoothed) == 1, f"entries={len(hz._smoothed)}")


print("\n─── 5. Severity scoring ───")

calm = compute_severity(
    fire_active=False, fire_boxes=[], fire_area_ratio=0.0,
    smoke_active=False, smoke_ratio=0.1, people_count=3,
    exits_total=2, exits_blocked=0,
)
check("calm building scores NOMINAL", calm["level"] == "NOMINAL",
      f"score={calm['score']}")

fire_small = compute_severity(
    fire_active=True, fire_boxes=[{}], fire_area_ratio=0.01,
    smoke_active=False, smoke_ratio=0.3, people_count=1,
    exits_total=2, exits_blocked=0,
)
check("small fire, empty room is ELEVATED or HIGH",
      fire_small["level"] in ("ELEVATED", "HIGH"), f"score={fire_small['score']}")

fire_bad = compute_severity(
    fire_active=True, fire_boxes=[{}], fire_area_ratio=0.09,
    smoke_active=True, smoke_ratio=1.4, people_count=9,
    exits_total=2, exits_blocked=1,
)
check("large fire, crowded, exit blocked is CRITICAL",
      fire_bad["level"] == "CRITICAL", f"score={fire_bad['score']}")

check("severity is monotonic", calm["score"] < fire_small["score"] < fire_bad["score"],
      f"{calm['score']} < {fire_small['score']} < {fire_bad['score']}")
check("score is bounded at 100", fire_bad["score"] <= 100, f"score={fire_bad['score']}")
check("components are exposed", set(fire_bad["components"]) ==
      {"fire", "extent", "smoke", "occupancy", "egress"})

# Occupancy must not inflate a calm building.
calm_crowd = compute_severity(
    fire_active=False, fire_boxes=[], fire_area_ratio=0.0,
    smoke_active=False, smoke_ratio=0.0, people_count=20,
    exits_total=2, exits_blocked=0,
)
check("crowd alone is not a hazard", calm_crowd["score"] == 0,
      f"score={calm_crowd['score']}")


print("\n─── 6. Confirmation window moves the gauge ───")

partial = compute_severity(
    fire_active=False, fire_boxes=[], fire_area_ratio=0.0,
    smoke_active=False, smoke_ratio=0.0, people_count=2,
    exits_total=2, exits_blocked=0, colour_strength=0.8,
)
check("unconfirmed fire colour lifts score off zero", partial["score"] > 0,
      f"score={partial['score']} during confirmation")
check("unconfirmed stays below a confirmed fire", partial["score"] < fire_small["score"],
      f"{partial['score']} < {fire_small['score']}")


failed = [r for r in results if r[0] == FAIL]
print(f"\n{'─' * 52}")
print(f"  {len(results) - len(failed)} / {len(results)} checks passed")
if failed:
    print("  FAILED:")
    for _, name, detail in failed:
        print(f"    · {name}  {detail}")
print(f"{'─' * 52}\n")

sys.exit(1 if failed else 0)
