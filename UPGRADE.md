# SAFEX AI — Detection Reliability Upgrade

Three changes, all aimed at one thing: the alarm must be **right** and it must
be **steady**. A safety system that flickers is not trusted, and a system that
is not trusted gets muted.

## 1. Classical fire fusion (`fire_fusion.py`)

`best.pt` is trained on 712 images. It misses frames the corpus does not
represent — a small flame near the lens, harsh indoor light, fire footage on a
screen. Rather than retrain (which needs data we do not have tonight), a second,
independent channel now runs alongside the network:

* **Colour** — combustion sits in a narrow, saturated red-orange-yellow band at
  high brightness, usually with a near-white hot core. The core only counts
  where it touches fire colour, which is what stops ceiling lights qualifying.
* **Flicker** — a flame's area wanders frame to frame. A red jacket, a poster or
  a traffic cone does not. This is measured as the coefficient of variation of
  the colour ratio over the last 12 frames.

Detections are now **graded** instead of simply thresholded:

| Network confidence | Colour evidence | Result |
|---|---|---|
| ≥ `FIRE_WEAK_CONF` (0.45) | anything | accepted, `source: model` |
| ≥ `FIRE_CONF` (0.25) but weak | corroborating | accepted, `source: model+colour` |
| ≥ `FIRE_CONF` but weak | none | **dropped** — as before |
| nothing at all | large **and** flickering | accepted, `source: colour` |

The last row is the recall path. The bar there is deliberately high and flicker
is mandatory — colour alone can never raise fire, or a hi-vis jacket would.

## 2. Temporal confirmation and hysteresis (`hazard_state.py`)

Per-frame detection has two failure modes with one cause — treating each frame
as an independent event:

* One bad frame raises a false alarm.
* One missed frame clears a real alarm, so the badge blinks.

The alarm now has memory. Fire must be seen on `FIRE_CONFIRM_FRAMES` (2) frames
to raise, and holds for `FIRE_HOLD_FRAMES` (12) after evidence stops. The
asymmetry is intentional: raising is cautious, clearing is slow. Hazard cells on
the map are retained through the hold, otherwise the twin would clear the fire
while the alarm is still up and route people back through it.

Everything downstream — the directive, cross-camera fusion, the feed wall — now
reads the **confirmed** state, not raw per-frame evidence.

## 3. Hazard severity score

One 0–100 number, because an operator under stress can act on one number and
cannot act on six.

| Term | Max | Rationale |
|---|---|---|
| Fire present | 40 | the dominant term |
| Fire extent | 15 | a spreading fire is worse than a contained one |
| Smoke | 15 | leading cause of casualties in real fires |
| Occupancy at risk | 15 | an empty building is property loss, not life safety |
| Exits compromised | 15 | losing egress is what turns an incident fatal |

Levels: `NOMINAL` < 25, `ELEVATED` < 50, `HIGH` < 75, `CRITICAL` ≥ 75.

Occupancy only scores when something is actually wrong — a crowded, safe
building scores zero. The component breakdown ships in the verdict and renders
on the dashboard, because an unexplained number is not actionable.

Also included: **box smoothing** (EMA, α 0.45) so the overlay tracks instead of
jittering.

## Verdict additions

```json
"severity":     { "score": 54, "level": "HIGH", "components": {...} },
"hazard_state": { "fire": {"active":true,"confirming":false,"holding":false,
                           "streak":4,"needs":2,"duration_s":1.8}, "smoke": {...} },
"evidence":     { "ratio":0.053, "flicker":0.71, "blobs":1, "strength":0.94 },
"fire":         { "detected": true, "raw": true, ... }
```

`fire.detected` is the confirmed alarm; `fire.raw` is this frame's evidence.

## Tests

```bash
python3 test_upgrade.py     # 31 checks — fusion, flicker, temporal, severity
python3 test_pipeline.py    # 24 checks — full verdict path, YOLO stubbed
```

Both run without torch installed.

## Tuning for the demo

Everything is environment-driven; no code edit needed.

| Variable | Default | Effect |
|---|---|---|
| `FIRE_FUSION_ENABLED` | 1 | Set 0 to revert to network-only detection |
| `FIRE_WEAK_CONF` | 0.45 | Below this a network hit needs corroboration |
| `FIRE_COLOUR_STANDALONE_RATIO` | 0.018 | **Lower to ~0.012 if fire is not detecting on stage** |
| `FIRE_COLOUR_STANDALONE_FLICKER` | 0.28 | Lower to ~0.20 to loosen the flicker requirement |
| `FIRE_CONFIRM_FRAMES` | 2 | Raise to 3–4 if false alarms appear |
| `FIRE_HOLD_FRAMES` | 12 | Raise for a steadier badge, lower to clear faster |
| `BOX_SMOOTHING_ALPHA` | 0.45 | Lower = smoother, higher = more responsive |

**Rollback:** set `FIRE_FUSION_ENABLED=0` and `FIRE_CONFIRM_FRAMES=1`,
`FIRE_HOLD_FRAMES=0`. Behaviour returns to the previous build with no code
change.

---

# Hotfix — rehearsal failures

Two failures on stage rehearsal, both caused by the section above, both fixed.

## What happened

| Observed | Cause |
|---|---|
| A face read as **FIRE 47%** | A network hit at or above `FIRE_WEAK_CONF` (0.45) bypassed corroboration entirely. 0.47 cleared it. Lit skin also sits inside the fire hue band, so the colour channel could not have objected anyway. |
| Fire on a phone screen **not detected** | Flicker measured the variance of the fire-coloured *area*. A full-screen fire video holds a near-constant area while the flames churn, so it scored ~0 and the standalone path never opened. Measured: cv = 0.074 against a 0.25 bar. |

## Fixes

**1. Skin is now removed from the fire mask.** HSV hue cannot separate skin from
flame — both sit at 5-25. YCrCb can, because skin occupies a tight chroma
cluster that combustion does not. A face now contributes a fire ratio of
**0.0000** where it previously contributed a large region.

**2. The hot core must sit inside the flame, not beside it.** The previous
version dilated the colour mask and accepted any bright desaturated pixel that
landed nearby — which reached the white wall behind a warm-lit face. The mask is
now eroded first, so the core has to be genuinely surrounded by flame colour.

**3. Flicker measures shape change, not area change.** Consecutive fire masks are
downscaled to a 32x24 binary signature and compared. A flame rewrites roughly a
third of its own outline every frame; a face or a traffic cone rewrites none.
This is what makes screen fire detectable and a static orange object still
rejected — the two cases the area metric got backwards.

**4. The bypass threshold moved from 0.45 to `FIRE_STRONG_CONF` 0.72**, and a
skin veto now runs *before* confidence is consulted. A box that is ≥22% skin in a
scene that is not churning is dropped whatever the network claims. The flicker
condition is what keeps a bystander standing in front of a real fire safe.

## Verification

```
test_regression.py   15/15   the two stage failures, reproduced then fixed
test_upgrade.py      31/31
test_pipeline.py     24/24
```

`test_regression.py` asserts the old metric would have missed the fire
(cv = 0.074), so the regression cannot silently return.

## Live tuning

| Symptom | Change |
|---|---|
| Fire still not detecting | `FIRE_COLOUR_STANDALONE_RATIO` 0.015 → 0.010 |
| | `FIRE_COLOUR_STANDALONE_FLICKER` 0.25 → 0.18 |
| Face still reading as fire | `FIRE_STRONG_CONF` 0.72 → 0.80 |
| | `FIRE_SKIN_VETO_RATIO` 0.22 → 0.15 |
| Alarm flickering | `FIRE_HOLD_FRAMES` 12 → 18 |

**Rollback:** `FIRE_FUSION_ENABLED=0`, `FIRE_CONFIRM_FRAMES=1`, `FIRE_HOLD_FRAMES=0`.
Note this also removes the skin veto — the face false positive is a property of
`best.pt` itself, and the veto is the only thing suppressing it.
