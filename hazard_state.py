"""
Temporal confirmation, alarm hysteresis and hazard severity scoring.

Why this exists
---------------
Per-frame detection has two failure modes that look opposite but share a cause —
treating each frame as an independent event:

  * A single bad frame raises a false alarm. One reflection, one frame, and the
    operator sees FIRE. Systems that do this get muted by their users.
  * A single missed frame clears a real alarm. The network detects on frames
    1 and 3 but not 2, and the badge blinks. A blinking alarm reads as broken,
    which is worse than an alarm that is simply wrong.

Both are fixed by giving the alarm memory. Fire must be seen on CONFIRM_FRAMES
consecutive-ish frames before it raises, and once raised it holds for
HOLD_FRAMES frames after evidence stops. The asymmetry is deliberate: raising is
cautious, clearing is slow. In a life-safety system those are the correct
directions to be wrong in.

The severity score then collapses the whole picture — fire, smoke, how much of
the frame is burning, how many people are inside, whether an exit is
compromised — into one 0-100 number, because an operator under stress can act on
one number and cannot act on six.
"""

from __future__ import annotations

import time
from collections import deque

import config

# ------------------------------------------------------------------- severity
LEVELS = (
    (75, "CRITICAL"),
    (50, "HIGH"),
    (25, "ELEVATED"),
    (0, "NOMINAL"),
)


def severity_level(score: int) -> str:
    for floor, name in LEVELS:
        if score >= floor:
            return name
    return "NOMINAL"


class HazardTracker:
    """
    Per-camera temporal state for one hazard channel (fire or smoke).

    Not thread-safe on its own; callers hold the session lock, which is the same
    lock that already guards every other piece of per-camera state.
    """

    def __init__(self, confirm_frames: int, hold_frames: int):
        self.confirm_frames = max(1, confirm_frames)
        self.hold_frames = max(0, hold_frames)

        self.streak = 0          # consecutive frames with evidence
        self.hold = 0            # frames remaining in hysteresis hold
        self.active = False      # confirmed alarm state
        self.raised_at = 0.0     # wall clock of the last raise
        self.last_evidence = 0.0 # wall clock of the last positive frame

    def update(self, evidence: bool) -> bool:
        """Feed one frame's raw detection; returns the confirmed alarm state."""
        now = time.time()

        if evidence:
            self.streak += 1
            self.hold = self.hold_frames
            self.last_evidence = now

            if not self.active and self.streak >= self.confirm_frames:
                self.active = True
                self.raised_at = now
        else:
            self.streak = 0

            if self.active:
                if self.hold > 0:
                    self.hold -= 1
                else:
                    self.active = False

        return self.active

    @property
    def confirming(self) -> bool:
        """Evidence is accumulating but the alarm has not raised yet."""
        return not self.active and self.streak > 0

    @property
    def holding(self) -> bool:
        """Alarm is up on hysteresis rather than on live evidence."""
        return self.active and self.streak == 0

    def duration(self) -> float:
        """Seconds the alarm has been up, 0 when clear."""
        return round(time.time() - self.raised_at, 1) if self.active else 0.0

    def state(self) -> dict:
        return {
            "active": self.active,
            "confirming": self.confirming,
            "holding": self.holding,
            "streak": self.streak,
            "needs": self.confirm_frames,
            "duration_s": self.duration(),
        }


class CameraHazard:
    """Everything temporal that one camera remembers about hazards."""

    def __init__(self):
        self.fire = HazardTracker(
            config.FIRE_CONFIRM_FRAMES, config.FIRE_HOLD_FRAMES
        )
        self.smoke = HazardTracker(
            config.SMOKE_CONFIRM_FRAMES, config.SMOKE_HOLD_FRAMES
        )
        # Recent fire-colour ratios, for the flicker measurement.
        self.colour_history = deque(maxlen=config.FIRE_COLOUR_HISTORY)
        # Smoothed detection boxes, keyed by a coarse position bucket.
        self._smoothed = {}

    def push_colour(self, ratio: float) -> None:
        self.colour_history.append(float(ratio))

    def smooth_boxes(self, boxes):
        """
        Exponentially smooth box coordinates between frames.

        Raw detector output jitters by several pixels every frame even on a
        static scene. On the overlay that reads as a nervous, unreliable box.
        Matching each box to its nearest previous position and easing toward it
        costs nothing and makes the overlay look like it is tracking rather than
        re-guessing.
        """
        if not config.BOX_SMOOTHING:
            return boxes

        alpha = config.BOX_SMOOTHING_ALPHA
        out = []
        used = set()

        for b in boxes:
            x1, y1, x2, y2 = b["box"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            key = (cx // 60, cy // 60)

            prev = self._smoothed.get(key)
            if prev and key not in used:
                x1 = int(prev[0] + alpha * (x1 - prev[0]))
                y1 = int(prev[1] + alpha * (y1 - prev[1]))
                x2 = int(prev[2] + alpha * (x2 - prev[2]))
                y2 = int(prev[3] + alpha * (y2 - prev[3]))

            self._smoothed[key] = (x1, y1, x2, y2)
            used.add(key)

            nb = dict(b)
            nb["box"] = [x1, y1, x2, y2]
            nb["center"] = [(x1 + x2) // 2, (y1 + y2) // 2]
            out.append(nb)

        # Drop buckets that no longer have a detection, so the cache cannot grow.
        for key in list(self._smoothed.keys()):
            if key not in used:
                del self._smoothed[key]

        return out


# ------------------------------------------------------------------- scoring
def compute_severity(
    *,
    fire_active: bool,
    fire_boxes,
    fire_area_ratio: float,
    smoke_active: bool,
    smoke_ratio: float,
    people_count: int,
    exits_total: int,
    exits_blocked: int,
    colour_strength: float = 0.0,
) -> dict:
    """
    Collapse the hazard picture into a 0-100 score with a named level.

    The weighting reflects what actually drives risk to life, not what is
    easiest to measure:

        Fire present        0-40   the dominant term, as it should be
        Fire extent         0-15   a spreading fire is worse than a contained one
        Smoke               0-15   the leading cause of casualties in real fires
        Occupancy at risk   0-15   an empty building is a property loss, not a
                                   life-safety emergency
        Exits compromised   0-15   losing egress is what turns an incident fatal

    Returns the score, level, and the component breakdown so the dashboard can
    show *why* — an unexplained number is not actionable.
    """
    components = {}

    # --- fire presence -----------------------------------------------------
    if fire_active:
        base = 40.0
    elif colour_strength > 0:
        # Unconfirmed but visible fire colour still lifts the score off zero,
        # which is what makes the gauge move during the confirmation window
        # instead of jumping from 0 to 40 in one frame.
        base = 18.0 * min(1.0, colour_strength)
    else:
        base = 0.0
    components["fire"] = round(base, 1)

    # --- fire extent -------------------------------------------------------
    extent = 15.0 * min(1.0, fire_area_ratio / max(config.SEVERITY_FULL_FIRE_RATIO, 1e-6))
    components["extent"] = round(extent if fire_active else 0.0, 1)

    # --- smoke -------------------------------------------------------------
    if smoke_active:
        smoke_term = 10.0 + 5.0 * min(1.0, smoke_ratio)
    else:
        smoke_term = 5.0 * min(1.0, smoke_ratio)
    components["smoke"] = round(smoke_term, 1)

    # --- occupancy ---------------------------------------------------------
    # Only counts when something is actually wrong; people in a safe building
    # are not a hazard.
    if fire_active or smoke_active:
        occ = 15.0 * min(1.0, people_count / max(config.SEVERITY_FULL_OCCUPANCY, 1))
    else:
        occ = 0.0
    components["occupancy"] = round(occ, 1)

    # --- egress ------------------------------------------------------------
    if exits_total > 0 and exits_blocked > 0:
        egress = 15.0 * (exits_blocked / exits_total)
    else:
        egress = 0.0
    components["egress"] = round(egress, 1)

    score = int(round(min(100.0, sum(components.values()))))

    return {
        "score": score,
        "level": severity_level(score),
        "components": components,
    }
