"""
Per-visitor state.

The original main.py kept `people_memory`, `memory_timer`, `frame_count` and
`last_alert` as module globals. On a single-user desktop script that is fine. On
a web server it is a correctness bug: every visitor would share one another's
detections and alert timers.

Those globals now live on a Session, keyed by an opaque id the browser sends.
A background sweeper drops idle sessions so long-running deployments don't grow
without bound.

One session can hold several cameras (cam_id 0, 1, ...), which is how the old
multi-camera CameraManager behaviour is preserved: open the page on a laptop and
a phone with the same session id and the two feeds are fused into one building
view.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

import config


@dataclass
class CameraState:
    """Everything the pipeline remembers between frames for ONE camera."""

    cam_id: int
    frame_count: int = 0

    # Cached people detections, reused on frames where detection is skipped.
    people_memory: list = field(default_factory=list)
    memory_timer: int = 0

    # Latest results, so other cameras in the session can be fused with these.
    people_grid: list = field(default_factory=list)  # [(grid_pos, (px, py)), ...]
    people_boxes: list = field(default_factory=list)  # frame-space boxes for drawing
    fire_grid: list = field(default_factory=list)  # [(row, col), ...]
    fire_boxes: list = field(default_factory=list)
    smoke_detected: bool = False

    last_seen: float = field(default_factory=time.time)

    def should_run_detection(self) -> bool:
        """Reproduces `frame_count % DETECT_STRIDE == 0` from the original."""
        stride = max(1, config.DETECT_STRIDE)
        return self.frame_count % stride == 0

    def remember_people(self, detected):
        """
        Frame-to-frame smoothing, unchanged in spirit from the original:
        a fresh detection replaces the cache, an empty one keeps the previous
        result alive for PEOPLE_MEMORY_FRAMES frames before clearing.

        In the original this cache was shared by every camera; here it is
        per-camera, which is what the code clearly intended.
        """
        if len(detected) > 0:
            self.people_memory = detected
            self.memory_timer = config.PEOPLE_MEMORY_FRAMES
        else:
            if self.memory_timer > 0:
                self.memory_timer -= 1
            else:
                self.people_memory = []

        return self.people_memory


@dataclass
class Session:
    session_id: str
    cameras: dict = field(default_factory=dict)
    last_alert: float = 0.0
    created: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def camera(self, cam_id: int) -> CameraState:
        cam = self.cameras.get(cam_id)
        if cam is None:
            cam = CameraState(cam_id=cam_id)
            self.cameras[cam_id] = cam
        return cam

    def drop_camera(self, cam_id: int) -> None:
        self.cameras.pop(cam_id, None)

    def touch(self) -> None:
        self.last_seen = time.time()

    def may_announce(self, now: float | None = None) -> bool:
        """True at most once every ALERT_COOLDOWN seconds."""
        now = now or time.time()
        if now - self.last_alert > config.ALERT_COOLDOWN:
            self.last_alert = now
            return True
        return False


class SessionStore:
    """Thread-safe session registry with idle expiry."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None) -> Session:
        if not session_id:
            session_id = uuid.uuid4().hex

        # Never let a client's string grow unbounded or carry junk.
        session_id = str(session_id)[:64]

        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = Session(session_id=session_id)
                self._sessions[session_id] = sess
            sess.touch()
            return sess

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def sweep(self, ttl: float | None = None) -> int:
        """Delete sessions idle for longer than the TTL. Returns how many went."""
        ttl = ttl if ttl is not None else config.SESSION_TTL_SECONDS
        cutoff = time.time() - ttl

        with self._lock:
            stale = [k for k, s in self._sessions.items() if s.last_seen < cutoff]
            for k in stale:
                del self._sessions[k]

        return len(stale)

    def stats(self) -> dict:
        with self._lock:
            return {
                "sessions": len(self._sessions),
                "cameras": sum(len(s.cameras) for s in self._sessions.values()),
            }


store = SessionStore()
