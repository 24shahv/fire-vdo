"""
SAFEX AI — web server.

Replaces the desktop main.py. There is no cv2.VideoCapture, no cv2.imshow and
no pyttsx3 anywhere in the running process: frames come from each visitor's
browser over a WebSocket, and the spoken alert is produced by the browser's
speech synthesiser.

Run locally:   python server.py
Run on Render: uvicorn server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import logging
import os
import time

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
import models_loader
from inference import FrameDecodeError, process_frame
from session import store

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("safex.server")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Bounds how many frames sit inside the models at once. Without this a handful
# of visitors can pile every request onto the CPU at the same moment.
_inference_slots = asyncio.Semaphore(max(1, config.MAX_CONCURRENT_INFERENCE))

_started_at = time.time()


async def _run_inference(session, cam_id: int, payload: bytes) -> dict:
    """Push the CPU-bound work off the event loop, with a concurrency cap."""
    async with _inference_slots:
        return await asyncio.to_thread(process_frame, session, cam_id, payload)


async def _sweeper() -> None:
    """Periodically drop idle sessions so memory stays flat."""
    while True:
        try:
            await asyncio.sleep(config.SESSION_SWEEP_SECONDS)
            removed = store.sweep()
            if removed:
                log.info("Swept %d idle session(s); %s", removed, store.stats())
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Session sweeper hiccup")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    if config.WARMUP_ON_START:
        log.info("Warming up YOLO models...")
        try:
            await asyncio.to_thread(models_loader.warmup)
        except Exception:
            log.exception("Warmup failed — models will load on first request")
    else:
        log.info("Warmup disabled (WARMUP_ON_START=0)")

    task = asyncio.create_task(_sweeper())
    log.info("SAFEX AI ready")

    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="SAFEX AI", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------- health
@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _started_at, 1),
        **store.stats(),
    }


@app.get("/api/config")
async def api_config():
    """What the frontend needs to know to match the server's expectations."""
    return {
        "frame": {"w": config.FRAME_WIDTH, "h": config.FRAME_HEIGHT},
        "grid": {"rows": config.GRID_ROWS, "cols": config.GRID_COLS},
        "exits": [list(e) for e in config.EXITS],
        "crowd_override_threshold": config.CROWD_OVERRIDE_THRESHOLD,
        "alert_cooldown": config.ALERT_COOLDOWN,
    }


# ----------------------------------------------------------------- REST path
@app.post("/api/detect")
async def api_detect(
    request: Request,
    file: UploadFile | None = File(default=None),
    session_id: str | None = Form(default=None),
    cam_id: int = Form(default=0),
):
    """
    Analyse a single frame.

    Accepts either:
      * multipart/form-data with `file` (plus optional session_id, cam_id), or
      * application/json  {"image": "<base64 or data URL>", "session_id", "cam_id"}
      * a raw image body, with session/cam supplied as query parameters.
    """
    payload: bytes | None = None

    if file is not None:
        payload = await file.read()
    else:
        content_type = (request.headers.get("content-type") or "").lower()
        raw = await request.body()

        if "application/json" in content_type:
            import json

            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                raise HTTPException(400, "invalid JSON body")

            session_id = body.get("session_id", session_id)
            cam_id = int(body.get("cam_id", cam_id) or 0)

            image = body.get("image") or ""
            if "," in image[:64] and image.lstrip().startswith("data:"):
                image = image.split(",", 1)[1]

            try:
                payload = base64.b64decode(image, validate=False)
            except (binascii.Error, ValueError):
                raise HTTPException(400, "image is not valid base64")
        elif raw:
            payload = raw

    if not payload:
        raise HTTPException(400, "no frame supplied")

    session_id = session_id or request.query_params.get("session")
    with contextlib.suppress(TypeError, ValueError):
        cam_id = int(request.query_params.get("cam", cam_id))

    session = store.get_or_create(session_id)

    try:
        result = await _run_inference(session, cam_id, payload)
    except FrameDecodeError as exc:
        raise HTTPException(400, str(exc))

    return JSONResponse(result)


# ------------------------------------------------------------ WebSocket path
@app.websocket("/ws")
async def ws_detect(websocket: WebSocket):
    """
    Realtime channel. The client sends a binary JPEG and waits for the JSON
    verdict before sending the next one — that request/response rhythm is the
    backpressure that keeps a slow server from building an unbounded queue.
    """
    await websocket.accept()

    session_id = websocket.query_params.get("session")
    try:
        cam_id = int(websocket.query_params.get("cam", 0))
    except (TypeError, ValueError):
        cam_id = 0

    session = store.get_or_create(session_id)
    log.info("WS connect session=%s cam=%s", session.session_id, cam_id)

    try:
        await websocket.send_json(
            {
                "type": "ready",
                "session_id": session.session_id,
                "cam_id": cam_id,
                "frame": {"w": config.FRAME_WIDTH, "h": config.FRAME_HEIGHT},
            }
        )

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            payload = message.get("bytes")

            if payload is None:
                text = message.get("text")
                if not text:
                    continue
                if text == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue
                # Also accept a base64 data URL over the text channel.
                if text.startswith("data:"):
                    text = text.split(",", 1)[-1]
                try:
                    payload = base64.b64decode(text, validate=False)
                except (binascii.Error, ValueError):
                    await websocket.send_json(
                        {"type": "error", "error": "bad base64 frame"}
                    )
                    continue

            session.touch()

            try:
                result = await _run_inference(session, cam_id, payload)
            except FrameDecodeError as exc:
                await websocket.send_json({"type": "error", "error": str(exc)})
                continue

            result["type"] = "result"
            await websocket.send_json(result)

    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("WebSocket error")
        with contextlib.suppress(Exception):
            await websocket.close(code=1011)
    finally:
        with contextlib.suppress(Exception):
            session.drop_camera(cam_id)
        log.info("WS disconnect session=%s cam=%s", session.session_id, cam_id)


# --------------------------------------------------------------- static site
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=bool(os.environ.get("DEV_RELOAD")),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
