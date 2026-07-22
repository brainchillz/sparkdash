"""SparkDash FastAPI app: serves the dashboard and broadcasts live snapshots.

Run on the head node — the Ray dashboard is bound to 127.0.0.1.

    uv run uvicorn sparkdash.app:app --host 0.0.0.0 --port 7862
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import config, history, logstream, store
from .admin_api import router as admin_router
from .collectors import Hub
from .metrics import render_prometheus

PROM_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="SparkDash")
app.include_router(admin_router)
hub = Hub()
clients: set[WebSocket] = set()
_broadcaster: asyncio.Task | None = None
_sampler: asyncio.Task | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _broadcaster, _sampler
    store.init_db()
    store.purge_expired_sessions()
    history.init_db()
    hub.start()
    _broadcaster = asyncio.create_task(_broadcast_loop())
    _sampler = asyncio.create_task(history.sampler_loop(hub))


@app.on_event("shutdown")
async def _shutdown() -> None:
    for task in (_broadcaster, _sampler):
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    await hub.stop()


async def _broadcast_loop() -> None:
    """Push the merged snapshot to every connected client on a fixed cadence."""
    while True:
        await asyncio.sleep(config.BROADCAST_INTERVAL)
        if not clients:
            continue
        snap = hub.snapshot()
        dead: list[WebSocket] = []
        # Iterate a copy: the set mutates if a client (dis)connects while a
        # send is awaited (seen as RuntimeError during shutdown).
        for ws in list(clients):
            try:
                await ws.send_json(snap)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)


@app.get("/api/snapshot")
@app.get("/api/v1/snapshot")
async def snapshot() -> JSONResponse:
    """One-shot merged snapshot for external consumers (stable schema).

    `/api/v1/snapshot` is the versioned alias; `/api/snapshot` tracks latest.
    """
    return JSONResponse(hub.snapshot())


@app.get("/api/history")
async def api_history(range: str = "24h") -> JSONResponse:
    """Historic metrics for the /history page: all series bucket-averaged
    to a shared time axis. Ranges: 1h, 6h, 24h, 7d, 30d, 1y."""
    if range not in history.RANGES:
        return JSONResponse(
            {"error": f"unknown range {range!r}",
             "ranges": sorted(history.RANGES)}, status_code=400)
    return JSONResponse(await asyncio.to_thread(history.query, range))


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus exposition of the merged snapshot (scrape target).

    Includes `sparkdash_node_vram_used_bytes`, the GB10 per-process VRAM that
    neither vLLM's nor Ray's own /metrics exposes.
    """
    return PlainTextResponse(render_prometheus(hub.snapshot()),
                             media_type=PROM_CONTENT_TYPE)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    clients.add(ws)
    try:
        await ws.send_json(hub.snapshot())  # immediate first paint
        while True:
            await ws.receive_text()  # keepalive / ignore client messages
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


@app.websocket("/api/logs/ws")
async def logs_ws(ws: WebSocket) -> None:
    """Stream the running recipe's head-container logs (docker logs -f)."""
    await ws.accept()
    recipe = hub.snapshot().get("recipe", {})
    if not recipe.get("running"):
        await ws.send_text("[sparkdash] No recipe is currently running.")
        await ws.close()
        return
    container = await logstream.find_container(recipe["id"])
    if not container:
        await ws.send_text(f"[sparkdash] Container for recipe "
                           f"{recipe.get('name')} not found.")
        await ws.close()
        return
    await ws.send_text(f"[sparkdash] Attached to {container} — recipe "
                       f"{recipe.get('name')}\n")

    stream = await logstream.start_stream(container, tail=400)

    async def pump() -> None:
        assert stream.proc.stdout is not None
        while True:
            raw = await stream.proc.stdout.readline()
            if not raw:
                break
            await ws.send_text(logstream.decode_line(raw))

    async def watch_close() -> None:
        # Resolves when the client disconnects (it never sends us data).
        try:
            while True:
                await ws.receive()
        except WebSocketDisconnect:
            pass

    pump_task = asyncio.create_task(pump())
    close_task = asyncio.create_task(watch_close())
    try:
        await asyncio.wait({pump_task, close_task},
                           return_when=asyncio.FIRST_COMPLETED)
    finally:
        pump_task.cancel()
        close_task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(pump_task, close_task, return_exceptions=True)
        # Shielded so the in-container tail is always killed, even if the
        # request task itself is being cancelled (e.g. server shutdown).
        with contextlib.suppress(Exception):
            await asyncio.shield(logstream.stop_stream(stream))


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/admin")
async def admin_page() -> FileResponse:
    return FileResponse(FRONTEND / "admin.html")


@app.get("/logs")
async def logs_page() -> FileResponse:
    return FileResponse(FRONTEND / "logs.html")


@app.get("/history")
async def history_page() -> FileResponse:
    return FileResponse(FRONTEND / "history.html")


# Static assets (css/js) if we split them out later.
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
