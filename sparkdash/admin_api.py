"""Auth, API-token, and certificate endpoints (the write/admin surface).

Split by protection level:
  * login/logout/me    - public (login is throttled)
  * operational writes - require_admin (session OR API token): /ping, and
                         Phase 2+ features like model preload will live here
  * administration     - require_session (interactive only): mint/revoke
                         tokens, replace the TLS cert
"""

from __future__ import annotations

import asyncio
import os
import signal

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import auth, backup, certs, chat, config, hf, recipe_ops, store

router = APIRouter()


class LoginBody(BaseModel):
    username: str
    password: str


class TokenCreateBody(BaseModel):
    name: str


class CertBody(BaseModel):
    cert: str
    key: str


class PreloadBody(BaseModel):
    model: str
    mirror: bool = True


class StartBody(BaseModel):
    recipe: str
    mode: str = "cluster"   # "solo" | "cluster"


class PreflightBody(BaseModel):
    recipe: str
    mode: str = "cluster"


class RestartBody(BaseModel):
    mode: str | None = None


class DeleteCacheBody(BaseModel):
    repo: str


class ChatBody(BaseModel):
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 512


class BackupTargetBody(BaseModel):
    base: str


class BackupBody(BaseModel):
    repo: str
    base: str


class RestoreBody(BaseModel):
    repo: str
    base: str
    mirror: bool = True


_BACKUP_BASE_KEY = "backup_base"


# -- auth --------------------------------------------------------------------

@router.post("/api/auth/login")
async def login(body: LoginBody, request: Request, response: Response) -> dict:
    auth.check_throttle(request)
    admin = store.get_admin()
    if not admin:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "No admin configured. Run: python -m sparkdash.admin set-password")
    ok = (body.username == admin["username"]) and auth.verify_password(body.password)
    if not ok:
        auth.record_failure(request)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    auth.clear_failures(request)
    auth.open_session(response)
    return {"ok": True, "username": admin["username"]}


@router.post("/api/auth/logout")
async def logout(request: Request, response: Response) -> dict:
    auth.close_session(request, response)
    return {"ok": True}


@router.get("/api/auth/me")
async def me(request: Request) -> dict:
    sid = request.cookies.get(config.SESSION_COOKIE)
    authed = bool(sid and store.get_session(sid))
    admin = store.get_admin()
    return {
        "authenticated": authed,
        "username": admin["username"] if authed and admin else None,
        "admin_configured": admin is not None,
    }


# -- API tokens --------------------------------------------------------------

@router.get("/api/admin/tokens", dependencies=[auth.AdminDep])
async def list_tokens() -> dict:
    return {"tokens": [
        {"id": t["id"], "name": t["name"], "created": t["created"],
         "last_used": t["last_used"], "revoked": bool(t["revoked"])}
        for t in store.list_tokens()
    ]}


@router.post("/api/admin/tokens", dependencies=[auth.SessionDep])
async def create_token(body: TokenCreateBody) -> dict:
    name = body.name.strip()
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Token name required")
    raw = auth.mint_token(name)
    # Returned exactly once — never retrievable again.
    return {"token": raw, "name": name}


@router.delete("/api/admin/tokens/{tid}", dependencies=[auth.SessionDep])
async def revoke_token(tid: str) -> dict:
    if not store.revoke_token(tid):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Token not found")
    return {"ok": True}


# -- TLS certificate ---------------------------------------------------------

@router.get("/api/admin/cert", dependencies=[auth.AdminDep])
async def cert_info() -> dict:
    return certs.info()


@router.post("/api/admin/cert", dependencies=[auth.SessionDep])
async def replace_cert(body: CertBody) -> dict:
    try:
        certs.install_custom(body.cert.encode(), body.key.encode())
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid certificate: {exc}")
    _schedule_restart()
    return {"ok": True, "restarting": True,
            "message": "Certificate installed; restarting to apply."}


def _schedule_restart() -> None:
    """Graceful self-restart so the TLS listener picks up the new cert.

    Under systemd (Restart=always) the process comes back within ~2s with the
    new cert; the browser's WebSocket auto-reconnects.
    """
    loop = asyncio.get_event_loop()
    loop.call_later(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM))


# -- model preload (operational write) --------------------------------------

@router.get("/api/admin/recipes", dependencies=[auth.AdminDep])
async def recipes() -> dict:
    """Recipe catalog (name + HF model + node needs) for the preload picker."""
    proc = await asyncio.create_subprocess_exec(
        "sparkrun", "list", "--json",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
    try:
        items = json.loads(out.decode() or "[]")
    except json.JSONDecodeError:
        items = []
    return {"recipes": [
        {"name": r.get("name"), "model": r.get("model"),
         "runtime": r.get("runtime"), "min_nodes": r.get("min_nodes"),
         "tp": r.get("tp")}
        for r in items if r.get("model")
    ]}


@router.get("/api/admin/cache", dependencies=[auth.AdminDep])
async def cache() -> dict:
    return await hf.list_cache()


@router.post("/api/admin/cache/delete", dependencies=[auth.AdminDep])
async def cache_delete(body: DeleteCacheBody) -> dict:
    try:
        results = await hf.delete_cached(body.repo.strip())
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    return {"ok": all(results.values()), "nodes": results}


@router.post("/api/admin/cache/mirror", dependencies=[auth.AdminDep])
async def cache_mirror(body: DeleteCacheBody) -> dict:
    """Push an already-cached model from the head to the node(s) missing it."""
    try:
        hf.preloader.start(body.repo.strip(), mirror=True, existing=True)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return {"ok": True, "status": hf.preloader.status()}


# -- model backup / restore (operational write) -----------------------------

@router.get("/api/admin/backup/target", dependencies=[auth.AdminDep])
async def backup_target() -> dict:
    return {"base": store.get_setting(_BACKUP_BASE_KEY, config.BACKUP_DEFAULT_BASE)}


@router.post("/api/admin/backup/target", dependencies=[auth.AdminDep])
async def set_backup_target(body: BackupTargetBody) -> dict:
    base = body.base.strip()
    if not base.startswith("/"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Target must be an absolute path")
    store.set_setting(_BACKUP_BASE_KEY, base)
    return {"ok": True, "base": base}


@router.get("/api/admin/backups", dependencies=[auth.AdminDep])
async def backups(base: str | None = None) -> dict:
    base = (base or store.get_setting(_BACKUP_BASE_KEY, config.BACKUP_DEFAULT_BASE)).strip()
    if not base.startswith("/"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Target must be an absolute path")
    return await backup.list_backups(base)


@router.post("/api/admin/backup", dependencies=[auth.AdminDep])
async def backup_start(body: BackupBody) -> dict:
    repo, base = body.repo.strip(), body.base.strip()
    if not repo:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Model id required")
    if not base.startswith("/"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Target must be an absolute path")
    store.set_setting(_BACKUP_BASE_KEY, base)
    try:
        backup.manager.start_backup(repo, base)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return {"ok": True, "status": backup.manager.status()}


@router.post("/api/admin/restore", dependencies=[auth.AdminDep])
async def restore_start(body: RestoreBody) -> dict:
    repo, base = body.repo.strip(), body.base.strip()
    if not repo:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Model id required")
    if not base.startswith("/"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Target must be an absolute path")
    store.set_setting(_BACKUP_BASE_KEY, base)
    try:
        backup.manager.start_restore(repo, base, body.mirror)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return {"ok": True, "status": backup.manager.status()}


@router.post("/api/admin/backup/verify", dependencies=[auth.AdminDep])
async def backup_verify(body: BackupBody) -> dict:
    repo, base = body.repo.strip(), body.base.strip()
    if not repo:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Model id required")
    if not base.startswith("/"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Target must be an absolute path")
    try:
        backup.manager.start_verify(repo, base)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return {"ok": True, "status": backup.manager.status()}


@router.post("/api/admin/backup/delete", dependencies=[auth.AdminDep])
async def backup_delete(body: BackupBody) -> dict:
    repo, base = body.repo.strip(), body.base.strip()
    if not repo:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Model id required")
    if not base.startswith("/"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Target must be an absolute path")
    try:
        await backup.delete_backup(repo, base)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    return {"ok": True}


@router.get("/api/admin/backup/status", dependencies=[auth.AdminDep])
async def backup_status() -> dict:
    return backup.manager.status()


@router.post("/api/admin/backup/cancel", dependencies=[auth.AdminDep])
async def backup_cancel() -> dict:
    await backup.manager.cancel()
    return {"ok": True}


@router.post("/api/admin/chat", dependencies=[auth.AdminDep])
async def chat_completions(body: ChatBody) -> StreamingResponse:
    return StreamingResponse(
        chat.stream_chat(body.messages, body.temperature, body.max_tokens),
        media_type="application/x-ndjson")


@router.post("/api/admin/preload", dependencies=[auth.AdminDep])
async def preload(body: PreloadBody) -> dict:
    model = body.model.strip()
    if not model:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Model id required")
    try:
        hf.preloader.start(model, mirror=body.mirror)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return {"ok": True, "status": hf.preloader.status()}


@router.get("/api/admin/preload/status", dependencies=[auth.AdminDep])
async def preload_status() -> dict:
    return hf.preloader.status()


@router.post("/api/admin/preload/cancel", dependencies=[auth.AdminDep])
async def preload_cancel() -> dict:
    await hf.preloader.cancel()
    return {"ok": True}


# -- recipe lifecycle control (operational write) ---------------------------

@router.get("/api/admin/recipe/current", dependencies=[auth.AdminDep])
async def recipe_current() -> dict:
    return await recipe_ops.current_recipe()


@router.get("/api/admin/recipe/last", dependencies=[auth.AdminDep])
async def recipe_last() -> dict:
    """The most recently stopped recipe (re-runnable even if not in a registry)."""
    return recipe_ops.last_recipe()


@router.get("/api/admin/recipe/saved", dependencies=[auth.AdminDep])
async def recipe_saved() -> dict:
    """Durably-saved custom recipes (not in any registry), selectable to start."""
    return {"recipes": recipe_ops.saved_recipes()}


@router.post("/api/admin/recipe/preflight", dependencies=[auth.AdminDep])
async def recipe_preflight(body: PreflightBody) -> dict:
    return await recipe_ops.preflight(body.recipe.strip(), body.mode)


@router.post("/api/admin/recipe/start", dependencies=[auth.AdminDep])
async def recipe_start(body: StartBody) -> dict:
    recipe = body.recipe.strip()
    if not recipe:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Recipe required")
    current = await recipe_ops.current_recipe()
    if current.get("running"):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "A recipe is already running — stop it first")
    try:
        recipe_ops.controller.start(recipe, body.mode)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return {"ok": True, "op": recipe_ops.controller.status()}


@router.post("/api/admin/recipe/stop", dependencies=[auth.AdminDep])
async def recipe_stop() -> dict:
    try:
        recipe_ops.controller.stop()
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return {"ok": True, "op": recipe_ops.controller.status()}


@router.post("/api/admin/recipe/restart", dependencies=[auth.AdminDep])
async def recipe_restart(body: RestartBody) -> dict:
    try:
        recipe_ops.controller.restart(body.mode)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return {"ok": True, "op": recipe_ops.controller.status()}


@router.get("/api/admin/recipe/op", dependencies=[auth.AdminDep])
async def recipe_op() -> dict:
    return recipe_ops.controller.status()
