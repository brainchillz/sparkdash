"""Authentication: password hashing, sessions, API tokens, and FastAPI gates.

Two levels of protection sit in front of the write surface:

  * require_admin   - accepts an interactive **session** OR a bearer **API
                      token**. Used for operational writes (model preload,
                      later recipe control) that automation may drive.
  * require_session - accepts a session only (interactive human). Used for
                      *administration* — minting/revoking tokens, replacing the
                      TLS cert, changing the password — so a leaked operational
                      token can't escalate into managing the system itself.

Reads (dashboard, /api/snapshot, /metrics, /ws) stay public.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

from fastapi import Depends, HTTPException, Request, Response, status

from . import config, store

_SCRYPT = dict(n=2**14, r=8, p=1, maxmem=64 * 1024 * 1024, dklen=32)


# -- password ----------------------------------------------------------------

def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), **_SCRYPT)
    return digest.hex(), salt


def verify_password(password: str) -> bool:
    admin = store.get_admin()
    if not admin:
        return False
    candidate, _ = hash_password(password, admin["pw_salt"])
    return hmac.compare_digest(candidate, admin["pw_hash"])


def set_password(username: str, password: str) -> None:
    store.init_db()
    pw_hash, pw_salt = hash_password(password)
    store.set_admin(username, pw_hash, pw_salt)


def admin_configured() -> bool:
    return store.get_admin() is not None


# -- sessions ----------------------------------------------------------------

def open_session(resp: Response) -> str:
    sid = secrets.token_urlsafe(32)
    store.create_session(sid, config.SESSION_TTL)
    resp.set_cookie(
        config.SESSION_COOKIE, sid,
        max_age=config.SESSION_TTL,
        httponly=True, samesite="strict",
        secure=True,  # always TLS in Phase 2
    )
    return sid


def close_session(request: Request, resp: Response) -> None:
    sid = request.cookies.get(config.SESSION_COOKIE)
    if sid:
        store.delete_session(sid)
    resp.delete_cookie(config.SESSION_COOKIE)


def _session_ok(request: Request) -> bool:
    sid = request.cookies.get(config.SESSION_COOKIE)
    return bool(sid and store.get_session(sid))


# -- API tokens --------------------------------------------------------------

def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def mint_token(name: str) -> str:
    raw = "spk_" + secrets.token_urlsafe(32)
    store.add_token(secrets.token_hex(8), name, _hash_token(raw))
    return raw  # shown to the caller exactly once


def _token_ok(request: Request) -> bool:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return False
    raw = auth[7:].strip()
    return bool(store.find_active_token(_hash_token(raw)))


# -- login throttle (per-IP, in-memory) --------------------------------------

_fails: dict[str, list[float]] = {}


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "?"


def check_throttle(request: Request) -> None:
    ip = _client_ip(request)
    now = time.time()
    hits = [t for t in _fails.get(ip, []) if now - t < config.LOGIN_COOLDOWN]
    _fails[ip] = hits
    if len(hits) >= config.LOGIN_MAX_FAILS:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            "Too many attempts; try again later.")


def record_failure(request: Request) -> None:
    _fails.setdefault(_client_ip(request), []).append(time.time())


def clear_failures(request: Request) -> None:
    _fails.pop(_client_ip(request), None)


# -- dependencies ------------------------------------------------------------

def require_admin(request: Request) -> None:
    """Session or API token — for operational writes."""
    if _session_ok(request) or _token_ok(request):
        return
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")


def require_session(request: Request) -> None:
    """Interactive session only — for administration."""
    if _session_ok(request):
        return
    raise HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "Interactive login required for this action",
    )


AdminDep = Depends(require_admin)
SessionDep = Depends(require_session)
