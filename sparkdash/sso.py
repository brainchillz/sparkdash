"""Optional single sign-on: verify an assertion from a Nexus SSO issuer.

OFF unless configured, and off is the default everywhere. The relying-party
pattern (and ed25519.py) follows the other Nexus apps: an assertion is
accepted at exactly ONE endpoint, /sso/callback, where it is exchanged for an
ordinary session cookie. It is never accepted as a bearer credential — API
tokens keep working exactly as before; this is a browser feature.

Configuration tiers, first match wins ("the environment wins"):

  1. SPARKDASH_SSO_* environment variables            (locked: UI read-only)
  2. an [sso] table in config.toml                    (locked: UI read-only)
  3. sso.json in the data dir, written by the admin UI after redeeming an
     enrollment code minted at the issuer               (UI-managed)

Scope of what a verified assertion can do:
  * It names a subject. That subject must equal this install's one admin
    username — SSO grants access to the account that exists, it never
    creates or elevates one.
  * It carries no role. SparkDash has a single role and the local account
    keeps it, exactly as after a password login.
So the worst a compromised issuer can do is sign in as the existing admin of
an install that opted in; it cannot invent access anywhere else.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import threading
import time

from . import config, ed25519

ALG = "EdDSA"
TYP = "nxa"
CLOCK_SKEW = 30

STORE = config.DATA_DIR / "sso.json"


def _env_cfg() -> dict | None:
    issuer = os.environ.get("SPARKDASH_SSO_ISSUER", "").rstrip("/")
    if not issuer:
        return None
    return {
        "issuer": issuer,
        "pubkey": os.environ.get("SPARKDASH_SSO_PUBKEY", ""),
        "kid": os.environ.get("SPARKDASH_SSO_KID", ""),
        "audience": os.environ.get("SPARKDASH_SSO_AUDIENCE", "") or _hostname(),
        "subject": os.environ.get("SPARKDASH_SSO_SUBJECT", ""),
        "auto_redirect": os.environ.get("SPARKDASH_SSO_AUTO_REDIRECT", "") in
                         ("1", "true", "yes"),
        "source": "env",
    }


def _toml_cfg() -> dict | None:
    t = config.SSO_TOML
    if not t.get("issuer"):
        return None
    return {
        "issuer": str(t["issuer"]).rstrip("/"),
        "pubkey": str(t.get("pubkey", "")),
        "kid": str(t.get("kid", "")),
        "audience": str(t.get("audience", "")) or _hostname(),
        "subject": str(t.get("subject", "")),
        "auto_redirect": bool(t.get("auto_redirect", False)),
        "source": "config",
    }


def _stored() -> dict:
    try:
        with open(STORE) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def get_config() -> dict | None:
    """Resolve the active configuration, or None. Env, then config.toml, then
    the UI-enrolled store — the install-time decision always wins."""
    cfg = _env_cfg() or _toml_cfg()
    if cfg:
        return cfg
    s = _stored()
    if s.get("issuer") and s.get("pubkey"):
        return {
            "issuer": str(s["issuer"]).rstrip("/"),
            "pubkey": str(s["pubkey"]),
            "kid": str(s.get("kid") or ""),
            "audience": str(s.get("audience") or _hostname()),
            "subject": str(s.get("subject") or ""),
            "auto_redirect": bool(s.get("auto_redirect")),
            "source": "stored",
        }
    return None


def expected_subject() -> str:
    """The one SSO subject this install signs in as its admin.

    An explicit `subject` in the configuration wins — a single-account app
    joining a fleet whose identity is e.g. "admin" maps that subject onto
    its local account. Default: the local admin username itself. Either way
    exactly ONE subject is accepted and it only ever unlocks the account
    that already exists — no provisioning, no role grant.
    """
    cfg = get_config() or {}
    if cfg.get("subject"):
        return cfg["subject"]
    from . import store
    admin = store.get_admin()
    return admin["username"] if admin else ""


def locked() -> bool:
    """True when env/config.toml fixes this and the UI must not edit it."""
    return bool(_env_cfg() or _toml_cfg())


def save_stored(issuer: str, pubkey: str, kid: str, aud: str) -> None:
    tmp = STORE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "issuer": str(issuer).rstrip("/"), "pubkey": pubkey,
        "kid": kid, "audience": aud,
    }))
    os.chmod(tmp, 0o600)
    os.replace(tmp, STORE)


def clear_stored() -> bool:
    try:
        os.unlink(STORE)
        return True
    except FileNotFoundError:
        return False


def _hostname() -> str:
    return socket.gethostname()


def _pubkey_bytes(cfg: dict | None = None) -> bytes:
    cfg = cfg if cfg is not None else get_config()
    s = (cfg or {}).get("pubkey", "")
    try:
        return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    except Exception:
        return b""


def enabled() -> bool:
    """True only when fully configured. A half-configured install behaves as
    if SSO were off rather than failing at login time."""
    cfg = get_config()
    return bool(cfg) and len(_pubkey_bytes(cfg)) == 32


def login_hint() -> dict | None:
    """What the login screen needs to offer SSO. Public values only — this is
    served to unauthenticated callers. None when SSO is not configured."""
    if not enabled():
        return None
    cfg = get_config() or {}
    return {"issuer": cfg.get("issuer", ""),
            "audience": cfg.get("audience", ""),
            "auto_redirect": bool(cfg.get("auto_redirect"))}


def authorize_url(next_path: str = "/") -> str:
    from urllib.parse import urlencode
    cfg = get_config() or {}
    return (cfg.get("issuer", "") + "/sso/authorize?"
            + urlencode({"aud": cfg.get("audience", ""),
                         "next": safe_next(next_path)}))


def redeem(issuer: str, code: str, callback: str,
           timeout: int = 15) -> tuple[dict | None, str | None]:
    """Redeem an enrollment code at `issuer`. Returns (result, None) or
    (None, error).

    stdlib urllib on purpose — enrollment must not add a dependency. The
    issuer's certificate may be self-signed, so verification is skipped; this
    stays safe because nothing secret is sent (the code is single-use and
    worthless once redeemed) and nothing secret comes back (the response is
    the same public key /sso/jwks serves to anyone).
    """
    import ssl as _ssl
    import urllib.error
    import urllib.request

    body = json.dumps({"code": code, "callback": callback}).encode()
    req = urllib.request.Request(
        issuer.rstrip("/") + "/sso/enroll", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    ctx = _ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = _ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as ex:
        try:
            return None, (json.loads(ex.read().decode()).get("error")
                          or "Issuer refused the code (HTTP %d)" % ex.code)
        except Exception:
            return None, "Issuer refused the code (HTTP %d)" % ex.code
    except Exception as ex:
        return None, "Could not reach the issuer: %s" % ex
    if not data.get("success"):
        return None, data.get("error") or "Enrollment failed"
    for k in ("issuer", "key", "audience"):
        if not data.get(k):
            return None, "Issuer response was missing %r" % k
    return data, None


def safe_next(value) -> str:
    """Reduce a caller-supplied path to something same-site. Anything that
    could send the browser elsewhere collapses to '/'."""
    if not value or not isinstance(value, str) or len(value) > 512:
        return "/"
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in value):
        return "/"
    value = value.replace("\\", "/")
    if not value.startswith("/") or value.startswith("//"):
        return "/"
    return value


# -- replay cache ------------------------------------------------------------
# Assertions are single-use. The cache only has to outlive an assertion's own
# lifetime, so entries are dropped once they cannot possibly still verify.

_seen: dict[str, int] = {}
_seen_lock = threading.Lock()


def _remember(jti: str, exp: int, now: int | None = None) -> bool:
    """Record a jti as spent. False if it was already spent. `now` comes from
    the caller so the cache and the expiry check share one clock."""
    now = int(now if now is not None else time.time())
    with _seen_lock:
        for k, v in list(_seen.items()):
            if v <= now:
                del _seen[k]
        if jti in _seen:
            return False
        _seen[jti] = exp
        return True


def _b64u_decode(s) -> bytes:
    if isinstance(s, str):
        s = s.encode()
    return base64.urlsafe_b64decode(s + b"=" * (-len(s) % 4))


def verify(token, now: int | None = None) -> str | None:
    """Verify an assertion and return its subject, or None.

    Never raises: every rejection path returns None, so the caller can hand
    it whatever arrived in the query string.
    """
    cfg = get_config()
    if not cfg or len(_pubkey_bytes(cfg)) != 32:
        return None
    now = int(now if now is not None else time.time())
    if not token or not isinstance(token, str) or token.count(".") != 2:
        return None
    h_b64, p_b64, s_b64 = token.split(".")
    try:
        header = json.loads(_b64u_decode(h_b64))
        payload = json.loads(_b64u_decode(p_b64))
        sig = _b64u_decode(s_b64)
    except Exception:
        return None
    if not isinstance(header, dict) or not isinstance(payload, dict):
        return None
    # The algorithm is pinned, so "alg": "none" and HMAC key confusion are
    # not reachable. No kid-directed key lookup either — this install knows
    # the one key it trusts.
    if header.get("alg") != ALG or header.get("typ") != TYP:
        return None
    if cfg.get("kid") and header.get("kid") != cfg["kid"]:
        return None
    if not ed25519.verify(_pubkey_bytes(cfg), (h_b64 + "." + p_b64).encode(), sig):
        return None

    # Claims are trusted only after the signature checks out.
    if payload.get("iss") != cfg["issuer"]:
        return None
    if payload.get("aud") != cfg["audience"]:
        return None
    exp, iat = payload.get("exp"), payload.get("iat")
    if not isinstance(exp, int) or not isinstance(iat, int):
        return None
    if now >= exp or iat > now + CLOCK_SKEW:
        return None
    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        return None
    jti = payload.get("jti")
    if not isinstance(jti, str) or not jti or not _remember(jti, exp, now):
        return None
    return sub
