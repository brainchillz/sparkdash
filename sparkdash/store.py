"""SQLite persistence for admin credential, sessions, and API tokens.

A fresh connection is opened per call — traffic is tiny and this sidesteps
sqlite3's thread-affinity entirely. WAL mode keeps concurrent reads snappy.
Secrets are never stored in the clear: the password is scrypt-hashed and API
tokens are stored only as SHA-256 digests.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS admin (
    id        INTEGER PRIMARY KEY CHECK (id = 1),
    username  TEXT NOT NULL,
    pw_hash   TEXT NOT NULL,
    pw_salt   TEXT NOT NULL,
    updated   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id        TEXT PRIMARY KEY,
    created   REAL NOT NULL,
    expires   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tokens (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    created    REAL NOT NULL,
    last_used  REAL,
    revoked    INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""


def init_db() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.executescript(_SCHEMA)


@contextmanager
def _conn():
    conn = sqlite3.connect(config.DB_FILE, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# -- admin credential --------------------------------------------------------

def set_admin(username: str, pw_hash: str, pw_salt: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO admin (id, username, pw_hash, pw_salt, updated) "
            "VALUES (1, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET username=excluded.username, "
            "pw_hash=excluded.pw_hash, pw_salt=excluded.pw_salt, "
            "updated=excluded.updated",
            (username, pw_hash, pw_salt, time.time()),
        )


def get_admin() -> sqlite3.Row | None:
    with _conn() as c:
        return c.execute("SELECT * FROM admin WHERE id = 1").fetchone()


# -- sessions ----------------------------------------------------------------

def create_session(sid: str, ttl: float) -> None:
    now = time.time()
    with _conn() as c:
        c.execute("INSERT INTO sessions (id, created, expires) VALUES (?, ?, ?)",
                  (sid, now, now + ttl))


def get_session(sid: str) -> sqlite3.Row | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        if row and row["expires"] < time.time():
            c.execute("DELETE FROM sessions WHERE id = ?", (sid,))
            return None
        return row


def delete_session(sid: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE id = ?", (sid,))


def purge_expired_sessions() -> None:
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))


# -- API tokens --------------------------------------------------------------

def add_token(tid: str, name: str, token_hash: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO tokens (id, name, token_hash, created) VALUES (?, ?, ?, ?)",
            (tid, name, token_hash, time.time()),
        )


def list_tokens() -> list[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT id, name, created, last_used, revoked FROM tokens "
            "ORDER BY created DESC"
        ).fetchall()


def revoke_token(tid: str) -> bool:
    with _conn() as c:
        cur = c.execute("UPDATE tokens SET revoked = 1 WHERE id = ?", (tid,))
        return cur.rowcount > 0


# -- settings (key/value) ----------------------------------------------------

def get_setting(key: str, default: str | None = None) -> str | None:
    with _conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def find_active_token(token_hash: str) -> sqlite3.Row | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM tokens WHERE token_hash = ? AND revoked = 0",
            (token_hash,),
        ).fetchone()
        if row:
            c.execute("UPDATE tokens SET last_used = ? WHERE id = ?",
                      (time.time(), row["id"]))
        return row
