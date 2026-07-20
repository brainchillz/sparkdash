"""SparkDash configuration.

Cluster topology (nodes, addresses, SSH user) is loaded from a TOML file so no
site-specific addresses live in the source. Search order:

  1. $SPARKDASH_CONFIG
  2. ~/.config/sparkdash/config.toml
  3. <repo>/sparkdash.toml

See sparkdash.example.toml for the format. With no config file, SparkDash
falls back to a single local node so it still imports and runs.

Control/API traffic (Ray, vLLM, sparkrun, the node probe) uses each node's
management address; an optional per-node `rdma` address is used for bulk file
movement (model mirroring) over a fast direct-connect link.
"""

import os
import tomllib
from pathlib import Path


def _load() -> dict:
    candidates = []
    if os.environ.get("SPARKDASH_CONFIG"):
        candidates.append(Path(os.environ["SPARKDASH_CONFIG"]))
    candidates.append(Path(os.path.expanduser("~/.config/sparkdash/config.toml")))
    candidates.append(Path(__file__).resolve().parent.parent / "sparkdash.toml")
    for p in candidates:
        try:
            if p.is_file():
                with open(p, "rb") as fh:
                    return tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            continue
    return {}


_cfg = _load()

# Node topology. Each node: name, ip (management address, matches the host keys
# in `sparkrun cluster monitor` output), rdma (fast-copy address), role, local.
_nodes = _cfg.get("nodes") or [
    {"name": "head", "ip": "localhost", "rdma": "localhost",
     "role": "head", "local": True},
]

NODES = [
    {"name": n["name"], "ip": n["ip"], "role": n.get("role", "worker"),
     "local": bool(n.get("local", False))}
    for n in _nodes
]

# Fast file-copy addresses, keyed by node name — used for model mirroring.
RDMA_IP = {n["name"]: n.get("rdma", n["ip"]) for n in _nodes}

SSH_USER = _cfg.get("ssh_user", "nvidia")

# Map monitor-stream host keys (management addresses) to node names.
IP_TO_NODE = {n["ip"]: n["name"] for n in NODES}

RAY_DASHBOARD = _cfg.get("ray_dashboard", "http://127.0.0.1:8265")
VLLM_BASE = _cfg.get("vllm_base", "http://localhost:8000")
SPARKRUN_CLUSTER = _cfg.get("sparkrun_cluster", "default")

# Poll / broadcast cadence (seconds).
SLOW_POLL = 5.0        # Ray, vLLM, node probe (VRAM + disk)
STATUS_POLL = 12.0     # `sparkrun status` (spawns SSH, keep it gentle)
BROADCAST_INTERVAL = 2.0

# -- Serving / data / auth ------------------------------------------------

PORT = int(os.environ.get("SPARKDASH_PORT", "7862"))

# Persistent state lives outside the repo so it can't be committed.
DATA_DIR = Path(os.environ.get(
    "SPARKDASH_DATA_DIR", os.path.expanduser("~/.local/share/sparkdash")))
CERT_DIR = DATA_DIR / "certs"
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE = CERT_DIR / "key.pem"
DB_FILE = DATA_DIR / "sparkdash.db"

# Admin identity. Username is non-default to slow trivial guessing; the
# password is set out-of-band via `python -m sparkdash.admin set-password`.
ADMIN_USER = os.environ.get("SPARKDASH_ADMIN_USER", _cfg.get("admin_user", "sparkadmin"))

SESSION_COOKIE = "sparkdash_session"
SESSION_TTL = 12 * 3600          # 12h interactive session
LOGIN_MAX_FAILS = 8              # per-IP failures before a cooldown
LOGIN_COOLDOWN = 300             # seconds

# Extra SAN entries for the auto-generated self-signed cert.
_cert = _cfg.get("cert", {})
CERT_HOSTNAMES = _cert.get("hostnames", ["localhost"])
CERT_IPS = _cert.get("ips", ["127.0.0.1"])

# -- Model preload --------------------------------------------------------

# sparkrun is installed as a separate uv tool; we call its model download /
# distribute functions with its own interpreter so we reuse the exact,
# fast-network-aware logic a recipe run uses (rather than reimplementing it).
SPARKRUN_PYTHON = os.environ.get(
    "SPARKRUN_PYTHON",
    os.path.expanduser("~/.local/share/uv/tools/sparkrun/bin/python"))

# -- Model backup / restore ----------------------------------------------
#
# Backups go to <base>/<BACKUP_SUBDIR>/models--<repo>/ so SparkDash's copies
# are cleanly separated from anything else on the (typically NFS) share. The
# base is entered in the UI and remembered in the store; this is only the
# initial default. The per-model dir keeps the HuggingFace hub naming so a
# restore is a plain copy back into each node's cache.
BACKUP_DEFAULT_BASE = os.environ.get("SPARKDASH_BACKUP_BASE", "/mnt/llm")
BACKUP_SUBDIR = "Sparkdash/Models"
BACKUP_MANIFEST = "sparkdash-backup.json"
