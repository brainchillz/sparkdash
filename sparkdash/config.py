"""SparkDash configuration.

Cluster topology (nodes, addresses, SSH user) is loaded from a TOML file so no
site-specific addresses live in the source. Search order:

  1. $SPARKDASH_CONFIG
  2. /etc/sparkdash/config.toml        (installed deployments)
  3. ~/.config/sparkdash/config.toml   (dev fallback)
  4. <repo>/sparkdash.toml

See sparkdash.example.toml for the format. With no config file, SparkDash
falls back to a single local node so it still imports and runs.

Runtime state (password DB, TLS certs, metrics history) likewise prefers the
system location: /var/lib/sparkdash when it exists (deploy/install.sh creates
it), else ~/.local/share/sparkdash for dev checkouts. This keeps state in a
stable place independent of whichever user happens to run a CLI command.

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
    candidates.append(Path("/etc/sparkdash/config.toml"))
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
MONITOR_STALE = 15.0   # no real monitor frame for this long -> stream is wedged
PROBE_STALE = 30.0     # VRAM/disk probe data older than this is dropped

# Speculative-decoding acceptance watch. Windowed over consecutive vLLM polls;
# a window only counts when it saw SPEC_MIN_STEPS drafter steps. Healthy DSpark
# on DeepSeek accepts 2-3.5 tokens/step; client sampling penalties (frequency/
# presence/repetition) push the whole engine below ~1.5 and halve throughput.
SPEC_MIN_STEPS = 25       # drafter steps a window needs before it counts
SPEC_COLLAPSE_LEN = 1.6   # accepted tokens/step below this = collapsed
SPEC_COLLAPSE_WINDOWS = 2 # consecutive low windows before flagging
SPEC_STALE = 300.0        # no qualifying window for this long -> state expires

# -- Serving / data / auth ------------------------------------------------

PORT = int(os.environ.get("SPARKDASH_PORT", "7862"))

# Persistent state lives outside the repo so it can't be committed. Installed
# deployments use /var/lib/sparkdash (created by deploy/install.sh) so state
# doesn't depend on any user's home directory; the home path is only a dev
# fallback when the system dir doesn't exist.
_SYSTEM_DATA = Path("/var/lib/sparkdash")
DATA_DIR = Path(
    os.environ.get("SPARKDASH_DATA_DIR")
    or (_SYSTEM_DATA if _SYSTEM_DATA.is_dir()
        else os.path.expanduser("~/.local/share/sparkdash")))
CERT_DIR = DATA_DIR / "certs"
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE = CERT_DIR / "key.pem"
DB_FILE = DATA_DIR / "sparkdash.db"

# -- Metrics history ------------------------------------------------------
#
# A background sampler persists one row per node (plus one cluster row for
# vLLM throughput) every HISTORY_INTERVAL seconds. Raw rows older than
# HISTORY_RAW_KEEP are folded into HISTORY_BUCKET-second min/avg/max rollups
# kept indefinitely (two nodes ≈ 1 MB/day raw, far less rolled up).
HISTORY_DB = DATA_DIR / "history.db"
HISTORY_INTERVAL = 30.0
HISTORY_RAW_KEEP = 7 * 86400
HISTORY_BUCKET = 300
HISTORY_MAX_POINTS = 400   # per-series cap on /api/history responses

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

# -- Single sign-on (Nexus SSO relying party) -----------------------------
#
# The [sso] table in config.toml is the install-time tier (issuer, pubkey,
# kid, audience, auto_redirect); SPARKDASH_SSO_* env vars override it, and a
# UI enrollment (sso.json in the data dir) is the fallback. See sso.py.
SSO_TOML = _cfg.get("sso", {})

# -- Alerting -------------------------------------------------------------
#
# POSTs a short message on cluster state transitions (healthy <-> degraded,
# model changes, node offline). `webhook_url` receives JSON {title, message,
# ts}; with style = "ntfy" the message is sent as a plain-text body with a
# Title header instead, which is what ntfy.sh-compatible servers expect.
_alerts = _cfg.get("alerts", {})
ALERT_WEBHOOK = _alerts.get("webhook_url", "")
ALERT_STYLE = _alerts.get("style", "json")          # "json" | "ntfy" | "gchat"

# -- Download rate cap ----------------------------------------------------
#
# Cap internet-sourced ingress on the head while a model preload downloads,
# so a 182 GB pull doesn't flatten the house uplink (LAN/RDMA/NFS traffic
# is exempt). `download_cap_mbit` here locks the value (UI read-only);
# leave it unset to manage the cap from the admin UI. `download_cap_iface`
# overrides the WAN interface (default: the default-route device).
DOWNLOAD_CAP_MBIT = int(_cfg.get("download_cap_mbit", 0) or 0)
DOWNLOAD_CAP_IFACE = _cfg.get("download_cap_iface", "")

# -- Peer installs --------------------------------------------------------
#
# Other SparkDash installs to summarise on the front page. Each: name, url
# (base URL, e.g. "https://pebbles.example:7862"). Read-only; their public
# snapshot endpoint is polled at PEER_POLL.
PEERS = [
    {"name": p.get("name", p.get("url", "?")), "url": p["url"].rstrip("/")}
    for p in _cfg.get("peers", []) if p.get("url")
]
PEER_POLL = 15.0

# -- Model canary ---------------------------------------------------------
#
# A tiny real chat completion that proves the model produces coherent output
# ("/health returns 200" is not that — vLLM binds the port before loading, and
# a garbling model serves 200s). Run from the admin UI; set canary_interval > 0
# (seconds) to also run it on a timer. Sampled greedily, so one model gives one
# answer; the token budget leaves room for the thinking preamble some chat
# templates put into the visible reply (GLM's does, even at low effort — 32
# tokens cut it off before the answer a few times a day).
CANARY_INTERVAL = float(_cfg.get("canary_interval", 0) or 0)
CANARY_MAX_TOKENS = 128

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

# -- Engine stacks (inference engines launched outside sparkrun) ----------
#
# A stack is a set of containers (one per node) started by an operator script
# rather than a sparkrun recipe — e.g. a hand-tuned vLLM build. SparkDash
# detects it from its container-name prefix on every node, shows it as the
# owner of the serving port, drives it through `command start|stop|restart`
# (streamed into the same operation panel as recipes), refuses to start a
# recipe on top of it (and vice versa), and the watchdog guards it the same
# way it guards a recipe. The script must: start = launch and block until the
# engine is healthy (exit non-zero otherwise); stop = tear down every
# container; restart = both. One [[stacks]] block per stack:
#
#   [[stacks]]
#   name    = "glm53k"                       # short id (container prefix sans '-')
#   label   = "GLM-5.3-Flash (knapcio TP4)"  # shown in the UI
#   command = "/usr/local/bin/glm"           # start|stop|restart|status|logs (not under a home dir)
#   prefix  = "glm53k-"                      # container-name prefix on every node
#   model   = "glm-5.3-flash"                # served model name (informational)
#   head    = "glm53k-r0"                    # container whose logs to stream (default: <prefix>r0)
#   stopped_marker = "/var/lib/glm/stopped"  # optional: present = stopped on purpose
STACKS = []
for _s in _cfg.get("stacks", []):
    if not (_s.get("name") and _s.get("command") and _s.get("prefix")):
        continue
    STACKS.append({
        "name": str(_s["name"]),
        "label": str(_s.get("label") or _s["name"]),
        "command": str(_s["command"]),
        "prefix": str(_s["prefix"]),
        "model": str(_s.get("model") or ""),
        "head": str(_s.get("head") or f"{_s['prefix']}r0"),
        "stopped_marker": str(_s.get("stopped_marker") or ""),
        "expected": int(_s.get("expected") or len(NODES)),   # containers (one per node)
    })

# -- Watchdog (auto-recovery) --------------------------------------------
#
# Guards whatever engine owns the serving port — a sparkrun recipe or a
# stack: every container up, /health answering past the startup grace, and
# the model canary passing. After WATCHDOG_FAIL_THRESHOLD consecutive failing
# checks it alerts (through the configured webhook), and when every node is
# reachable and no automatic restart happened within the cooldown, restarts
# the engine through the same operation path the admin UI uses. A deliberate
# stop needs no marker: an engine that is no longer there is not guarded.
# `enabled` here locks the setting (UI read-only); otherwise the admin UI
# toggles it (stored in the settings table, default on).
_wd = _cfg.get("watchdog", {})
WATCHDOG_LOCKED = "enabled" in _wd
WATCHDOG_ENABLED_DEFAULT = bool(_wd.get("enabled", True))
WATCHDOG_INTERVAL = float(_wd.get("interval", 60) or 60)          # seconds between checks
WATCHDOG_FAIL_THRESHOLD = int(_wd.get("fail_threshold", 2) or 2)  # consecutive failing checks
WATCHDOG_GRACE_MIN = float(_wd.get("startup_grace_min", 30) or 30)  # a fresh engine may take this long
WATCHDOG_COOLDOWN_MIN = float(_wd.get("restart_cooldown_min", 40) or 40)
WATCHDOG_AUTO_RESTART = bool(_wd.get("auto_restart", True))        # false = alert only

# Launch watch: a sparkdash-driven start/restart follows the serve log and the
# serve PID inside the head container instead of only polling /health (see
# launch.py). A launch that logs nothing for stall_min while /health fails is
# called dead; one that keeps logging may take up to max_min.
_launch = _cfg.get("launch", {}) or {}
LAUNCH_STALL_MIN = float(_launch.get("stall_min", 10) or 10)
LAUNCH_MAX_MIN = float(_launch.get("max_min", 45) or 45)
