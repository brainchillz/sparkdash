"""Internet download rate cap for model preloads.

A 182 GB checkpoint pulled at line rate flattens the house uplink for
everyone else. When a cap is configured, the Preloader brackets the
download phase with a tc ingress policer on the WAN-facing interface of
this host (the head — sparkrun downloads once here, then mirrors to the
workers over RDMA, so the internet only ever flows through this box).

The policer drops inbound packets above the configured rate and TCP settles
underneath it. Traffic from RFC1918 sources (LAN clients, the NFS share,
the RDMA mirror) is exempted with pass filters ahead of the policer, so
only internet-sourced ingress is throttled.

Configuration tiers, first match wins (mirrors alerts.py):

  1. `download_cap_mbit` in config.toml            (locked: UI read-only)
  2. the settings table in the admin DB, managed from the admin UI

Privileges: under the systemd unit the service holds CAP_NET_ADMIN
(AmbientCapabilities — NoNewPrivileges blocks sudo there), so tc runs
directly; a dev shell falls back to passwordless sudo. If neither works
the preload proceeds uncapped and the error is surfaced in the preload
status rather than blocking the download.
"""

from __future__ import annotations

import asyncio
import re

from . import config, store

_TC = "/usr/sbin/tc"
#: Pass filters ahead of the policer: sources that must never be throttled.
_EXEMPT_SRC = ("192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12")
MIN_MBIT, MAX_MBIT = 50, 100_000


# -- configuration -----------------------------------------------------------

def get_mbit() -> int:
    """Resolve the active cap in Mbit/s, 0 = uncapped."""
    if config.DOWNLOAD_CAP_MBIT > 0:
        return config.DOWNLOAD_CAP_MBIT
    try:
        return int(store.get_setting("netcap.mbit") or 0)
    except ValueError:
        return 0


def locked() -> bool:
    """True when config.toml fixes this and the UI must not edit it."""
    return config.DOWNLOAD_CAP_MBIT > 0


def save_stored(mbit: int) -> None:
    store.set_setting("netcap.mbit", str(mbit))


def clear_stored() -> bool:
    return store.delete_setting("netcap.mbit")


# -- tc plumbing -------------------------------------------------------------

async def _run(*argv: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace").strip()


async def _tc(*args: str) -> tuple[int, str]:
    """Run a privileged tc command. Under the systemd unit the service holds
    CAP_NET_ADMIN (NoNewPrivileges blocks sudo there), so plain tc works;
    a dev shell without the capability falls back to passwordless sudo."""
    rc, out = await _run(_TC, *args)
    if rc != 0 and ("not permitted" in out.lower()
                    or "permission denied" in out.lower()):
        rc, out = await _run("sudo", "-n", _TC, *args)
    return rc, out


async def iface() -> str:
    """WAN-facing interface: config override, else the default-route device."""
    if config.DOWNLOAD_CAP_IFACE:
        return config.DOWNLOAD_CAP_IFACE
    rc, out = await _run("ip", "-o", "route", "show", "default")
    m = re.search(r"\bdev\s+(\S+)", out) if rc == 0 else None
    return m.group(1) if m else ""


async def apply(mbit: int) -> str | None:
    """Install the ingress policer. Returns an error string, or None."""
    dev = await iface()
    if not dev:
        return "no default-route interface found (set download_cap_iface)"
    # Idempotent: tear down any previous cap first.
    await _tc("qdisc", "del", "dev", dev, "ingress")
    rc, out = await _tc("qdisc", "add", "dev", dev,
                       "handle", "ffff:", "ingress")
    if rc != 0:
        return f"tc qdisc add failed: {out[-200:]}"
    for prio, src in enumerate(_EXEMPT_SRC, start=1):
        rc, out = await _tc(
            "filter", "add", "dev", dev, "parent", "ffff:",
            "protocol", "ip", "prio", str(prio), "u32",
            "match", "ip", "src", src, "flowid", ":1")
        if rc != 0:
            await remove()
            return f"tc exempt filter failed: {out[-200:]}"
    # ~100 ms of line rate as burst so TCP can breathe without overshooting.
    burst_kb = max(256, mbit * 128 // 10)
    rc, out = await _tc(
        "filter", "add", "dev", dev, "parent", "ffff:",
        "protocol", "ip", "prio", "9", "u32", "match", "u32", "0", "0",
        "police", "rate", f"{mbit}mbit", "burst", f"{burst_kb}k",
        # GRO hands ingress up-to-64KB superpackets; the police default of
        # mtu 2Kb would drop them all outright (observed: 20 B/s throughput).
        "mtu", "65535",
        "drop", "flowid", ":2")
    if rc != 0:
        await remove()
        return f"tc police filter failed: {out[-200:]}"
    return None


async def remove() -> None:
    """Tear down the policer (no-op if absent)."""
    dev = await iface()
    if dev:
        await _tc("qdisc", "del", "dev", dev, "ingress")


async def active() -> dict:
    """Whether a cap is currently installed (reads need no sudo)."""
    dev = await iface()
    if not dev:
        return {"active": False, "iface": ""}
    rc, out = await _run(_TC, "filter", "show", "dev", dev, "parent", "ffff:")
    m = re.search(r"rate (\d+)([KMG])bit", out) if rc == 0 else None
    mbit = 0
    if m:
        mbit = int(m.group(1)) * {"K": 1, "M": 1000, "G": 1000_000}[m.group(2)] // 1000
    return {"active": bool(m), "iface": dev, "mbit": mbit}
