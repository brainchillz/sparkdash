"""Persistent metrics history: sampler, rollup, and range queries.

A background task snapshots the hub every HISTORY_INTERVAL seconds into a
dedicated SQLite file (WAL, one connection per call — same pattern as
store.py, kept separate so the auth DB stays tiny). Raw rows are kept for
HISTORY_RAW_KEEP seconds, then averaged into HISTORY_BUCKET-second
min/avg/max rollups kept indefinitely — a year of two-node history is tens
of MB, and temperature/util spikes survive the averaging via min/max.

Generation throughput is derived here (delta of vLLM's cumulative token
counter between samples) so history survives counter resets and restarts.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from contextlib import contextmanager

from . import config

log = logging.getLogger(__name__)

# Per-node columns sampled from each snapshot node dict (name → column).
NODE_COLS = [
    "cpu_pct", "cpu_temp_c",
    "gpu_util_pct", "gpu_temp_c", "gpu_power_w",
    "mem_used_mb", "vram_used_mb",
    "disk_used", "disk_total",
]
# Cluster-wide columns (one row per sample, node key "cluster" in rollups).
CLUSTER_COLS = ["tokens_per_sec", "kv_cache_pct", "requests_running"]

RANGES = {
    "1h": 3600, "6h": 6 * 3600, "24h": 86400,
    "7d": 7 * 86400, "30d": 30 * 86400, "1y": 365 * 86400,
}

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS node_samples (
    ts    INTEGER NOT NULL,
    node  TEXT NOT NULL,
    {", ".join(c + " REAL" for c in NODE_COLS)},
    PRIMARY KEY (ts, node)
);
CREATE TABLE IF NOT EXISTS cluster_samples (
    ts    INTEGER PRIMARY KEY,
    {", ".join(c + " REAL" for c in CLUSTER_COLS)}
);
CREATE TABLE IF NOT EXISTS rollup (
    bucket INTEGER NOT NULL,
    node   TEXT NOT NULL,
    metric TEXT NOT NULL,
    avg REAL, min REAL, max REAL, n INTEGER,
    PRIMARY KEY (bucket, node, metric)
);
"""


def init_db() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.executescript(_SCHEMA)


@contextmanager
def _conn():
    conn = sqlite3.connect(config.HISTORY_DB, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# -- sampling ---------------------------------------------------------------

_last_gen: tuple[float, float] | None = None  # (ts, generation_tokens_total)


def record(snap: dict) -> None:
    """Persist one snapshot. Offline nodes are skipped — the missing row
    renders as a gap, which is the honest display."""
    global _last_gen
    ts = int(snap.get("ts") or time.time())

    node_rows = [
        (ts, n["name"], *[n.get(c) for c in NODE_COLS])
        for n in snap.get("nodes", []) if n.get("online")
    ]

    cluster_row = None
    vllm = snap.get("vllm") or {}
    if vllm.get("reachable"):
        m = vllm.get("metrics") or {}
        gen = m.get("vllm:generation_tokens_total")
        tok_s = None
        if gen is not None:
            if _last_gen is not None:
                dt, dv = ts - _last_gen[0], gen - _last_gen[1]
                # dv < 0 means the counter reset (vLLM restart) — unknown rate.
                if dt > 0 and dv >= 0:
                    tok_s = dv / dt
            _last_gen = (ts, gen)
        kv = m.get("vllm:kv_cache_usage_perc")
        if kv is None:
            kv = m.get("vllm:gpu_cache_usage_perc")
        cluster_row = (ts, tok_s,
                       kv * 100 if kv is not None else None,
                       m.get("vllm:num_requests_running"))

    with _conn() as c:
        if node_rows:
            c.executemany(
                f"INSERT OR REPLACE INTO node_samples VALUES "
                f"({','.join('?' * (len(NODE_COLS) + 2))})", node_rows)
        if cluster_row:
            c.execute(
                f"INSERT OR REPLACE INTO cluster_samples VALUES "
                f"({','.join('?' * (len(CLUSTER_COLS) + 1))})", cluster_row)


# -- rollup / retention -----------------------------------------------------

def rollup_and_prune() -> None:
    """Fold raw rows older than HISTORY_RAW_KEEP into 5-min rollups, then
    delete them. The cutoff is snapped to a bucket boundary so every rolled
    bucket is complete — reruns can never produce a half-filled duplicate."""
    b = config.HISTORY_BUCKET
    cutoff = (int(time.time()) - config.HISTORY_RAW_KEEP) // b * b
    with _conn() as c:
        for col in NODE_COLS:
            c.execute(
                f"INSERT OR IGNORE INTO rollup "
                f"SELECT (ts/{b})*{b}, node, ?, AVG({col}), MIN({col}), "
                f"MAX({col}), COUNT({col}) FROM node_samples "
                f"WHERE ts < ? AND {col} IS NOT NULL GROUP BY 1, 2",
                (col, cutoff))
        for col in CLUSTER_COLS:
            c.execute(
                f"INSERT OR IGNORE INTO rollup "
                f"SELECT (ts/{b})*{b}, 'cluster', ?, AVG({col}), MIN({col}), "
                f"MAX({col}), COUNT({col}) FROM cluster_samples "
                f"WHERE ts < ? AND {col} IS NOT NULL GROUP BY 1",
                (col, cutoff))
        c.execute("DELETE FROM node_samples WHERE ts < ?", (cutoff,))
        c.execute("DELETE FROM cluster_samples WHERE ts < ?", (cutoff,))


# -- queries ----------------------------------------------------------------

def query(range_key: str) -> dict:
    """All series for a range, bucket-averaged server-side to a fixed time
    axis of ≤ HISTORY_MAX_POINTS points (uPlot-ready aligned arrays)."""
    span = RANGES[range_key]  # KeyError → 400 at the endpoint
    end = int(time.time())
    step = max(int(config.HISTORY_INTERVAL),
               math.ceil(span / config.HISTORY_MAX_POINTS))
    # Snap to clean bucket multiples: the sampler cadence while raw data is
    # in play, the rollup bucket once the range reaches into rollups.
    unit = int(config.HISTORY_INTERVAL) if step <= config.HISTORY_BUCKET \
        else config.HISTORY_BUCKET
    step = math.ceil(step / unit) * unit
    start = (end - span) // step * step
    axis = list(range(start, end + 1, step))

    node_names = [n["name"] for n in config.NODES]
    node_series = {m: {n: [None] * len(axis) for n in node_names}
                   for m in NODE_COLS}
    cluster_series = {m: [None] * len(axis) for m in CLUSTER_COLS}

    def put(bucket: int, node: str, metric: str, val) -> None:
        if val is None:
            return
        i = (bucket - start) // step
        if not 0 <= i < len(axis):
            return
        if node == "cluster":
            if metric in cluster_series:
                cluster_series[metric][i] = val
        elif metric in node_series and node in node_series[metric]:
            node_series[metric][node][i] = val

    with _conn() as c:
        sel = ", ".join(f"AVG({col})" for col in NODE_COLS)
        for row in c.execute(
                f"SELECT (ts/{step})*{step} AS b, node, {sel} "
                f"FROM node_samples WHERE ts >= ? GROUP BY b, node", (start,)):
            for col, val in zip(NODE_COLS, list(row)[2:]):
                put(row["b"], row["node"], col, val)
        sel = ", ".join(f"AVG({col})" for col in CLUSTER_COLS)
        for row in c.execute(
                f"SELECT (ts/{step})*{step} AS b, {sel} "
                f"FROM cluster_samples WHERE ts >= ? GROUP BY b", (start,)):
            for col, val in zip(CLUSTER_COLS, list(row)[1:]):
                put(row["b"], "cluster", col, val)
        # Rollups only hold data older than the raw window, so buckets can
        # never collide with the raw fills above.
        for row in c.execute(
                "SELECT (bucket/{s})*{s} AS b, node, metric, "
                "SUM(avg*n)/SUM(n) AS v FROM rollup WHERE bucket >= ? "
                "GROUP BY b, node, metric".format(s=step), (start,)):
            put(row["b"], row["node"], row["metric"], row["v"])

    return {
        "range": range_key, "start": start, "end": end, "step": step,
        "ts": axis, "nodes": node_names,
        "node_series": node_series, "cluster_series": cluster_series,
    }


# -- background task --------------------------------------------------------

async def sampler_loop(hub) -> None:
    import asyncio
    last_rollup = 0.0
    while True:
        await asyncio.sleep(config.HISTORY_INTERVAL)
        try:
            snap = hub.snapshot()
            await asyncio.to_thread(record, snap)
            if time.time() - last_rollup >= 3600:
                last_rollup = time.time()
                await asyncio.to_thread(rollup_and_prune)
        except Exception:
            log.exception("history sampler iteration failed")
