"""Prometheus exposition of the merged snapshot.

Re-exports SparkDash's collected state as Prometheus text so Grafana/Prometheus
can scrape everything from one endpoint — including the GB10 per-process VRAM
figure, which neither vLLM's nor Ray's own /metrics exposes (unified memory).

Hand-rolled (no prometheus_client dependency) to keep full control over the
GB10-specific series. Values that are None are omitted rather than zero-filled,
so a missing probe doesn't look like a real reading.
"""

from __future__ import annotations

MIB = 1024 * 1024


def _esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(pairs: dict[str, str]) -> str:
    if not pairs:
        return ""
    inner = ",".join(f'{k}="{_esc(str(val))}"' for k, val in pairs.items())
    return "{" + inner + "}"


class _Doc:
    """Accumulates metric families with HELP/TYPE emitted once each."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._declared: set[str] = set()

    def add(self, name: str, value, help: str, labels: dict | None = None,
            type_: str = "gauge") -> None:
        if value is None:
            return
        if isinstance(value, bool):
            value = 1 if value else 0
        if name not in self._declared:
            self._lines.append(f"# HELP {name} {help}")
            self._lines.append(f"# TYPE {name} {type_}")
            self._declared.add(name)
        self._lines.append(f"{name}{_labels(labels or {})} {value}")

    def render(self) -> str:
        return "\n".join(self._lines) + "\n"


def render_prometheus(snap: dict) -> str:
    doc = _Doc()

    doc.add("sparkdash_cluster_healthy", snap.get("cluster_healthy"),
            "1 if Ray, vLLM and both nodes are all healthy.")

    ray = snap.get("ray", {})
    doc.add("sparkdash_ray_reachable", ray.get("reachable"),
            "1 if the Ray dashboard responded.")
    doc.add("sparkdash_ray_nodes_alive", ray.get("nodes_alive"),
            "Ray nodes in ALIVE state.")
    doc.add("sparkdash_ray_nodes_total", ray.get("nodes_total"),
            "Ray nodes known to the cluster.")

    vllm = snap.get("vllm", {})
    doc.add("sparkdash_vllm_reachable", vllm.get("reachable"),
            "1 if the vLLM endpoint responded.")
    doc.add("sparkdash_vllm_healthy", vllm.get("healthy"),
            "1 if vLLM /health returned 200.")
    if vllm.get("model"):
        doc.add("sparkdash_vllm_info", 1,
                "Loaded model as a label (value always 1).",
                labels={"model": vllm["model"]})
    m = vllm.get("metrics", {})
    doc.add("sparkdash_vllm_requests_running", m.get("vllm:num_requests_running"),
            "Requests currently executing in vLLM.")
    doc.add("sparkdash_vllm_requests_waiting", m.get("vllm:num_requests_waiting"),
            "Requests waiting to be scheduled in vLLM.")
    doc.add("sparkdash_vllm_kv_cache_usage_ratio",
            m.get("vllm:kv_cache_usage_perc", m.get("vllm:gpu_cache_usage_perc")),
            "KV-cache utilisation (0-1).")
    doc.add("sparkdash_vllm_prefix_cache_hit_ratio", m.get("prefix_cache_hit_rate"),
            "Prefix-cache hit rate (0-1).")

    recipe = snap.get("recipe", {})
    doc.add("sparkdash_recipe_running", bool(recipe.get("running")),
            "1 if a sparkrun recipe is running.")
    if recipe.get("running"):
        doc.add("sparkdash_recipe_info", 1,
                "Running recipe as labels (value always 1).",
                labels={"name": recipe.get("name", ""), "tp": recipe.get("tp", "")})

    for n in snap.get("nodes", []):
        lb = {"node": n["name"], "role": n["role"]}
        doc.add("sparkdash_node_online", n.get("online"),
                "1 if the node is reporting via the monitor stream.", lb)
        doc.add("sparkdash_node_cpu_percent", n.get("cpu_pct"),
                "Node CPU utilisation (0-100).", lb)
        doc.add("sparkdash_node_cpu_load1", n.get("cpu_load_1m"),
                "1-minute load average.", lb)
        doc.add("sparkdash_node_cpu_temp_celsius", n.get("cpu_temp_c"),
                "CPU package temperature.", lb)
        doc.add("sparkdash_node_memory_used_bytes",
                _mb(n.get("mem_used_mb")), "RAM used.", lb)
        doc.add("sparkdash_node_memory_total_bytes",
                _mb(n.get("mem_total_mb")), "RAM total.", lb)
        # The GB10 unified-memory workaround: per-process VRAM sum.
        doc.add("sparkdash_node_vram_used_bytes",
                _mb(n.get("vram_used_mb")),
                "VRAM in use (sum of per-process compute-app memory; GB10 "
                "unified memory has no aggregate query).", lb)
        doc.add("sparkdash_node_gpu_utilization_percent", n.get("gpu_util_pct"),
                "GPU utilisation (0-100).", lb)
        doc.add("sparkdash_node_gpu_temp_celsius", n.get("gpu_temp_c"),
                "GPU temperature.", lb)
        doc.add("sparkdash_node_gpu_power_watts", n.get("gpu_power_w"),
                "GPU power draw.", lb)
        doc.add("sparkdash_node_gpu_clock_mhz", n.get("gpu_clock_mhz"),
                "GPU core clock.", lb)
        doc.add("sparkdash_node_disk_used_bytes", n.get("disk_used"),
                "Root filesystem used.", lb)
        doc.add("sparkdash_node_disk_total_bytes", n.get("disk_total"),
                "Root filesystem total.", lb)
        doc.add("sparkdash_node_uptime_seconds", n.get("uptime_sec"),
                "Node uptime.", lb, type_="counter")

    return doc.render()


def _mb(mib: float | None) -> float | None:
    return None if mib is None else mib * MIB
