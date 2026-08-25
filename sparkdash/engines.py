"""Inference-engine adapters.

Each supported engine exposes Prometheus metrics under its own prefix with
its own names; everything downstream of the collector — frontend tiles,
history sampler, Prometheus re-exporter, alerts — consumes one normalized
schema (the vllm: names, kept for compatibility with existing history data).
An adapter is just the translation table plus the quirks that can't be
expressed as a rename. Adding an engine means adding an entry here and
nothing else.

Detection is by metrics prefix: whichever engine's prefix appears in the
scrape names the engine. No config required, and a host that switches
engines (pebbles has run vLLM, SGLang, and llama.cpp on one port) is picked
up on the next poll.
"""

from __future__ import annotations

import re

# Normalized names every consumer understands. Values are 0-1 for the cache
# ratio, monotonic counters for the *_total names.
_METRIC_RE = re.compile(
    r"^((?:vllm|sglang|llamacpp):[a-zA-Z_]+)(?:\{[^}]*\})?\s+([0-9eE.+\-]+)$")

_WANTED = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
}

# SGLang serves the same concepts under its own prefix (run with
# --enable-metrics).
_SGLANG_MAP = {
    "sglang:num_running_reqs": "vllm:num_requests_running",
    "sglang:num_queue_reqs": "vllm:num_requests_waiting",
    "sglang:token_usage": "vllm:kv_cache_usage_perc",
    "sglang:prompt_tokens_total": "vllm:prompt_tokens_total",
    "sglang:generation_tokens_total": "vllm:generation_tokens_total",
    "sglang:num_requests_total": "vllm:request_success_total",
    "sglang:cache_hit_rate": "prefix_cache_hit_rate",   # already a 0-1 ratio
}

# llama.cpp's llama-server (run with --metrics). Names verified against the
# OTNOPS b10472 build. Its spec-decode counters carry the same semantics as
# vLLM's, so the acceptance watch works unchanged. No KV-usage gauge in this
# build — that tile just shows "—". prompt_tokens_cached_total feeds the
# prefix-cache hit rate with prompt_tokens_total as the denominator (see
# parse_metrics).
_LLAMACPP_MAP = {
    "llamacpp:requests_processing": "vllm:num_requests_running",
    "llamacpp:requests_deferred": "vllm:num_requests_waiting",
    "llamacpp:prompt_tokens_total": "vllm:prompt_tokens_total",
    "llamacpp:prompt_tokens_cached_total": "vllm:prefix_cache_hits_total",
    "llamacpp:tokens_predicted_total": "vllm:generation_tokens_total",
    "llamacpp:spec_decode_num_drafts_total": "vllm:spec_decode_num_drafts_total",
    "llamacpp:spec_decode_num_draft_tokens_total": "vllm:spec_decode_num_draft_tokens_total",
    "llamacpp:spec_decode_num_accepted_tokens_total": "vllm:spec_decode_num_accepted_tokens_total",
}

_MAPS = {"sglang": _SGLANG_MAP, "llamacpp": _LLAMACPP_MAP, "vllm": {}}


def parse_metrics(text: str) -> tuple[dict[str, float], str | None]:
    """(normalized metrics, engine) from a Prometheus scrape.

    The engine is whichever known prefix appears in the scrape ("vllm",
    "sglang", "llamacpp"), or None when nothing recognizable was found.
    """
    out: dict[str, float] = {}
    engine: str | None = None
    for line in text.splitlines():
        if line.startswith("#") or "_created" in line:
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        name, value = m.group(1), m.group(2)
        prefix = name.split(":", 1)[0]
        engine = engine or prefix
        name = _MAPS[prefix].get(name, name)
        if name in _WANTED or name == "prefix_cache_hit_rate":
            try:
                out[name] = out.get(name, 0.0) + float(value)
            except ValueError:
                pass
    # Derived: prefix-cache hit rate (vLLM exposes the two counters instead;
    # llama.cpp has no queries counter — every prompt token is a query).
    q = out.get("vllm:prefix_cache_queries_total", 0.0)
    if engine == "llamacpp" and not q:
        q = out.get("vllm:prompt_tokens_total", 0.0)
    h = out.get("vllm:prefix_cache_hits_total", 0.0)
    if q > 0:
        out["prefix_cache_hit_rate"] = h / q
    return out, engine


def display_model(model_id: str | None, engine: str | None) -> str | None:
    """The model name as the dashboard should show it.

    llama.cpp reports the model as the .gguf file path it was launched with;
    a path is an implementation detail, so show the basename without the
    extension. Repo-id style names (vLLM/SGLang) pass through untouched.
    """
    if model_id and engine == "llamacpp" and model_id.endswith(".gguf"):
        return model_id.rsplit("/", 1)[-1][:-len(".gguf")]
    return model_id
