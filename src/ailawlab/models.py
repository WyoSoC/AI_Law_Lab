"""The chat models a role can be played by: what Ollama has installed on every Spark.

A role-play agent may name its own model, so a run can set, say, qwen3.6 against gemma4,
or swap them to see whether the outcome follows the model rather than the case. A model
is offered only when every Spark has it, because the router sends a call to whichever
Spark frees a slot first, and only when its name matches none of settings.hidden_models
(the uncensored variants installed for other work). Capabilities come from Ollama's /api/show: a model without
"thinking" (hermes3) must be called with think off, or Ollama refuses the request.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from .config import settings

log = logging.getLogger(__name__)

CACHE_S = 300
_cache: dict[str, Any] = {"at": 0.0, "models": []}
_lock = asyncio.Lock()


async def _host_models(client: httpx.AsyncClient, base: str) -> dict[str, dict]:
    r = await client.get(f"{base}/api/tags")
    r.raise_for_status()
    return {m["name"]: m for m in r.json().get("models", [])}


async def _show(client: httpx.AsyncClient, base: str, name: str) -> dict:
    r = await client.post(f"{base}/api/show", json={"model": name})
    r.raise_for_status()
    return r.json()


def hidden(name: str) -> bool:
    return any(h.lower() in name.lower() for h in settings.hidden_models if h)


def _describe(name: str, tag: dict, show: dict) -> dict | None:
    caps = show.get("capabilities") or []
    if "completion" not in caps:            # embedding models
        return None
    info = show.get("model_info") or {}
    context = max((v for k, v in info.items() if k.endswith("context_length") and isinstance(v, int)),
                  default=0)
    return {"name": name,
            "params": (show.get("details") or tag.get("details") or {}).get("parameter_size", ""),
            "size_gb": round((tag.get("size") or 0) / 1e9, 1),
            "thinking": "thinking" in caps, "tools": "tools" in caps,
            "context": context}


async def chat_models(refresh: bool = False) -> list[dict]:
    """Chat models installed on every Spark and not hidden, the lab default first, then by name. A Spark
    that cannot be reached is left out of the intersection rather than emptying the list."""
    async with _lock:
        if not refresh and _cache["models"] and time.monotonic() - _cache["at"] < CACHE_S:
            return _cache["models"]
        async with httpx.AsyncClient(timeout=10) as client:
            per_host = await asyncio.gather(*(_host_models(client, b) for b in settings.base_urls()),
                                            return_exceptions=True)
            reachable = [(b, h) for b, h in zip(settings.base_urls(), per_host, strict=True)
                         if isinstance(h, dict)]
            if not reachable:
                log.warning("no Spark answered for its model list")
                return _cache["models"]
            names = {n for n in set.intersection(*(set(h) for _, h in reachable)) if not hidden(n)}
            base, tags = reachable[0]
            shows = await asyncio.gather(*(_show(client, base, n) for n in sorted(names)),
                                         return_exceptions=True)
        models = [m for n, s in zip(sorted(names), shows, strict=True) if isinstance(s, dict)
                  if (m := _describe(n, tags[n], s))]
        models.sort(key=lambda m: (m["name"] != settings.chat_model, m["name"]))
        _cache.update(at=time.monotonic(), models=models)
        return models


async def model_info(name: str) -> dict | None:
    return next((m for m in await chat_models() if m["name"] == name), None)


def thinks(model: dict | None) -> bool:
    """Whether to ask this model to reason first. An unknown model is assumed to, as gemma4
    does; the lab's models without thinking are listed, so they are known."""
    return True if model is None else bool(model.get("thinking"))
