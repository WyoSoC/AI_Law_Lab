"""Bounded-queue LLM router across the DGX Spark units.

Why a queue and not round-robin
-------------------------------
Benchmarking both Sparks (2026-07-23) showed 50 concurrent requests complete without
error, but aggregate throughput flattens at ~160 tok/s from about 16 concurrent onward
while p50 latency climbs from 1.9s to 5.2s. The extra requests are queueing inside
Ollama, not running. Naive round-robin dispatch would happily push 50 requests at a
host and then report the resulting latency as if it were model speed.

So the router admits at most `slots_per_host` in-flight requests per host, queues the
rest itself, and reports queue-wait separately from model-eval time. That keeps
experiment timings interpretable and gives backpressure a place to live.

Host selection is least-loaded-first among healthy hosts, which self-balances better
than round-robin when requests have wildly different prompt sizes -- as they do when
one agent is summarizing a 200-page contract and another is emitting a one-line reply.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import settings

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """One completed model call, with timings split for the logging tier."""
    text: str
    thinking: str | None
    host: str
    queue_wait_ms: int
    eval_ms: int
    prompt_tokens: int
    output_tokens: int
    done_reason: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_s(self) -> float:
        return (self.output_tokens / (self.eval_ms / 1000)) if self.eval_ms else 0.0

    @property
    def truncated(self) -> bool:
        """Hit the token ceiling before finishing.

        Worth checking explicitly: gemma4 reasons before answering, so a stingy
        num_predict can be consumed entirely by the thinking block and return empty
        content with done_reason='length'. Silent empty strings are a nasty failure
        mode in a multi-step graph.
        """
        return self.done_reason == "length"


@dataclass
class HostState:
    base_url: str
    slots: asyncio.Semaphore
    in_flight: int = 0
    healthy: bool = True
    consecutive_failures: int = 0
    total_requests: int = 0
    total_errors: int = 0
    last_checked: float = 0.0

    @property
    def name(self) -> str:
        return self.base_url.split("//", 1)[-1].split(":", 1)[0]


class NoHealthyHostsError(RuntimeError):
    pass


class LLMRouter:
    """Routes chat/embed calls across Spark hosts with per-host admission control."""

    def __init__(self, base_urls: list[str] | None = None, slots_per_host: int | None = None):
        urls = base_urls if base_urls is not None else settings.base_urls()
        self.slots_per_host = slots_per_host or settings.slots_per_host
        self.hosts = [
            HostState(base_url=u, slots=asyncio.Semaphore(self.slots_per_host)) for u in urls
        ]
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()
        self._global_capacity = self.slots_per_host * len(self.hosts)
        self._global_slots = asyncio.Semaphore(self._global_capacity)

    # ------------------------------------------------------------------ lifecycle

    async def __aenter__(self) -> LLMRouter:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def start(self) -> None:
        if self._client is None:
            # Connection pool sized to total admitted concurrency, plus headroom.
            limit = self.slots_per_host * len(self.hosts) + 8
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.request_timeout_s, connect=10.0),
                limits=httpx.Limits(max_connections=limit, max_keepalive_connections=limit),
            )
        await self.health_check()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("LLMRouter used before start(); call `await router.start()`")
        return self._client

    # ------------------------------------------------------------------ health

    async def health_check(self) -> dict[str, bool]:
        """Probe every host. A host that fails is parked until it recovers."""
        async def probe(h: HostState) -> None:
            try:
                r = await self.client.get(f"{h.base_url}/api/version", timeout=5.0)
                r.raise_for_status()
                if not h.healthy:
                    log.info("host %s recovered", h.name)
                h.healthy = True
                h.consecutive_failures = 0
            except Exception as e:  # noqa: BLE001 - any failure means "don't route here"
                h.healthy = False
                log.warning("host %s unhealthy: %s", h.name, e)
            h.last_checked = time.time()

        await asyncio.gather(*(probe(h) for h in self.hosts))
        return {h.name: h.healthy for h in self.hosts}

    # ------------------------------------------------------------------ admission

    @asynccontextmanager
    async def _acquire(self):
        """Admit one request, yielding (host, queue_wait_ms).

        Two stages, deliberately:

        1. A single global semaphore is the queue. Its capacity is the sum of healthy
           host capacity, so total in-flight work never exceeds what the cluster can
           actually run concurrently. Waiting here *is* the backpressure, and the time
           spent is what gets reported as queue_wait_ms.
        2. Once admitted, pick the least-loaded healthy host. By construction a slot is
           essentially always free, so this rarely blocks -- but the per-host semaphore
           is still held to keep the invariant true if a host is parked mid-flight.

        Racing acquires across hosts would shave a little latency but makes cancellation
        semantics subtle for no measurable gain at this scale.
        """
        t0 = time.perf_counter()
        await self._resize_global()

        async with self._global_slots:
            healthy = [h for h in self.hosts if h.healthy]
            if not healthy:
                await self.health_check()
                healthy = [h for h in self.hosts if h.healthy]
                if not healthy:
                    raise NoHealthyHostsError(
                        f"no healthy Ollama hosts among {[h.name for h in self.hosts]}"
                    )

            host = min(healthy, key=lambda h: h.in_flight)
            await host.slots.acquire()
            try:
                yield host, int((time.perf_counter() - t0) * 1000)
            finally:
                host.slots.release()

    async def _resize_global(self) -> None:
        """Keep global queue capacity in step with how many hosts are healthy.

        asyncio.Semaphore has no resize, so this rebuilds it when the healthy count
        changes. Only ever grows capacity for waiters already queued on the old object;
        in-flight requests hold the old semaphore and drain normally.
        """
        want = max(1, self.slots_per_host * sum(1 for h in self.hosts if h.healthy))
        async with self._lock:
            if want != self._global_capacity:
                log.info("resizing global capacity %d -> %d", self._global_capacity, want)
                self._global_capacity = want
                self._global_slots = asyncio.Semaphore(want)

    # ------------------------------------------------------------------ calls

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        think: bool | None = None,
        num_ctx: int = 32768,
        temperature: float = 0.2,
        options: dict[str, Any] | None = None,
    ) -> LLMResult:
        """One chat completion, routed to whichever Spark frees a slot first.

        `think` is passed through explicitly rather than left to the model default.
        gemma4 reasons by default, which is what we want for research traces but costs
        roughly an order of magnitude more output tokens on short mechanical answers
        (measured: 2 tokens with think=False vs 32 with it on, for the same reply).
        Nodes that just classify or route should pass think=False.
        """
        body: dict[str, Any] = {
            "model": model or settings.chat_model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": num_ctx, "temperature": temperature, **(options or {})},
        }
        if tools:
            body["tools"] = tools
        if think is not None:
            body["think"] = think

        async with self._acquire() as (host, queue_wait_ms):
            host.in_flight += 1
            host.total_requests += 1
            try:
                r = await self.client.post(f"{host.base_url}/api/chat", json=body)
                r.raise_for_status()
                data = r.json()
            except Exception:
                host.total_errors += 1
                host.consecutive_failures += 1
                if host.consecutive_failures >= 3:
                    host.healthy = False
                    log.warning("host %s parked after 3 consecutive failures", host.name)
                raise
            finally:
                host.in_flight -= 1

        host.consecutive_failures = 0
        msg = data.get("message", {}) or {}
        return LLMResult(
            text=msg.get("content", "") or "",
            thinking=msg.get("thinking") or None,
            host=host.name,
            queue_wait_ms=queue_wait_ms,
            eval_ms=int(data.get("eval_duration", 0) / 1e6),
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            output_tokens=int(data.get("eval_count", 0) or 0),
            done_reason=data.get("done_reason", "") or "",
            tool_calls=msg.get("tool_calls", []) or [],
            raw=data,
        )

    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Embed a batch of texts. Ollama's /api/embed accepts a list directly."""
        if not texts:
            return []
        body = {"model": model or settings.embed_model, "input": texts}
        async with self._acquire() as (host, _):
            host.in_flight += 1
            host.total_requests += 1
            try:
                r = await self.client.post(f"{host.base_url}/api/embed", json=body)
                r.raise_for_status()
                return r.json()["embeddings"]
            except Exception:
                host.total_errors += 1
                raise
            finally:
                host.in_flight -= 1

    async def embed_one(self, text: str, *, model: str | None = None) -> list[float]:
        return (await self.embed([text], model=model))[0]

    # ------------------------------------------------------------------ introspection

    def stats(self) -> dict[str, Any]:
        return {
            "slots_per_host": self.slots_per_host,
            "total_slots": self.slots_per_host * sum(1 for h in self.hosts if h.healthy),
            "hosts": [
                {
                    "name": h.name,
                    "url": h.base_url,
                    "healthy": h.healthy,
                    "in_flight": h.in_flight,
                    "free_slots": self.slots_per_host - h.in_flight,
                    "total_requests": h.total_requests,
                    "total_errors": h.total_errors,
                }
                for h in self.hosts
            ],
        }


_router: LLMRouter | None = None


async def get_router() -> LLMRouter:
    """Process-wide router singleton."""
    global _router
    if _router is None:
        _router = LLMRouter()
        await _router.start()
    return _router
