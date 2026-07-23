"""Router tests. These hit the real Sparks -- they are the integration check that the
admission control behaves against actual hardware, not a mock of it."""
from __future__ import annotations

import asyncio

import pytest

from ailawlab.router import LLMRouter, NoHealthyHostsError


@pytest.fixture
async def router():
    r = LLMRouter()
    await r.start()
    yield r
    await r.aclose()


async def test_hosts_healthy(router):
    health = await router.health_check()
    assert any(health.values()), f"no healthy hosts: {health}"


async def test_chat_returns_timings(router):
    res = await router.chat(
        [{"role": "user", "content": "Reply with exactly: OK"}],
        think=False,
        options={"num_predict": 32},
    )
    assert res.text.strip()
    assert res.host
    assert res.eval_ms >= 0
    assert res.queue_wait_ms >= 0
    assert res.output_tokens > 0
    assert not res.truncated


async def test_thinking_is_captured_separately(router):
    """gemma4's reasoning must land in .thinking, not be mixed into the answer --
    the logging tier stores them in different columns."""
    res = await router.chat(
        [{"role": "user", "content": "Is a verbal contract enforceable? One sentence."}],
        think=True,
        options={"num_predict": 400},
    )
    assert res.thinking, "expected a reasoning trace with think=True"
    assert res.text.strip(), "reasoning consumed the whole budget; answer was empty"
    assert res.thinking not in res.text


async def test_think_false_suppresses_reasoning(router):
    """Cheap path for mechanical nodes: no reasoning tokens at all."""
    res = await router.chat(
        [{"role": "user", "content": "Say the number 7 and nothing else."}],
        think=False,
        options={"num_predict": 32},
    )
    assert not res.thinking
    assert "7" in res.text


async def test_truncation_is_detectable(router):
    """A budget too small to finish must be visible, not a silent empty string."""
    res = await router.chat(
        [{"role": "user", "content": "Explain promissory estoppel in detail."}],
        think=True,
        options={"num_predict": 8},
    )
    assert res.truncated


async def test_embed_dimension(router):
    vecs = await router.embed(["a legal test sentence", "another one"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 768


async def test_admission_never_exceeds_slots():
    """The core invariant: in-flight per host stays within slots_per_host under load."""
    r = LLMRouter(slots_per_host=4)
    await r.start()
    observed_max = 0

    async def one(i: int):
        nonlocal observed_max
        res = await r.chat(
            [{"role": "user", "content": f"Say the number {i} and nothing else."}],
            think=False,
            options={"num_predict": 32},
        )
        observed_max = max(observed_max, max(h.in_flight for h in r.hosts))
        return res

    # Sample in-flight while the burst is running, not just after.
    async def sampler(stop: asyncio.Event):
        nonlocal observed_max
        while not stop.is_set():
            observed_max = max(observed_max, max(h.in_flight for h in r.hosts))
            await asyncio.sleep(0.01)

    stop = asyncio.Event()
    sampler_task = asyncio.create_task(sampler(stop))
    results = await asyncio.gather(*(one(i) for i in range(24)))
    stop.set()
    await sampler_task
    await r.aclose()

    assert len(results) == 24
    assert all(res.text for res in results)
    assert observed_max <= 4, f"admission control breached: saw {observed_max} in flight"


async def test_queue_wait_recorded_under_saturation():
    """With one slot, requests must queue -- and that wait must show up in queue_wait_ms."""
    r = LLMRouter(base_urls=[LLMRouter().hosts[0].base_url], slots_per_host=1)
    await r.start()
    results = await asyncio.gather(
        *(r.chat([{"role": "user", "content": "Count to three."}], options={"num_predict": 24})
          for _ in range(4))
    )
    await r.aclose()
    # The first request goes straight through; later ones must have waited.
    assert max(res.queue_wait_ms for res in results) > 0


async def test_no_healthy_hosts_raises():
    r = LLMRouter(base_urls=["http://127.0.0.1:9"], slots_per_host=2)
    await r.start()
    with pytest.raises(NoHealthyHostsError):
        await r.chat([{"role": "user", "content": "hi"}])
    await r.aclose()
