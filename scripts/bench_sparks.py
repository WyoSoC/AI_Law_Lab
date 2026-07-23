#!/usr/bin/env python3
"""Measure the concurrency ceiling of the Spark cluster.

The number that matters is not how many requests complete without error -- Ollama will
admit far more than it can run -- but where aggregate throughput stops rising. Past that
point extra concurrency only adds queue latency, and any experiment timing taken there
measures the queue rather than the model.

Usage:
    python scripts/bench_sparks.py                      # both hosts, default levels
    python scripts/bench_sparks.py --host test-spark3 --levels 1,4,16,32,50
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ailawlab.config import settings
from ailawlab.router import LLMRouter


async def bench_host(base_url: str, levels: list[int], num_predict: int) -> None:
    # slots_per_host is set to the level under test so the router never throttles
    # below what we are trying to measure.
    name = base_url.split("//", 1)[-1].split(":", 1)[0]
    print(f"\n--- {name} ---")
    print(f"{'conc':>5} {'wall_s':>8} {'ok':>4} {'err':>4} {'tok/s_agg':>10} "
          f"{'p50_lat':>9} {'p95_lat':>9}")

    for level in levels:
        r = LLMRouter(base_urls=[base_url], slots_per_host=level)
        await r.start()

        # Warm the model so a cold load does not land inside the measurement.
        await r.chat([{"role": "user", "content": "hi"}], think=False,
                     options={"num_predict": 4})

        # `router=r` binds this iteration's router; a bare closure over `r` would
        # capture whatever the loop variable points at by the time the task runs.
        async def one(i: int, router: LLMRouter = r):
            t0 = time.perf_counter()
            try:
                res = await router.chat(
                    [{"role": "user", "content": f"Count from {i} to {i + 5}, comma separated."}],
                    think=False, temperature=0.0, options={"num_predict": num_predict},
                )
                return time.perf_counter() - t0, res.output_tokens, None
            except Exception as e:  # noqa: BLE001 - an error is a data point, not a crash
                return time.perf_counter() - t0, 0, str(e)[:60]

        t0 = time.perf_counter()
        results = await asyncio.gather(*(one(i) for i in range(level)))
        wall = time.perf_counter() - t0
        await r.aclose()

        lats = sorted(x[0] for x in results)
        toks = sum(x[1] for x in results)
        errs = [x[2] for x in results if x[2]]
        p50 = statistics.median(lats)
        p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))]
        print(f"{level:>5} {wall:>8.2f} {len(results) - len(errs):>4} {len(errs):>4} "
              f"{toks / wall:>10.1f} {p50:>9.2f} {p95:>9.2f}")
        if errs:
            print(f"      first error: {errs[0]}")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", action="append", help="hostname (repeatable); default: all configured")
    ap.add_argument("--levels", default="1,4,16,32,50")
    ap.add_argument("--num-predict", type=int, default=48)
    args = ap.parse_args()

    urls = ([f"http://{h}:{settings.ollama_port}" for h in args.host]
            if args.host else settings.base_urls())
    levels = [int(x) for x in args.levels.split(",")]

    for url in urls:
        await bench_host(url, levels, args.num_predict)

    print("\nRead the tok/s_agg column: the level where it stops rising is the real "
          "concurrency ceiling.\nSet AILAWLAB_SLOTS_PER_HOST to that value.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
