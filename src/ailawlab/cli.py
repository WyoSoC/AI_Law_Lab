"""Command line entry points for AI Law Lab."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path


async def _cmd_status() -> int:
    from .router import get_router

    router = await get_router()
    await router.health_check()
    stats = router.stats()
    print(f"slots per host: {stats['slots_per_host']}   total capacity: {stats['total_slots']}")
    for h in stats["hosts"]:
        state = "healthy" if h["healthy"] else "UNREACHABLE"
        print(f"  {h['name']:<16} {state:<12} {h['free_slots']}/{stats['slots_per_host']} free")
    await router.aclose()
    return 0 if any(h["healthy"] for h in stats["hosts"]) else 1


async def _cmd_ingest(paths: list[str], corpus: str) -> int:
    from .db import close_pool
    from .rag import Corpus
    from .router import get_router

    router = await get_router()
    store = Corpus(router, name=corpus)
    added = skipped = 0
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            print(f"  missing: {p}", file=sys.stderr)
            continue
        if p.suffix.lower() == ".pdf":
            doc_id = await store.add_pdf(p)
        else:
            doc_id = await store.add_document(title=p.stem, text=p.read_text(errors="replace"),
                                              source_uri=str(p), doc_type="text")
        if doc_id is None:
            skipped += 1
            print(f"  skipped (already present or no text): {p.name}")
        else:
            added += 1
            print(f"  ingested: {p.name}")
    print(f"\n{added} added, {skipped} skipped. {await store.stats()}")
    await close_pool()
    return 0


async def _cmd_run(experiment_id: str, inputs: str) -> int:
    from . import experiments
    from .db import close_pool

    result = await experiments.execute_run(
        (await experiments.create_run(experiment_id, json.loads(inputs)))["id"].__str__()
    )
    print(json.dumps(result, indent=2, default=str))
    await close_pool()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="ailawlab", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve", help="run the web interface")
    sub.add_parser("status", help="check the inference cluster")

    ing = sub.add_parser("ingest", help="add documents to a corpus")
    ing.add_argument("paths", nargs="+")
    ing.add_argument("--corpus", default="default")

    run = sub.add_parser("run", help="execute one run of an experiment")
    run.add_argument("experiment_id")
    run.add_argument("--inputs", default="{}")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "serve":
        from .web.app import main as serve

        serve()
        return 0
    if args.cmd == "status":
        return asyncio.run(_cmd_status())
    if args.cmd == "ingest":
        return asyncio.run(_cmd_ingest(args.paths, args.corpus))
    if args.cmd == "run":
        return asyncio.run(_cmd_run(args.experiment_id, args.inputs))
    return 1


if __name__ == "__main__":
    sys.exit(main())
