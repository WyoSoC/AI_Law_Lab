"""Async Postgres access: connection pool + helpers.

A pool (not one shared connection) is the point here -- the original SQLite agent used a
single connection, which serializes hard and raises under the ~32 concurrent agents this
cluster can sustain.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pgvector import Vector
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .config import settings

_pool: AsyncConnectionPool | None = None


async def _configure(conn: AsyncConnection) -> None:
    conn.row_factory = dict_row
    # Register the pgvector type so `vector` columns round-trip as Python lists.
    from pgvector.psycopg import register_vector_async

    await register_vector_async(conn)


async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            settings.database_url,
            min_size=2,
            max_size=20,
            configure=_configure,
            open=False,
        )
        await _pool.open(wait=True, timeout=30)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def fetch_all(sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    pool = await get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(sql, params)
        return await cur.fetchall()


async def fetch_one(sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    pool = await get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(sql, params)
        return await cur.fetchone()


async def execute(sql: str, params: Iterable[Any] = ()) -> None:
    pool = await get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(sql, params)


def jsonb(value: Any) -> str:
    """psycopg needs dicts destined for jsonb columns serialized explicitly."""
    return json.dumps(value, default=str)


def vec(values: Iterable[float]) -> Vector:
    """Wrap an embedding for binding to a `vector` column.

    A bare Python list is adapted as double precision[], which has no `<=>` operator,
    so every similarity query would fail at runtime. Wrap at the DB boundary rather
    than making the rest of the codebase hold pgvector types.
    """
    return Vector(list(values))
