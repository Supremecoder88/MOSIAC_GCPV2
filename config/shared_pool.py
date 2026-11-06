import json
import asyncpg

_shared_pool: asyncpg.Pool | None = None


async def register_pgvector_codec(conn: asyncpg.Connection) -> None:
    """Register pgvector conversion for every pooled connection."""
    await conn.set_type_codec(
        "vector",
        encoder=lambda value: json.dumps(value),
        decoder=lambda value: json.loads(value),
        schema="public",
        format="text",
    )


def set_shared_pool(pool: asyncpg.Pool | None) -> None:
    global _shared_pool
    _shared_pool = pool


def get_shared_pool() -> asyncpg.Pool | None:
    return _shared_pool