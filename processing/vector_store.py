import asyncio
import json
from typing import Any

import asyncpg

from config.logging_config import setup_logging
from config.settings import settings
from processing.embedder import EmbeddedChunk


logger = setup_logging(__name__)

POOL_MIN_SIZE = 2
POOL_MAX_SIZE = 10
TOP_K_DEFAULT = 5


class VectorStore:
    """Store embedded chunks in Cloud SQL and perform semantic search."""

    def __init__(self):
        self._pool: asyncpg.Pool | None = None

    async def __aenter__(self) -> "VectorStore":
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def init(self) -> None:
        """Create the connection pool and register the pgvector codec."""

        logger.info(
            f"Connecting to Cloud SQL | "
            f"host={settings.db_host} | "
            f"database={settings.db_name}"
        )

        self._pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=int(settings.db_port),
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            init=self._init_connection,
        )

        logger.info("Connection pool created successfully")

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Register the pgvector codec for Python/PostgreSQL conversion."""

        await conn.set_type_codec(
            "vector",
            encoder=lambda v: json.dumps(v),
            decoder=lambda v: json.loads(v),
            schema="public",
            format="text",
        )

    async def close(self) -> None:
        """Close the connection pool."""

        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Connection pool closed")

    async def save_embedded_chunks(
        self,
        chunks: list[EmbeddedChunk],
    ) -> int:
        """Save embedded chunks, skipping conflicts and logging failures."""

        if not chunks:
            logger.warning("save_embedded_chunks called with empty list")
            return 0

        saved_count = 0

        async with self._pool.acquire() as conn:
            for chunk in chunks:
                try:
                    await conn.execute(
                        """
                        INSERT INTO chunks
                            (nct_id, chunk_text, embedding, chunk_index, source)
                        VALUES
                            ($1, $2, $3, $4, $5)
                        ON CONFLICT DO NOTHING
                        """,
                        chunk.nct_id,
                        chunk.chunk_text,
                        chunk.embedding,
                        chunk.chunk_index,
                        chunk.source,
                    )

                    saved_count += 1

                except Exception as e:
                    logger.error(
                        f"Failed to save chunk | "
                        f"chunk_id={chunk.chunk_id} | "
                        f"error={e}"
                    )

        logger.info(
            f"Chunks saved | "
            f"saved={saved_count} | "
            f"total_input={len(chunks)} | "
            f"skipped={len(chunks) - saved_count}"
        )

        return saved_count

    async def save_study(self, study_data: dict[str, Any]) -> None:
        """Insert or update a study record."""

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO studies
                    (nct_id, title, sponsor, phase, status,
                     conditions, interventions, primary_outcome,
                     secondary_outcomes, start_date, completion_date,
                     results_posted, enrollment, gcs_path)
                VALUES
                    ($1, $2, $3, $4, $5,
                     $6, $7, $8,
                     $9, $10, $11,
                     $12, $13, $14)
                ON CONFLICT (nct_id) DO UPDATE SET
                    title              = EXCLUDED.title,
                    sponsor            = EXCLUDED.sponsor,
                    phase              = EXCLUDED.phase,
                    status             = EXCLUDED.status,
                    conditions         = EXCLUDED.conditions,
                    interventions      = EXCLUDED.interventions,
                    primary_outcome    = EXCLUDED.primary_outcome,
                    secondary_outcomes = EXCLUDED.secondary_outcomes,
                    start_date         = EXCLUDED.start_date,
                    completion_date    = EXCLUDED.completion_date,
                    results_posted     = EXCLUDED.results_posted,
                    enrollment         = EXCLUDED.enrollment,
                    gcs_path           = EXCLUDED.gcs_path
                """,
                study_data.get("nct_id"),
                study_data.get("title"),
                study_data.get("sponsor"),
                study_data.get("phase"),
                study_data.get("status"),
                study_data.get("conditions", []),
                study_data.get("interventions", []),
                study_data.get("primary_outcome"),
                study_data.get("secondary_outcomes", []),
                study_data.get("start_date"),
                study_data.get("completion_date"),
                study_data.get("results_posted"),
                study_data.get("enrollment"),
                study_data.get("gcs_path"),
            )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K_DEFAULT,
        source_filter: str | None = None,
        nct_id_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find the most relevant chunks using cosine distance."""

        conditions = []
        params: list[Any] = [query_embedding]
        param_count = 1

        if source_filter:
            param_count += 1
            conditions.append(f"source = ${param_count}")
            params.append(source_filter)

        if nct_id_filter:
            param_count += 1
            conditions.append(f"nct_id = ${param_count}")
            params.append(nct_id_filter)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        param_count += 1
        params.append(top_k)

        query = f"""
            SELECT
                nct_id,
                chunk_text,
                chunk_index,
                source,
                embedding <=> $1 AS distance
            FROM chunks
            {where_clause}
            ORDER BY distance ASC
            LIMIT ${param_count}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results = [dict(row) for row in rows]

        logger.info(
            f"Semantic search complete | "
            f"results_found={len(results)} | "
            f"top_k={top_k} | "
            f"source_filter={source_filter} | "
            f"nct_id_filter={nct_id_filter}"
        )

        return results

    async def get_chunk_count(self) -> int:
        """Return the total number of chunks in the database."""

        async with self._pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM chunks")

        logger.info(f"Total chunks in database: {result}")
        return result

    async def study_exists(self, nct_id: str) -> bool:
        """Check whether a study has chunks stored in the database."""

        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE nct_id = $1",
                nct_id,
            )

        return count > 0