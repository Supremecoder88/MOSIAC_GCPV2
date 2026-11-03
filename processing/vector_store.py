from datetime import datetime, timezone
import asyncio
import json
from typing import Any

import asyncpg

from config.logging_config import setup_logging
from config.settings import settings
from processing.chunker import TextChunk
from processing.embedder import EmbeddedChunk


logger = setup_logging(__name__)

POOL_MIN_SIZE = 2
POOL_MAX_SIZE = 10
TOP_K_DEFAULT = 5


class VectorStore:
    """Store embedded study and paper chunks in Cloud SQL."""

    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        self._init_lock = asyncio.Lock()

    async def __aenter__(self) -> "VectorStore":
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # --------------------------------------------------
    # CONNECTION MANAGEMENT
    # --------------------------------------------------

    async def init(self) -> None:
        """Create the connection pool safely."""

        if self._pool is not None:
            return

        async with self._init_lock:
            if self._pool is not None:
                return

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

    async def _ensure_pool(self) -> None:
        """Initialize the pool if it does not exist."""

        if self._pool is None:
            await self.init()

    async def _init_connection(
        self,
        conn: asyncpg.Connection,
    ) -> None:
        """Register the pgvector codec."""

        await conn.set_type_codec(
            "vector",
            encoder=lambda value: json.dumps(value),
            decoder=lambda value: json.loads(value),
            schema="public",
            format="text",
        )

    async def close(self) -> None:
        """Close the connection pool."""

        if self._pool is not None:
            await self._pool.close()
            self._pool = None

            logger.info("Connection pool closed")

    # --------------------------------------------------
    # INCREMENTAL EMBEDDING
    # --------------------------------------------------

    async def get_chunks_needing_embedding(
        self,
        chunks: list[TextChunk],
    ) -> list[TextChunk]:
        """
        Return only new or modified chunks.

        Existing chunks with identical text are skipped.
        This method performs read-only database queries.
        """

        if not chunks:
            return []

        await self._ensure_pool()

        study_ids = list({
            chunk.nct_id
            for chunk in chunks
            if chunk.source == "study" and chunk.nct_id
        })

        paper_ids = list({
            chunk.pmid
            for chunk in chunks
            if chunk.source == "paper" and chunk.pmid
        })

        existing = {}

        async with self._pool.acquire() as conn:

            if study_ids:
                rows = await conn.fetch(
                    """
                    SELECT nct_id, chunk_index, chunk_text
                    FROM chunks
                    WHERE source = 'study'
                      AND nct_id = ANY($1::text[])
                    """,
                    study_ids,
                )

                for row in rows:
                    key = (
                        "study",
                        row["nct_id"],
                        row["chunk_index"],
                    )

                    existing[key] = row["chunk_text"]

            if paper_ids:
                rows = await conn.fetch(
                    """
                    SELECT pmid, chunk_index, chunk_text
                    FROM chunks
                    WHERE source = 'paper'
                      AND pmid = ANY($1::text[])
                    """,
                    paper_ids,
                )

                for row in rows:
                    key = (
                        "paper",
                        row["pmid"],
                        row["chunk_index"],
                    )

                    existing[key] = row["chunk_text"]

        chunks_to_embed = []

        for chunk in chunks:

            if chunk.source == "study":

                if not chunk.nct_id or chunk.pmid is not None:
                    raise ValueError(
                        "Invalid study chunk identifiers."
                    )

                document_id = chunk.nct_id

            elif chunk.source == "paper":

                if not chunk.pmid or chunk.nct_id is not None:
                    raise ValueError(
                        "Invalid paper chunk identifiers."
                    )

                document_id = chunk.pmid

            else:
                raise ValueError(
                    f"Unsupported chunk source: {chunk.source}"
                )

            key = (
                chunk.source,
                document_id,
                chunk.chunk_index,
            )

            existing_text = existing.get(key)

            if existing_text != chunk.chunk_text:
                chunks_to_embed.append(chunk)

        logger.info(
            f"Incremental embedding check | "
            f"total_chunks={len(chunks)} | "
            f"already_current="
            f"{len(chunks) - len(chunks_to_embed)} | "
            f"needing_embedding={len(chunks_to_embed)}"
        )

        return chunks_to_embed

    # --------------------------------------------------
    # SAVE EMBEDDED CHUNKS
    # --------------------------------------------------

    async def save_embedded_chunks(
        self,
        chunks: list[EmbeddedChunk],
    ) -> int:
        """
        Insert or update embedded study and paper chunks.

        Study chunks use (nct_id, chunk_index).
        Paper chunks use (pmid, chunk_index).

        PostgreSQL generates chunk_id using its UUID default.
        """

        if not chunks:
            logger.warning(
                "save_embedded_chunks called with empty list"
            )
            return 0

        await self._ensure_pool()

        saved_count = 0
        failed_count = 0

        async with self._pool.acquire() as conn:

            for chunk in chunks:

                try:

                    if chunk.source == "study":

                        if not chunk.nct_id or chunk.pmid is not None:
                            raise ValueError(
                                "Study chunks require nct_id "
                                "and must not contain pmid."
                            )

                        await conn.execute(
                            """
                            INSERT INTO chunks
                                (
                                    nct_id,
                                    pmid,
                                    chunk_text,
                                    embedding,
                                    chunk_index,
                                    source
                                )
                            VALUES ($1, NULL, $2, $3, $4, 'study')

                            ON CONFLICT (nct_id, chunk_index)
                                WHERE source = 'study'

                            DO UPDATE SET
                                chunk_text = EXCLUDED.chunk_text,
                                embedding = EXCLUDED.embedding
                            """,
                            chunk.nct_id,
                            chunk.chunk_text,
                            chunk.embedding,
                            chunk.chunk_index,
                        )

                    elif chunk.source == "paper":

                        if not chunk.pmid or chunk.nct_id is not None:
                            raise ValueError(
                                "Paper chunks require pmid "
                                "and must not contain nct_id."
                            )

                        await conn.execute(
                            """
                            INSERT INTO chunks
                                (
                                    nct_id,
                                    pmid,
                                    chunk_text,
                                    embedding,
                                    chunk_index,
                                    source
                                )
                            VALUES (NULL, $1, $2, $3, $4, 'paper')

                            ON CONFLICT (pmid, chunk_index)
                                WHERE source = 'paper'

                            DO UPDATE SET
                                chunk_text = EXCLUDED.chunk_text,
                                embedding = EXCLUDED.embedding
                            """,
                            chunk.pmid,
                            chunk.chunk_text,
                            chunk.embedding,
                            chunk.chunk_index,
                        )

                    else:
                        raise ValueError(
                            f"Unsupported chunk source: {chunk.source}"
                        )

                    saved_count += 1

                except Exception as e:

                    failed_count += 1

                    logger.error(
                        f"Failed to save chunk | "
                        f"chunk_id={chunk.chunk_id} | "
                        f"source={chunk.source} | "
                        f"error={e}"
                    )

        logger.info(
            f"Chunk storage complete | "
            f"processed={len(chunks)} | "
            f"inserted_or_updated={saved_count} | "
            f"failed={failed_count}"
        )

        return saved_count

    # --------------------------------------------------
    # PAPER METADATA
    # --------------------------------------------------

    async def save_paper(
        self,
        paper_data: dict[str, Any],
    ) -> int:
        """
        Insert or update paper metadata and link it
        to clinical studies already present in the database.
        """

        pmid = paper_data.get("pmid")
        parsed_at = paper_data.get("parsed_at")

        if isinstance(parsed_at, str):
            parsed_at = datetime.fromisoformat(
                parsed_at.replace("Z", "+00:00")
            )

            if parsed_at.tzinfo is None:
                parsed_at = parsed_at.replace(tzinfo=timezone.utc)

        if not pmid:
            raise ValueError(
                "Paper data must contain a PMID."
            )

        await self._ensure_pool()

        nct_ids = list(
            dict.fromkeys(
                paper_data.get("nct_ids_referenced") or []
            )
        )

        async with self._pool.acquire() as conn:

            async with conn.transaction():

                await conn.execute(
                    """
                    INSERT INTO papers
                        (
                            pmid,
                            title,
                            abstract,
                            journal,
                            pub_date,
                            authors,
                            nct_ids_referenced,
                            source,
                            word_count,
                            parsed_at
                        )
                    VALUES
                        ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)

                    ON CONFLICT (pmid) DO UPDATE SET
                        title = EXCLUDED.title,
                        abstract = EXCLUDED.abstract,
                        journal = EXCLUDED.journal,
                        pub_date = EXCLUDED.pub_date,
                        authors = EXCLUDED.authors,
                        nct_ids_referenced =
                            EXCLUDED.nct_ids_referenced,
                        source = EXCLUDED.source,
                        word_count = EXCLUDED.word_count,
                        parsed_at = EXCLUDED.parsed_at
                    """,
                    pmid,
                    paper_data.get("title"),
                    paper_data.get("abstract"),
                    paper_data.get("journal"),
                    paper_data.get("pub_date"),
                    paper_data.get("authors") or [],
                    nct_ids,
                    paper_data.get("source") or "pubmed",
                    paper_data.get("word_count"),
                    parsed_at,
                )

                linked_count = 0

                if nct_ids:

                    status = await conn.execute(
                        """
                        INSERT INTO paper_studies (pmid, nct_id)

                        SELECT $1, studies.nct_id
                        FROM studies
                        WHERE studies.nct_id = ANY($2::text[])

                        ON CONFLICT (pmid, nct_id) DO NOTHING
                        """,
                        pmid,
                        nct_ids,
                    )

                    linked_count = int(status.split()[-1])

        logger.info(
            f"Paper saved | pmid={pmid} | "
            f"referenced_trials={len(nct_ids)} | "
            f"new_links={linked_count}"
        )

        return linked_count

    # --------------------------------------------------
    # STUDY METADATA
    # --------------------------------------------------

    
    async def save_study(
        self,
        study_data: dict[str, Any],
    ) -> None:
        """Insert/update a study and synchronize affected sponsor profiles."""

        await self._ensure_pool()

        nct_id = study_data.get("nct_id")
        sponsor = study_data.get("sponsor")

        if not nct_id:
            raise ValueError("Cannot save a study without an NCT ID.")

        new_sponsor = (
            sponsor.strip()
            if isinstance(sponsor, str) and sponsor.strip()
            else None
        )

        async with self._pool.acquire() as conn:
            async with conn.transaction():

                # Find the previous sponsor before updating the study.
                old_sponsor = await conn.fetchval(
                    """
                    SELECT NULLIF(TRIM(sponsor), '')
                    FROM studies
                    WHERE nct_id = $1
                    """,
                    nct_id,
                )

                # Insert or update the study.
                await conn.execute(
                    """
                    INSERT INTO studies (
                        nct_id,
                        title,
                        sponsor,
                        phase,
                        status,
                        conditions,
                        interventions,
                        primary_outcome,
                        secondary_outcomes,
                        start_date,
                        completion_date,
                        results_posted,
                        enrollment,
                        gcs_path
                    )
                    VALUES (
                        $1, $2, $3, $4, $5, $6, $7,
                        $8, $9, $10, $11, $12, $13, $14
                    )
                    ON CONFLICT (nct_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        sponsor = EXCLUDED.sponsor,
                        phase = EXCLUDED.phase,
                        status = EXCLUDED.status,
                        conditions = EXCLUDED.conditions,
                        interventions = EXCLUDED.interventions,
                        primary_outcome = EXCLUDED.primary_outcome,
                        secondary_outcomes = EXCLUDED.secondary_outcomes,
                        start_date = EXCLUDED.start_date,
                        completion_date = EXCLUDED.completion_date,
                        results_posted = EXCLUDED.results_posted,
                        enrollment = EXCLUDED.enrollment,
                        gcs_path = EXCLUDED.gcs_path
                    """,
                    nct_id,
                    study_data.get("title"),
                    sponsor,
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

                # Only recalculate profiles affected by this study.
                affected_sponsors = {
                    name for name in (old_sponsor, new_sponsor)
                    if name
                }

                for sponsor_name in affected_sponsors:

                    stats = await conn.fetchrow(
                        """
                        SELECT
                            COUNT(DISTINCT nct_id) AS total_studies,
                            COUNT(DISTINCT nct_id) FILTER (
                                WHERE results_posted IS TRUE
                            ) AS results_posted,
                            COUNT(DISTINCT nct_id) FILTER (
                                WHERE status = 'COMPLETED'
                                  AND results_posted IS FALSE
                            ) AS results_missing
                        FROM studies
                        WHERE TRIM(sponsor) = $1
                        """,
                        sponsor_name,
                    )

                    total = int(stats["total_studies"] or 0)
                    posted = int(stats["results_posted"] or 0)
                    missing = int(stats["results_missing"] or 0)

                    if total == 0:
                        await conn.execute(
                            """
                            DELETE FROM sponsor_profiles
                            WHERE sponsor = $1
                            """,
                            sponsor_name,
                        )
                        continue

                    await conn.execute(
                        """
                        INSERT INTO sponsor_profiles (
                            sponsor,
                            credibility_score,
                            total_studies,
                            results_posted,
                            results_missing,
                            broken_promises,
                            avg_delay_days,
                            last_updated
                        )
                        VALUES (
                            $1,
                            $2,
                            $3,
                            $4,
                            $5,
                            0,
                            0,
                            NOW()
                        )
                        ON CONFLICT (sponsor) DO UPDATE SET
                            total_studies = EXCLUDED.total_studies,
                            results_posted = EXCLUDED.results_posted,
                            results_missing = EXCLUDED.results_missing,
                            credibility_score = GREATEST(
                                0.0,
                                LEAST(
                                    1.0,
                                    (
                                        EXCLUDED.results_posted::DOUBLE PRECISION
                                        / NULLIF(EXCLUDED.total_studies, 0)
                                    ) * 0.7
                                    - (
                                        sponsor_profiles.broken_promises
                                        * 0.1
                                    )
                                )
                            ),
                            last_updated = NOW()
                        """,
                        sponsor_name,
                        (posted / total) * 0.7,
                        total,
                        posted,
                        missing,
                    )

        logger.info(
            f"Study saved and sponsor profiles synchronized "
            f"| nct_id={nct_id} "
            f"| previous_sponsor={old_sponsor} "
            f"| current_sponsor={new_sponsor}"
        )

    # --------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K_DEFAULT,
        source_filter: str | None = None,
        nct_id_filter: str | None = None,
        max_distance: float | None = None,
        pmid_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find relevant chunks using cosine distance.

        Set max_distance=None to disable relevance filtering.
        """

        await self._ensure_pool()

        if top_k < 1:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if max_distance is not None and not 0 <= max_distance <= 2:
            raise ValueError(
                "max_distance must be between 0 and 2."
            )

        conditions = []
        params: list[Any] = [query_embedding]
        param_count = 1

        if source_filter:
            param_count += 1
            conditions.append(
                f"source = ${param_count}"
            )
            params.append(source_filter)

        if nct_id_filter:
            param_count += 1
            conditions.append(
                f"nct_id = ${param_count}"
            )
            params.append(nct_id_filter)

        if pmid_filter:
            param_count += 1
            conditions.append(
                f"pmid = ${param_count}"
            )
            params.append(pmid_filter)

        if max_distance is not None:
            param_count += 1
            conditions.append(
                f"(embedding <=> $1) <= ${param_count}"
            )
            params.append(max_distance)

        where_clause = ""

        if conditions:
            where_clause = (
                "WHERE " + " AND ".join(conditions)
            )

        param_count += 1
        params.append(top_k)

        query = f"""
            SELECT
                nct_id,
                pmid,
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

        results = [
            dict(row)
            for row in rows
        ]

        logger.info(
            f"Semantic search complete | "
            f"results_found={len(results)} | "
            f"top_k={top_k} | "
            f"max_distance={max_distance} | "
            f"source_filter={source_filter} | "
            f"nct_id_filter={nct_id_filter} | "
            f"pmid_filter={pmid_filter}"
        )

        return results

    # --------------------------------------------------
    # STUDY CHUNK RETRIEVAL
    # --------------------------------------------------

    async def get_chunks_for_study(
        self,
        nct_id: str,
    ) -> list[dict[str, Any]]:
        """Retrieve all chunks belonging to a clinical trial."""

        await self._ensure_pool()

        async with self._pool.acquire() as conn:

            rows = await conn.fetch(
                """
                SELECT
                    nct_id,
                    pmid,
                    chunk_text,
                    chunk_index,
                    source
                FROM chunks
                WHERE nct_id = $1
                  AND source = 'study'
                ORDER BY chunk_index ASC
                """,
                nct_id,
            )

        results = [
            dict(row)
            for row in rows
        ]

        logger.info(
            f"Study chunks retrieved | "
            f"nct_id={nct_id} | "
            f"count={len(results)}"
        )

        return results

    # --------------------------------------------------
    # PAPER CHUNK RETRIEVAL
    # --------------------------------------------------

    async def get_chunks_for_paper(
        self,
        pmid: str,
    ) -> list[dict[str, Any]]:
        """Retrieve all chunks belonging to a PubMed paper."""

        await self._ensure_pool()

        async with self._pool.acquire() as conn:

            rows = await conn.fetch(
                """
                SELECT
                    nct_id,
                    pmid,
                    chunk_text,
                    chunk_index,
                    source
                FROM chunks
                WHERE pmid = $1
                  AND source = 'paper'
                ORDER BY chunk_index ASC
                """,
                pmid,
            )

        results = [
            dict(row)
            for row in rows
        ]

        logger.info(
            f"Paper chunks retrieved | "
            f"pmid={pmid} | "
            f"count={len(results)}"
        )

        return results

    # --------------------------------------------------
    # DATABASE STATISTICS
    # --------------------------------------------------

    async def get_chunk_count(self) -> int:
        """Return the total number of stored chunks."""

        await self._ensure_pool()

        async with self._pool.acquire() as conn:

            result = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks"
            )

        logger.info(
            f"Total chunks in database: {result}"
        )

        return result or 0

    async def study_exists(
        self,
        nct_id: str,
    ) -> bool:
        """Check whether a study has stored study chunks."""

        await self._ensure_pool()

        async with self._pool.acquire() as conn:

            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE nct_id = $1
                  AND source = 'study'
                """,
                nct_id,
            )

        return count > 0
