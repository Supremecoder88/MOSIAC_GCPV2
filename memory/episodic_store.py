import json
import uuid
from datetime import datetime, timezone

import asyncpg
from openai import AsyncOpenAI

from config.settings import settings
from config.logging_config import setup_logging


logger = setup_logging(__name__)


class EpisodicStore:
    """Store and search agent reasoning sessions using embeddings."""

    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self._embedding_model = settings.openai_embedding_model

        logger.info("EpisodicStore initialised")

    async def _ensure_pool(self) -> None:
        """Create the database connection pool when needed."""
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=int(settings.db_port),
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            min_size=1,
            max_size=5,
            init=self._init_connection,
        )

        logger.info("EpisodicStore pool created")

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        """Register the pgvector codec."""
        await conn.set_type_codec(
            "vector",
            encoder=lambda value: json.dumps(value),
            decoder=lambda value: json.loads(value),
            schema="public",
            format="text",
        )

    async def _embed(self, text: str) -> list[float]:
        """Convert text into a vector embedding."""
        response = await self._openai.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        return response.data[0].embedding

    async def save_episode(
        self,
        agent_name: str,
        content: str,
        nct_id: str | None = None,
        outcome: str | None = None,
    ) -> str:
        """Save an agent session and return its ID."""
        await self._ensure_pool()

        episode_id = str(uuid.uuid4())
        embedding = await self._embed(content)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO episodes (
                    episode_id, agent_name, nct_id, content,
                    outcome, embedding, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                episode_id,
                agent_name,
                nct_id,
                content,
                outcome,
                embedding,
                datetime.now(timezone.utc),
            )

        logger.info(
            f"Episode saved | agent={agent_name} | "
            f"nct_id={nct_id} | outcome={outcome} | "
            f"episode_id={episode_id}"
        )

        return episode_id

    async def search_episodes(
        self,
        query: str,
        agent_name: str | None = None,
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> list[dict]:
        """Find past episodes by semantic similarity."""
        await self._ensure_pool()
        query_embedding = await self._embed(query)

        sql = """
            SELECT
                episode_id,
                agent_name,
                nct_id,
                content,
                outcome,
                created_at,
                1 - (embedding <=> $1) AS similarity
            FROM episodes
            WHERE 1 - (embedding <=> $1) >= $2
        """

        params = [query_embedding, min_similarity]
        param_idx = 3

        if agent_name:
            sql += f" AND agent_name = ${param_idx}"
            params.append(agent_name)
            param_idx += 1

        sql += f"""
            ORDER BY embedding <=> $1
            LIMIT ${param_idx}
        """
        params.append(top_k)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        episodes = [
            {
                "episode_id": row["episode_id"],
                "agent_name": row["agent_name"],
                "nct_id": row["nct_id"],
                "content": row["content"],
                "outcome": row["outcome"],
                "similarity": round(float(row["similarity"]), 3),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

        logger.info(
            f"Episode search complete | query='{query[:50]}' | "
            f"agent_filter={agent_name} | results_found={len(episodes)}"
        )

        return episodes

    async def get_recent_episodes(
        self,
        agent_name: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Return the most recent episodes."""
        await self._ensure_pool()

        if agent_name:
            sql = """
                SELECT episode_id, agent_name, nct_id,
                       content, outcome, created_at
                FROM episodes
                WHERE agent_name = $1
                ORDER BY created_at DESC
                LIMIT $2
            """
            params = [agent_name, limit]
        else:
            sql = """
                SELECT episode_id, agent_name, nct_id,
                       content, outcome, created_at
                FROM episodes
                ORDER BY created_at DESC
                LIMIT $1
            """
            params = [limit]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [
            {
                "episode_id": row["episode_id"],
                "agent_name": row["agent_name"],
                "nct_id": row["nct_id"],
                "content": row["content"],
                "outcome": row["outcome"],
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    async def count_episodes(
        self,
        agent_name: str | None = None,
    ) -> int:
        """Count stored episodes."""
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            if agent_name:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM episodes WHERE agent_name = $1",
                    agent_name,
                )
            else:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM episodes"
                )

        return count or 0

    async def close(self) -> None:
        """Close the database connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("EpisodicStore pool closed")