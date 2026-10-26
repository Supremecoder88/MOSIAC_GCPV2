from typing import Any

import asyncpg

from config.settings import settings
from config.logging_config import setup_logging


logger = setup_logging(__name__)


class SemanticStore:
    """Manage sponsor profiles and accumulated knowledge."""

    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        logger.info("SemanticStore initialised")

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
        )

        logger.info("SemanticStore pool created")

    async def get_sponsor_profile(
        self,
        sponsor: str,
    ) -> dict[str, Any] | None:
        """Retrieve a sponsor's profile."""
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                FROM sponsor_profiles
                WHERE sponsor = $1
                """,
                sponsor,
            )

        if row is None:
            logger.info(
                f"No profile found for sponsor | sponsor={sponsor}"
            )
            return None

        return {
            "sponsor": row["sponsor"],
            "credibility_score": float(
                row["credibility_score"] or 0.0
            ),
            "total_studies": int(row["total_studies"] or 0),
            "results_posted": int(row["results_posted"] or 0),
            "results_missing": int(row["results_missing"] or 0),
            "broken_promises": int(row["broken_promises"] or 0),
            "avg_delay_days": float(row["avg_delay_days"] or 0.0),
            "last_updated": str(row["last_updated"]),
        }

    async def update_sponsor_knowledge(
        self,
        sponsor: str,
        results_posted: bool = False,
        had_broken_promise: bool = False,
        delay_days: int = 0,
    ) -> None:
        """Update or create a sponsor profile after study analysis."""
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            # Insert a new profile or update the existing one.
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
                VALUES ($1, 0.5, 1, $2, $3, $4, $5, NOW())
                ON CONFLICT (sponsor) DO UPDATE SET
                    total_studies = sponsor_profiles.total_studies + 1,
                    results_posted = sponsor_profiles.results_posted + $2,
                    results_missing = sponsor_profiles.results_missing + $3,
                    broken_promises = sponsor_profiles.broken_promises + $4,
                    avg_delay_days = (
                        (sponsor_profiles.avg_delay_days *
                         sponsor_profiles.total_studies) + $5
                    ) / (sponsor_profiles.total_studies + 1),
                    last_updated = NOW()
                """,
                sponsor,
                int(results_posted),
                int(not results_posted),
                int(had_broken_promise),
                float(delay_days),
            )

            # Recalculate credibility using the updated counts.
            await conn.execute(
                """
                UPDATE sponsor_profiles
                SET credibility_score = GREATEST(
                    0.0,
                    LEAST(
                        1.0,
                        (
                            CASE
                                WHEN total_studies = 0 THEN 0.5
                                ELSE (
                                    results_posted::float / total_studies
                                ) * 0.7
                            END
                        ) - (broken_promises * 0.1)
                    )
                )
                WHERE sponsor = $1
                """,
                sponsor,
            )

        logger.info(
            f"Sponsor knowledge updated | "
            f"sponsor={sponsor} | "
            f"results_posted={results_posted} | "
            f"broken_promise={had_broken_promise} | "
            f"delay_days={delay_days}"
        )

    async def get_low_credibility_sponsors(
        self,
        threshold: float = 0.6,
        min_studies: int = 3,
    ) -> list[dict]:
        """Return sufficiently studied sponsors below the threshold."""
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                FROM sponsor_profiles
                WHERE credibility_score < $1
                  AND total_studies >= $2
                ORDER BY credibility_score ASC
                """,
                threshold,
                min_studies,
            )

        sponsors = [
            {
                "sponsor": row["sponsor"],
                "credibility_score": float(
                    row["credibility_score"] or 0.0
                ),
                "total_studies": int(row["total_studies"] or 0),
                "results_posted": int(row["results_posted"] or 0),
                "results_missing": int(row["results_missing"] or 0),
                "broken_promises": int(row["broken_promises"] or 0),
                "avg_delay_days": float(row["avg_delay_days"] or 0.0),
                "last_updated": str(row["last_updated"]),
            }
            for row in rows
        ]

        logger.info(
            f"Low credibility sponsors found | "
            f"count={len(sponsors)} | "
            f"threshold={threshold} | "
            f"min_studies={min_studies}"
        )

        return sponsors

    async def get_all_sponsor_profiles(
        self,
        limit: int = 50,
    ) -> list[dict]:
        """Return sponsor profiles ordered by credibility."""
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                FROM sponsor_profiles
                ORDER BY credibility_score ASC
                LIMIT $1
                """,
                limit,
            )

        return [
            {
                "sponsor": row["sponsor"],
                "credibility_score": float(
                    row["credibility_score"] or 0.0
                ),
                "total_studies": int(row["total_studies"] or 0),
                "results_posted": int(row["results_posted"] or 0),
                "results_missing": int(row["results_missing"] or 0),
                "broken_promises": int(row["broken_promises"] or 0),
                "avg_delay_days": float(row["avg_delay_days"] or 0.0),
                "last_updated": str(row["last_updated"]),
            }
            for row in rows
        ]

    async def sponsor_exists(self, sponsor: str) -> bool:
        """Check whether a sponsor profile exists."""
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM sponsor_profiles
                WHERE sponsor = $1
                """,
                sponsor,
            )

        return (count or 0) > 0

    async def close(self) -> None:
        """Close the database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("SemanticStore pool closed")