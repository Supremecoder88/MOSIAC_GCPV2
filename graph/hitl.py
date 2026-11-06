
import uuid
import json
from datetime import datetime
from typing import Any

import asyncpg

from memory.procedural_store import ProceduralStore
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)


CONFIDENCE_THRESHOLDS = {
    "broken_promises_agent": 0.60,
    "missing_results_agent": 0.65,
    "track_record_agent": 0.70,
    "pattern_finder_agent": 0.65,
    "side_effect_agent": 0.55,
    "timeline_agent": 0.60,
}


class HITLGate:
    """
    Routes signals to either direct save or human review queue
    based on the agent's confidence score.
    """

    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        self._procedural_store = ProceduralStore()

    async def _ensure_pool(self) -> None:
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            min_size=1,
            max_size=5,
        )

    async def _resolve_nct_id(
        self,
        conn: asyncpg.Connection,
        raw_nct_id: Any,
    ) -> str | None:
        """Return an existing study ID or None if unavailable."""

        if raw_nct_id is None:
            return None

        nct_id = str(raw_nct_id).strip().upper()

        if nct_id in {
            "",
            "N/A",
            "NA",
            "NONE",
            "NULL",
            "UNKNOWN",
            "NOT AVAILABLE",
        }:
            return None

        exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM studies
                WHERE nct_id = $1
            )
            """,
            nct_id,
        )

        if not exists:
            logger.warning(
                "Unknown NCT ID %s. Saving signal with NULL study reference.",
                nct_id,
            )
            return None

        return nct_id

    async def process_signal(
        self,
        signal: dict[str, Any],
    ) -> dict[str, Any]:

        await self._ensure_pool()

        agent_name = signal.get("agent", "unknown")
        confidence = float(signal.get("confidence", 0.0))
        threshold = CONFIDENCE_THRESHOLDS.get(agent_name, 0.65)

        if confidence >= threshold:
            signal_id = await self._save_signal(signal)

            logger.info(
                "Signal saved directly | agent=%s | confidence=%.2f",
                agent_name,
                confidence,
            )

            return {
                "action": "saved_directly",
                "signal_id": signal_id,
            }

        queue_id = await self._send_to_review_queue(signal)

        logger.info(
            "Signal sent to review | agent=%s | confidence=%.2f | threshold=%.2f",
            agent_name,
            confidence,
            threshold,
        )

        return {
            "action": "sent_to_review",
            "queue_id": queue_id,
        }

    async def _save_signal(
        self,
        signal: dict[str, Any],
    ) -> str:
        """Insert a high-confidence signal."""

        signal_id = str(uuid.uuid4())

        async with self._pool.acquire() as conn:
            nct_id = await self._resolve_nct_id(
                conn,
                signal.get("nct_id"),
            )

            await conn.execute(
                """
                INSERT INTO signals (
                    signal_id,
                    nct_id,
                    agent,
                    signal_type,
                    summary,
                    evidence,
                    confidence,
                    status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (signal_id) DO NOTHING
                """,
                signal_id,
                nct_id,
                signal.get("agent", ""),
                signal.get("signal_type", ""),
                signal.get("summary", ""),
                json.dumps(signal.get("evidence", [])),
                signal.get("confidence", 0.0),
                "approved",
            )

        return signal_id

    async def _send_to_review_queue(
        self,
        signal: dict[str, Any],
    ) -> str:
        """Insert a signal and its pending review entry."""

        queue_id = str(uuid.uuid4())
        signal_id = str(uuid.uuid4())

        async with self._pool.acquire() as conn:
            async with conn.transaction():

                nct_id = await self._resolve_nct_id(
                    conn,
                    signal.get("nct_id"),
                )

                await conn.execute(
                    """
                    INSERT INTO signals (
                        signal_id,
                        nct_id,
                        agent,
                        signal_type,
                        summary,
                        evidence,
                        confidence,
                        status
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    signal_id,
                    nct_id,
                    signal.get("agent", ""),
                    signal.get("signal_type", ""),
                    signal.get("summary", ""),
                    json.dumps(signal.get("evidence", [])),
                    signal.get("confidence", 0.0),
                    "pending",
                )

                await conn.execute(
                    """
                    INSERT INTO hitl_reviews (
                        review_id,
                        signal_id,
                        decision
                    )
                    VALUES ($1, $2, $3)
                    """,
                    queue_id,
                    signal_id,
                    "pending",
                )

        return queue_id

    async def process_human_decision(
        self,
        queue_id: str,
        decision: str,
        reviewer: str,
        rejection_reason: str = "",
        edit_summary: str = "",
    ) -> dict[str, Any]:

        await self._ensure_pool()

        if decision not in ("approve", "reject", "edit"):
            return {
                "success": False,
                "error": "Invalid decision",
            }

        async with self._pool.acquire() as conn:
            async with conn.transaction():

                row = await conn.fetchrow(
                    """
                    SELECT signal_id
                    FROM hitl_reviews
                    WHERE review_id = $1
                    FOR UPDATE
                    """,
                    queue_id,
                )

                if not row:
                    return {
                        "success": False,
                        "error": "Review not found",
                    }

                signal_id = row["signal_id"]

                new_status = (
                    "approved"
                    if decision in ("approve", "edit")
                    else "rejected"
                )

                await conn.execute(
                    """
                    UPDATE signals
                    SET status = $1
                    WHERE signal_id = $2
                    """,
                    new_status,
                    signal_id,
                )

                await conn.execute(
                    """
                    UPDATE hitl_reviews
                    SET decision = $1,
                        reviewer = $2,
                        rejection_reason = $3,
                        edit_summary = $4,
                        reviewed_at = NOW()
                    WHERE review_id = $5
                    """,
                    decision,
                    reviewer,
                    rejection_reason,
                    edit_summary,
                    queue_id,
                )

                if decision == "edit" and edit_summary:
                    await conn.execute(
                        """
                        UPDATE signals
                        SET summary = $1
                        WHERE signal_id = $2
                        """,
                        edit_summary,
                        signal_id,
                    )

        if decision == "reject" and rejection_reason:
            agent_row = await self._get_agent_for_signal(signal_id)

            if agent_row:
                await self._procedural_store.update_from_feedback(
                    agent_name=agent_row,
                    rejection_reason=rejection_reason,
                )

                logger.info(
                    "Procedural memory updated from rejection | agent=%s",
                    agent_row,
                )

        return {
            "success": True,
            "decision": decision,
            "signal_id": signal_id,
            "queue_id": queue_id,
        }

    async def _get_agent_for_signal(
        self,
        signal_id: str,
    ) -> str | None:

        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                """
                SELECT agent
                FROM signals
                WHERE signal_id = $1
                """,
                signal_id,
            )

    async def get_review_queue(self) -> list[dict]:
        """Returns all signals awaiting human review."""

        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    hr.review_id,
                    hr.signal_id,
                    s.agent,
                    s.signal_type,
                    s.summary,
                    s.confidence,
                    s.nct_id,
                    hr.decision,
                    hr.reviewed_at
                FROM hitl_reviews hr
                JOIN signals s ON hr.signal_id = s.signal_id
                WHERE hr.decision = 'pending'
                ORDER BY s.confidence ASC
                """
            )

        return [dict(row) for row in rows]

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
