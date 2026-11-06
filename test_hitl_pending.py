
import asyncio
from graph.hitl import HITLGate


async def main():
    gate = HITLGate()

    signal = {
        "agent": "track_record_agent",
        "signal_type": "low_credibility",
        "nct_id": "N/A",
        "summary": "HITL pending persistence test",
        "evidence": ["Testing NULL NCT handling"],
        "confidence": 0.50,
    }

    try:
        result = await gate.process_signal(signal)
        print("HITL result:", result)

        async with gate._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT s.nct_id, s.status, hr.decision
                FROM hitl_reviews hr
                JOIN signals s ON s.signal_id = hr.signal_id
                WHERE hr.review_id = $1
                """,
                result["queue_id"],
            )

            print("Database record:", dict(row) if row else None)

    finally:
        await gate.close()


asyncio.run(main())
