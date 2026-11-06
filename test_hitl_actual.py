
import asyncio
from graph.hitl import HITLGate


async def main():
    gate = HITLGate()

    test_signals = [
        {
            "agent": "track_record_agent",
            "signal_type": "low_credibility",
            "nct_id": "NCT00006453",
            "summary": "Test valid NCT ID",
            "evidence": ["Persistence test"],
            "confidence": 0.95,
        },
        {
            "agent": "track_record_agent",
            "signal_type": "low_credibility",
            "nct_id": "N/A",
            "summary": "Test missing NCT ID",
            "evidence": ["Persistence test"],
            "confidence": 0.95,
        },
        {
            "agent": "track_record_agent",
            "signal_type": "low_credibility",
            "nct_id": "NCT99999999",
            "summary": "Test unknown NCT ID",
            "evidence": ["Persistence test"],
            "confidence": 0.95,
        },
    ]

    try:
        for signal in test_signals:
            result = await gate.process_signal(signal)
            print(result)

        print("\nAll three HITL insertion calls completed.")

    finally:
        await gate.close()


asyncio.run(main())
