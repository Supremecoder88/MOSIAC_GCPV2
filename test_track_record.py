
import asyncio
import json

from agents.track_record_agent import track_record_node


async def main():
    state = {
        "task": (
            "Evaluate sponsor credibility using historical "
            "clinical trial behavior and sponsor profiles."
        ),
        "nct_ids": [],
        "signals": [],
        "agents_activated": [],
        "error_log": [],
    }

    result = await track_record_node(state)

    print("\n========== TRACK RECORD AGENT ==========")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
