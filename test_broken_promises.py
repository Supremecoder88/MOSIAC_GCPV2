
import asyncio
import json

from agents.broken_promises_agent import broken_promises_node


async def main():
    state = {
        "task": (
            "Investigate NCT00567554 for changes "
            "to its registered primary and secondary outcomes. "
            "Do not report a violation without historical evidence."
        ),
        "nct_ids": ["NCT00567554"],
        "signals": [],
        "agents_activated": [],
        "error_log": [],
    }

    result = await broken_promises_node(state)

    print("\n========== AGENT RESULT ==========")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
