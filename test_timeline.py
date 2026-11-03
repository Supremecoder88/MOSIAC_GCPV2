
import asyncio
import json

from agents.timeline_agent import timeline_node


async def main():
    state = {
        "task": (
            "Identify clinical trials that are more than "
            "180 days past their actual completion dates "
            "without a verified amendment explaining the delay."
        ),
        "nct_ids": ["NCT00567554"],
        "signals": [],
        "agents_activated": [],
        "error_log": [],
    }

    result = await timeline_node(state)

    print("\n========== TIMELINE AGENT ==========")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
