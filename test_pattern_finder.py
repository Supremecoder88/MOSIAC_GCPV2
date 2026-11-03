
import asyncio
import json

from agents.pattern_finder_agent import pattern_finder_node


async def main():
    state = {
        "task": (
            "Identify recurring patterns across clinical trials "
            "involving similar drugs, sponsors, and outcomes. "
            "Use evidence from at least three distinct studies."
        ),
        "nct_ids": [],
        "signals": [],
        "agents_activated": [],
        "error_log": [],
    }

    result = await pattern_finder_node(state)

    print("\n========== PATTERN FINDER AGENT ==========")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
