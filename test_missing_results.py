
import asyncio
import json

from agents.missing_results_agent import missing_results_node


async def main():
    state = {
        "task": (
            "Find completed clinical trials that have not "
            "posted their results. Verify each finding "
            "using the live API."
        ),
        "nct_ids": ["NCT00567554"],
        "signals": [],
        "agents_activated": [],
        "error_log": [],
    }

    result = await missing_results_node(state)

    print("\n========== MISSING RESULTS AGENT ==========")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
