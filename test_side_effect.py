
import asyncio
import json

from agents.side_effect_agent import side_effect_node


async def main():
    state = {
        "task": (
            "Compare official safety reporting with published "
            "research papers and identify verified discrepancies."
        ),
        "nct_ids": ["NCT00567554"],
        "signals": [],
        "agents_activated": [],
        "error_log": [],
    }

    result = await side_effect_node(state)

    print("\n========== SIDE EFFECT AGENT ==========")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
