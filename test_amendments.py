
import asyncio
import json

from tools.clinical_tools import get_study_amendments


async def main():
    result = await get_study_amendments.ainvoke({
        "nct_id": "NCT00567554"
    })

    data = json.loads(result)

    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
