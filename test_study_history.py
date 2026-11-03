import asyncio
import json

from ingestion.clinical_trials_client import ClinicalTrialsClient


async def main():

    nct_id = "NCT00567554"

    async with ClinicalTrialsClient() as client:

        history = await client.fetch_study_history(nct_id)

        print(json.dumps(history, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())