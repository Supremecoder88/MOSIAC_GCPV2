
import asyncio
import asyncpg
from config.settings import settings


async def main():
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=5433,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )

    test_ids = [
        "NCT00006453",
        "N/A",
        "NCT99999999",
    ]

    for raw_id in test_ids:
        nct_id = None if raw_id in {
            "N/A", "NCT99999999"
        } else raw_id

        if nct_id:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM studies WHERE nct_id=$1)",
                nct_id,
            )
            if not exists:
                nct_id = None

        print(f"Input: {raw_id} -> Database value: {nct_id}")

    await conn.close()


asyncio.run(main())
