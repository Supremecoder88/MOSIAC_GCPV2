
import asyncio
import asyncpg

from config.settings import settings


async def main():
    conn = await asyncpg.connect(
        host=settings.db_host,
        port=int(settings.db_port),
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )

    try:
        print("\n========== DATABASE SUMMARY ==========")

        total = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks"
        )

        unique_studies = await conn.fetchval(
            "SELECT COUNT(DISTINCT nct_id) FROM chunks"
        )

        print(f"Total chunks: {total}")
        print(f"Unique studies: {unique_studies}")

        print("\n========== REMDESIVIR SEARCH ==========")

        remdesivir = await conn.fetch(
            """
            SELECT nct_id, chunk_text
            FROM chunks
            WHERE chunk_text ILIKE '%remdesivir%'
            LIMIT 10
            """
        )

        print(f"Remdesivir matches: {len(remdesivir)}")

        for row in remdesivir:
            print(f"\nNCT ID: {row['nct_id']}")
            print(row["chunk_text"][:500])

        print("\n========== COVID-19 SEARCH ==========")

        covid = await conn.fetch(
            """
            SELECT nct_id, chunk_text
            FROM chunks
            WHERE chunk_text ILIKE '%COVID-19%'
            LIMIT 10
            """
        )

        print(f"COVID-19 matches: {len(covid)}")

        for row in covid:
            print(f"\nNCT ID: {row['nct_id']}")
            print(row["chunk_text"][:350])

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
