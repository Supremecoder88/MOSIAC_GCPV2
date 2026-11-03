
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
        print("\n========== EXISTING DATA ==========")

        tables = [
            "studies",
            "chunks",
            "signals",
            "hitl_reviews",
            "episodes",
            "procedures",
            "sponsor_profiles",
        ]

        for table in tables:
            count = await conn.fetchval(
                f'SELECT COUNT(*) FROM "{table}"'
            )
            print(f"{table}: {count}")

        print("\n========== DUPLICATE CHUNK CHECK ==========")

        duplicates = await conn.fetch("""
            SELECT nct_id, chunk_index, source, COUNT(*) AS copies
            FROM chunks
            GROUP BY nct_id, chunk_index, source
            HAVING COUNT(*) > 1
            ORDER BY copies DESC;
        """)

        print(f"Duplicate logical chunks: {len(duplicates)}")

        for row in duplicates:
            print(dict(row))

        print("\n========== NULL CHECK ==========")

        nulls = await conn.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE nct_id IS NULL) AS null_nct_ids,
                COUNT(*) FILTER (WHERE chunk_index IS NULL) AS null_indexes,
                COUNT(*) FILTER (WHERE source IS NULL) AS null_sources
            FROM chunks;
        """)

        print(dict(nulls))

        print("\nPre-migration inspection complete.")
        print("No database records were modified.")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
