
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
        print("\n========== COLUMN DEFAULTS ==========")

        rows = await conn.fetch("""
            SELECT
                table_name,
                column_name,
                column_default
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name IN ('studies', 'chunks')
            ORDER BY table_name, ordinal_position;
        """)

        for row in rows:
            print(
                f"{row['table_name']}.{row['column_name']} "
                f"| default={row['column_default']}"
            )

        print("\n========== FOREIGN KEYS ==========")

        rows = await conn.fetch("""
            SELECT
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table,
                ccu.column_name AS foreign_column,
                tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.constraint_schema = kcu.constraint_schema
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.constraint_schema = tc.constraint_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = 'public'
            ORDER BY tc.table_name;
        """)

        for row in rows:
            print(
                f"{row['table_name']}.{row['column_name']} "
                f"-> {row['foreign_table']}.{row['foreign_column']} "
                f"| {row['constraint_name']}"
            )

        print("\n========== INDEXES ==========")

        rows = await conn.fetch("""
            SELECT
                tablename,
                indexname,
                indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename IN ('studies', 'chunks')
            ORDER BY tablename, indexname;
        """)

        for row in rows:
            print(f"\n{row['tablename']} | {row['indexname']}")
            print(row['indexdef'])

        print("\n========== CHUNK SOURCES ==========")

        rows = await conn.fetch("""
            SELECT source, COUNT(*) AS total
            FROM chunks
            GROUP BY source
            ORDER BY source;
        """)

        for row in rows:
            print(f"{row['source']}: {row['total']}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
