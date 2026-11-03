
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
        tables = await conn.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)

        print("\n========== DATABASE TABLES ==========")

        for table in tables:
            table_name = table["table_name"]
            print(f"\nTABLE: {table_name}")

            columns = await conn.fetch("""
                SELECT
                    column_name,
                    data_type,
                    udt_name,
                    is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = $1
                ORDER BY ordinal_position;
            """, table_name)

            for col in columns:
                print(
                    f"  {col['column_name']:<25} "
                    f"type={col['udt_name']:<15} "
                    f"nullable={col['is_nullable']}"
                )

            constraints = await conn.fetch("""
                SELECT
                    constraint_name,
                    constraint_type
                FROM information_schema.table_constraints
                WHERE table_schema = 'public'
                  AND table_name = $1;
            """, table_name)

            for constraint in constraints:
                print(
                    f"  CONSTRAINT: {constraint['constraint_name']} "
                    f"({constraint['constraint_type']})"
                )

        print("\n========== EXTENSIONS ==========")

        extensions = await conn.fetch("""
            SELECT extname, extversion
            FROM pg_extension
            ORDER BY extname;
        """)

        for ext in extensions:
            print(f"{ext['extname']} | version={ext['extversion']}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
