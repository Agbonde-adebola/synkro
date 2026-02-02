"""Database client for verification queries.

Supports both PostgreSQL (asyncpg) and SQLite (aiosqlite).
"""

from typing import Any

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore

try:
    import aiosqlite
except ImportError:
    aiosqlite = None  # type: ignore


class SQL:
    """Database client wrapper supporting PostgreSQL and SQLite."""

    def __init__(self, dsn: str):
        """Initialize with database connection string.

        Args:
            dsn: Connection string:
                - PostgreSQL: "postgresql://user:pass@host/db"
                - SQLite: "sqlite:///path/to/db.sqlite" or just "/path/to/db.sqlite"
        """
        self.dsn = dsn
        self._pool: Any = None
        self._sqlite_conn: Any = None

        # Detect database type
        if dsn.startswith("postgresql://") or dsn.startswith("postgres://"):
            self._db_type = "postgresql"
            if asyncpg is None:
                raise RuntimeError("asyncpg not installed. Install with: pip install asyncpg")
        else:
            self._db_type = "sqlite"
            if aiosqlite is None:
                raise RuntimeError("aiosqlite not installed. Install with: pip install aiosqlite")
            # Strip sqlite:/// prefix if present
            if dsn.startswith("sqlite:///"):
                self.dsn = dsn[10:]

    @staticmethod
    def render(template: str, params: dict[str, str]) -> str:
        """Render SQL template with parameters.

        Uses {{param_name}} placeholders. Values are escaped to prevent SQL injection.

        Args:
            template: SQL template with {{param}} placeholders
            params: Parameter values to substitute

        Returns:
            Rendered SQL string
        """
        sql = template
        for k, v in params.items():
            # Escape single quotes for SQL safety
            safe_value = str(v).replace("'", "''")
            sql = sql.replace(f"{{{{{k}}}}}", safe_value)
        return sql

    async def _ensure_connection(self) -> None:
        """Create connection if not exists."""
        if self._db_type == "postgresql":
            if self._pool is None:
                self._pool = await asyncpg.create_pool(self.dsn)
        else:
            if self._sqlite_conn is None:
                self._sqlite_conn = await aiosqlite.connect(self.dsn)
                self._sqlite_conn.row_factory = aiosqlite.Row

    async def execute(self, template: str, params: dict[str, str]) -> list[dict]:
        """Execute SQL template with parameters.

        Args:
            template: SQL template with {{param}} placeholders
            params: Parameter values to substitute

        Returns:
            List of row dicts
        """
        await self._ensure_connection()
        sql = self.render(template, params)

        if self._db_type == "postgresql":
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql)
                return [dict(r) for r in rows]
        else:
            async with self._sqlite_conn.execute(sql) as cursor:
                rows = await cursor.fetchall()
                # Convert Row objects to dicts
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in rows]

    async def execute_raw(self, sql: str) -> None:
        """Execute raw SQL (for CREATE TABLE, INSERT, etc.)."""
        await self._ensure_connection()

        if self._db_type == "postgresql":
            async with self._pool.acquire() as conn:
                await conn.execute(sql)
        else:
            await self._sqlite_conn.execute(sql)
            await self._sqlite_conn.commit()

    async def execute_batch(
        self,
        queries: list[tuple[str, dict[str, str]]],
    ) -> list[list[dict]]:
        """Execute multiple queries efficiently using connection pool.

        Reuses the same connection for all queries in the batch.

        Args:
            queries: List of (template, params) tuples

        Returns:
            List of results, one per query
        """
        await self._ensure_connection()

        results = []

        if self._db_type == "postgresql":
            async with self._pool.acquire() as conn:
                for template, params in queries:
                    sql = self.render(template, params)
                    rows = await conn.fetch(sql)
                    results.append([dict(r) for r in rows])
        else:
            for template, params in queries:
                sql = self.render(template, params)
                async with self._sqlite_conn.execute(sql) as cursor:
                    rows = await cursor.fetchall()
                    columns = [d[0] for d in cursor.description]
                    results.append([dict(zip(columns, row)) for row in rows])

        return results

    async def close(self) -> None:
        """Close connection."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        if self._sqlite_conn:
            await self._sqlite_conn.close()
            self._sqlite_conn = None
