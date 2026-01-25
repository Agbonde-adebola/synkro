"""Database storage for session persistence.

Supports both SQLite (local) and PostgreSQL (remote) via SQLAlchemy async.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import Column, DateTime, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class SessionRecord(Base):
    """Session state stored in DB."""

    __tablename__ = "synkro_sessions"

    session_id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data_json = Column(Text, default="{}")


class Storage:
    """Unified storage - works with SQLite (local) or Postgres (remote).

    Examples:
        >>> store = Storage()  # SQLite at ~/.synkro/sessions.db
        >>> store = Storage("sqlite:///./my.db")
        >>> store = Storage("postgresql://user:pass@host/db")
        >>> store = Storage(os.environ["DATABASE_URL"])
    """

    def __init__(self, url: str | None = None):
        """Initialize storage with database URL.

        Args:
            url: Database URL. If None, uses SQLite at ~/.synkro/sessions.db.
                 Supports SQLite and PostgreSQL URLs.
        """
        if url is None:
            db_path = Path("~/.synkro/sessions.db").expanduser()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite+aiosqlite:///{db_path}"
        elif url.startswith("postgres://"):
            # Fix common Heroku-style URLs
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://") and "asyncpg" not in url:
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

        self._engine = create_async_engine(url)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self._init = False

    async def _ensure_init(self):
        """Create tables if they don't exist."""
        if self._init:
            return
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._init = True

    async def create(self, session_id: str | None = None) -> str:
        """Create a new session.

        If a session_id is provided and already exists, it will be deleted
        and recreated (idempotent behavior for testing/development).

        Args:
            session_id: Optional custom ID. If None, generates a random 8-char ID.

        Returns:
            The session ID.
        """
        await self._ensure_init()
        sid = session_id or uuid.uuid4().hex[:8]
        async with self._session_factory() as db:
            # Delete existing session if using custom ID (idempotent)
            if session_id:
                existing = await db.get(SessionRecord, session_id)
                if existing:
                    await db.delete(existing)
                    await db.commit()
            db.add(SessionRecord(session_id=sid))
            await db.commit()
        return sid

    async def load(self, session_id: str) -> dict | None:
        """Load session data by ID.

        Args:
            session_id: The session ID to load.

        Returns:
            Session data dict, or None if not found.
        """
        await self._ensure_init()
        async with self._session_factory() as db:
            record = await db.get(SessionRecord, session_id)
            return json.loads(record.data_json) if record else None

    async def save(self, session_id: str, data: dict) -> None:
        """Save session data.

        Args:
            session_id: The session ID.
            data: Session data dict to save.
        """
        await self._ensure_init()
        async with self._session_factory() as db:
            record = await db.get(SessionRecord, session_id)
            if record:
                record.data_json = json.dumps(data)
                record.updated_at = datetime.utcnow()
            else:
                db.add(SessionRecord(session_id=session_id, data_json=json.dumps(data)))
            await db.commit()

    async def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        await self._ensure_init()
        async with self._session_factory() as db:
            record = await db.get(SessionRecord, session_id)
            if record:
                await db.delete(record)
                await db.commit()
                return True
            return False

    async def list(self, limit: int = 50) -> list[dict]:
        """List recent sessions.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of session metadata dicts.
        """
        await self._ensure_init()
        async with self._session_factory() as db:
            result = await db.execute(
                select(SessionRecord).order_by(SessionRecord.updated_at.desc()).limit(limit)
            )
            return [
                {
                    "session_id": r.session_id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in result.scalars()
            ]
