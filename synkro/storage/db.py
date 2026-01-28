"""Database storage for session persistence.

Supports both SQLite (local) and PostgreSQL (remote) via SQLAlchemy async.

Schema:
- sessions: Top-level container for dataset generation runs
- rules: Extracted rules from policy (normalized)
- scenarios: Generated test scenarios
- traces: Conversation traces with grading results
- taxonomy: Category hierarchy
- session_stats: Pre-computed statistics
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.types import JSON, TypeDecorator

Base = declarative_base()


# ---------------------------------------------------------------------------
# Custom Types
# ---------------------------------------------------------------------------


class StringArray(TypeDecorator):
    """Array of strings - uses ARRAY on Postgres, JSON on SQLite."""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(ARRAY(String(20)))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        import json

        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return []
        if dialect.name == "postgresql":
            return value or []
        if isinstance(value, str):
            import json

            return json.loads(value)
        return value or []


class JSONType(TypeDecorator):
    """JSON type - uses JSONB on Postgres, JSON on SQLite."""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        import json

        return json.dumps(value) if isinstance(value, (dict, list)) else value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        if isinstance(value, str):
            import json

            return json.loads(value)
        return value


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class SessionRecord(Base):
    """Top-level container for a dataset generation run."""

    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    policy_text = Column(Text, nullable=False, default="")
    policy_hash = Column(String(64))  # SHA256 for dedup
    dataset_type = Column(String(20), default="conversation")
    model = Column(String(100))
    grading_model = Column(String(100))
    base_url = Column(String(500))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    rules = relationship("RuleRecord", back_populates="session", cascade="all, delete-orphan")
    scenarios = relationship(
        "ScenarioRecord", back_populates="session", cascade="all, delete-orphan"
    )
    traces = relationship("TraceRecord", back_populates="session", cascade="all, delete-orphan")
    taxonomy = relationship(
        "TaxonomyRecord", back_populates="session", cascade="all, delete-orphan"
    )
    stats = relationship(
        "SessionStatsRecord",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )


class RuleRecord(Base):
    """Extracted rules from policy document."""

    __tablename__ = "rules"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    rule_id = Column(String(20), nullable=False)  # R001, R002
    text = Column(Text, nullable=False)
    condition = Column(Text)
    action = Column(Text)
    category = Column(String(50))
    dependencies = Column(StringArray(), default=list)  # ['R001', 'R002']
    position = Column(SmallInteger)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("SessionRecord", back_populates="rules")

    __table_args__ = (
        Index("idx_rules_session", "session_id"),
        Index("idx_rules_session_rule", "session_id", "rule_id", unique=True),
    )


class ScenarioRecord(Base):
    """Generated test scenarios."""

    __tablename__ = "scenarios"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    scenario_id = Column(String(20), nullable=False)  # S001, S002
    type = Column(String(20), nullable=False)  # positive, negative, edge_case, irrelevant
    category = Column(String(100))
    subcategory = Column(String(100))
    description = Column(Text, nullable=False)
    context = Column(Text)
    expected_outcome = Column(Text)
    rules_tested = Column(StringArray(), default=list)  # ['R001', 'R003']
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("SessionRecord", back_populates="scenarios")

    __table_args__ = (
        Index("idx_scenarios_session", "session_id"),
        Index("idx_scenarios_type", "session_id", "type"),
    )


class TraceRecord(Base):
    """Conversation traces with grading results."""

    __tablename__ = "traces"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    scenario_id = Column(String(36), ForeignKey("scenarios.id", ondelete="SET NULL"))

    # Messages stored as JSON array (atomic unit for refinement)
    messages = Column(JSONType(), nullable=False)  # [{"role": "user", "content": "..."}]

    # Grading results
    grade_passed = Column(Boolean)
    grade_feedback = Column(Text)
    grade_score = Column(Float)  # 0.0-1.0

    # Rule tracking
    rules_applied = Column(StringArray(), default=list)
    rules_violated = Column(StringArray(), default=list)

    # Metadata
    turn_count = Column(SmallInteger)
    reasoning_chain = Column(JSONType())  # Chain of thought
    verified_at = Column(DateTime)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("SessionRecord", back_populates="traces")

    __table_args__ = (
        Index("idx_traces_session", "session_id"),
        Index("idx_traces_session_grade", "session_id", "grade_passed"),
        Index("idx_traces_session_created", "session_id", "created_at"),
    )


class TaxonomyRecord(Base):
    """Category hierarchy for organizing scenarios."""

    __tablename__ = "taxonomy"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    category = Column(String(100), nullable=False)
    subcategory = Column(String(100))
    description = Column(Text)
    scenario_count = Column(Integer, default=0)

    session = relationship("SessionRecord", back_populates="taxonomy")

    __table_args__ = (Index("idx_taxonomy_session", "session_id"),)


class SessionStatsRecord(Base):
    """Pre-computed statistics to avoid aggregating many rows."""

    __tablename__ = "session_stats"

    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True)
    rule_count = Column(Integer, default=0)
    scenario_count = Column(Integer, default=0)
    trace_count = Column(Integer, default=0)
    passed_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    pass_rate = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("SessionRecord", back_populates="stats")


# ---------------------------------------------------------------------------
# Storage Class
# ---------------------------------------------------------------------------


class Storage:
    """Unified storage - works with SQLite (local) or Postgres (remote).

    Features:
        - Connection pooling for better performance
        - Bulk operations for batch inserts
        - Lazy loading - metadata loads fast, data loads on demand

    Examples:
        >>> store = Storage()  # SQLite at ~/.synkro/sessions.db
        >>> store = Storage("sqlite:///./my.db")
        >>> store = Storage("postgresql://user:pass@host/db")
        >>> store = Storage(os.environ["DATABASE_URL"])
    """

    def __init__(self, url: str | None = None):
        """Initialize storage with database URL and connection pool."""
        if url is None:
            db_path = Path("~/.synkro/sessions.db").expanduser()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite+aiosqlite:///{db_path}"
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://") and "asyncpg" not in url:
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

        self._url = url
        self._is_postgres = "postgresql" in url

        # Connection pooling settings
        pool_kwargs: dict = {
            "pool_pre_ping": True,  # Verify connections before use
        }
        if self._is_postgres:
            # Postgres: use connection pool
            pool_kwargs.update(
                {
                    "pool_size": 5,  # Base connections
                    "max_overflow": 10,  # Extra connections under load
                    "pool_recycle": 3600,  # Recycle connections after 1 hour
                }
            )
        else:
            # SQLite: use StaticPool + optimized PRAGMAs
            from sqlalchemy.pool import StaticPool

            pool_kwargs["poolclass"] = StaticPool
            pool_kwargs["connect_args"] = {"check_same_thread": False}

        self._engine = create_async_engine(url, **pool_kwargs)

        # SQLite optimizations via PRAGMAs
        if not self._is_postgres:
            import sqlalchemy.event

            @sqlalchemy.event.listens_for(self._engine.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")  # ~5x faster writes
                cursor.execute("PRAGMA synchronous=NORMAL")  # ~2x faster, still safe
                cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
                cursor.execute("PRAGMA busy_timeout=5000")  # 5s timeout
                cursor.execute("PRAGMA temp_store=MEMORY")  # Temp tables in RAM
                cursor.close()

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

    # -----------------------------------------------------------------------
    # Session CRUD
    # -----------------------------------------------------------------------

    async def create_session(
        self,
        session_id: str | None = None,
        policy_text: str = "",
        model: str | None = None,
        grading_model: str | None = None,
        base_url: str | None = None,
    ) -> str:
        """Create a new session."""
        await self._ensure_init()
        sid = session_id or uuid.uuid4().hex[:8]
        policy_hash = hashlib.sha256(policy_text.encode()).hexdigest() if policy_text else None

        async with self._session_factory() as db:
            # Delete existing if using custom ID (idempotent)
            if session_id:
                existing = await db.get(SessionRecord, session_id)
                if existing:
                    await db.delete(existing)
                    await db.commit()

            session = SessionRecord(
                id=sid,
                policy_text=policy_text,
                policy_hash=policy_hash,
                model=model,
                grading_model=grading_model,
                base_url=base_url,
            )
            db.add(session)
            db.add(SessionStatsRecord(session_id=sid))
            await db.commit()
        return sid

    async def load_session(self, session_id: str) -> dict | None:
        """Load session metadata only (fast, lazy loading).

        Does NOT load rules/scenarios/traces - use get_rules(), get_scenarios(),
        get_traces() to load those on demand.
        """
        await self._ensure_init()
        async with self._session_factory() as db:
            session = await db.get(SessionRecord, session_id)
            if not session:
                return None

            stats = await db.get(SessionStatsRecord, session_id)

            return {
                "session_id": session.id,
                "policy_text": session.policy_text,
                "model": session.model,
                "grading_model": session.grading_model,
                "base_url": session.base_url,
                "dataset_type": session.dataset_type,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                "stats": {
                    "rule_count": stats.rule_count if stats else 0,
                    "scenario_count": stats.scenario_count if stats else 0,
                    "trace_count": stats.trace_count if stats else 0,
                    "passed_count": stats.passed_count if stats else 0,
                    "failed_count": stats.failed_count if stats else 0,
                    "pass_rate": stats.pass_rate if stats else 0.0,
                    "total_cost": stats.total_cost if stats else 0.0,
                }
                if stats
                else None,
            }

    async def load_session_full(self, session_id: str) -> dict | None:
        """Load session with ALL related data (slower, use for export/migration)."""
        await self._ensure_init()
        async with self._session_factory() as db:
            session = await db.get(SessionRecord, session_id)
            if not session:
                return None

            # Load all related data in parallel-ish (same connection)
            rules = (
                (
                    await db.execute(
                        select(RuleRecord)
                        .where(RuleRecord.session_id == session_id)
                        .order_by(RuleRecord.position)
                    )
                )
                .scalars()
                .all()
            )

            scenarios = (
                (
                    await db.execute(
                        select(ScenarioRecord).where(ScenarioRecord.session_id == session_id)
                    )
                )
                .scalars()
                .all()
            )

            traces = (
                (
                    await db.execute(
                        select(TraceRecord)
                        .where(TraceRecord.session_id == session_id)
                        .order_by(TraceRecord.created_at)
                    )
                )
                .scalars()
                .all()
            )

            taxonomy = (
                (
                    await db.execute(
                        select(TaxonomyRecord).where(TaxonomyRecord.session_id == session_id)
                    )
                )
                .scalars()
                .all()
            )

            stats = await db.get(SessionStatsRecord, session_id)

            return {
                "session_id": session.id,
                "policy_text": session.policy_text,
                "model": session.model,
                "grading_model": session.grading_model,
                "base_url": session.base_url,
                "dataset_type": session.dataset_type,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                "rules": [
                    {
                        "rule_id": r.rule_id,
                        "text": r.text,
                        "condition": r.condition,
                        "action": r.action,
                        "category": r.category,
                        "dependencies": r.dependencies or [],
                    }
                    for r in rules
                ],
                "scenarios": [
                    {
                        "id": s.id,
                        "scenario_id": s.scenario_id,
                        "type": s.type,
                        "category": s.category,
                        "subcategory": s.subcategory,
                        "description": s.description,
                        "context": s.context,
                        "expected_outcome": s.expected_outcome,
                        "rules_tested": s.rules_tested or [],
                    }
                    for s in scenarios
                ],
                "traces": [
                    {
                        "id": t.id,
                        "scenario_id": t.scenario_id,
                        "messages": t.messages,
                        "grade_passed": t.grade_passed,
                        "grade_feedback": t.grade_feedback,
                        "grade_score": t.grade_score,
                        "rules_applied": t.rules_applied or [],
                        "rules_violated": t.rules_violated or [],
                        "turn_count": t.turn_count,
                        "reasoning_chain": t.reasoning_chain,
                    }
                    for t in traces
                ],
                "taxonomy": [
                    {
                        "id": t.id,
                        "category": t.category,
                        "subcategory": t.subcategory,
                        "description": t.description,
                        "scenario_count": t.scenario_count,
                    }
                    for t in taxonomy
                ],
                "stats": {
                    "rule_count": stats.rule_count if stats else 0,
                    "scenario_count": stats.scenario_count if stats else 0,
                    "trace_count": stats.trace_count if stats else 0,
                    "passed_count": stats.passed_count if stats else 0,
                    "failed_count": stats.failed_count if stats else 0,
                    "pass_rate": stats.pass_rate if stats else 0.0,
                    "total_cost": stats.total_cost if stats else 0.0,
                }
                if stats
                else None,
            }

    async def update_session(
        self,
        session_id: str,
        model: str | None = None,
        grading_model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Update session metadata."""
        await self._ensure_init()
        async with self._session_factory() as db:
            session = await db.get(SessionRecord, session_id)
            if session:
                if model is not None:
                    session.model = model
                if grading_model is not None:
                    session.grading_model = grading_model
                if base_url is not None:
                    session.base_url = base_url
                session.updated_at = datetime.now(timezone.utc)
                await db.commit()

    async def delete_session(self, session_id: str) -> bool:
        """Delete session and all related data (CASCADE)."""
        await self._ensure_init()
        async with self._session_factory() as db:
            session = await db.get(SessionRecord, session_id)
            if session:
                await db.delete(session)
                await db.commit()
                return True
            return False

    async def list_sessions(self, limit: int = 50) -> list[dict]:
        """List recent sessions with stats."""
        await self._ensure_init()
        async with self._session_factory() as db:
            result = await db.execute(
                select(SessionRecord, SessionStatsRecord)
                .outerjoin(SessionStatsRecord)
                .order_by(SessionRecord.updated_at.desc())
                .limit(limit)
            )
            return [
                {
                    "session_id": s.id,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                    "rule_count": st.rule_count if st else 0,
                    "scenario_count": st.scenario_count if st else 0,
                    "trace_count": st.trace_count if st else 0,
                    "pass_rate": st.pass_rate if st else 0.0,
                }
                for s, st in result.all()
            ]

    # -----------------------------------------------------------------------
    # Rules CRUD
    # -----------------------------------------------------------------------

    async def save_rules(self, session_id: str, rules: list[dict]) -> None:
        """Replace all rules for a session (bulk insert)."""
        await self._ensure_init()
        async with self._session_factory() as db:
            # Delete existing rules
            await db.execute(
                RuleRecord.__table__.delete().where(RuleRecord.session_id == session_id)
            )
            # Bulk insert new rules
            if rules:
                db.add_all(
                    [
                        RuleRecord(
                            session_id=session_id,
                            rule_id=r.get("rule_id", f"R{i+1:03d}"),
                            text=r.get("text", ""),
                            condition=r.get("condition"),
                            action=r.get("action"),
                            category=r.get("category"),
                            dependencies=r.get("dependencies", []),
                            position=i,
                        )
                        for i, r in enumerate(rules)
                    ]
                )
            await db.commit()
        await self._update_stats(session_id)

    async def get_rules(self, session_id: str) -> list[dict]:
        """Get all rules for a session."""
        await self._ensure_init()
        async with self._session_factory() as db:
            result = await db.execute(
                select(RuleRecord)
                .where(RuleRecord.session_id == session_id)
                .order_by(RuleRecord.position)
            )
            return [
                {
                    "rule_id": r.rule_id,
                    "text": r.text,
                    "condition": r.condition,
                    "action": r.action,
                    "category": r.category,
                    "dependencies": r.dependencies or [],
                }
                for r in result.scalars()
            ]

    # -----------------------------------------------------------------------
    # Scenarios CRUD
    # -----------------------------------------------------------------------

    async def save_scenarios(self, session_id: str, scenarios: list[dict]) -> None:
        """Replace all scenarios for a session (bulk insert)."""
        await self._ensure_init()
        async with self._session_factory() as db:
            # Delete existing scenarios
            await db.execute(
                ScenarioRecord.__table__.delete().where(ScenarioRecord.session_id == session_id)
            )
            # Bulk insert new scenarios
            if scenarios:
                db.add_all(
                    [
                        ScenarioRecord(
                            session_id=session_id,
                            scenario_id=s.get("scenario_id", f"S{i+1:03d}"),
                            type=s.get("type", s.get("scenario_type", "positive")),
                            category=s.get("category"),
                            subcategory=s.get("subcategory"),
                            description=s.get("description", ""),
                            context=s.get("context"),
                            expected_outcome=s.get("expected_outcome"),
                            rules_tested=s.get("rules_tested", s.get("target_rule_ids", [])),
                        )
                        for i, s in enumerate(scenarios)
                    ]
                )
            await db.commit()
        await self._update_stats(session_id)

    async def get_scenarios(self, session_id: str, type_filter: str | None = None) -> list[dict]:
        """Get scenarios for a session, optionally filtered by type."""
        await self._ensure_init()
        async with self._session_factory() as db:
            query = select(ScenarioRecord).where(ScenarioRecord.session_id == session_id)
            if type_filter:
                query = query.where(ScenarioRecord.type == type_filter)
            result = await db.execute(query.order_by(ScenarioRecord.scenario_id))
            return [
                {
                    "id": s.id,
                    "scenario_id": s.scenario_id,
                    "type": s.type,
                    "category": s.category,
                    "subcategory": s.subcategory,
                    "description": s.description,
                    "context": s.context,
                    "expected_outcome": s.expected_outcome,
                    "rules_tested": s.rules_tested or [],
                }
                for s in result.scalars()
            ]

    # -----------------------------------------------------------------------
    # Traces CRUD
    # -----------------------------------------------------------------------

    async def save_traces(self, session_id: str, traces: list[dict]) -> None:
        """Replace all traces for a session (bulk insert)."""
        await self._ensure_init()
        now = datetime.now(timezone.utc)
        async with self._session_factory() as db:
            # Delete existing traces
            await db.execute(
                TraceRecord.__table__.delete().where(TraceRecord.session_id == session_id)
            )
            # Bulk insert new traces
            if traces:
                db.add_all(
                    [
                        TraceRecord(
                            session_id=session_id,
                            scenario_id=t.get("scenario_id"),
                            messages=t.get("messages", []),
                            grade_passed=t.get("grade_passed"),
                            grade_feedback=t.get("grade_feedback"),
                            grade_score=t.get("grade_score"),
                            rules_applied=t.get("rules_applied", []),
                            rules_violated=t.get("rules_violated", []),
                            turn_count=len(t.get("messages", [])) // 2,
                            reasoning_chain=t.get("reasoning_chain"),
                            verified_at=now if t.get("grade_passed") is not None else None,
                        )
                        for t in traces
                    ]
                )
            await db.commit()
        await self._update_stats(session_id)

    async def add_trace(self, session_id: str, trace: dict) -> str:
        """Add a single trace. Returns trace ID."""
        await self._ensure_init()
        trace_id = generate_uuid()
        messages = trace.get("messages", [])
        async with self._session_factory() as db:
            db.add(
                TraceRecord(
                    id=trace_id,
                    session_id=session_id,
                    scenario_id=trace.get("scenario_id"),
                    messages=messages,
                    grade_passed=trace.get("grade_passed"),
                    grade_feedback=trace.get("grade_feedback"),
                    grade_score=trace.get("grade_score"),
                    rules_applied=trace.get("rules_applied", []),
                    rules_violated=trace.get("rules_violated", []),
                    turn_count=len(messages) // 2 if messages else 0,
                    reasoning_chain=trace.get("reasoning_chain"),
                )
            )
            await db.commit()
        await self._update_stats(session_id)
        return trace_id

    async def update_trace(self, trace_id: str, updates: dict) -> None:
        """Update a single trace (e.g., after grading)."""
        await self._ensure_init()
        async with self._session_factory() as db:
            trace = await db.get(TraceRecord, trace_id)
            if trace:
                for key, value in updates.items():
                    if hasattr(trace, key):
                        setattr(trace, key, value)
                if "grade_passed" in updates:
                    trace.verified_at = datetime.now(timezone.utc)
                await db.commit()
                await self._update_stats(trace.session_id)

    async def get_traces(
        self,
        session_id: str,
        passed_only: bool = False,
        failed_only: bool = False,
    ) -> list[dict]:
        """Get traces for a session."""
        await self._ensure_init()
        async with self._session_factory() as db:
            query = select(TraceRecord).where(TraceRecord.session_id == session_id)
            if passed_only:
                query = query.where(TraceRecord.grade_passed == True)  # noqa: E712
            elif failed_only:
                query = query.where(TraceRecord.grade_passed == False)  # noqa: E712
            result = await db.execute(query.order_by(TraceRecord.created_at))
            return [
                {
                    "id": t.id,
                    "scenario_id": t.scenario_id,
                    "messages": t.messages,
                    "grade_passed": t.grade_passed,
                    "grade_feedback": t.grade_feedback,
                    "grade_score": t.grade_score,
                    "rules_applied": t.rules_applied or [],
                    "rules_violated": t.rules_violated or [],
                    "turn_count": t.turn_count,
                    "reasoning_chain": t.reasoning_chain,
                }
                for t in result.scalars()
            ]

    async def get_traces_by_rule(self, session_id: str, rule_id: str) -> list[dict]:
        """Get traces that apply a specific rule."""
        await self._ensure_init()
        async with self._session_factory() as db:
            # For Postgres, use array contains. For SQLite, use JSON.
            if self._is_postgres:
                query = select(TraceRecord).where(
                    TraceRecord.session_id == session_id,
                    TraceRecord.rules_applied.contains([rule_id]),
                )
            else:
                # SQLite JSON contains
                query = select(TraceRecord).where(
                    TraceRecord.session_id == session_id,
                    func.json_extract(TraceRecord.rules_applied, "$").contains(rule_id),
                )
            result = await db.execute(query)
            return [
                {
                    "id": t.id,
                    "messages": t.messages,
                    "grade_passed": t.grade_passed,
                }
                for t in result.scalars()
            ]

    # -----------------------------------------------------------------------
    # Taxonomy CRUD
    # -----------------------------------------------------------------------

    async def save_taxonomy(self, session_id: str, taxonomy: list[dict]) -> None:
        """Replace taxonomy for a session (bulk insert)."""
        await self._ensure_init()
        async with self._session_factory() as db:
            await db.execute(
                TaxonomyRecord.__table__.delete().where(TaxonomyRecord.session_id == session_id)
            )
            if taxonomy:
                db.add_all(
                    [
                        TaxonomyRecord(
                            session_id=session_id,
                            category=t.get("category", ""),
                            subcategory=t.get("subcategory"),
                            description=t.get("description"),
                            scenario_count=t.get("scenario_count", 0),
                        )
                        for t in taxonomy
                    ]
                )
            await db.commit()

    async def get_taxonomy(self, session_id: str) -> list[dict]:
        """Get taxonomy for a session."""
        await self._ensure_init()
        async with self._session_factory() as db:
            result = await db.execute(
                select(TaxonomyRecord).where(TaxonomyRecord.session_id == session_id)
            )
            return [
                {
                    "id": t.id,
                    "category": t.category,
                    "subcategory": t.subcategory,
                    "description": t.description,
                    "scenario_count": t.scenario_count,
                }
                for t in result.scalars()
            ]

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    async def _update_stats(self, session_id: str) -> None:
        """Recompute and save session stats."""
        async with self._session_factory() as db:
            # Count rules
            rule_count = (
                await db.execute(select(func.count()).where(RuleRecord.session_id == session_id))
            ).scalar() or 0

            # Count scenarios
            scenario_count = (
                await db.execute(
                    select(func.count()).where(ScenarioRecord.session_id == session_id)
                )
            ).scalar() or 0

            # Count traces
            trace_count = (
                await db.execute(select(func.count()).where(TraceRecord.session_id == session_id))
            ).scalar() or 0

            # Count passed/failed
            passed_count = (
                await db.execute(
                    select(func.count()).where(
                        TraceRecord.session_id == session_id,
                        TraceRecord.grade_passed == True,  # noqa: E712
                    )
                )
            ).scalar() or 0

            failed_count = (
                await db.execute(
                    select(func.count()).where(
                        TraceRecord.session_id == session_id,
                        TraceRecord.grade_passed == False,  # noqa: E712
                    )
                )
            ).scalar() or 0

            pass_rate = passed_count / trace_count if trace_count > 0 else 0.0

            # Update or insert stats
            stats = await db.get(SessionStatsRecord, session_id)
            if stats:
                stats.rule_count = rule_count
                stats.scenario_count = scenario_count
                stats.trace_count = trace_count
                stats.passed_count = passed_count
                stats.failed_count = failed_count
                stats.pass_rate = pass_rate
                stats.updated_at = datetime.now(timezone.utc)
            else:
                db.add(
                    SessionStatsRecord(
                        session_id=session_id,
                        rule_count=rule_count,
                        scenario_count=scenario_count,
                        trace_count=trace_count,
                        passed_count=passed_count,
                        failed_count=failed_count,
                        pass_rate=pass_rate,
                    )
                )
            await db.commit()

    async def update_cost(self, session_id: str, cost: float) -> None:
        """Add to total cost for a session."""
        await self._ensure_init()
        async with self._session_factory() as db:
            stats = await db.get(SessionStatsRecord, session_id)
            if stats:
                stats.total_cost = (stats.total_cost or 0.0) + cost
                stats.updated_at = datetime.now(timezone.utc)
                await db.commit()

    async def get_stats(self, session_id: str) -> dict:
        """Get stats for a session."""
        await self._ensure_init()
        async with self._session_factory() as db:
            stats = await db.get(SessionStatsRecord, session_id)
            if not stats:
                return {}
            return {
                "rule_count": stats.rule_count,
                "scenario_count": stats.scenario_count,
                "trace_count": stats.trace_count,
                "passed_count": stats.passed_count,
                "failed_count": stats.failed_count,
                "pass_rate": stats.pass_rate,
                "total_cost": stats.total_cost,
            }
