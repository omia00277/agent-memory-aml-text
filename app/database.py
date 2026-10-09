import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

Base = declarative_base()

engine_kwargs = {"pool_pre_ping": True, "future": True}
if settings.database_url.startswith("sqlite"):
    # SQLite requires this flag for multi-threaded usage (FastAPI runs handlers in a thread pool)
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.database_url, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class RawChunk(Base):
    """Persist raw memory chunks for audit and downstream enrichment."""

    __tablename__ = "raw_chunks"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(String(255), nullable=False, index=True, unique=True)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    messages = Column(JSON, nullable=True)
    source_ts = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class MemoryUnit(Base):
    """Consolidated atomic memory facts that are returned by Search."""

    __tablename__ = "memory_units"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    source_request_id = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    unit_type = Column(String(32), nullable=False, default="fact")
    source_ts = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Structured fields for conflict detection / current-state tracking
    entity = Column(String(255), nullable=True, index=True)
    attribute = Column(String(255), nullable=True, index=True)
    value = Column(Text, nullable=True)
    valid = Column(Integer, nullable=False, default=1, index=True)  # 1 = valid, 0 = superseded
    superseded_by = Column(String(64), nullable=True, index=True)


def init_db():
    Base.metadata.create_all(bind=engine)
    _migrate_sqlite()


def _migrate_sqlite():
    """Add columns introduced after the initial schema to existing SQLite tables."""
    if not settings.database_url.startswith("sqlite"):
        return
    inspector = inspect(engine)
    if "raw_chunks" in inspector.get_table_names():
        existing = {c["name"] for c in inspector.get_columns("raw_chunks")}
        if "source_ts" not in existing:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE raw_chunks ADD COLUMN source_ts INTEGER"))
    if "memory_units" in inspector.get_table_names():
        existing = {c["name"] for c in inspector.get_columns("memory_units")}
        new_cols = [
            ("entity", "VARCHAR(255)"),
            ("attribute", "VARCHAR(255)"),
            ("value", "TEXT"),
            ("valid", "INTEGER DEFAULT 1"),
            ("superseded_by", "VARCHAR(64)"),
        ]
        for col, dtype in new_cols:
            if col not in existing:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE memory_units ADD COLUMN {col} {dtype}"))
