import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, JSON, String, Text, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

Base = declarative_base()
engine_kwargs = {"pool_pre_ping": True, "future": True}
if settings.database_url.startswith("sqlite"):
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
    __tablename__ = "raw_chunks"
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(String(255), nullable=False, index=True, unique=True)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    messages = Column(JSON, nullable=True)
    source_ts = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Episode(Base):
    __tablename__ = "episodes"
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    source_id = Column(String(255), nullable=True, index=True)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class MemoryUnit(Base):
    __tablename__ = "memory_units"
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    source_request_id = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    unit_type = Column(String(32), nullable=False, default="fact")
    evidence = Column(Text, nullable=True)
    source_id = Column(String(255), nullable=True, index=True)
    role = Column(String(32), nullable=True)
    subject = Column(String(500), nullable=True)
    question = Column(Text, nullable=True)
    action = Column(String(500), nullable=True)
    fact_type = Column(String(32), nullable=True)
    event_title = Column(String(500), nullable=True)
    event_subject = Column(String(500), nullable=True)
    event_problem = Column(Text, nullable=True)
    event_time = Column(String(500), nullable=True)
    outcome = Column(Text, nullable=True)
    fact_metadata = Column("metadata", JSON, nullable=True)
    episode_id = Column(String(64), nullable=True, index=True)
    source_ts = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


def init_db():
    Base.metadata.create_all(bind=engine)
    _migrate_sqlite()


def _migrate_sqlite():
    if not settings.database_url.startswith("sqlite"):
        return
    inspector = inspect(engine)
    with engine.begin() as conn:
        if "raw_chunks" in inspector.get_table_names():
            existing = {c["name"] for c in inspector.get_columns("raw_chunks")}
            if "source_ts" not in existing:
                conn.execute(text("ALTER TABLE raw_chunks ADD COLUMN source_ts INTEGER"))
        if "memory_units" in inspector.get_table_names():
            existing = {c["name"] for c in inspector.get_columns("memory_units")}
            additions = {
                "evidence": "TEXT", "source_id": "VARCHAR(255)", "role": "VARCHAR(32)",
                "subject": "VARCHAR(500)", "question": "TEXT", "action": "VARCHAR(500)",
                "fact_type": "VARCHAR(32)", "event_title": "VARCHAR(500)",
                "event_subject": "VARCHAR(500)", "event_problem": "TEXT",
                "event_time": "VARCHAR(500)", "outcome": "TEXT",
                "metadata": "JSON", "episode_id": "VARCHAR(64)",
            }
            for name, type_sql in additions.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE memory_units ADD COLUMN {name} {type_sql}"))
