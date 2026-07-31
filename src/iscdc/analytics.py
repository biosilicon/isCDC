from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Engine,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    event,
    func,
    select,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

logger = logging.getLogger(__name__)

ANALYTICS_SCHEMA_VERSION = "1"
SESSION_COOKIE_NAME = "iscdc_session"

EVENT_METRICS = {
    "page_view": "page_view_count",
    "catalogue_search": "search_count",
    "database_detail_view": "database_detail_count",
    "challenge_detail_view": "challenge_detail_count",
    "download": "download_count",
}

AUTOMATION_PATTERN = re.compile(
    r"bot|spider|crawler|slurp|curl|wget|python-requests|python-httpx|"
    r"headless|monitoring|uptime",
    re.IGNORECASE,
)


class AnalyticsBase(DeclarativeBase):
    pass


class AnalyticsMetadata(AnalyticsBase):
    __tablename__ = "analytics_metadata"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(64), nullable=False)


class VisitSession(AnalyticsBase):
    __tablename__ = "visit_sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    counted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    initial_automated: Mapped[bool] = mapped_column(Boolean, nullable=False)


class VisitEvent(AnalyticsBase):
    __tablename__ = "visit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("visit_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    occurred_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    route_name: Mapped[str] = mapped_column(String(100), nullable=False)
    path: Mapped[str] = mapped_column(String(2048), nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    referrer: Mapped[str | None] = mapped_column(Text, nullable=True)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    automated: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)


class DailyAnalytics(AnalyticsBase):
    __tablename__ = "daily_analytics"

    day: Mapped[date] = mapped_column(Date, primary_key=True)
    visit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_view_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    search_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    database_detail_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    challenge_detail_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    download_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    automated_event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class AnalyticsSchemaError(RuntimeError):
    """Raised when an analytics database has an unsupported schema version."""


@dataclass(frozen=True)
class SessionResult:
    session_id: str
    set_cookie: bool


def utc_now() -> datetime:
    """Return a naive datetime whose value is UTC, for consistent SQLite storage."""

    return datetime.now(UTC).replace(tzinfo=None)


def is_automated_user_agent(user_agent: str | None) -> bool:
    return not user_agent or bool(AUTOMATION_PATTERN.search(user_agent))


def _normalize_session_id(value: str | None) -> str | None:
    if not value or len(value) > 36:
        return None
    try:
        return str(UUID(value))
    except ValueError:
        return None


def _increment_daily(session: Session, day: date, column_name: str) -> None:
    values = {
        "day": day,
        "visit_count": 0,
        "page_view_count": 0,
        "search_count": 0,
        "database_detail_count": 0,
        "challenge_detail_count": 0,
        "download_count": 0,
        "automated_event_count": 0,
    }
    values[column_name] = 1
    column = getattr(DailyAnalytics, column_name)
    statement = insert(DailyAnalytics).values(**values)
    session.execute(
        statement.on_conflict_do_update(
            index_elements=[DailyAnalytics.day],
            set_={column_name: column + 1},
        )
    )


def create_analytics_engine(database_path: Path) -> Engine:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False, "timeout": 5},
    )

    @event.listens_for(engine, "connect")
    def configure_connection(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    # The default data directory may be on a network filesystem, where WAL's
    # shared-memory coordination is not reliable. DELETE mode uses ordinary
    # file locking and also checkpoints databases created by older releases.
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=DELETE")

    return engine


def initialize_analytics_database(engine: Engine) -> None:
    AnalyticsBase.metadata.create_all(engine)
    with engine.begin() as connection:
        version = connection.scalar(
            select(AnalyticsMetadata.value).where(AnalyticsMetadata.key == "schema_version")
        )
        if version is None:
            connection.execute(
                AnalyticsMetadata.__table__.insert().values(
                    key="schema_version", value=ANALYTICS_SCHEMA_VERSION
                )
            )
        elif version != ANALYTICS_SCHEMA_VERSION:
            raise AnalyticsSchemaError(
                f"Unsupported analytics schema version {version!r}; "
                f"expected {ANALYTICS_SCHEMA_VERSION!r}."
            )


class AnalyticsService:
    def __init__(self, engine: Engine, retention_days: int) -> None:
        self.engine = engine
        self.session_factory = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False
        )
        self.retention_days = retention_days
        self._cleanup_lock = threading.Lock()
        self._last_cleanup_day: date | None = None

    def maybe_cleanup(self, now: datetime | None = None) -> None:
        current = now or utc_now()
        if self._last_cleanup_day == current.date():
            return
        with self._cleanup_lock:
            if self._last_cleanup_day == current.date():
                return
            self._last_cleanup_day = current.date()
            try:
                self.cleanup(current)
            except Exception:  # analytics cleanup must never interrupt request handling
                logger.exception("Analytics retention cleanup failed")

    def cleanup(self, now: datetime | None = None) -> None:
        cutoff = (now or utc_now()) - timedelta(days=self.retention_days)
        with self.session_factory.begin() as session:
            session.execute(delete(VisitEvent).where(VisitEvent.occurred_at < cutoff))
            session.execute(delete(VisitSession).where(VisitSession.last_seen_at < cutoff))

    def start_session(
        self,
        cookie_value: str | None,
        *,
        automated: bool,
        now: datetime | None = None,
    ) -> SessionResult:
        current = now or utc_now()
        self.maybe_cleanup(current)
        supplied_session_id = _normalize_session_id(cookie_value)
        with self.session_factory.begin() as session:
            visit = session.get(VisitSession, supplied_session_id) if supplied_session_id else None
            set_cookie = visit is None
            if visit is None:
                visit = VisitSession(
                    session_id=str(uuid4()),
                    started_at=current,
                    last_seen_at=current,
                    counted=not automated,
                    initial_automated=automated,
                )
                session.add(visit)
                if not automated:
                    _increment_daily(session, current.date(), "visit_count")
            else:
                visit.last_seen_at = current
                if not automated and not visit.counted:
                    visit.counted = True
                    _increment_daily(session, current.date(), "visit_count")
        return SessionResult(visit.session_id, set_cookie)

    def record_event(
        self,
        *,
        session_id: str,
        event_type: str,
        route_name: str,
        path: str,
        details: dict[str, Any],
        ip_address: str | None,
        user_agent: str | None,
        referrer: str | None,
        status_code: int,
        duration_ms: int,
        automated: bool,
        now: datetime | None = None,
    ) -> None:
        if event_type not in EVENT_METRICS:
            raise ValueError(f"Unknown analytics event type: {event_type}")
        current = now or utc_now()
        with self.session_factory.begin() as session:
            visit = session.get(VisitSession, session_id)
            if visit is None:
                raise ValueError(f"Unknown analytics session: {session_id}")
            visit.last_seen_at = current
            session.add(
                VisitEvent(
                    session_id=session_id,
                    occurred_at=current,
                    event_type=event_type,
                    route_name=route_name[:100],
                    path=path[:2048],
                    details=details,
                    ip_address=ip_address[:45] if ip_address else None,
                    user_agent=user_agent[:1024] if user_agent else None,
                    referrer=referrer[:2048] if referrer else None,
                    status_code=status_code,
                    duration_ms=max(0, duration_ms),
                    automated=automated,
                )
            )
            metric = "automated_event_count" if automated else EVENT_METRICS[event_type]
            _increment_daily(session, current.date(), metric)

    def total_visits(self) -> int:
        with self.session_factory() as session:
            return int(
                session.scalar(select(func.coalesce(func.sum(DailyAnalytics.visit_count), 0)))
                or 0
            )

    def summary(self, start: date, end: date) -> dict[str, Any]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(DailyAnalytics)
                .where(DailyAnalytics.day.between(start, end))
                .order_by(DailyAnalytics.day)
            ).all()
        daily = [self._daily_record(row) for row in rows]
        total_keys = (
            "visits",
            "page_views",
            "searches",
            "database_details",
            "challenge_details",
            "downloads",
            "automated_events",
        )
        return {
            "from": start.isoformat(),
            "to": end.isoformat(),
            "lifetime_visits": self.total_visits(),
            "totals": {key: sum(day[key] for day in daily) for key in total_keys},
            "daily": daily,
        }

    @staticmethod
    def _daily_record(row: DailyAnalytics) -> dict[str, Any]:
        return {
            "day": row.day.isoformat(),
            "visits": row.visit_count,
            "page_views": row.page_view_count,
            "searches": row.search_count,
            "database_details": row.database_detail_count,
            "challenge_details": row.challenge_detail_count,
            "downloads": row.download_count,
            "automated_events": row.automated_event_count,
        }

    def event_records(self, start: date, end: date) -> Iterator[dict[str, Any]]:
        lower = datetime.combine(start, time.min)
        upper = datetime.combine(end + timedelta(days=1), time.min)
        with self.session_factory() as session:
            events = session.scalars(
                select(VisitEvent)
                .where(VisitEvent.occurred_at >= lower, VisitEvent.occurred_at < upper)
                .order_by(VisitEvent.occurred_at, VisitEvent.id)
                .execution_options(yield_per=500)
            )
            for visit_event in events:
                yield {
                    "id": visit_event.id,
                    "occurred_at": f"{visit_event.occurred_at.isoformat(timespec='milliseconds')}Z",
                    "session_id": visit_event.session_id,
                    "event_type": visit_event.event_type,
                    "route_name": visit_event.route_name,
                    "path": visit_event.path,
                    "details": visit_event.details,
                    "ip_address": visit_event.ip_address,
                    "user_agent": visit_event.user_agent,
                    "referrer": visit_event.referrer,
                    "status_code": visit_event.status_code,
                    "duration_ms": visit_event.duration_ms,
                    "automated": visit_event.automated,
                }


def create_analytics_service(database_path: Path, retention_days: int) -> AnalyticsService:
    engine = create_analytics_engine(database_path)
    try:
        initialize_analytics_database(engine)
        service = AnalyticsService(engine, retention_days)
        service.maybe_cleanup()
        return service
    except Exception:
        engine.dispose()
        raise
