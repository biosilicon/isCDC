from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta

import httpx
import pytest
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select

from iscdc.analytics import (
    SESSION_COOKIE_NAME,
    DailyAnalytics,
    VisitEvent,
    VisitSession,
    create_analytics_service,
    utc_now,
)
from iscdc.app import create_app
from iscdc.cli import main
from iscdc.importer import import_dataset

HUMAN_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"


def test_footer_tolerates_visit_count_missing_from_an_old_application(settings):
    class TemplateRequest:
        @staticmethod
        def url_for(_name, **_path_params):  # noqa: ANN001, ANN003, ANN202
            return "/"

    templates = Jinja2Templates(directory=settings.templates_dir)
    rendered = templates.get_template("index.html").render(
        request=TemplateRequest(), database_count=0, challenge_count=0
    )

    assert "Visits:" in rendered
    assert "unavailable" in rendered


@pytest.mark.anyio
async def test_browser_session_is_counted_once_and_shown_in_every_footer(settings):
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"User-Agent": HUMAN_USER_AGENT},
    ) as client:
        first = await client.get("/")
        assert first.status_code == 200
        assert "Visits: 1" in first.text
        assert SESSION_COOKIE_NAME in client.cookies
        cookie_header = first.headers["set-cookie"]
        assert "HttpOnly" in cookie_header
        assert "SameSite=lax" in cookie_header
        assert "Max-Age" not in cookie_header
        assert "expires=" not in cookie_header.lower()

        second = await client.get("/databases")
        assert second.status_code == 200
        assert "Visits: 1" in second.text
        assert "set-cookie" not in second.headers

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"User-Agent": HUMAN_USER_AGENT},
    ) as another_client:
        third = await another_client.get("/")
        assert "Visits: 2" in third.text

    analytics = app.state.analytics
    assert analytics is not None
    assert analytics.total_visits() == 2
    with analytics.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(VisitSession)) == 2
        assert session.scalar(select(func.count()).select_from(VisitEvent)) == 3


@pytest.mark.anyio
async def test_core_behaviors_are_recorded_but_api_and_failed_requests_are_not(
    settings, write_h5mu, write_metadata
):
    import_dataset(write_h5mu(), write_metadata(), settings)
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={
            "User-Agent": HUMAN_USER_AGENT,
            "Referer": "https://example.test/discovery?source=lab",
        },
    ) as client:
        assert (await client.get("/")).status_code == 200
        assert (
            await client.get(
                "/databases",
                params={"q": "kidney", "organism": "Homo sapiens", "page": 1},
            )
        ).status_code == 200
        assert (await client.get("/databases/test_rna_protein")).status_code == 200
        assert (await client.get("/downloads/test_rna_protein/metadata")).status_code == 200
        assert (await client.get("/api/databases")).status_code == 200
        assert (await client.get("/databases/unknown")).status_code == 404
        assert (await client.get("/databases?page=0")).status_code == 422
        assert (await client.get("/healthz")).status_code == 200

    analytics = app.state.analytics
    assert analytics is not None
    with analytics.session_factory() as session:
        events = session.scalars(select(VisitEvent).order_by(VisitEvent.id)).all()
    assert [event.event_type for event in events] == [
        "page_view",
        "catalogue_search",
        "database_detail_view",
        "download",
    ]
    assert events[1].details == {
        "catalogue": "databases",
        "filters": {"q": "kidney", "organism": "Homo sapiens"},
        "page": 1,
    }
    assert events[2].details == {"dataset_id": "test_rna_protein"}
    assert events[3].details == {"dataset_id": "test_rna_protein", "kind": "metadata"}
    assert all(event.referrer == "https://example.test/discovery?source=lab" for event in events)


@pytest.mark.anyio
async def test_automated_requests_are_logged_without_counting_and_forwarded_ip_is_ignored(
    settings,
):
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app, client=("192.0.2.44", 4321))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"User-Agent": "Googlebot/2.1", "X-Forwarded-For": "203.0.113.8"},
    ) as client:
        health = await client.get("/healthz")
        assert "set-cookie" not in health.headers
        page = await client.get("/")
        assert "Visits: 0" in page.text
        assert SESSION_COOKIE_NAME in client.cookies
        assert (await client.get("/api/databases")).status_code == 200

    analytics = app.state.analytics
    assert analytics is not None
    assert analytics.total_visits() == 0
    with analytics.session_factory() as session:
        events = session.scalars(select(VisitEvent)).all()
        daily = session.scalars(select(DailyAnalytics)).one()
    assert len(events) == 1
    assert events[0].automated is True
    assert events[0].ip_address == "192.0.2.44"
    assert daily.automated_event_count == 1
    assert daily.visit_count == 0


def test_retention_deletes_raw_records_but_keeps_permanent_counts(tmp_path):
    service = create_analytics_service(tmp_path / "analytics.db", retention_days=30)
    now = utc_now()
    old = now - timedelta(days=31)
    visit = service.start_session(None, automated=False, now=old)
    service.record_event(
        session_id=visit.session_id,
        event_type="page_view",
        route_name="home",
        path="/",
        details={"page": "home"},
        ip_address="192.0.2.10",
        user_agent=HUMAN_USER_AGENT,
        referrer=None,
        status_code=200,
        duration_ms=4,
        automated=False,
        now=old,
    )

    service.cleanup(now)

    assert service.total_visits() == 1
    with service.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(VisitSession)) == 0
        assert session.scalar(select(func.count()).select_from(VisitEvent)) == 0
        assert session.scalar(select(func.count()).select_from(DailyAnalytics)) == 1
    service.engine.dispose()


def test_analytics_uses_network_filesystem_compatible_journal_mode(tmp_path):
    service = create_analytics_service(tmp_path / "analytics.db", retention_days=30)
    with service.engine.connect() as connection:
        journal_mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar_one()

    assert journal_mode == "delete"
    service.engine.dispose()


@pytest.mark.anyio
async def test_analytics_initialization_failure_does_not_break_the_catalogue(
    tmp_path, settings, caplog
):
    invalid_database_path = tmp_path / "analytics-directory"
    invalid_database_path.mkdir()
    app = create_app(replace(settings, analytics_database_path=invalid_database_path))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"User-Agent": HUMAN_USER_AGENT},
    ) as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert "Visits: unavailable" in response.text
        assert (await client.get("/healthz")).json() == {"status": "ok"}

    assert app.state.analytics is None
    assert "Analytics initialization failed" in caplog.text


@pytest.mark.anyio
async def test_runtime_analytics_failures_do_not_change_successful_responses(
    settings, monkeypatch, caplog
):
    app = create_app(settings)
    analytics = app.state.analytics
    assert analytics is not None

    def fail_event_write(**_kwargs):  # noqa: ANN003, ANN202
        raise RuntimeError("simulated analytics write failure")

    monkeypatch.setattr(analytics, "record_event", fail_event_write)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"User-Agent": HUMAN_USER_AGENT},
    ) as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert "Visits: 1" in response.text

        def fail_counter_read():
            raise RuntimeError("simulated analytics read failure")

        monkeypatch.setattr(analytics, "total_visits", fail_counter_read)
        response = await client.get("/databases")
        assert response.status_code == 200
        assert "Visits: unavailable" in response.text

    assert "Analytics event write failed" in caplog.text
    assert "Analytics counter read failed" in caplog.text


def test_analytics_cli_summarizes_and_exports_csv_and_jsonl(tmp_path, monkeypatch, capsys):
    database_path = tmp_path / "analytics.db"
    monkeypatch.setenv("ISCDC_ANALYTICS_DATABASE_PATH", str(database_path))
    service = create_analytics_service(database_path, retention_days=30)
    now = utc_now()
    visit = service.start_session(None, automated=False, now=now)
    service.record_event(
        session_id=visit.session_id,
        event_type="download",
        route_name="download_file",
        path="/downloads/example/metadata",
        details={"dataset_id": "example", "kind": "metadata"},
        ip_address="198.51.100.7",
        user_agent=HUMAN_USER_AGENT,
        referrer=None,
        status_code=200,
        duration_ms=12,
        automated=False,
        now=now,
    )
    service.engine.dispose()
    day = now.date().isoformat()

    assert main(["analytics", "summary", "--from", day, "--to", day]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["lifetime_visits"] == 1
    assert summary["totals"]["downloads"] == 1

    csv_path = tmp_path / "events.csv"
    assert (
        main(
            [
                "analytics",
                "export",
                "--format",
                "csv",
                "--from",
                day,
                "--to",
                day,
                "--output",
                str(csv_path),
            ]
        )
        == 0
    )
    assert "198.51.100.7" in csv_path.read_text(encoding="utf-8")
    assert main(["analytics", "export", "--format", "csv", "--output", str(csv_path)]) == 1
    assert "already exists" in capsys.readouterr().err

    jsonl_path = tmp_path / "events.jsonl"
    assert (
        main(
            [
                "analytics",
                "export",
                "--format",
                "jsonl",
                "--from",
                day,
                "--to",
                day,
                "--output",
                str(jsonl_path),
            ]
        )
        == 0
    )
    exported = json.loads(jsonl_path.read_text(encoding="utf-8"))
    assert exported["details"] == {"dataset_id": "example", "kind": "metadata"}
