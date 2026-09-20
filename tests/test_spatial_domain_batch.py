from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from iscdc import spatial_domain_batch as batch


def test_eta_calibrates_per_method_and_excludes_terminal_jobs():
    jobs = [
        {"state": "reused", "method": "BANKSY", "work_units": 100, "inference_seconds": 120},
        {"state": "running", "method": "BANKSY", "work_units": 200},
        {"state": "pending", "method": "BANKSY", "work_units": 100},
        {"state": "skipped", "method": None},
        {"state": "failed", "method": "GraphST", "work_units": 999999},
    ]
    progress = batch.estimate_progress(jobs, current_elapsed=100, heartbeat=60)
    assert progress["remaining_seconds_estimate"] == 240
    assert progress["completed"] == 3
    assert progress["eta_calibration_samples"] == {"BANKSY": 1, "GraphST": 0}
    assert batch.estimate_progress(jobs, current_elapsed=9999)["remaining_seconds_estimate"] == 180


@pytest.fixture
def prepared(monkeypatch, tmp_path, settings):
    with sqlite3.connect(settings.database_path) as conn:
        conn.execute("CREATE TABLE snapshot_marker (value TEXT)")
        conn.execute("INSERT INTO snapshot_marker VALUES ('read-only source')")
    monkeypatch.setattr(batch.Settings, "from_environment", lambda: settings)
    source = tmp_path / "project"
    (source / "src" / "iscdc").mkdir(parents=True)
    (source / "src" / "iscdc" / "marker.py").write_text("VERSION = 1\n")
    monkeypatch.setattr(batch, "PROJECT_ROOT", source)
    records = [
        {
            "dataset_id": name,
            "spatial_unit": "single_cell",
            "n_obs": 10,
            "sample_ids": ["sample"],
            "sha256": "a" * 64,
            "coordinate_dimensions": 2,
            "modalities": {"rna": {"value_type": "counts", "n_obs": 10, "n_vars": 20}},
        }
        for name in ("first", "second", "no_rna")
    ]
    records[2]["modalities"] = {}
    monkeypatch.setattr(batch, "catalogue_records", lambda settings: records)
    root = tmp_path / "run"
    batch.prepare(root, python=Path(sys.executable))
    return root, records


def test_prepare_freezes_inputs_and_marks_missing_rna(prepared, settings):
    root, records = prepared
    plan = batch.read_json(root / "plan.json")
    assert len(plan["jobs"]) == 3
    excluded = next(j for j in plan["jobs"] if j["dataset_id"] == "no_rna")
    assert excluded["state"] == "skipped"
    assert excluded["reason"] == "missing_rna"
    with sqlite3.connect(root / "catalog.db") as conn:
        conn.execute("UPDATE snapshot_marker SET value='snapshot only'")
    with sqlite3.connect(settings.database_path) as conn:
        assert conn.execute("SELECT value FROM snapshot_marker").fetchone()[0] == "read-only source"
    with pytest.raises(ValueError, match="Frozen catalogue changed"):
        batch.run(root)


def test_queue_continues_after_failure_and_does_not_repeat_complete_jobs(prepared, monkeypatch):
    root, records = prepared
    completed, attempts = set(), []

    def load(root, record):
        if record["dataset_id"] not in completed:
            raise ValueError("No published success")
        return SimpleNamespace(generation_id="run-1", report={"elapsed_seconds": 12})

    def execute(command, env, log_path, heartbeat, on_tick):
        dataset = command[5]
        attempts.append(dataset)
        assert env["ISCDC_DATABASE_PATH"] == str(root / "catalog.db")
        assert env["PYTHONUNBUFFERED"] == "1"
        on_tick({"elapsed_seconds": 1, "rss_bytes": 4096})
        log_path.write_text("fixture worker\n")
        if dataset == "first":
            batch.ct.publish_failure(root / "sidecars", dataset, "Fixture resource limit")
            return 1, 2, 4096
        completed.add(dataset)
        return 0, 3, 8192

    monkeypatch.setattr(batch, "load_spatial_domain_visualization", load)
    monkeypatch.setattr(batch, "execute_job", execute)
    assert batch.run(root) == 0
    state = batch.read_json(root / "progress.json")
    assert state["state"] == "complete"
    assert state["progress"]["counts"] == {"skipped": 1, "failed": 1, "success": 1}
    assert state["progress"]["remaining_seconds_estimate"] == 0
    assert attempts == ["first", "second"]
    assert batch.run(root) == 0
    assert attempts == ["first", "second"]
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    finishes = [e for e in events if e["event"] == "dataset_finished"]
    assert finishes[0]["current"]["returncode"] == 1
    assert finishes[0]["current"]["reason"] == "Fixture resource limit"
    assert finishes[1]["current"]["wall_seconds"] == 3


def test_resume_recovers_atomically_published_success(prepared, monkeypatch):
    root, records = prepared
    state = batch.read_json(root / "progress.json")
    for job in state["jobs"]:
        if job["state"] == "pending":
            job["state"] = "running"
    batch.atomic_json(root / "progress.json", state)
    monkeypatch.setattr(
        batch,
        "load_spatial_domain_visualization",
        lambda root, record: SimpleNamespace(
            generation_id="recovered", report={"elapsed_seconds": 10}
        ),
    )
    monkeypatch.setattr(
        batch, "execute_job", lambda *args: pytest.fail("must not recompute success")
    )
    batch.run(root)
    state = batch.read_json(root / "progress.json")
    assert state["progress"]["counts"] == {"skipped": 1, "success": 2}


def test_snapshot_modification_prevents_execution(prepared):
    root, records = prepared
    (root / "code" / "src" / "iscdc" / "marker.py").write_text("VERSION = 2\n")
    with pytest.raises(ValueError, match="Frozen source changed"):
        batch.run(root)


def test_process_log_heartbeat_and_exit_are_preserved(monkeypatch, tmp_path):
    class Gone(Exception):
        pass

    def no_process(pid):
        raise Gone()

    monkeypatch.setitem(
        sys.modules, "psutil", SimpleNamespace(Process=no_process, NoSuchProcess=Gone)
    )
    ticks = []
    log = tmp_path / "worker.log"
    code, elapsed, peak = batch.execute_job(
        [
            sys.executable,
            "-u",
            "-c",
            "import time; print('stage marker'); time.sleep(.15); raise SystemExit(2)",
        ],
        os.environ.copy(),
        log,
        0.02,
        ticks.append,
    )
    assert code == 2
    assert elapsed >= 0.15
    assert len(ticks) >= 2
    assert any(tick["last_output"] == "stage marker" for tick in ticks)
    assert "END returncode=2" in log.read_text()
