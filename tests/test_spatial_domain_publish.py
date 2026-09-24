from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import httpx
import pytest
from test_spatial_domain_visualization import generation

from iscdc import spatial_domain_publish as publisher
from iscdc.config import Settings
from iscdc.spatial_domain_visualization import load_spatial_domain_visualization, publish_generation


def quiet(*args, **kwargs):
    pass


def test_spatialglue_partial_release_preserves_rna_and_other_combinations(
    completed, tmp_path, monkeypatch
):
    from test_spatialglue import combination_generation

    from iscdc import spatialglue_config as config
    from iscdc import spatialglue_publish as glue
    from iscdc.spatial_domain_visualization import publish_method_failure

    root, settings, record = completed
    record["modalities"] = {
        m: {"value_type": "counts", "n_obs": 3, "n_vars": 2}
        for m in ("rna", "protein", "atac", "histone")
    }
    monkeypatch.setattr(glue, "catalogue_records", lambda _: [copy.deepcopy(record)])
    monkeypatch.setattr(glue, "frozen_adapter_sha256", lambda _: "c" * 64)
    lock = root / "code/annotation/spatial_domain/gpu/requirements.lock.txt"
    lock.parent.mkdir(parents=True)
    lock.write_text("locked fixture")
    jobs = []
    (root / "configs").mkdir()
    for index, modalities in enumerate(config.expand_combinations(record, config.DEFAULTS)):
        combination = config.combination_id(modalities)
        row = {
            "dataset_id": "example",
            "combination_id": combination,
            "modalities": modalities,
            "source_sha256": record["sha256"],
        }
        jobs.append(row)
        (root / "configs" / f"{index:04d}.yaml").write_text("defaults: {}\n")
        if index == 3:
            publish_method_failure(
                root / "sidecars",
                "example",
                "failed input",
                method_family="spatialglue",
                combination_id=combination,
            )
            continue
        _, manifest, files = combination_generation(record, modalities)
        manifest["provenance"].update(
            environment_lock_sha256=publisher.ct._file_digest(lock)[1],
            parameters=config.load_parameters(None, "example", modalities=modalities),
        )
        publish_generation(
            root / "sidecars",
            record,
            manifest,
            files,
            method_family="spatialglue",
            combination_id=combination,
        )
    publisher.ct._atomic_json(root / "tasks.json", {"jobs": jobs, "excluded": []})
    publisher.ct._atomic_json(root / "run_state.json", {"state": "incomplete"})
    publisher.ct._atomic_json(
        root / "plan.json",
        {
            "total": 4,
            "files": {
                "tasks.json": publisher.ct._file_digest(root / "tasks.json")[1],
                "catalog.db": publisher.ct._file_digest(root / "catalog.db")[1],
            },
        },
    )
    with pytest.raises(ValueError, match="尚未完成"):
        glue.audit_run(root, settings, quiet)
    audit, records = glue.audit_run(root, settings, quiet, completed_only=True)
    assert len(audit["selected"]) == 3
    assert audit["deferred"][0]["combination_id"] == jobs[-1]["combination_id"]
    target, release = tmp_path / "public", tmp_path / "release"
    import shutil

    shutil.copytree(root / "sidecars", target)
    before = publisher.tree_hashes(target)
    release.mkdir()
    stage = publisher.stage_release(root, target, release, audit, records, quiet)
    assert publisher.tree_hashes(stage) == before
    assert load_spatial_domain_visualization(stage, record).generation_id == "run-1"
    assert len({str(publisher.result_directory(stage, item)) for item in audit["selected"]}) == 3
    (settings.data_root / "example/dataset.h5mu").write_bytes(b"changed")
    with pytest.raises(ValueError, match="源 H5MU 摘要"):
        glue.audit_run(root, settings, quiet, completed_only=True)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    (root / "catalog.db").write_bytes(b"frozen catalogue")
    (root / "code").mkdir()
    (root / "code/adapter.py").write_text("version = 1\n")
    source = tmp_path / "datasets/example/dataset.h5mu"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source input bytes")
    record, _, _ = generation()
    record.update(
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        storage_dir="example",
        modalities={"rna": {"n_obs": 3, "n_vars": 2, "value_type": "counts"}},
    )
    record, manifest, files = generation(record)
    publish_generation(root / "sidecars", record, manifest, files)
    jobs = [
        {
            "dataset_id": "example",
            "state": "success",
            "generation_id": "run-1",
            "source_sha256": record["sha256"],
        }
    ]
    publisher.ct._atomic_json(root / "progress.json", {"state": "complete", "jobs": jobs})
    publisher.ct._atomic_json(
        root / "plan.json",
        {
            "jobs": jobs,
            "catalogue_sha256": publisher.ct._file_digest(root / "catalog.db")[1],
            "code_sha256": {"adapter.py": publisher.ct._file_digest(root / "code/adapter.py")[1]},
        },
    )
    settings = replace(
        Settings.from_environment(),
        database_path=tmp_path / "current.db",
        data_root=tmp_path / "datasets",
        spatial_domain_visualization_root=tmp_path / "public",
    )
    monkeypatch.setattr(publisher, "catalogue_records", lambda settings: [copy.deepcopy(record)])
    monkeypatch.setattr(publisher, "resource_lock_path", lambda: tmp_path / "resource.lock")
    return root, settings, record


def test_completed_audit_and_check_only_preserve_inputs(completed, tmp_path, monkeypatch):
    root, settings, record = completed
    before = publisher.tree_hashes(root)
    monkeypatch.setattr(
        publisher.subprocess, "run", lambda *a, **kw: pytest.fail("no service action")
    )
    publisher.execute(root, settings, tmp_path / "public", tmp_path / "report", check_only=True)
    audit = publisher.read_json(tmp_path / "report/audit.json")
    assert audit["selected"][0]["generation_id"] == "run-1"
    assert audit["skipped"] == []
    assert not (tmp_path / "public").exists()
    after = publisher.tree_hashes(root)
    after.pop(".supervisor.lock")
    after.pop("sidecars/example/.generation.lock")
    assert after == before
    assert (
        publisher.ct._file_digest(settings.data_root / "example/dataset.h5mu")[1]
        == record["sha256"]
    )


@pytest.mark.parametrize(
    "fault", ["running", "failed", "skipped", "source", "artifact", "code", "catalogue"]
)
def test_audit_refuses_unfinished_failed_or_stale_results(completed, tmp_path, monkeypatch, fault):
    root, settings, record = completed
    progress = publisher.read_json(root / "progress.json")
    if fault == "running":
        progress["state"] = "running"
    elif fault in {"failed", "skipped"}:
        progress["jobs"][0]["state"] = fault
    elif fault == "source":
        (settings.data_root / "example/dataset.h5mu").write_bytes(b"changed")
    elif fault == "artifact":
        path = root / "sidecars/example/generations/run-1/points/sample_0.bin"
        path.write_bytes(b"bad")
    elif fault == "code":
        (root / "code/adapter.py").write_text("version = 2\n")
    else:
        changed = {**record, "sha256": "b" * 64}
        monkeypatch.setattr(
            publisher,
            "catalogue_records",
            lambda s: [record if s.database_path == root / "catalog.db" else changed],
        )
    publisher.ct._atomic_json(root / "progress.json", progress)
    monkeypatch.setattr(
        publisher.subprocess, "run", lambda *a, **kw: pytest.fail("no service action")
    )
    with pytest.raises(ValueError):
        publisher.execute(root, settings, tmp_path / "public", tmp_path / "report")
    assert not (tmp_path / "public").exists()


def test_eligibility_skips_are_recorded_without_blocking(completed, monkeypatch):
    root, settings, record = completed
    skipped = {**record, "dataset_id": "no_rna", "modalities": {}}
    monkeypatch.setattr(publisher, "catalogue_records", lambda settings: [record, skipped])
    for name in ("plan.json", "progress.json"):
        value = publisher.read_json(root / name)
        value["jobs"].append({"dataset_id": "no_rna", "state": "skipped", "reason": "missing_rna"})
        publisher.ct._atomic_json(root / name, value)
    audit, _ = publisher.audit_run(root, settings, quiet)
    assert audit["skipped"] == [{"dataset_id": "no_rna", "reason": "missing_rna"}]


def test_staging_preserves_previous_outputs_and_detects_generation_conflicts(completed, tmp_path):
    root, settings, record = completed
    target, release = tmp_path / "public", tmp_path / "release"
    target.mkdir()
    (target / "unrelated.txt").write_text("keep existing results")
    release.mkdir()
    audit, records = publisher.audit_run(root, settings, quiet)
    stage = publisher.stage_release(root, target, release, audit, records, quiet)
    assert (stage / "unrelated.txt").read_text() == "keep existing results"
    assert load_spatial_domain_visualization(stage, record).generation_id == "run-1"
    assert not (target / "example").exists()
    (stage / "example/generations/run-1/report.json").write_text("corrupt old generation")
    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(ValueError, match="generation 冲突"):
        publisher.stage_release(root, stage, second, audit, records, quiet)


@pytest.mark.parametrize("kind", ["absent", "directory", "symlink"])
@pytest.mark.parametrize("failure", [None, "http", "start", "interrupt", "rollback_stop"])
def test_switch_and_rollback_preserve_previous_binding(tmp_path, kind, failure):
    target, release = tmp_path / "public", tmp_path / "release"
    release.mkdir()
    stage = release / "sidecars"
    stage.mkdir()
    (stage / "result").write_text("new")
    old = tmp_path / "old"
    old.mkdir()
    (old / "result").write_text("old")
    if kind == "directory":
        target.mkdir()
        (target / "result").write_text("old")
    elif kind == "symlink":
        target.symlink_to("old", target_is_directory=True)
    actions = []

    def service(action):
        actions.append(action)
        if action == "start" and failure == "start" and actions.count("start") == 1:
            raise RuntimeError("start failed")
        if action == "stop" and failure == "rollback_stop" and actions.count("stop") == 2:
            raise RuntimeError("no healthy service to stop")

    def verify():
        assert (target / "result").read_text() == "new"
        if failure in {"http", "rollback_stop"}:
            raise ValueError("HTTP failed")
        if failure == "interrupt":
            raise KeyboardInterrupt()

    if failure:
        with pytest.raises((ValueError, RuntimeError, KeyboardInterrupt)):
            publisher.switch_release(target, stage, release, service, verify, quiet)
        assert publisher.read_json(release / "transaction.json")["state"] == "rolled_back"
        if kind == "absent":
            assert not target.exists()
        else:
            assert (target / "result").read_text() == "old"
        assert target.is_symlink() == (kind == "symlink")
        assert actions[-2:] == ["start", "status"]
    else:
        publisher.switch_release(target, stage, release, service, verify, quiet)
        assert target.is_symlink() and (target / "result").read_text() == "new"
        assert actions == ["status", "stop", "start"]
        assert publisher.read_json(release / "transaction.json")["state"] == "published"
        if kind == "directory":
            assert (release / "previous/result").read_text() == "old"
    assert (stage / "result").read_text() == "new"


@pytest.mark.parametrize("fault", [None, "page", "point", "asset"])
def test_http_verification_checks_generation_pages_and_exact_point_bytes(
    completed, monkeypatch, fault
):
    root, settings, _ = completed
    audit, _ = publisher.audit_run(root, settings, quiet)
    calls = []

    def request(req):
        calls.append(req.url.path)
        if req.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        if req.url.path.startswith("/static/"):
            content = (
                publisher.PROJECT_ROOT / "assets/static/cell_type_visualization.js"
            ).read_bytes()
            return httpx.Response(200, content=b"wrong bundle" if fault == "asset" else content)
        if req.url.path == "/databases/example":
            config = {
                "views": [
                    {
                        "kind": "spatial_domain",
                        "datasetId": "example",
                        "generationId": "wrong" if fault == "page" else "run-1",
                        "samples": [{"key": "sample_0"}],
                    }
                ]
            }
            return httpx.Response(
                200,
                text='<script id="cell-type-visualization-config">'
                + json.dumps(config)
                + "</script>",
            )
        assert req.headers["accept-encoding"] == "identity"
        content = (root / "sidecars/example/generations/run-1/points/sample_0.bin").read_bytes()
        return httpx.Response(
            200,
            content=b"corrupt" if fault == "point" else content,
            headers={"content-type": publisher.POINT_MEDIA_TYPE},
        )

    with httpx.Client(
        transport=httpx.MockTransport(request), base_url="http://127.0.0.1"
    ) as client:
        if fault:
            with pytest.raises(ValueError):
                publisher.verify_http(client, audit["selected"], quiet)
        else:
            publisher.verify_http(client, audit["selected"], quiet)
            assert len(calls) == 4


def test_execute_connects_audit_staging_deployment_and_verification(
    completed, tmp_path, monkeypatch
):
    root, settings, record = completed
    target, report = tmp_path / "public", tmp_path / "report"
    commands, verified = [], []

    def run(command, **kwargs):
        commands.append(command[-1])
        assert kwargs["env"]["ISCDC_SPATIAL_DOMAIN_VISUALIZATION_ROOT"] == str(target)
        assert kwargs["env"]["ISCDC_DATABASE_PATH"] == str(settings.database_path)
        assert kwargs["env"]["ISCDC_DEPLOY_START_TIMEOUT"] == "600"

    def verify(client, items, log):
        verified.append(load_spatial_domain_visualization(target, record).generation_id)

    monkeypatch.setattr(publisher.subprocess, "run", run)
    monkeypatch.setattr(publisher, "verify_http", verify)
    publisher.execute(root, settings, target, report)
    assert commands == ["status", "stop", "start"] and verified == ["run-1"]
    assert publisher.read_json(report / "published.json")["successful"] == 1
    assert target.is_symlink()


def test_publication_refuses_an_owned_supervisor_lock(completed, tmp_path):
    import fcntl

    root, settings, _ = completed
    with (root / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(BlockingIOError):
            publisher.execute(root, settings, tmp_path / "public", tmp_path / "report")
    assert not (tmp_path / "public").exists()


def test_disconnected_stdout_still_records_persistent_audit(completed, tmp_path, monkeypatch):
    root, settings, _ = completed

    def disconnected(*args, **kwargs):
        raise BrokenPipeError("terminal disconnected")

    monkeypatch.setattr(publisher, "print", disconnected, raising=False)
    publisher.execute(root, settings, tmp_path / "public", tmp_path / "report", check_only=True)
    events = [
        json.loads(line) for line in (tmp_path / "report/events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "check_only_complete"


@pytest.mark.parametrize("state", ["pending", "running", "failed", "interrupted"])
def test_partial_publication_runs_alongside_supervisor_and_preserves_deferred_results(
    completed, tmp_path, monkeypatch, state
):
    import fcntl

    root, settings, record = completed
    other = {**record, "dataset_id": "unfinished"}
    monkeypatch.setattr(publisher, "catalogue_records", lambda s: [record, other])
    for file in ("progress.json", "plan.json"):
        doc = publisher.read_json(root / file)
        doc["state"] = "running"
        doc["jobs"].append({"dataset_id": "unfinished", "state": state, "reason": "deferred"})
        publisher.ct._atomic_json(root / file, doc)
    target = tmp_path / "public"
    target.mkdir()
    (target / "unfinished").mkdir()
    (target / "unfinished/old.txt").write_text("keep old result")
    commands = []
    monkeypatch.setattr(
        publisher.subprocess, "run", lambda command, **kw: commands.append(command[-1])
    )
    monkeypatch.setattr(
        publisher,
        "verify_http",
        lambda client, selected, log: (
            selected[0]["dataset_id"] == "example" or pytest.fail("wrong selection")
        ),
    )
    with (
        (root / ".supervisor.lock").open("a") as supervisor,
        publisher.resource_lock_path().open("a") as lease,
    ):
        fcntl.flock(supervisor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publisher.execute(root, settings, target, tmp_path / "report", completed_only=True)
    assert commands == ["status", "stop", "start"]
    result = publisher.read_json(tmp_path / "report/published.json")
    assert result["successful"] == 1 and result["completed_only"]
    assert result["deferred"] == [
        {"dataset_id": "unfinished", "state": state, "reason": "deferred"}
    ]
    assert (target / "unfinished/old.txt").read_text() == "keep old result"
    assert publisher.read_json(root / "progress.json")["state"] == "running"


def test_partial_cohort_is_frozen_and_does_not_pick_up_later_results(completed, monkeypatch):
    root, settings, record = completed
    other = {**record, "dataset_id": "unfinished"}
    monkeypatch.setattr(publisher, "catalogue_records", lambda s: [record, other])
    snapshot = publisher.read_json(root / "progress.json")
    snapshot["state"] = "running"
    snapshot["jobs"].append({"dataset_id": "unfinished", "state": "running"})
    plan = publisher.read_json(root / "plan.json")
    plan["jobs"] = snapshot["jobs"]
    publisher.ct._atomic_json(root / "plan.json", plan)
    latest = copy.deepcopy(snapshot)
    latest["jobs"][1]["state"] = "success"  # No sidecar: would fail if the cohort were reread.
    publisher.ct._atomic_json(root / "progress.json", latest)
    audit, _ = publisher.audit_run(root, settings, quiet, completed_only=True, progress=snapshot)
    assert len(audit["selected"]) == 1 and audit["deferred"][0]["state"] == "running"


def test_partial_publication_refuses_concurrent_rewrite_of_selected_generation(completed, tmp_path):
    import fcntl

    root, settings, _ = completed
    with (root / "sidecars/example/.generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            publisher.execute(
                root, settings, tmp_path / "public", tmp_path / "report", completed_only=True
            )
    assert not (tmp_path / "public").exists()
