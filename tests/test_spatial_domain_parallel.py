from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path

import pytest

from iscdc import spatial_domain_parallel as parallel
from iscdc import spatial_domain_resources as resources


def job(name, cpu=8, memory=8, work=100, state="pending"):
    return {
        "dataset_id": name,
        "method": "GraphST",
        "state": state,
        "work_units": work,
        "resources": {"cpu_count": cpu, "memory_gb": memory},
    }


def test_packing_respects_all_limits_and_keeps_room_for_light_jobs():
    jobs = [job("large", 16, 112, 9999), job("second_large", 16, 79, 9998)]
    jobs += [job(str(i), 8, 8, 1000 - i) for i in range(12)]
    selected = parallel.choose_jobs(jobs, {}, 64, 192, 8)
    assert selected[0]["dataset_id"] == "large"
    assert len(selected) == 7
    assert sum(j["resources"]["cpu_count"] for j in selected) == 64
    assert sum(j["resources"]["memory_gb"] for j in selected) <= 192
    active = {"large": {"job": jobs[0]}}
    jobs[0]["state"] = "running"
    selected = parallel.choose_jobs(jobs, active, 48, 192, 3)
    assert len(selected) == 2
    assert all(j["resources"]["memory_gb"] <= 48 for j in selected)


def test_large_jobs_are_not_starved_when_no_light_job_fits():
    jobs = [job("large", 16, 100, 9999), job("large2", 16, 80, 9998)]
    assert len(parallel.choose_jobs(jobs, {}, 64, 192, 8)) == 2
    assert len(parallel.choose_jobs(jobs, {}, 64, 128, 8)) == 1


def test_explicit_repair_retries_have_priority_within_resource_budget():
    large = job("large", cpu=16, work=9999)
    retry = job("retry", cpu=8, work=100)
    retry["attempt_history"] = [{"state": "failed"}]
    assert parallel.choose_jobs([large, retry], {}, 16, 192, 8) == [retry]


def test_eta_accounts_for_memory_serialization():
    jobs = [job("a", 8, 100, 100), job("b", 8, 100, 100)]
    policy = {"max_workers": 8, "cpu_budget": 64, "memory_gb": 192}
    serial = parallel.parallel_progress(jobs, {}, policy, 60)["remaining_seconds_estimate"]
    policy["memory_gb"] = 256
    concurrent = parallel.parallel_progress(jobs, {}, policy, 60)["remaining_seconds_estimate"]
    assert serial == 2 * concurrent


def test_requirements_cover_preflight_and_assign_method_threads():
    r = {"spatial_unit": "single_cell", "modalities": {"rna": {"n_obs": 100000, "n_vars": 3000}}}
    req = parallel.requirements(r, {"n_top_genes": 3000}, 64)
    assert req["cpu_count"] == 16
    assert req["memory_gb"] * 2**30 > req["preflight_estimate_bytes"]
    r["spatial_unit"] = "spot_level"
    assert parallel.requirements(r, {"n_top_genes": 3000}, 64)["cpu_count"] == 8


def test_managed_workers_share_inherited_lease_without_unlocking_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(resources.tempfile, "gettempdir", lambda: str(tmp_path))
    cpus = sorted(os.sched_getaffinity(0))[:2]
    lease_path = resources.resource_lock_path()
    with lease_path.open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = os.environ.copy()
        env.update(
            TMPDIR=str(tmp_path),
            PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
            ISCDC_DOMAIN_CPU_IDS=",".join(map(str, cpus)),
            ISCDC_DOMAIN_RESOURCE_FD=str(lease.fileno()),
        )
        code = (
            "from iscdc.spatial_domain_resources import managed_cpu_ids,resource_guard; "
            "assert len(managed_cpu_ids()) == 2;\n"
            "with resource_guard(): print('lease accepted')"
        )
        children = [
            subprocess.Popen(
                [sys.executable, "-c", code],
                env=env,
                pass_fds=(lease.fileno(),),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        for child in children:
            stdout, stderr = child.communicate(timeout=20)
            assert child.returncode == 0, stderr
            assert "lease accepted" in stdout
        with lease_path.open("a") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_managed_worker_rejects_unrelated_descriptor(tmp_path, monkeypatch):
    monkeypatch.setattr(resources.tempfile, "gettempdir", lambda: str(tmp_path))
    resources.resource_lock_path().touch()
    with (tmp_path / "wrong.lock").open("a") as wrong:
        monkeypatch.setenv("ISCDC_DOMAIN_CPU_IDS", str(min(os.sched_getaffinity(0))))
        monkeypatch.setenv("ISCDC_DOMAIN_RESOURCE_FD", str(wrong.fileno()))
        with pytest.raises(ValueError, match="global resource lock"):
            resources.managed_cpu_ids()


def test_worker_launch_passes_resource_lease_and_preserves_scientific_parameters(
    tmp_path, monkeypatch
):
    import yaml

    from iscdc.spatial_domain_annotation import DEFAULTS

    config = tmp_path / "code/assets/spatial_domain/defaults.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({"defaults": DEFAULTS, "datasets": {}}))
    (tmp_path / "logs").mkdir()
    captured = {}

    def spawn(command, **kwargs):
        captured.update(command=command, **kwargs)
        return object()

    monkeypatch.setattr(parallel.subprocess, "Popen", spawn)
    item_job = job("example", 8, 12)
    item_job["log_path"] = "logs/example.log"
    allocation = [0, 1, 2, 3, 4, 5, 6, 7]
    item = parallel.start_process(
        tmp_path, {"python": sys.executable, "data_root": "/source"}, item_job, allocation, 123
    )
    item["log"].close()
    assert captured["pass_fds"] == (123,)
    assert captured["start_new_session"]
    assert captured["env"]["ISCDC_DOMAIN_WORKER"] == "1"
    assert captured["env"]["OMP_NUM_THREADS"] == "8"
    assert captured["env"]["PYTHONHASHSEED"] == "42"
    assert captured["env"]["ISCDC_DOMAIN_CPU_IDS"] == "0,1,2,3,4,5,6,7"
    actual = parallel.load_parameters(tmp_path / "job_configs/example.yaml", "example")
    assert actual["threads"] == 8 and actual["memory_budget_gb"] == 12
    assert {k: v for k, v in actual.items() if k not in {"threads", "memory_budget_gb"}} == {
        k: v for k, v in DEFAULTS.items() if k not in {"threads", "memory_budget_gb"}
    }


@pytest.mark.parametrize("retry_failed", [False, True])
def test_continuation_preserves_completed_and_failed_jobs(tmp_path, monkeypatch, retry_failed):
    import hashlib
    from types import SimpleNamespace

    import yaml

    previous = tmp_path / "previous"
    previous.mkdir()
    for folder in (
        "sidecars",
        "logs",
        "code/assets/spatial_domain",
        "code/annotation/spatial_domain",
    ):
        (previous / folder).mkdir(parents=True, exist_ok=True)
    (previous / "catalog.db").write_bytes(b"catalogue fixture")
    (previous / "code/assets/spatial_domain/defaults.yaml").write_text(
        yaml.safe_dump({"defaults": {}})
    )
    (previous / "code/annotation/spatial_domain/requirements.lock.txt").write_text("fixture==1\n")
    source = tmp_path / "project/src/iscdc"
    source.mkdir(parents=True)
    (source / "marker.py").write_text("VERSION=2\n")
    monkeypatch.setattr(parallel, "PROJECT_ROOT", tmp_path / "project")
    jobs = [
        job("done", state="success"),
        job("bad", state="failed"),
        job("interrupted", state="interrupted"),
    ]
    for item in jobs:
        item["log_path"] = f"logs/{item['dataset_id']}.log"
        (previous / item["log_path"]).write_text("previous log\n")
    jobs[0]["inference_seconds"] = 10
    jobs[1]["reason"] = "Original failure"
    parallel.batch.atomic_json(previous / "progress.json", {"state": "interrupted", "jobs": jobs})
    parallel.batch.atomic_json(
        previous / "plan.json",
        {
            "python": sys.executable,
            "data_root": "/data",
            "catalogue_sha256": hashlib.sha256(b"catalogue fixture").hexdigest(),
        },
    )
    records = [
        {
            "dataset_id": item["dataset_id"],
            "spatial_unit": "spot_level",
            "modalities": {"rna": {"n_obs": 1000, "n_vars": 3000}},
        }
        for item in jobs
    ]
    monkeypatch.setattr(parallel, "catalogue_records", lambda settings: records)
    monkeypatch.setattr(
        parallel, "load_spatial_domain_visualization", lambda root, record: SimpleNamespace()
    )
    monkeypatch.setattr(parallel.os, "sched_getaffinity", lambda pid: set(range(80)))
    root = tmp_path / "new"
    parallel.continue_run(previous, root, retry_failed=retry_failed)
    state = parallel.batch.read_json(root / "progress.json")
    assert [item["state"] for item in state["jobs"]] == [
        "success",
        "pending" if retry_failed else "failed",
        "pending",
    ]
    if retry_failed:
        assert state["jobs"][1]["attempt_history"][0]["reason"] == "Original failure"
        assert "reason" not in state["jobs"][1]
    else:
        assert state["jobs"][1]["reason"] == "Original failure"
    assert state["jobs"][0]["carried_from"] == str(previous)
    assert (root / "logs/done.log").read_text() == "previous log\n"
    assert state["scheduler"]["cpu_budget"] == 64


def test_eta_does_not_collapse_overdue_jobs_to_one_heartbeat(monkeypatch):
    monkeypatch.setattr(parallel.time, "monotonic", lambda: 3600)
    item = job("slow", state="running")
    active = {"slow": {"job": item, "started": 0}}
    policy = {"max_workers": 8, "cpu_budget": 64, "memory_gb": 192}
    report = parallel.parallel_progress([item], active, policy, 60)
    assert report["remaining_seconds_estimate"] >= 1800
    assert report["eta_overdue_jobs"] == ["slow"]


def test_eta_accounts_for_large_banksy_observation_graph():
    completed = job("reference", state="success")
    completed.update(method="BANKSY", n_obs=70000, inference_seconds=660, work_units=210000000)
    pending = job("large", work=280000000)
    pending.update(method="BANKSY", n_obs=700000)
    policy = {"max_workers": 8, "cpu_budget": 64, "memory_gb": 192}
    report = parallel.parallel_progress([completed, pending], {}, policy, 60)
    assert report["remaining_seconds_estimate"] >= 20000
