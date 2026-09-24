"""Audit the frozen SpatialGLUE queue for the shared domain release publisher."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone

from . import spatial_domain_publish as publisher
from . import spatialglue_config as config
from .spatial_domain_annotation import catalogue_records


def frozen_adapter_sha256(root):
    digest = hashlib.sha256()
    for name in (
        "spatialglue.py",
        "spatialglue_config.py",
        "spatialglue_resources.py",
        "spatialglue_cli.py",
        "spatialglue_preprocessing.py",
        "spatialglue_graph.py",
        "spatialglue_runtime.py",
        "spatialglue_cache.py",
        "spatial_domain_annotation.py",
        "spatial_domain_visualization.py",
    ):
        digest.update(name.encode())
        digest.update((root / "code/src/iscdc" / name).read_bytes())
    return digest.hexdigest()


def snapshot_progress(root):
    jobs = []
    for row in publisher.read_json(root / "tasks.json")["jobs"]:
        item = {**row, "method_family": "spatialglue"}
        path = publisher.result_directory(root / "sidecars", item) / "status.json"
        status = publisher.read_json(path) if path.exists() else {"state": "pending"}
        reason = None
        if status["state"] == "failure":
            report_path, _, _ = publisher.ct._validate_file_record(
                path.parent, status["report"], "failure report"
            )
            reason = publisher.read_json(report_path).get("error")
        item.update(
            state="failed" if status["state"] == "failure" else status["state"],
            generation_id=status.get("generation_id"),
            reason=reason,
        )
        jobs.append(item)
    return {
        "state": publisher.read_json(root / "run_state.json")["state"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "jobs": jobs,
    }


def audit_run(root, settings, log, *, completed_only=False, progress=None):
    p = publisher
    progress = progress if progress is not None else snapshot_progress(root)
    plan = p.read_json(root / "plan.json")
    if not completed_only:
        p.require(progress["state"] == "complete", "SpatialGLUE 分析尚未完成")
    for relative, expected in plan["files"].items():
        p.require(p.ct._file_digest(root / relative)[1] == expected, f"冻结文件已改变：{relative}")
    document = p.read_json(root / "tasks.json")
    frozen = catalogue_records(replace(settings, database_path=root / "catalog.db"))
    current = catalogue_records(settings)
    p.require(
        [p.binding(r) for r in frozen] == [p.binding(r) for r in current],
        "当前目录的来源或科学元数据已改变",
    )
    records = {r["dataset_id"]: r for r in current}
    jobs = progress["jobs"]
    identities = [(j["dataset_id"], j["combination_id"]) for j in jobs]
    p.require(
        len(jobs) == plan["total"] and len(set(identities)) == len(jobs),
        "SpatialGLUE 任务数量或组合身份不一致",
    )
    p.require(
        identities == [(j["dataset_id"], j["combination_id"]) for j in document["jobs"]],
        "SpatialGLUE 任务清单已改变",
    )
    adapter = frozen_adapter_sha256(root)
    lock_hash = p.ct._file_digest(
        root / "code/annotation/spatial_domain/gpu/requirements.lock.txt"
    )[1]
    selected, deferred, sources = [], [], {}
    for index, job in enumerate(jobs):
        name = job["dataset_id"]
        if job["state"] != "success":
            p.require(
                completed_only and job["state"] in {"pending", "running", "failed", "interrupted"},
                f"SpatialGLUE 任务未成功：{name}/{job['combination_id']}",
            )
            deferred.append(
                {k: job.get(k) for k in ("dataset_id", "combination_id", "state", "reason")}
            )
            continue
        record = records[name]
        expected = config.load_parameters(
            root / "configs" / f"{index:04d}.yaml", name, modalities=job["modalities"]
        )
        config.eligibility(record, expected)
        p.require(
            config.combination_id(job["modalities"]) == job["combination_id"], "组合身份不一致"
        )
        p.require(record["sha256"] == job["source_sha256"], f"队列来源摘要不一致：{name}")
        log(
            "audit_dataset",
            index=index + 1,
            total=len(jobs),
            dataset_id=name,
            combination_id=job["combination_id"],
        )
        source = settings.data_root / record["storage_dir"] / "dataset.h5mu"
        if name not in sources:
            before = p.signature(source)
            p.require(p.ct._file_digest(source)[1] == record["sha256"], f"源 H5MU 摘要不符：{name}")
            p.require(p.signature(source) == before, f"源文件在审计中改变：{name}")
            sources[name] = before
        snapshot = p.load_result(root / "sidecars", record, job)
        provenance = snapshot.manifest["provenance"]
        p.require(snapshot.generation_id == job["generation_id"], f"成功 generation 已改变：{name}")
        p.require(
            provenance["adapter_sha256"] == adapter
            and provenance["environment_lock_sha256"] == lock_hash
            and provenance["parameters"] == expected,
            f"计算代码、环境或参数与冻结计划不一致：{name}",
        )
        selected.append(
            {
                "dataset_id": name,
                "method_family": "spatialglue",
                "combination_id": job["combination_id"],
                "generation_id": snapshot.generation_id,
                "source_path": str(source),
                "source_signature": sources[name],
                "status_sha256": p.ct._file_digest(
                    p.result_directory(root / "sidecars", job) / "status.json"
                )[1],
                "manifest": dict(snapshot.manifest),
            }
        )
    p.require(selected, "没有可发布结果")
    log(
        "audit_passed",
        successful=len(selected),
        datasets=len(sources),
        deferred=len(deferred),
        skipped=len(document["excluded"]),
    )
    return {
        "selected": selected,
        "skipped": document["excluded"],
        "deferred": deferred,
        "completed_only": completed_only,
        "progress_snapshot_at": progress["updated_at"],
        "catalogue": [p.binding(r) for r in current],
    }, records
