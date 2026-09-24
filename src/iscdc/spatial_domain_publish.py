"""Audit and publish a completed domain run, with service verification and rollback."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import uuid
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

import httpx

from . import cell_type_visualization as ct
from .config import PROJECT_ROOT, Settings
from .spatial_domain_annotation import catalogue_records, eligibility
from .spatial_domain_resources import resource_lock_path
from .spatial_domain_visualization import (
    POINT_MEDIA_TYPE,
    domain_directory,
    load_spatial_domain_visualization,
)


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def binding(record):
    keys = (
        "dataset_id",
        "dataset_type",
        "sha256",
        "n_obs",
        "sample_ids",
        "spatial_unit",
        "coordinate_dimensions",
        "coordinate_unit",
        "modalities",
    )
    return {key: record[key] for key in keys}


def audit_run(root, settings, log, *, completed_only=False, progress=None):
    progress = progress if progress is not None else read_json(root / "progress.json")
    plan = read_json(root / "plan.json")
    if not completed_only:
        require(
            progress["state"] == "complete",
            "分析尚未完成，拒绝发布；请等待 progress.json state=complete",
        )
    jobs = progress["jobs"]
    bad = [j["dataset_id"] for j in jobs if j["state"] not in {"success", "reused", "skipped"}]
    if not completed_only:
        require(not bad, f"仍有失败或未完成任务，拒绝发布：{bad}")
    require(ct._file_digest(root / "catalog.db")[1] == plan["catalogue_sha256"], "冻结目录摘要不符")
    for name, digest in plan["code_sha256"].items():
        require(ct._file_digest(root / "code" / name)[1] == digest, f"冻结代码已改变：{name}")
    frozen = catalogue_records(replace(settings, database_path=root / "catalog.db"))
    current = catalogue_records(settings)
    records = {r["dataset_id"]: r for r in current}
    require(len(jobs) == len({j["dataset_id"] for j in jobs}), "队列有重复 Database ID")
    expected_ids = {r["dataset_id"] for r in frozen}
    require(
        expected_ids
        == set(records)
        == {j["dataset_id"] for j in jobs}
        == {j["dataset_id"] for j in plan["jobs"]},
        "当前目录、冻结目录和任务清单的 Database 集合不一致",
    )
    require(
        [binding(r) for r in frozen] == [binding(r) for r in current],
        "当前目录的来源或科学元数据已改变",
    )
    selected, skipped, deferred = [], [], []
    for index, job in enumerate(jobs, 1):
        record = records[job["dataset_id"]]
        reason = eligibility(record)
        if job["state"] == "skipped":
            require(
                reason is not None and job.get("reason") == reason,
                f"跳过原因不一致：{record['dataset_id']}",
            )
            skipped.append({"dataset_id": record["dataset_id"], "reason": reason})
            continue
        if completed_only and job["state"] not in {"success", "reused"}:
            require(
                job["state"] in {"running", "pending", "failed", "interrupted"},
                f"未知任务状态：{job['state']}",
            )
            deferred.append(
                {
                    "dataset_id": record["dataset_id"],
                    "state": job["state"],
                    "reason": job.get("reason"),
                }
            )
            continue
        name = ct._safe_name(record["dataset_id"], "dataset_id")
        require(reason is None, f"成功任务已不符合输入资格：{name}")
        require(job["source_sha256"] == record["sha256"], f"队列来源摘要不一致：{name}")
        log("audit_dataset", index=index, total=len(jobs), dataset_id=name)
        source = settings.data_root / record["storage_dir"] / "dataset.h5mu"
        before = signature(source)
        require(ct._file_digest(source)[1] == record["sha256"], f"源 H5MU 摘要不符：{name}")
        require(signature(source) == before, f"源文件在审计中改变：{name}")
        snapshot = load_spatial_domain_visualization(root / "sidecars", record)
        require(
            snapshot.generation_id == job["generation_id"], f"队列与成功 generation 不一致：{name}"
        )
        selected.append(
            {
                "dataset_id": name,
                "generation_id": snapshot.generation_id,
                "source_path": str(source),
                "source_signature": before,
                "status_sha256": ct._file_digest(root / "sidecars" / name / "status.json")[1],
                "manifest": dict(snapshot.manifest),
            }
        )
    require(selected, "没有可发布结果")
    log("audit_passed", successful=len(selected), skipped=len(skipped), deferred=len(deferred))
    return {
        "selected": selected,
        "skipped": skipped,
        "deferred": deferred,
        "completed_only": completed_only,
        "progress_snapshot_at": progress.get("updated_at"),
        "catalogue": [binding(r) for r in current],
    }, records


def regular_tree(root):
    """Refuse nested links; the publication root itself may be a release pointer."""
    for path in root.rglob("*"):
        require(not path.is_symlink(), f"结果目录包含嵌套符号链接：{path}")
        require(path.is_file() or path.is_dir(), f"结果目录包含特殊文件：{path}")


def tree_hashes(root):
    return {str(p.relative_to(root)): ct._file_digest(p)[1] for p in root.rglob("*") if p.is_file()}


def result_directory(root, item):
    return domain_directory(
        root,
        item["dataset_id"],
        item.get("method_family", "rna"),
        combination_id=item.get("combination_id"),
    )


def load_result(root, record, item):
    return load_spatial_domain_visualization(
        root,
        record,
        method_family=item.get("method_family", "rna"),
        combination_id=item.get("combination_id"),
    )


def stage_release(root, target, release, audit, records, log):
    source_root, stage = root / "sidecars", release / "sidecars"
    sources = [
        result_directory(source_root, item) / "generations" / item["generation_id"]
        for item in audit["selected"]
    ]
    require(not target.is_symlink() or target.is_dir(), "正式结果指针已损坏")
    require(not target.exists() or target.is_dir(), "正式结果路径不是目录")
    trees = ([target] if target.exists() else []) + sources
    size = 0
    for tree in trees:
        regular_tree(tree)
        size += sum(p.stat().st_size for p in tree.rglob("*") if p.is_file())
    require(shutil.disk_usage(release).free >= size * 1.1 + 64 * 2**20, "发布磁盘空间不足")
    if target.exists():
        shutil.copytree(target, stage)
    else:
        stage.mkdir()
    for index, item in enumerate(audit["selected"], 1):
        name, generation = item["dataset_id"], item["generation_id"]
        log("stage_dataset", index=index, total=len(sources), dataset_id=name)
        source = result_directory(source_root, item) / "generations" / generation
        destination = result_directory(stage, item) / "generations" / generation
        if destination.exists():
            require(
                tree_hashes(source) == tree_hashes(destination),
                f"不可变 generation 冲突：{name}/{generation}",
            )
        else:
            shutil.copytree(source, destination)
        status = result_directory(source_root, item) / "status.json"
        payload = status.read_bytes()
        require(
            hashlib.sha256(payload).hexdigest() == item["status_sha256"],
            f"分析结果在发布中改变：{name}",
        )
        (result_directory(stage, item) / "status.json").write_bytes(payload)
        load_result(stage, records[name], item)
    log("staging_verified", directory=str(stage), datasets=len(sources))
    return stage


class PageConfig(HTMLParser):
    def __init__(self):
        super().__init__()
        self.capture, self.parts = False, []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.capture = dict(attrs).get("id") == "cell-type-visualization-config"

    def handle_endtag(self, tag):
        if tag == "script":
            self.capture = False

    def handle_data(self, data):
        if self.capture:
            self.parts.append(data)


def verify_http(client, selected, log):
    health = client.get("/healthz")
    health.raise_for_status()
    require(health.json().get("status") == "ok", "网站健康检查失败")
    asset = client.get("/static/cell_type_visualization.js")
    asset.raise_for_status()
    require(
        hashlib.sha256(asset.content).hexdigest()
        == ct._file_digest(PROJECT_ROOT / "assets/static/cell_type_visualization.js")[1],
        "网站前端 bundle 与本地版本不一致",
    )
    for index, item in enumerate(selected, 1):
        name, generation = item["dataset_id"], item["generation_id"]
        path = f"/databases/{quote(name, safe='')}"
        response = client.get(path)
        response.raise_for_status()
        parser = PageConfig()
        parser.feed(response.text)
        config = json.loads("".join(parser.parts))
        family = item.get("method_family", "rna")
        combination = item.get("combination_id")
        views = [
            view
            for view in config["views"]
            if view["kind"] == "spatial_domain"
            and view.get("methodFamily", "spatial_domain")
            == ("spatialglue" if family == "spatialglue" else "spatial_domain")
            and (not combination or view.get("viewId") == f"spatialglue:{combination}")
        ]
        require(
            len(views) == 1 and views[0]["generationId"] == generation, f"页面未加载新结果：{name}"
        )
        require(views[0]["datasetId"] == name, f"页面 Dataset 不一致：{name}")
        require(
            {s["key"] for s in views[0]["samples"]}
            == {s["key"] for s in item["manifest"]["samples"]},
            f"页面样本不完整：{name}",
        )
        for sample in item["manifest"]["samples"]:
            url = f"{path}/spatial-domain-visualization/{generation}/{sample['key']}"
            if family == "spatialglue":
                url = (
                    f"{path}/spatial-domain-visualization/spatialglue/"
                    f"{combination}/{generation}/{sample['key']}"
                )
            point = client.get(url, headers={"Accept-Encoding": "identity"})
            point.raise_for_status()
            expected = sample["representations"]["identity"]
            require(
                point.headers.get("content-type", "").split(";")[0] == POINT_MEDIA_TYPE,
                f"点位 MIME 类型不正确：{name}",
            )
            require(
                len(point.content) == expected["size"]
                and hashlib.sha256(point.content).hexdigest() == expected["sha256"],
                f"线上点位摘要不符：{name}/{sample['key']}",
            )
        log("http_verified", index=index, total=len(selected), dataset_id=name)


def switch_release(target, stage, release, service, verify, log):
    """Stop before moving a real directory; restore the old binding on any caught failure."""
    previous = {"kind": "absent"}
    if target.is_symlink():
        previous = {"kind": "symlink", "link": os.readlink(target)}
    elif target.exists():
        previous = {"kind": "directory", "backup": str(release / "previous")}
    journal = {
        "target": str(target),
        "stage": str(stage),
        "previous": previous,
        "state": "prepared",
    }

    def save(state):
        journal["state"] = state
        ct._atomic_json(release / "transaction.json", journal)
        log(state, target=str(target))

    service("status")  # Require a healthy existing managed service before stopping it.
    save("prepared")
    stopped, moved, switched = False, False, False
    temporary = release / "next-link"
    try:
        stopped = True  # A failed stop can still have stopped the service.
        service("stop")
        save("service_stopped")
        temporary.symlink_to(stage, target_is_directory=True)
        if previous["kind"] == "directory":
            target.rename(previous["backup"])
            moved = True
        os.replace(temporary, target)
        switched = True
        save("switched")
        service("start")
        verify()
        save("published")
    except BaseException as error:
        journal["error"] = str(error)
        log("publication_failed", error=str(error))
        try:
            if stopped:
                try:
                    service("stop")
                except Exception as stop_error:
                    # Still restore the files if a failed start left no healthy service.
                    log("rollback_stop_failed", error=str(stop_error))
            # A signal can arrive between a successful rename and its Python flag assignment.
            moved = moved or (previous["kind"] == "directory" and Path(previous["backup"]).exists())
            switched = switched or (target.is_symlink() and target.resolve() == stage.resolve())
            if switched:
                require(
                    target.is_symlink() and target.resolve() == stage.resolve(),
                    "正式指针被外部修改，拒绝自动覆盖",
                )
                target.unlink()
            if previous["kind"] == "directory" and moved:
                Path(previous["backup"]).rename(target)
            elif previous["kind"] == "symlink" and switched:
                target.symlink_to(previous["link"], target_is_directory=True)
            if stopped:
                service("start")
                service("status")
            save("rolled_back")
        except BaseException as rollback_error:
            journal["rollback_error"] = str(rollback_error)
            save("rollback_failed")
            raise RuntimeError(
                f"发布及回滚失败，请保留并检查 {release / 'transaction.json'}"
            ) from error
        raise


def execute(
    root, settings, target, report_dir, *, check_only=False, completed_only=False, method="rna"
):
    report_dir.mkdir(parents=True, exist_ok=False)

    def log(event, **details):
        row = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        text = json.dumps(row, ensure_ascii=False)
        with (report_dir / "events.jsonl").open("a") as stream:
            stream.write(text + "\n")
        try:
            print(text, flush=True)
        except OSError:
            pass  # A disconnected terminal must not prevent rollback or persistent logging.

    log(
        "started",
        run_root=str(root),
        target=str(target),
        check_only=check_only,
        completed_only=completed_only,
    )
    try:
        # Freeze the cohort once; jobs finishing later are picked up on the next invocation.
        audit_function = audit_run
        if method == "spatialglue":
            from .spatialglue_publish import audit_run as audit_function
            from .spatialglue_publish import snapshot_progress

            progress = snapshot_progress(root)
        else:
            progress = read_json(root / "progress.json")
        if not completed_only:
            require(progress["state"] == "complete", "分析尚未完成，未执行发布或重启")
        with ExitStack() as stack:
            target_key = hashlib.sha256(str(target).encode()).hexdigest()[:24]
            publication_lock = resource_lock_path().with_name(
                f"iscdc-domain-publish-{os.getuid()}-{target_key}.lock"
            )
            paths = [publication_lock]
            if not completed_only:
                queue_lock = "run.lock" if method == "spatialglue" else ".supervisor.lock"
                paths.extend((root / queue_lock, resource_lock_path()))
            for job in progress["jobs"]:
                if job["state"] in {"success", "reused"}:
                    directory = result_directory(root / "sidecars", job)
                    require(not directory.is_symlink(), f"不安全的结果目录：{directory}")
                    paths.append(directory / ".generation.lock")
            for path in paths:
                lock = stack.enter_context(path.open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            audit, records = audit_function(
                root, settings, log, completed_only=completed_only, progress=progress
            )
            ct._atomic_json(report_dir / "audit.json", audit)
            if check_only:
                log("check_only_complete", report=str(report_dir / "audit.json"))
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            require(
                target != root and root not in target.parents and target not in root.parents,
                "正式目录与计算目录不能互相包含",
            )
            release = target.parent / f".{target.name}-releases" / report_dir.name
            release.mkdir(parents=True, exist_ok=False)
            stage = stage_release(root, target, release, audit, records, log)
            require(
                [binding(r) for r in catalogue_records(settings)] == audit["catalogue"],
                "审计后 catalogue 已改变",
            )
            for item in audit["selected"]:
                require(
                    signature(Path(item["source_path"])) == item["source_signature"],
                    f"审计后源文件已改变：{item['dataset_id']}",
                )
            env = os.environ.copy()
            env.update(
                ISCDC_PYTHON=sys.executable,
                ISCDC_DATABASE_PATH=str(settings.database_path),
                ISCDC_DATA_ROOT=str(settings.data_root),
                ISCDC_SPATIAL_DOMAIN_VISUALIZATION_ROOT=str(target),
            )
            env.setdefault("ISCDC_DEPLOY_START_TIMEOUT", "600")

            def service(action):
                log("service", action=action)
                with (report_dir / "service.log").open("a") as output:
                    subprocess.run(
                        ["bash", str(PROJECT_ROOT / "deploy_test.sh"), action],
                        cwd=PROJECT_ROOT,
                        env=env,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=660,
                    )

            port = int(env.get("ISCDC_DEPLOY_PORT", "5000"))
            require(1 <= port <= 65535, "无效服务端口")
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=60
            ) as client:
                switch_release(
                    target,
                    stage,
                    release,
                    service,
                    lambda: verify_http(client, audit["selected"], log),
                    log,
                )
            ct._atomic_json(
                report_dir / "published.json",
                {
                    "run_root": str(root),
                    "release": str(release),
                    "target": str(target),
                    "successful": len(audit["selected"]),
                    "skipped": audit["skipped"],
                    "deferred": audit["deferred"],
                    "completed_only": completed_only,
                    "progress_snapshot_at": audit["progress_snapshot_at"],
                },
            )
            log("complete", published=len(audit["selected"]), report=str(report_dir))
    except BaseException as error:
        log("failed", error=str(error), error_type=type(error).__name__)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, help="默认读取 temp/spatial_domain_full_latest.json"
    )
    parser.add_argument("--check-only", action="store_true", help="只审计，写报告，不发布或重启")
    parser.add_argument("--method", choices=("rna", "spatialglue"), default="rna")
    parser.add_argument(
        "--completed-only",
        action="store_true",
        help="发布本次开始时已成功的结果，暂缓其他任务，允许后台计算继续运行",
    )
    args = parser.parse_args(argv)
    if args.method == "spatialglue" and args.run_root is None:
        parser.error("SpatialGLUE 发布必须显式指定 --run-root")
    settings = Settings.from_environment()
    root = args.run_root or Path(
        read_json(PROJECT_ROOT / "temp/spatial_domain_full_latest.json")["run_root"]
    )
    # Keep the last component lexical: it may already be the public release symlink.
    configured = Path(
        os.environ.get(
            "ISCDC_SPATIAL_DOMAIN_VISUALIZATION_ROOT",
            str(PROJECT_ROOT / "data/spatial_domain_visualizations"),
        )
    ).expanduser()
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    target = configured.parent.resolve() / configured.name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    report_dir = PROJECT_ROOT / "temp" / f"spatial_domain_publication_{stamp}"

    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        execute(
            root.resolve(),
            settings,
            target,
            report_dir,
            check_only=args.check_only,
            completed_only=args.completed_only,
            method=args.method,
        )
    except (Exception, KeyboardInterrupt) as error:
        print(f"发布未完成：{error}\n日志：{report_dir}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
