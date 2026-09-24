"""Resource-limited SpatialGlue worker entry; no algorithm imports on website startup."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import spatialglue_config as config
from .spatial_domain_annotation import catalogue_records
from .spatial_domain_resources import managed_cpu_ids, resource_guard
from .spatial_domain_visualization import domain_directory, publish_method_failure


def execute(args, settings, root):
    if (
        Path(sys.prefix).name != "iscdc-spatial-domain-gpu"
        and os.environ.get("CONDA_DEFAULT_ENV") != "iscdc-spatial-domain-gpu"
    ):
        raise ValueError("Run SpatialGlue in the isolated iscdc-spatial-domain-gpu environment")
    config_path = args.config or config.CONFIG_PATH
    params = config.load_parameters(config_path, args.dataset_id)
    record = catalogue_records(settings, [args.dataset_id])[0]
    config.eligibility(record, params)
    groups = config.expand_combinations(record, params)
    requested = getattr(args, "combination", None)
    if requested:
        groups = [g for g in groups if config.combination_id(g) == requested]
        if not groups:
            raise ValueError("Requested combination is not in this Database's execution plan")
    cpus = managed_cpu_ids()
    if cpus is None:
        available = sorted(os.sched_getaffinity(0))
        cpus = available[: min(params["threads"], max(1, len(available) - 16))]
    failures = 0
    original_force = args.force
    for modalities in groups:
        args.force = original_force
        combination = config.combination_id(modalities)
        args.combination = combination
        directory = domain_directory(
            root, args.dataset_id, "spatialglue", combination_id=combination
        )
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".generation.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another SpatialGlue job owns this combination") from exc
            if (directory / "status.json").exists() and not args.force:
                if not getattr(args, "resume", False):
                    raise ValueError("Existing SpatialGlue status; use --resume or --force")
                from .spatial_domain_visualization import load_spatial_domain_visualization

                try:
                    snapshot = load_spatial_domain_visualization(
                        root, record, method_family="spatialglue", combination_id=combination
                    )
                except (ValueError, OSError, KeyError):
                    pass
                else:
                    from .spatialglue import adapter_sha256

                    resolved = config.load_parameters(
                        config_path, args.dataset_id, modalities=modalities
                    )
                    if (
                        snapshot.manifest["provenance"]["adapter_sha256"] == adapter_sha256()
                        and snapshot.manifest["provenance"]["parameters"] == resolved
                    ):
                        print(
                            json.dumps({"combination_id": combination, "state": "already_complete"})
                        )
                        continue
                args.force = True
            if os.environ.get("ISCDC_GLUE_WORKER") == "1":
                from .spatialglue import generate

                os.sched_setaffinity(0, cpus)
                os.nice(5)
                try:
                    result = generate(
                        record, settings, root, config_path, force=args.force, modalities=modalities
                    )
                    print(json.dumps(result, indent=2), flush=True)
                except (ValueError, RuntimeError, OSError) as exc:
                    print(f"SpatialGlue {combination}: {exc}", file=sys.stderr, flush=True)
                    failures += 1
            else:
                with resource_guard():
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    failures += int(run_worker(args, root, config_path, params, cpus) != 0)
    return int(failures > 0)


def run_worker(args, root, config_path, params, cpus):
    import psutil

    env = os.environ.copy()
    env.update(
        ISCDC_GLUE_WORKER="1",
        PYTHONHASHSEED=str(params["seed"]),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        NUMBA_THREADING_LAYER="workqueue",
    )
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[key] = str(len(cpus))
    env.setdefault("MPLCONFIGDIR", "/tmp/iscdc-spatialglue-mpl")
    descriptors = tuple(
        int(env[key]) for key in ("ISCDC_DOMAIN_RESOURCE_FD", "ISCDC_GLUE_GPU_FD") if key in env
    )
    command = [
        sys.executable,
        "-m",
        "iscdc.spatial_domain_annotation",
        "generate-spatial-domain-visualization",
        args.dataset_id,
        "--method",
        "spatialglue",
        "--config",
        str(config_path),
        "--output-root",
        str(root),
        "--combination",
        args.combination,
    ]
    if args.force:
        command.append("--force")
    process = subprocess.Popen(command, env=env, start_new_session=True, pass_fds=descriptors)
    monitored = psutil.Process(process.pid)
    interrupted = []

    def stop(signum=None, frame=None):
        if signum is not None:
            interrupted.append(signum)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    peak, last_tick = 0, time.monotonic()
    try:
        while process.poll() is None:
            try:
                rss = monitored.memory_info().rss + sum(
                    p.memory_info().rss for p in monitored.children(recursive=True)
                )
                peak = max(peak, rss)
                if rss > params["memory_budget_gb"] * 2**30:
                    stop()
                    publish_method_failure(
                        root,
                        args.dataset_id,
                        "Runtime memory budget exceeded",
                        method_family="spatialglue",
                        combination_id=args.combination,
                        details={"peak_rss_bytes": peak},
                    )
                    return 1
                if time.monotonic() - last_tick >= 60:
                    print(
                        json.dumps(
                            {
                                "event": "spatialglue_heartbeat",
                                "dataset_id": args.dataset_id,
                                "rss_bytes": rss,
                                "peak_rss_bytes": peak,
                            }
                        ),
                        flush=True,
                    )
                    last_tick = time.monotonic()
            except psutil.NoSuchProcess:
                pass
            time.sleep(1)
        if process.returncode != 0:
            # OOM/SIGKILL/import failures may prevent the child from writing a report.
            base = domain_directory(
                root, args.dataset_id, "spatialglue", combination_id=args.combination
            )
            status = base / "status.json"
            state = json.loads(status.read_text()).get("state") if status.exists() else None
            if state != "failure" or interrupted:
                publish_method_failure(
                    root,
                    args.dataset_id,
                    f"SpatialGlue worker exited {process.returncode}",
                    method_family="spatialglue",
                    combination_id=args.combination,
                    details={"peak_rss_bytes": peak},
                )
        return process.returncode
    finally:
        stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
