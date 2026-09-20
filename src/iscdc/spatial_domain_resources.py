"""CPU leases for a single resource-owning offline supervisor."""

from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def resource_lock_path():
    return Path(tempfile.gettempdir()) / f"iscdc-spatial-domain-{os.getuid()}.lock"


def managed_cpu_ids():
    value = os.environ.get("ISCDC_DOMAIN_CPU_IDS")
    if value is None:
        return None
    ids = [int(part) for part in value.split(",")]
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= os.sched_getaffinity(0):
        raise ValueError("Invalid managed CPU allocation")
    fd = int(os.environ["ISCDC_DOMAIN_RESOURCE_FD"])
    inherited, expected = os.fstat(fd), resource_lock_path().stat()
    if (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("Managed resource lease does not refer to the global resource lock")
    # The inherited open file description must own the supervisor's exclusive lock.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return ids


@contextmanager
def resource_guard():
    if managed_cpu_ids() is not None:
        yield  # Never unlock the inherited supervisor lease from a worker.
        return
    with resource_lock_path().open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield
