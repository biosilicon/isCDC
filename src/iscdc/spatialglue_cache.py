"""Content-addressed, locked inference cache; never authoritative source annotations."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


def cache_key(kind, values, implementation):
    payload = json.dumps(
        {
            "kind": kind,
            "values": values,
            "implementation": hashlib.sha256(Path(implementation).read_bytes()).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def cached_artifact(key):
    root = os.environ.get("ISCDC_GLUE_CACHE_ROOT")
    if not root:
        yield None, False
        return
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("Unsafe inference cache")
    target, record = directory / f"{key}.npz", directory / f"{key}.json"
    with (directory / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        valid = False
        if target.is_file() and not target.is_symlink() and record.is_file():
            try:
                valid = json.loads(record.read_text())["sha256"] == digest(target)
            except (ValueError, KeyError, OSError):
                pass
        yield target, valid


def commit_cache(target, writer):
    temporary = target.with_name(target.stem + ".tmp.npz")
    writer(temporary)
    temporary.replace(target)
    record = target.with_suffix(".json")
    temporary = record.with_suffix(".tmp")
    temporary.write_text(json.dumps({"sha256": digest(target), "size": target.stat().st_size}))
    temporary.replace(record)
