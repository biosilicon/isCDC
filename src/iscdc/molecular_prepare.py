"""Offline preparation, audit and atomic publication of molecular views.

Run in the ordinary iscdc environment. No catalogue or source files are written.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings
from .molecular_search import filter_index, searchable_features
from .molecular_visualization import DTYPES, SAFE_KEY, VERSION, encode_array, safe_path

PREPARATION_CODE = Path(__file__).read_bytes()
ENCODING_CODE = Path(__file__).with_name("molecular_visualization.py").read_bytes()
SEARCH_CODE = Path(__file__).with_name("molecular_search.py").read_bytes()
CODE_DIGESTS = {
    "molecular_prepare.py": hashlib.sha256(PREPARATION_CODE).hexdigest(),
    "molecular_visualization.py": hashlib.sha256(ENCODING_CODE).hexdigest(),
    "molecular_search.py": hashlib.sha256(SEARCH_CODE).hexdigest(),
}

ALIASES = (
    "feature_name",
    "gene_symbol",
    "target_name",
    "antibody_target",
    "source_feature_name",
    "name",
    "secondary_name",
    "source_feature_id",
    "mz",
    "mass_to_charge",
    "cdr3_aa",
    "junction_aa",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                document, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive(root: Path):
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def catalogue(settings: Settings) -> list[dict]:
    with sqlite3.connect(f"{settings.database_path.resolve().as_uri()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        records = [
            dict(r)
            for r in db.execute(
                "SELECT dataset_id,storage_dir,sha256,n_obs,coordinate_dimensions,sample_ids "
                "FROM datasets WHERE dataset_type='full' ORDER BY dataset_id"
            )
        ]
    for record in records:
        record["sample_ids"] = json.loads(record["sample_ids"])
    return records


def _index(group):
    from anndata.io import read_elem

    return read_elem(group[group.attrs["_index"]]).astype(str)


def _order_digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _feature_index(group, path: Path):
    import pandas as pd
    from anndata.io import read_elem

    ids = _index(group)
    if len(ids) == 0 or len(set(ids)) != len(ids) or any(not s.strip() for s in ids):
        raise ValueError("Empty or duplicate feature identity")
    columns = {key: read_elem(group[key]) for key in ALIASES if key in group}
    first = None
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE features(idx INTEGER PRIMARY KEY, feature_id TEXT NOT NULL,
                label TEXT NOT NULL, folded_id TEXT NOT NULL, folded_label TEXT NOT NULL);
            CREATE VIRTUAL TABLE search USING fts5(aliases, tokenize='trigram');
        """)
        batch, search_batch = [], []
        for index, feature_id in enumerate(ids):
            aliases = [
                str(column[index])
                for column in columns.values()
                if pd.notna(column[index]) and str(column[index]).strip()
            ]
            label = aliases[0] if aliases else feature_id
            # Numeric/sequence annotations are searchable without replacing stable IDs.
            if columns and next(iter(columns)) in {
                "mz",
                "mass_to_charge",
                "cdr3_aa",
                "junction_aa",
            }:
                label = feature_id
            batch.append((index, feature_id, label, feature_id.casefold(), label.casefold()))
            search_batch.append(
                (index, "\n".join(dict.fromkeys([feature_id, *aliases])).casefold())
            )
            if first is None:
                first = {"key": str(index), "id": feature_id, "label": label}
            if len(batch) == 1000:
                db.executemany("INSERT INTO features VALUES(?,?,?,?,?)", batch)
                db.executemany("INSERT INTO search(rowid,aliases) VALUES(?,?)", search_batch)
                batch.clear()
                search_batch.clear()
        db.executemany("INSERT INTO features VALUES(?,?,?,?,?)", batch)
        db.executemany("INSERT INTO search(rowid,aliases) VALUES(?,?)", search_batch)
        db.executescript("""
            CREATE INDEX feature_id_lookup ON features(folded_id);
            CREATE INDEX feature_label_lookup ON features(folded_label);
            INSERT INTO search(search) VALUES('optimize');
        """)
    return len(ids), _order_digest(ids), first


def _write_checked(handle, name, values):
    import numpy as np

    output = handle.create_dataset(
        name, data=values, compression="gzip", compression_opts=1, shuffle=True
    )
    # Exhaustively compare bounded chunks; do not rely on a few sampled features.
    block = max(1, 8 * 1024**2 // max(values.dtype.itemsize, 1))
    for start in range(0, len(values), block):
        np.testing.assert_array_equal(output[start : start + block], values[start : start + block])


def _write_matrix(group, path, top_ids, sample_rows, *, memory_gib):
    import h5py
    import numpy as np
    import pandas as pd
    from anndata.io import read_elem
    from scipy import sparse

    names = _index(group["obs"])
    if len(set(names)) != len(names):
        raise ValueError("Duplicate modality observations")
    positions = pd.Index(top_ids).get_indexer(names)
    if (positions < 0).any():
        raise ValueError("Foreign modality observation")
    mapping = np.full(len(top_ids), -1, dtype=np.int64)
    mapping[positions] = np.arange(len(names))
    # Full Databases should not contain composite zero-padding masks. Refuse such
    # unexpected input instead of presenting unmeasured features as measured zeros.
    if "feature_measured_by_source" in group.get("varm", {}):
        mask = read_elem(group["varm/feature_measured_by_source"])
        if not np.all(mask):
            raise ValueError("Source-dependent feature coverage needs an explicit full-data recipe")
    source = group["X"]
    shape = source.attrs["shape"] if isinstance(source, h5py.Group) else source.shape
    if tuple(shape) != (len(names), len(_index(group["var"]))):
        raise ValueError("Matrix shape differs from observation/feature identities")
    with h5py.File(path, "w") as output:
        output.attrs["n_obs"] = len(names)
        for key, rows in sample_rows.items():
            _write_checked(output, f"samples/{key}", mapping[rows])
        if isinstance(source, h5py.Group):
            estimate = (
                sum(
                    source[k].size * source[k].dtype.itemsize for k in ("data", "indices", "indptr")
                )
                * 3
            )
            if estimate > memory_gib * 1024**3:
                raise ValueError(f"Sparse conversion exceeds {memory_gib} GiB memory budget")
            matrix = read_elem(source)
            if not sparse.issparse(matrix) or matrix.dtype.name not in DTYPES:
                raise ValueError("Unsupported sparse matrix")
            csc = matrix.tocsc()
            csc.sum_duplicates()
            csc.sort_indices()
            output.attrs["encoding"] = "csc"
            output.attrs["n_vars"] = csc.shape[1]
            for name, values in (
                ("data", csc.data),
                ("indices", csc.indices),
                ("indptr", csc.indptr),
            ):
                _write_checked(output, name, values)
            dtype = csc.dtype.name
            del matrix, csc
        else:
            if source.ndim != 2 or source.dtype.name not in DTYPES:
                raise ValueError("Unsupported dense matrix")
            n_obs, n_vars = source.shape
            output.attrs["encoding"] = "dense"
            output.attrs["n_vars"] = n_vars
            values = output.create_dataset(
                "values",
                (n_vars, n_obs),
                dtype=source.dtype,
                chunks=(1, min(n_obs, 65536)),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            block = max(1, 32 * 1024**2 // max(n_obs * source.dtype.itemsize, 1))
            for start in range(0, n_vars, block):
                chunk = source[:, start : start + block].T
                values[start : start + block] = chunk
                np.testing.assert_array_equal(values[start : start + block], chunk)
            dtype = source.dtype.name
    gc.collect()
    return dtype, _order_digest(names)


def prepare_generation(settings, record, root, *, memory_gib=16):
    import h5py
    import numpy as np
    from anndata.io import read_elem

    if record["coordinate_dimensions"] != 2 or not SAFE_KEY.fullmatch(record["dataset_id"]):
        raise ValueError("Molecular visualization requires a safe, full 2D Database")
    source = safe_path(settings.data_root, record["storage_dir"] + "/dataset.h5mu")
    if sha256(source) != record["sha256"]:
        raise ValueError("Source checksum differs from catalogue")
    generation_id = f"v{VERSION}-{record['sha256'][:16]}-{uuid.uuid4().hex[:12]}"
    relative = f"{record['dataset_id']}/generations/{generation_id}"
    directory = safe_path(root, relative)
    directory.mkdir(parents=True)
    started = time.monotonic()
    manifest = {
        "version": VERSION,
        "dataset_id": record["dataset_id"],
        "generation_id": generation_id,
        "source_sha256": record["sha256"],
        "generated_at": datetime.now(UTC).isoformat(),
        "y_axis": "up",
        "samples": [],
        "modalities": [],
        "files": {},
        "generator": {
            "code_sha256": CODE_DIGESTS,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "h5py", "anndata")
            },
        },
    }
    with h5py.File(source, "r") as handle:
        top_ids = _index(handle["obs"])
        samples = np.asarray(read_elem(handle["obs/sample_id"])).astype(str)
        spatial = np.asarray(handle["obsm/spatial"], dtype=np.float64)
        if (
            len(top_ids) != record["n_obs"]
            or len(set(top_ids)) != len(top_ids)
            or spatial.shape != (len(top_ids), 2)
            or not np.isfinite(spatial).all()
            or set(samples) != set(record["sample_ids"])
        ):
            raise ValueError("Invalid spatial observation/sample identities")
        manifest["obs_order_sha256"] = _order_digest(top_ids)
        sample_rows = {}
        for i, sample_id in enumerate(record["sample_ids"]):
            key = f"s{i}"
            rows = np.flatnonzero(samples == sample_id)
            sample_rows[key] = rows
            filename = f"coordinates-{key}.bin"
            payload = encode_array(spatial[rows].T, kind=1)
            (directory / filename).write_bytes(payload)
            manifest["samples"].append(
                {
                    "key": key,
                    "id": sample_id,
                    "count": len(rows),
                    "coordinates": filename,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "obs_order_sha256": _order_digest(top_ids[rows]),
                }
            )
        for i, (name, group) in enumerate(handle["mod"].items()):
            if not SAFE_KEY.fullmatch(name):
                raise ValueError("Unsafe modality name")
            print(f"  {record['dataset_id']} / {name}", flush=True)
            index_name, matrix_name = f"features-{i}.sqlite", f"values-{i}.h5"
            count, digest, first = _feature_index(group["var"], directory / index_name)
            dtype, obs_digest = _write_matrix(
                group, directory / matrix_name, top_ids, sample_rows, memory_gib=memory_gib
            )
            search = filter_index(
                directory / index_name, searchable_features(directory / matrix_name)
            )
            assay = read_elem(group["uns/assay"])
            manifest["modalities"].append(
                {
                    "name": name,
                    "n_vars": count,
                    "feature_order_sha256": digest,
                    "obs_order_sha256": obs_digest,
                    "first_feature": search["first_feature"] or first,
                    "n_searchable": search["n_searchable"],
                    "value_type": str(assay["value_type"]),
                    "dtype": dtype,
                    "index": index_name,
                    "matrix": matrix_name,
                }
            )
    # Freeze both ends of conversion, including metadata-only source replacements.
    if sha256(source) != record["sha256"]:
        raise ValueError("Source changed while molecular artifacts were prepared")
    for path in sorted(directory.iterdir()):
        manifest["files"][path.name] = {"sha256": sha256(path), "size": path.stat().st_size}
    manifest["validation"] = {
        "status": "passed",
        "matrix_check": "all stored values",
        "source_unchanged": True,
        "seconds": round(time.monotonic() - started, 3),
    }
    atomic_json(directory / "manifest.json", manifest)
    return {
        "directory": relative,
        "manifest_sha256": sha256(directory / "manifest.json"),
        "source_sha256": record["sha256"],
        "state": "success",
    }


def audit_generation(settings, record, root, item, *, check_source=True):
    directory = safe_path(root, item["directory"])
    if sha256(directory / "manifest.json") != item["manifest_sha256"]:
        raise ValueError("Molecular manifest checksum mismatch")
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["version"] != VERSION
        or manifest["dataset_id"] != record["dataset_id"]
        or manifest["source_sha256"] != record["sha256"]
        or manifest["validation"]["status"] != "passed"
        or sum(s["count"] for s in manifest["samples"]) != record["n_obs"]
        or [s["id"] for s in manifest["samples"]] != record["sample_ids"]
    ):
        raise ValueError("Molecular source identity mismatch")
    if check_source:
        source = safe_path(settings.data_root, record["storage_dir"] + "/dataset.h5mu")
        if sha256(source) != record["sha256"]:
            raise ValueError("Source checksum mismatch during molecular audit")
    for relative, expected in manifest["files"].items():
        path = safe_path(directory, relative)
        if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Molecular artifact checksum mismatch: {relative}")
    return manifest


def prepare_batch(settings, output_root, *, dataset_id=None, memory_gib=16):
    records = catalogue(settings)
    if dataset_id:
        records = [r for r in records if r["dataset_id"] == dataset_id]
    if not records:
        raise ValueError("No full Database selected")
    with exclusive(output_root):
        code_root = output_root / "code"
        code_root.mkdir(exist_ok=True)
        for name, content in (
            ("molecular_prepare.py", PREPARATION_CODE),
            ("molecular_visualization.py", ENCODING_CODE),
            ("molecular_search.py", SEARCH_CODE),
        ):
            snapshot = code_root / name
            if snapshot.exists() and snapshot.read_bytes() != content:
                raise ValueError("Preparation code changed; use a new output directory")
            if not snapshot.exists():
                snapshot.write_bytes(content)
        batch_path = output_root / "batch.json"
        if batch_path.exists():
            batch = json.loads(batch_path.read_text())
            if batch["records"] != records or batch["version"] != VERSION:
                raise ValueError("Frozen batch differs from catalogue; use a new output directory")
        else:
            batch = {
                "version": VERSION,
                "records": records,
                "datasets": {},
                "scope": "single" if dataset_id else "all",
                "status": "preparing",
            }
        atomic_json(batch_path, batch)
        for number, record in enumerate(records, 1):
            did = record["dataset_id"]
            print(f"[{number}/{len(records)}] {did}", flush=True)
            try:
                existing = batch["datasets"].get(did)
                if existing and existing["state"] == "success":
                    audit_generation(settings, record, output_root, existing)
                    print("  resumed verified generation", flush=True)
                    continue
                batch["datasets"][did] = prepare_generation(
                    settings, record, output_root, memory_gib=memory_gib
                )
            except Exception as exc:
                batch["datasets"][did] = {"state": "failed", "error": str(exc)}
                print(f"  FAILED: {exc}", flush=True)
            atomic_json(batch_path, batch)
        batch["status"] = (
            "prepared"
            if all(d["state"] == "success" for d in batch["datasets"].values())
            else "failed"
        )
        atomic_json(batch_path, batch)
        return batch


def audit_batch(settings, root):
    batch = json.loads((root / "batch.json").read_text())
    current = catalogue(settings)
    if batch["scope"] == "single":
        ids = {r["dataset_id"] for r in batch["records"]}
        current = [r for r in current if r["dataset_id"] in ids]
    if current != batch["records"]:
        raise ValueError("Catalogue changed since batch preparation")
    report = {
        "version": VERSION,
        "status": "passed",
        "datasets": {},
        "audited_at": datetime.now(UTC).isoformat(),
    }
    for i, record in enumerate(current, 1):
        did = record["dataset_id"]
        print(f"Audit [{i}/{len(current)}] {did}", flush=True)
        try:
            item = batch["datasets"][did]
            if item["state"] != "success":
                raise ValueError("Generation was not successful")
            manifest = audit_generation(settings, record, root, item)
            report["datasets"][did] = {
                "status": "passed",
                "modalities": len(manifest["modalities"]),
                "samples": len(manifest["samples"]),
            }
        except Exception as exc:
            report["datasets"][did] = {"status": "failed", "error": str(exc)}
            report["status"] = "failed"
    atomic_json(root / "audit.json", report)
    return report


def publish_batch(settings, root, destination):
    """Copy immutable generations, verify the copies, then atomically switch one index."""
    report = audit_batch(settings, root)
    if report["status"] != "passed":
        raise ValueError("Molecular batch audit failed; publication was not changed")
    batch = json.loads((root / "batch.json").read_text())
    with exclusive(destination):
        publication = destination / "publication.json"
        previous = (
            json.loads(publication.read_text())
            if publication.exists()
            else {"version": VERSION, "datasets": {}}
        )
        next_index = {
            "version": VERSION,
            "published_at": datetime.now(UTC).isoformat(),
            "datasets": dict(previous["datasets"]) if batch["scope"] == "single" else {},
        }
        for record in batch["records"]:
            item = batch["datasets"][record["dataset_id"]]
            source = safe_path(root, item["directory"])
            target = safe_path(destination, item["directory"])
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".staging")
                if temporary.exists():
                    shutil.rmtree(temporary)
                shutil.copytree(source, temporary)
                os.replace(temporary, target)
            audit_generation(settings, record, destination, item, check_source=False)
            next_index["datasets"][record["dataset_id"]] = item
        if catalogue(settings) != catalogue_snapshot_for_publish(settings, batch):
            raise ValueError("Catalogue changed during molecular publication")
        atomic_json(destination / "previous-publication.json", previous)
        atomic_json(publication, next_index)
    return next_index


def catalogue_snapshot_for_publish(settings, batch):
    if batch["scope"] == "all":
        return batch["records"]
    current = catalogue(settings)
    pinned = {r["dataset_id"]: r for r in batch["records"]}
    return [pinned.get(r["dataset_id"], r) for r in current]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    scope = prepare.add_mutually_exclusive_group(required=True)
    scope.add_argument("--dataset-id")
    scope.add_argument("--all", action="store_true")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--memory-gib", type=int, default=16, choices=range(1, 257))
    for command in ("audit", "publish"):
        sub = commands.add_parser(command)
        sub.add_argument("--input-root", type=Path, required=True)
    commands.add_parser("rollback")
    args = parser.parse_args(argv)
    settings = Settings.from_environment()
    destination = settings.molecular_visualization_root
    if args.command == "prepare":
        if args.output_root.resolve() == destination.resolve():
            parser.error("Prepare into an isolated output directory")
        result = prepare_batch(
            settings, args.output_root, dataset_id=args.dataset_id, memory_gib=args.memory_gib
        )
        return 0 if result["status"] == "prepared" else 1
    if args.command == "audit":
        return 0 if audit_batch(settings, args.input_root)["status"] == "passed" else 1
    if args.command == "publish":
        publish_batch(settings, args.input_root, destination)
        print("Published molecular artifacts. Restart and verify the website.", flush=True)
        return 0
    with exclusive(destination):
        previous = json.loads((destination / "previous-publication.json").read_text())
        atomic_json(destination / "publication.json", previous)
    print("Restored previous molecular publication. Restart the website.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
