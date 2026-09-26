"""Offline all-zero feature filtering and generation-bound search publication."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import uuid
from pathlib import Path

from .molecular_visualization import safe_path

SEARCH_VERSION = 1


def searchable_features(path: Path, *, block_bytes=8 * 1024**2):
    """Keep any value != 0, including negatives and unknown/non-finite values."""
    import h5py
    import numpy as np

    with h5py.File(path, "r") as matrix:
        n_vars, n_obs = int(matrix.attrs["n_vars"]), int(matrix.attrs["n_obs"])
        # No observations is unknown coverage, not proof that every value is zero.
        keep = np.full(n_vars, n_obs == 0, dtype=bool)
        if n_obs == 0:
            return keep
        if matrix.attrs["encoding"] == "csc":
            pointers = matrix["indptr"][:]
            values = matrix["data"]
            block = max(1, block_bytes // values.dtype.itemsize)
            for start in range(0, len(values), block):
                positions = np.flatnonzero(values[start : start + block] != 0) + start
                keep[np.searchsorted(pointers, positions, side="right") - 1] = True
        else:
            values = matrix["values"]
            block = max(1, block_bytes // max(n_obs * values.dtype.itemsize, 1))
            for start in range(0, n_vars, block):
                keep[start : start + block] = np.any(values[start : start + block] != 0, axis=1)
        return keep


def filter_index(path: Path, keep) -> dict:
    """Filter an unpublished index without renumbering source feature keys."""
    import numpy as np

    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        db.execute("CREATE TEMP TABLE hidden(idx INTEGER PRIMARY KEY)")
        db.executemany("INSERT INTO hidden VALUES(?)", ((int(i),) for i in np.flatnonzero(~keep)))
        db.execute("DELETE FROM search WHERE rowid IN (SELECT idx FROM hidden)")
        db.execute("DELETE FROM features WHERE idx IN (SELECT idx FROM hidden)")
        actual = np.fromiter(
            (r[0] for r in db.execute("SELECT idx FROM features ORDER BY idx")), int
        )
        np.testing.assert_array_equal(actual, np.flatnonzero(keep))
        first = db.execute("SELECT * FROM features ORDER BY idx LIMIT 1").fetchone()
        db.execute("INSERT INTO search(search) VALUES('optimize')")
        db.commit()
        db.execute("VACUUM")
    return {
        "n_searchable": int(keep.sum()),
        "first_feature": (
            {"key": str(first["idx"]), "id": first["feature_id"], "label": first["label"]}
            if first is not None
            else None
        ),
    }


def prepare_search(settings, output_root: Path):
    from .molecular_prepare import atomic_json, audit_generation, catalogue, exclusive, sha256

    source_root = settings.molecular_visualization_root
    if output_root.resolve().is_relative_to(source_root.resolve()):
        raise ValueError("Prepare search indices into an isolated output directory")
    records = catalogue(settings)
    publication_path = source_root / "publication.json"
    publication = json.loads(publication_path.read_text())
    if set(publication["datasets"]) != {r["dataset_id"] for r in records}:
        raise ValueError(
            "Prepare molecular artifacts for the full catalogue before search filtering"
        )
    with exclusive(output_root):
        if (output_root / "batch.json").exists():
            raise ValueError("Search batch already exists; use a new output directory")
        directory = f"search-indices/v{SEARCH_VERSION}-{uuid.uuid4().hex}"
        batch = {
            "version": SEARCH_VERSION,
            "directory": directory,
            "records": records,
            "parent_publication_sha256": sha256(publication_path),
            "status": "preparing",
            "datasets": {},
            "files": {},
        }
        target = safe_path(output_root, directory)
        target.mkdir(parents=True)
        code = target / "molecular_search.py"
        code.write_bytes(Path(__file__).read_bytes())
        batch["files"][code.name] = sha256(code)
        atomic_json(output_root / "batch.json", batch)
        for number, record in enumerate(records, 1):
            did = record["dataset_id"]
            print(f"Filter [{number}/{len(records)}] {did}", flush=True)
            item = publication["datasets"][did]
            manifest = audit_generation(settings, record, source_root, item, check_source=False)
            parent = safe_path(source_root, item["directory"])
            entry = {
                "parent_generation_id": manifest["generation_id"],
                "parent_manifest_sha256": item["manifest_sha256"],
                "modalities": {},
            }
            safe_path(target, did).mkdir()
            for modality in manifest["modalities"]:
                name = modality["name"]
                relative = f"{did}/{modality['index']}"
                path = safe_path(target, relative)
                shutil.copyfile(safe_path(parent, modality["index"]), path)
                keep = searchable_features(safe_path(parent, modality["matrix"]))
                stats = filter_index(path, keep)
                entry["modalities"][name] = {
                    **stats,
                    "n_vars": modality["n_vars"],
                    "index": relative,
                }
                batch["files"][relative] = sha256(path)
                print(
                    f"  {name}: {stats['n_searchable']}/{modality['n_vars']} searchable", flush=True
                )
            batch["datasets"][did] = entry
            atomic_json(output_root / "batch.json", batch)
        if (
            sha256(publication_path) != batch["parent_publication_sha256"]
            or catalogue(settings) != records
        ):
            raise ValueError("Molecular publication or catalogue changed while filtering")
        batch["status"] = "prepared"
        atomic_json(output_root / "batch.json", batch)
        return batch


def audit_search(settings, root: Path):
    from .molecular_prepare import audit_generation, catalogue, sha256

    batch = json.loads((root / "batch.json").read_text())
    source_root = settings.molecular_visualization_root
    publication_path = source_root / "publication.json"
    if (
        batch["version"] != SEARCH_VERSION
        or batch["status"] != "prepared"
        or batch["records"] != catalogue(settings)
        or batch["parent_publication_sha256"] != sha256(publication_path)
        or set(batch["datasets"]) != {r["dataset_id"] for r in batch["records"]}
    ):
        raise ValueError("Search batch or its molecular source publication changed")
    publication = json.loads(publication_path.read_text())
    for record in batch["records"]:
        did = record["dataset_id"]
        item, entry = publication["datasets"][did], batch["datasets"][did]
        manifest = audit_generation(settings, record, source_root, item, check_source=False)
        if (
            entry["parent_generation_id"] != manifest["generation_id"]
            or entry["parent_manifest_sha256"] != item["manifest_sha256"]
        ):
            raise ValueError("Search index parent generation changed")
    directory = safe_path(root, batch["directory"])
    for path, digest in batch["files"].items():
        if sha256(safe_path(directory, path)) != digest:
            raise ValueError("Filtered search index checksum mismatch")
    return batch


def publish_search(settings, root: Path):
    from .molecular_prepare import atomic_json, catalogue, exclusive, sha256

    destination = settings.molecular_visualization_root
    with exclusive(destination):
        batch = audit_search(settings, root)
        source = safe_path(root, batch["directory"])
        target = safe_path(destination, batch["directory"])
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
        for path, digest in batch["files"].items():
            if sha256(safe_path(target, path)) != digest:
                raise ValueError("Published search index checksum mismatch")
        if (
            sha256(destination / "publication.json") != batch["parent_publication_sha256"]
            or catalogue(settings) != batch["records"]
        ):
            raise ValueError("Molecular publication changed during search publication")
        publication = destination / "search-publication.json"
        previous = (
            json.loads(publication.read_text())
            if publication.exists()
            else {"version": SEARCH_VERSION, "datasets": {}}
        )
        atomic_json(destination / "previous-search-publication.json", previous)
        atomic_json(publication, {k: batch[k] for k in ("version", "directory", "datasets")})
        return batch


def main(argv=None):
    from .config import Settings
    from .molecular_prepare import atomic_json, exclusive

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").add_argument("--output-root", type=Path, required=True)
    for name in ("audit", "publish"):
        commands.add_parser(name).add_argument("--input-root", type=Path, required=True)
    commands.add_parser("rollback")
    args = parser.parse_args(argv)
    settings = Settings.from_environment()
    if args.command == "prepare":
        prepare_search(settings, args.output_root)
    elif args.command == "audit":
        audit_search(settings, args.input_root)
    elif args.command == "publish":
        publish_search(settings, args.input_root)
        print("Published filtered search indices. Restart and verify the website.", flush=True)
    else:
        root = settings.molecular_visualization_root
        with exclusive(root):
            previous = json.loads((root / "previous-search-publication.json").read_text())
            atomic_json(root / "search-publication.json", previous)
        print("Restored previous search publication. Restart the website.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
