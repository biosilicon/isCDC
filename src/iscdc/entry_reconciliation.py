"""Explicit, reversible correction of catalogue entry IDs."""

from __future__ import annotations

import copy
import json
import os
import secrets
import shutil
import sqlite3
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import yaml
from pydantic import ValidationError

from .config import Settings
from .entry_migration import (
    _atomic_restore,
    _catalogue_digest,
    _link_or_copy,
    _read_json,
    _read_yaml,
    _sha256,
    _sqlite_backup,
    _write_json_atomic,
)
from .schemas import SAFE_IDENTIFIER_PATTERN, MetadataDocument

RECONCILIATION_NAME = "entry_id_reconciliation"
ARTIFACT_NAMES = ("dataset.h5mu", "metadata.yaml", "manifest.json", "checksum.sha256")


class EntryIdReconciliationError(RuntimeError):
    """Raised when an entry reconciliation cannot be completed safely."""


@dataclass(frozen=True)
class EntryIdReconciliationInventory:
    explicit_full_changes: int
    propagated_derived_changes: int
    affected_challenges: int
    affected_visualizations: int
    snapshot_sha256_updates: int
    resulting_full_entries: int

    def as_dict(self) -> dict[str, int]:
        return {
            "explicit_full_changes": self.explicit_full_changes,
            "propagated_derived_changes": self.propagated_derived_changes,
            "affected_challenges": self.affected_challenges,
            "affected_visualizations": self.affected_visualizations,
            "snapshot_sha256_updates": self.snapshot_sha256_updates,
            "resulting_full_entries": self.resulting_full_entries,
        }


@dataclass(frozen=True)
class EntryIdReconciliationResult:
    report_path: Path
    backup_path: Path
    inventory: EntryIdReconciliationInventory


def _safe_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SAFE_IDENTIFIER_PATTERN.fullmatch(value):
        raise EntryIdReconciliationError(f"{field} must be a safe identifier")
    return value


def _load_mapping(path: Path) -> dict[str, dict[str, Any]]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EntryIdReconciliationError(f"Unable to read mapping {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("version") != "1.0":
        raise EntryIdReconciliationError("Mapping version must be '1.0'")
    changes = document.get("changes")
    if not isinstance(changes, list) or not changes:
        raise EntryIdReconciliationError("Mapping changes must be a non-empty list")
    result: dict[str, dict[str, Any]] = {}
    for index, change in enumerate(changes):
        if not isinstance(change, dict):
            raise EntryIdReconciliationError(f"changes[{index}] must be an object")
        dataset_id = _safe_identifier(change.get("dataset_id"), f"changes[{index}].dataset_id")
        expected = _safe_identifier(
            change.get("expected_entry_id"), f"changes[{index}].expected_entry_id"
        )
        target = _safe_identifier(
            change.get("target_entry_id"), f"changes[{index}].target_entry_id"
        )
        evidence = change.get("evidence")
        if not isinstance(evidence, list) or not evidence or not all(
            isinstance(item, str) and item.strip() for item in evidence
        ):
            raise EntryIdReconciliationError(
                f"changes[{index}].evidence must contain at least one non-blank value"
            )
        if dataset_id in result:
            raise EntryIdReconciliationError(f"Duplicate mapping for {dataset_id!r}")
        if expected == target:
            raise EntryIdReconciliationError(f"Mapping for {dataset_id!r} makes no change")
        result[dataset_id] = {
            "dataset_id": dataset_id,
            "expected_entry_id": expected,
            "target_entry_id": target,
            "evidence": [item.strip() for item in evidence],
        }
    return result


def _catalogue_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with sqlite3.connect(path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT dataset_id, entry_id, dataset_type, split_id, derivation, "
                "storage_dir, file_size, sha256, n_obs, coordinate_dimensions, sample_ids "
                "FROM datasets ORDER BY dataset_id"
            ).fetchall()
    except sqlite3.Error as exc:
        raise EntryIdReconciliationError(f"Unable to read catalogue: {exc}") from exc
    result = []
    for row in rows:
        record = dict(row)
        if isinstance(record["derivation"], str):
            record["derivation"] = json.loads(record["derivation"])
        if isinstance(record["sample_ids"], str):
            record["sample_ids"] = json.loads(record["sample_ids"])
        result.append(record)
    return result


def _dataset_directory(settings: Settings, row: dict[str, Any]) -> Path:
    storage_dir = Path(str(row["storage_dir"]))
    if storage_dir.name != str(storage_dir) or storage_dir in {Path("."), Path("..")}:
        raise EntryIdReconciliationError(f"Unsafe storage_dir {storage_dir!s}")
    root = settings.data_root.resolve()
    directory = (root / storage_dir).resolve()
    if directory.parent != root:
        raise EntryIdReconciliationError(f"Dataset path escapes data root: {directory}")
    return directory


def _desired_entry_ids(
    rows: list[dict[str, Any]], mapping: dict[str, dict[str, Any]]
) -> dict[str, str]:
    by_id = {str(row["dataset_id"]): row for row in rows}
    desired = {dataset_id: str(row["entry_id"]) for dataset_id, row in by_id.items()}
    for dataset_id, change in mapping.items():
        row = by_id.get(dataset_id)
        if row is None:
            raise EntryIdReconciliationError(f"Mapped dataset does not exist: {dataset_id}")
        if row["dataset_type"] != "full":
            raise EntryIdReconciliationError(f"Mapped dataset is not full: {dataset_id}")
        if row["entry_id"] != change["expected_entry_id"]:
            raise EntryIdReconciliationError(
                f"Current entry_id differs for {dataset_id!r}: {row['entry_id']!r}"
            )
        desired[dataset_id] = change["target_entry_id"]

    derived_by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["dataset_type"] in {"train", "test"}:
            split_id = _safe_identifier(row["split_id"], f"{row['dataset_id']}.split_id")
            derived_by_split.setdefault(split_id, []).append(row)
    for split_id, challenge_rows in derived_by_split.items():
        source_ids: set[str] = set()
        for row in challenge_rows:
            derivation = row["derivation"]
            if not isinstance(derivation, dict):
                raise EntryIdReconciliationError(
                    f"Derived dataset lacks derivation: {row['dataset_id']}"
                )
            values = derivation.get("source_dataset_ids")
            if not isinstance(values, list) or not values:
                raise EntryIdReconciliationError(
                    f"Derived dataset lacks sources: {row['dataset_id']}"
                )
            source_ids.update(map(str, values))
        missing = source_ids - by_id.keys()
        if missing:
            raise EntryIdReconciliationError(
                f"Challenge {split_id!r} references missing sources: {sorted(missing)}"
            )
        source_entries = {desired[source_id] for source_id in source_ids}
        target = next(iter(source_entries)) if len(source_entries) == 1 else split_id
        for row in challenge_rows:
            desired[str(row["dataset_id"])] = target
    return desired


def _inspect_current_artifacts(
    settings: Settings, row: dict[str, Any], expected_entry_id: str
) -> dict[str, str]:
    directory = _dataset_directory(settings, row)
    paths = {name: directory / name for name in ARTIFACT_NAMES}
    for path in paths.values():
        if not path.is_file():
            raise EntryIdReconciliationError(f"Dataset artifact is missing: {path}")
    if paths["dataset.h5mu"].stat().st_size != row["file_size"]:
        raise EntryIdReconciliationError(f"MuData size differs: {directory}")
    if _sha256(paths["dataset.h5mu"]) != row["sha256"]:
        raise EntryIdReconciliationError(f"MuData SHA-256 differs: {directory}")
    metadata = _read_yaml(paths["metadata.yaml"])
    manifest = _read_json(paths["manifest.json"])
    if metadata["database"].get("entry_id") != expected_entry_id:
        raise EntryIdReconciliationError(f"Metadata entry_id differs: {directory}")
    if manifest.get("database", {}).get("entry_id") != expected_entry_id:
        raise EntryIdReconciliationError(f"Manifest entry_id differs: {directory}")
    h5mu_record = manifest.get("files", {}).get("h5mu", {})
    if h5mu_record.get("sha256") != row["sha256"] or h5mu_record.get("size") != row["file_size"]:
        raise EntryIdReconciliationError(f"Manifest MuData record differs: {directory}")
    if paths["checksum.sha256"].read_text(encoding="utf-8") != (
        f"{row['sha256']}  dataset.h5mu\n"
    ):
        raise EntryIdReconciliationError(f"Checksum file differs: {directory}")
    try:
        with h5py.File(paths["dataset.h5mu"], "r") as handle:
            value = handle["uns/database/entry_id"][()]
            h5_entry_id = value.decode() if isinstance(value, bytes) else str(value)
    except (KeyError, OSError, RuntimeError, UnicodeError) as exc:
        raise EntryIdReconciliationError(f"Unable to inspect MuData {directory}: {exc}") from exc
    if h5_entry_id != expected_entry_id:
        raise EntryIdReconciliationError(f"MuData entry_id differs: {directory}")
    return {name: _sha256(path) for name, path in paths.items()}


def _replace_h5mu_entry_id(path: Path, entry_id: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    path.chmod(mode | stat.S_IWUSR)
    try:
        with h5py.File(path, "r+") as handle:
            database = handle["uns/database"]
            del database["entry_id"]
            entry = database.create_dataset(
                "entry_id", data=entry_id, dtype=h5py.string_dtype(encoding="utf-8")
            )
            entry.attrs["encoding-type"] = "string"
            entry.attrs["encoding-version"] = "0.2.0"
            handle.flush()
    finally:
        path.chmod(mode)


def _stage_dataset(
    settings: Settings,
    row: dict[str, Any],
    target_entry_id: str,
    stage_root: Path,
    old_hashes: dict[str, str],
    *,
    validate: bool = True,
) -> dict[str, Any]:
    directory = _dataset_directory(settings, row)
    staged = stage_root / "datasets" / str(row["storage_dir"])
    staged.mkdir(parents=True)
    shutil.copy2(directory / "dataset.h5mu", staged / "dataset.h5mu")
    _replace_h5mu_entry_id(staged / "dataset.h5mu", target_entry_id)
    new_size = (staged / "dataset.h5mu").stat().st_size
    new_sha256 = _sha256(staged / "dataset.h5mu")

    metadata = _read_yaml(directory / "metadata.yaml")
    metadata["database"]["entry_id"] = target_entry_id
    if validate:
        try:
            MetadataDocument.model_validate(metadata)
        except ValidationError as exc:
            raise EntryIdReconciliationError(
                f"Reconciled metadata is invalid for {row['dataset_id']!r}: {exc}"
            ) from exc
    (staged / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    manifest = _read_json(directory / "manifest.json")
    manifest["database"]["entry_id"] = target_entry_id
    manifest["files"]["h5mu"]["size"] = new_size
    manifest["files"]["h5mu"]["sha256"] = new_sha256
    (staged / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staged / "checksum.sha256").write_text(
        f"{new_sha256}  dataset.h5mu\n", encoding="utf-8"
    )
    return {
        "dataset_id": row["dataset_id"],
        "dataset_type": row["dataset_type"],
        "split_id": row["split_id"],
        "storage_dir": row["storage_dir"],
        "old_entry_id": row["entry_id"],
        "new_entry_id": target_entry_id,
        "old_size": row["file_size"],
        "new_size": new_size,
        "old_sha256": row["sha256"],
        "new_sha256": new_sha256,
        "n_obs": row["n_obs"],
        "coordinate_dimensions": row["coordinate_dimensions"],
        "sample_ids": row["sample_ids"],
        "files": [
            {
                "name": name,
                "path": str((directory / name).resolve()),
                "staged_path": str((staged / name).resolve()),
                "old_sha256": old_hashes[name],
                "new_sha256": _sha256(staged / name),
            }
            for name in ARTIFACT_NAMES
        ],
    }


def rebind_difficulty_snapshot_sha256(
    document: dict[str, Any], sha_changes: dict[str, tuple[str, str]]
) -> tuple[dict[str, Any], int]:
    """Return a copy with only matching train/test SHA-256 fields replaced."""
    rebound = copy.deepcopy(document)
    updated: set[str] = set()
    rows = rebound.get("challenges")
    if not isinstance(rows, list):
        raise EntryIdReconciliationError("Difficulty snapshot challenges must be a list")
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "success":
            continue
        for side_name in ("train", "test"):
            side = row.get(side_name)
            if not isinstance(side, dict):
                raise EntryIdReconciliationError("Successful difficulty row lacks a side")
            dataset_id = side.get("dataset_id")
            if dataset_id not in sha_changes:
                continue
            old_sha256, new_sha256 = sha_changes[dataset_id]
            if side.get("sha256") != old_sha256:
                raise EntryIdReconciliationError(
                    f"Difficulty snapshot has unexpected SHA-256 for {dataset_id!r}"
                )
            side["sha256"] = new_sha256
            updated.add(dataset_id)
    if updated != set(sha_changes):
        missing = sorted(set(sha_changes) - updated)
        raise EntryIdReconciliationError(
            "Difficulty snapshot does not contain changed derived datasets: " + ", ".join(missing)
        )

    normalized_old = copy.deepcopy(document)
    normalized_new = copy.deepcopy(rebound)
    for normalized in (normalized_old, normalized_new):
        for row in normalized["challenges"]:
            if isinstance(row, dict):
                for side_name in ("train", "test"):
                    side = row.get(side_name)
                    if isinstance(side, dict) and side.get("dataset_id") in sha_changes:
                        side["sha256"] = "<rebound>"
    if normalized_old != normalized_new:
        raise EntryIdReconciliationError(
            "Difficulty snapshot changed outside train/test SHA-256 fields"
        )
    return rebound, len(updated)


def _stage_snapshot(
    snapshot_path: Path,
    stage_root: Path,
    changed_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    derived = {
        record["dataset_id"]: (record["old_sha256"], record["new_sha256"])
        for record in changed_records
        if record["dataset_type"] in {"train", "test"}
    }
    if not derived or not snapshot_path.is_file():
        return None
    original = _read_json(snapshot_path)
    rebound, update_count = rebind_difficulty_snapshot_sha256(original, derived)
    staged = stage_root / "challenge_difficulty.json"
    staged.write_text(
        json.dumps(rebound, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(snapshot_path.resolve()),
        "staged_path": str(staged.resolve()),
        "old_sha256": _sha256(snapshot_path),
        "new_sha256": _sha256(staged),
        "updated_dataset_count": update_count,
        "metrics_recomputed": False,
    }


def _stage_visualization(
    root: Path,
    record: dict[str, Any],
    stage_root: Path,
    stamp: str,
    *,
    validate: bool = True,
) -> dict[str, Any] | None:
    dataset_id = record["dataset_id"]
    status_path = root / dataset_id / "status.json"
    if not status_path.is_file():
        return None
    status = _read_json(status_path)
    if status.get("state") != "success" or status.get("dataset_id") != dataset_id:
        return None
    old_generation_id = _safe_identifier(status.get("generation_id"), "generation_id")
    old_generation = status_path.parent / "generations" / old_generation_id
    manifest = _read_json(old_generation / "manifest.json")
    report = _read_json(old_generation / "report.json")
    manifest_path = old_generation / "manifest.json"
    if validate and status.get("manifest_sha256") != _sha256(manifest_path):
        raise EntryIdReconciliationError(
            f"Visualization status digest differs for {dataset_id!r}"
        )
    source = manifest.get("source", {})
    if validate and source.get("sha256") != report.get("source_sha256"):
        raise EntryIdReconciliationError(
            f"Visualization manifest/report SHA-256 differs for {dataset_id!r}"
        )
    if validate and (
        source.get("observation_count") != record["n_obs"]
        or source.get("coordinate_dimensions") != record["coordinate_dimensions"]
        or source.get("sample_ids") != record["sample_ids"]
    ):
        raise EntryIdReconciliationError(
            f"Visualization source identity differs for {dataset_id!r}"
        )
    token = secrets.token_hex(5)
    new_generation_id = f"{stamp.replace('-', '').replace(':', '').replace('.', '')}-{token}"
    new_generation_id = new_generation_id[:128]
    staged_generation = stage_root / "visualizations" / dataset_id / new_generation_id
    staged_generation.mkdir(parents=True)
    for source in old_generation.rglob("*"):
        if source.is_file() and source.name not in {"manifest.json", "report.json"}:
            relative = source.relative_to(old_generation)
            _link_or_copy(source, staged_generation / relative)
    now = datetime.now(UTC).isoformat()
    report["generation_id"] = new_generation_id
    report["source_sha256"] = record["new_sha256"]
    report_path = staged_generation / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["generation_id"] = new_generation_id
    manifest["generated_at"] = now
    manifest["source"]["sha256"] = record["new_sha256"]
    manifest["report"]["sha256"] = _sha256(report_path)
    manifest["report"]["size"] = report_path.stat().st_size
    manifest_path = staged_generation / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    new_status = copy.deepcopy(status)
    new_status["generation_id"] = new_generation_id
    new_status["manifest_sha256"] = _sha256(manifest_path)
    new_status["updated_at"] = now
    staged_status = staged_generation.parent / "status.json"
    staged_status.write_text(
        json.dumps(new_status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "dataset_id": dataset_id,
        "status_path": str(status_path.resolve()),
        "staged_status_path": str(staged_status.resolve()),
        "old_status_sha256": _sha256(status_path),
        "old_generation_id": old_generation_id,
        "new_generation_id": new_generation_id,
        "staged_generation_path": str(staged_generation.resolve()),
        "generation_path": str(
            (status_path.parent / "generations" / new_generation_id).resolve()
        ),
    }


def _update_catalogue_copy(path: Path, records: list[dict[str, Any]]) -> None:
    try:
        with sqlite3.connect(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for record in records:
                cursor = connection.execute(
                    "UPDATE datasets SET entry_id = ?, file_size = ?, sha256 = ? "
                    "WHERE dataset_id = ? AND entry_id = ? AND sha256 = ?",
                    (
                        record["new_entry_id"],
                        record["new_size"],
                        record["new_sha256"],
                        record["dataset_id"],
                        record["old_entry_id"],
                        record["old_sha256"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise EntryIdReconciliationError(
                        f"Catalogue row changed for {record['dataset_id']!r}"
                    )
            connection.commit()
    except sqlite3.Error as exc:
        raise EntryIdReconciliationError(f"Unable to update catalogue copy: {exc}") from exc


def inspect_entry_id_reconciliation(
    settings: Settings,
    mapping_path: Path,
    *,
    verify_artifacts: bool = True,
    skip_difficulty_snapshot: bool = False,
) -> tuple[EntryIdReconciliationInventory, dict[str, str]]:
    mapping = _load_mapping(mapping_path)
    rows = _catalogue_rows(settings.database_path)
    desired = _desired_entry_ids(rows, mapping)
    changed = [row for row in rows if desired[row["dataset_id"]] != row["entry_id"]]
    if not changed:
        raise EntryIdReconciliationError("Mapping produces no catalogue changes")
    if verify_artifacts:
        for row in changed:
            _inspect_current_artifacts(settings, row, str(row["entry_id"]))
    full_changes = [row for row in changed if row["dataset_type"] == "full"]
    derived_changes = [row for row in changed if row["dataset_type"] in {"train", "test"}]
    visualization_count = sum(
        (
            settings.cell_type_visualization_root
            / str(row["dataset_id"])
            / "status.json"
        ).is_file()
        for row in full_changes
        if settings.cell_type_visualization_root is not None
    )
    snapshot_path = settings.database_path.parent / "challenge_difficulty.json"
    snapshot_updates = (
        len(derived_changes)
        if snapshot_path.is_file() and not skip_difficulty_snapshot
        else 0
    )
    resulting_entries = len(
        {desired[row["dataset_id"]] for row in rows if row["dataset_type"] == "full"}
    )
    return (
        EntryIdReconciliationInventory(
            explicit_full_changes=len(full_changes),
            propagated_derived_changes=len(derived_changes),
            affected_challenges=len({row["split_id"] for row in derived_changes}),
            affected_visualizations=visualization_count,
            snapshot_sha256_updates=snapshot_updates,
            resulting_full_entries=resulting_entries,
        ),
        desired,
    )


def reconcile_entry_ids(
    settings: Settings,
    mapping_path: Path,
    *,
    dry_run: bool = False,
    skip_difficulty_snapshot: bool = False,
    skip_validation: bool = False,
) -> EntryIdReconciliationInventory | EntryIdReconciliationResult:
    inventory, desired = inspect_entry_id_reconciliation(
        settings,
        mapping_path,
        verify_artifacts=dry_run,
        skip_difficulty_snapshot=skip_difficulty_snapshot,
    )
    if dry_run:
        return inventory
    rows = _catalogue_rows(settings.database_path)
    changed_rows = [row for row in rows if desired[row["dataset_id"]] != row["entry_id"]]
    source_digest = _catalogue_digest(settings.database_path)
    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    parent = settings.database_path.parent
    stage_root = parent / f".entry_reconciliation_staging_{stamp}"
    backup_path = parent / "backups" / f"entry_reconciliation_{stamp}"
    report_path = parent / "migrations" / f"entry_reconciliation_{stamp}.json"
    for path in (stage_root, backup_path, report_path):
        if path.exists():
            raise EntryIdReconciliationError(f"Reconciliation target exists: {path}")

    mapping = _load_mapping(mapping_path)
    report: dict[str, Any] = {
        "migration": RECONCILIATION_NAME,
        "status": "staging",
        "started_at": started.isoformat(),
        "completed_at": None,
        "backup_deleted_at": None,
        "catalogue_path": str(settings.database_path.resolve()),
        "data_root": str(settings.data_root.resolve()),
        "mapping_path": str(mapping_path.resolve()),
        "mapping_sha256": _sha256(mapping_path),
        "source_catalogue_content_sha256": source_digest,
        "backup_path": str(backup_path.resolve()),
        "inventory": inventory.as_dict(),
        "evidence": list(mapping.values()),
        "datasets": [],
        "visualizations": [],
        "difficulty_snapshot": None,
        "difficulty_snapshot_skipped": skip_difficulty_snapshot,
        "validation_skipped": skip_validation,
    }
    activated_files: list[dict[str, Any]] = []
    activated_visualizations: list[dict[str, Any]] = []
    snapshot_activated = False
    catalogue_activated = False
    try:
        stage_root.mkdir(parents=True)
        backup_path.mkdir(parents=True)
        staged_catalogue = stage_root / "catalog.db"
        _sqlite_backup(settings.database_path, staged_catalogue)
        for row in changed_rows:
            if skip_validation:
                directory = _dataset_directory(settings, row)
                hashes = {
                    "dataset.h5mu": str(row["sha256"]),
                    **{
                        name: _sha256(directory / name)
                        for name in ARTIFACT_NAMES
                        if name != "dataset.h5mu"
                    },
                }
            else:
                hashes = _inspect_current_artifacts(
                    settings, row, str(row["entry_id"])
                )
            report["datasets"].append(
                _stage_dataset(
                    settings,
                    row,
                    desired[row["dataset_id"]],
                    stage_root,
                    hashes,
                    validate=not skip_validation,
                )
            )
        _update_catalogue_copy(staged_catalogue, report["datasets"])

        if not skip_difficulty_snapshot:
            report["difficulty_snapshot"] = _stage_snapshot(
                settings.database_path.parent / "challenge_difficulty.json",
                stage_root,
                report["datasets"],
            )
        if settings.cell_type_visualization_root is not None:
            for record in report["datasets"]:
                if record["dataset_type"] == "full":
                    visualization = _stage_visualization(
                        settings.cell_type_visualization_root,
                        record,
                        stage_root,
                        started.strftime("%Y%m%dT%H%M%S%fZ"),
                        validate=not skip_validation,
                    )
                    if visualization is not None:
                        report["visualizations"].append(visualization)

        if _catalogue_digest(settings.database_path) != source_digest:
            raise EntryIdReconciliationError("Catalogue changed during staging")
        if not skip_validation:
            for record in report["datasets"]:
                for file_record in record["files"]:
                    if _sha256(Path(file_record["path"])) != file_record["old_sha256"]:
                        raise EntryIdReconciliationError(
                            f"Dataset artifact changed during staging: {file_record['path']}"
                        )
        snapshot = report["difficulty_snapshot"]
        if snapshot and _sha256(Path(snapshot["path"])) != snapshot["old_sha256"]:
            raise EntryIdReconciliationError("Difficulty snapshot changed during staging")

        _sqlite_backup(settings.database_path, backup_path / "catalog.db")
        for record in report["datasets"]:
            backup_directory = backup_path / "datasets" / record["storage_dir"]
            for file_record in record["files"]:
                backup_file = backup_directory / file_record["name"]
                _link_or_copy(Path(file_record["path"]), backup_file)
                file_record["backup_path"] = str(backup_file.resolve())
        for visualization in report["visualizations"]:
            backup = backup_path / "visualizations" / visualization["dataset_id"] / "status.json"
            _link_or_copy(Path(visualization["status_path"]), backup)
            visualization["backup_status_path"] = str(backup.resolve())
        if snapshot:
            backup = backup_path / "challenge_difficulty.json"
            _link_or_copy(Path(snapshot["path"]), backup)
            snapshot["backup_path"] = str(backup.resolve())

        report["status"] = "staged"
        _write_json_atomic(report_path, report)
        for record in report["datasets"]:
            for file_record in record["files"]:
                os.replace(file_record["staged_path"], file_record["path"])
                activated_files.append(file_record)
        for visualization in report["visualizations"]:
            destination = Path(visualization["generation_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(visualization["staged_generation_path"], destination)
            os.replace(visualization["staged_status_path"], visualization["status_path"])
            activated_visualizations.append(visualization)
        if snapshot:
            os.replace(snapshot["staged_path"], snapshot["path"])
            snapshot_activated = True
        os.replace(staged_catalogue, settings.database_path)
        catalogue_activated = True

        if not skip_validation:
            active_rows = {
                row["dataset_id"]: row
                for row in _catalogue_rows(settings.database_path)
            }
            for record in report["datasets"]:
                row = active_rows[record["dataset_id"]]
                expected = (
                    record["new_entry_id"],
                    record["new_size"],
                    record["new_sha256"],
                )
                if (row["entry_id"], row["file_size"], row["sha256"]) != expected:
                    raise EntryIdReconciliationError(
                        f"Active catalogue differs for {record['dataset_id']!r}"
                    )
                _inspect_current_artifacts(settings, row, record["new_entry_id"])
            if snapshot and _sha256(Path(snapshot["path"])) != snapshot["new_sha256"]:
                raise EntryIdReconciliationError("Active difficulty snapshot differs")
        report["status"] = "complete"
        report["completed_at"] = datetime.now(UTC).isoformat()
        _write_json_atomic(report_path, report)
        shutil.rmtree(stage_root)
        return EntryIdReconciliationResult(report_path, backup_path, inventory)
    except Exception as exc:
        rollback_errors: list[str] = []
        if catalogue_activated:
            try:
                _atomic_restore(backup_path / "catalog.db", settings.database_path)
            except Exception as rollback_exc:
                rollback_errors.append(f"catalogue rollback failed: {rollback_exc}")
        if snapshot_activated and report.get("difficulty_snapshot"):
            try:
                snapshot = report["difficulty_snapshot"]
                _atomic_restore(Path(snapshot["backup_path"]), Path(snapshot["path"]))
            except Exception as rollback_exc:
                rollback_errors.append(f"snapshot rollback failed: {rollback_exc}")
        for visualization in reversed(activated_visualizations):
            try:
                _atomic_restore(
                    Path(visualization["backup_status_path"]),
                    Path(visualization["status_path"]),
                )
                shutil.rmtree(visualization["generation_path"], ignore_errors=True)
            except Exception as rollback_exc:
                rollback_errors.append(f"visualization rollback failed: {rollback_exc}")
        for file_record in reversed(activated_files):
            try:
                _atomic_restore(Path(file_record["backup_path"]), Path(file_record["path"]))
            except Exception as rollback_exc:
                rollback_errors.append(f"artifact rollback failed: {rollback_exc}")
        report["status"] = "rollback_failed" if rollback_errors else "rolled_back"
        report["error"] = str(exc)
        report["rollback_errors"] = rollback_errors
        try:
            _write_json_atomic(report_path, report)
        except OSError:
            pass
        shutil.rmtree(stage_root, ignore_errors=True)
        detail = f"Entry reconciliation failed and was rolled back: {exc}"
        if rollback_errors:
            detail += "; " + "; ".join(rollback_errors)
        raise EntryIdReconciliationError(detail) from exc


def finalize_entry_id_reconciliation(report_path: Path, settings: Settings) -> Path:
    report = _read_json(report_path)
    if report.get("migration") != RECONCILIATION_NAME or report.get("status") != "complete":
        raise EntryIdReconciliationError("Report is not a completed entry reconciliation")
    if report.get("backup_deleted_at") is not None:
        raise EntryIdReconciliationError("Reconciliation backup was already finalized")
    if Path(report.get("catalogue_path", "")).resolve() != settings.database_path.resolve():
        raise EntryIdReconciliationError("Report belongs to a different catalogue")
    backup = Path(report["backup_path"]).resolve()
    backups_root = (settings.database_path.parent / "backups").resolve()
    if backup.parent != backups_root or not backup.name.startswith("entry_reconciliation_"):
        raise EntryIdReconciliationError(f"Unsafe backup path: {backup}")
    if not (backup / "catalog.db").is_file():
        raise EntryIdReconciliationError(f"Incomplete backup: {backup}")
    rows = {row["dataset_id"]: row for row in _catalogue_rows(settings.database_path)}
    for record in report["datasets"]:
        row = rows.get(record["dataset_id"])
        if row is None or row["entry_id"] != record["new_entry_id"]:
            raise EntryIdReconciliationError(
                f"Active reconciliation differs for {record['dataset_id']!r}"
            )
        _inspect_current_artifacts(settings, row, record["new_entry_id"])
    snapshot = report.get("difficulty_snapshot")
    if snapshot and _sha256(Path(snapshot["path"])) != snapshot["new_sha256"]:
        raise EntryIdReconciliationError("Active difficulty snapshot differs")
    shutil.rmtree(backup)
    report["backup_deleted_at"] = datetime.now(UTC).isoformat()
    _write_json_atomic(report_path, report)
    return backup
