from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import yaml
from pydantic import ValidationError

from .config import PROJECT_ROOT, Settings
from .schemas import MetadataDocument

SOURCE_CATALOGUE_VERSION = "3"
TARGET_CATALOGUE_VERSION = "4"


class CatalogueV4MigrationError(RuntimeError):
    """Raised when the catalogue v4 migration cannot complete safely."""


def _validate_license_free_v4_metadata(value: dict[str, Any]) -> None:
    """Validate v4 metadata without making the later v5 entry tag part of v4."""
    candidate = dict(value)
    database = dict(candidate.get("database", {}))
    database.setdefault("entry_id", database.get("dataset_id"))
    candidate["database"] = database
    MetadataDocument.model_validate(candidate)


@dataclass(frozen=True)
class CatalogueV4Inventory:
    dataset_count: int
    formal_metadata_count: int
    supplemental_metadata_count: int
    license_field_count: int
    non_null_license_count: int
    h5mu_count: int
    manifest_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "dataset_count": self.dataset_count,
            "formal_metadata_count": self.formal_metadata_count,
            "supplemental_metadata_count": self.supplemental_metadata_count,
            "license_field_count": self.license_field_count,
            "non_null_license_count": self.non_null_license_count,
            "h5mu_count": self.h5mu_count,
            "manifest_count": self.manifest_count,
        }


@dataclass(frozen=True)
class CatalogueV4MigrationResult:
    report_path: Path
    backup_path: Path
    inventory: CatalogueV4Inventory


@dataclass(frozen=True)
class _MetadataTarget:
    path: Path
    group: str
    relative_path: Path
    formal: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogueV4MigrationError(f"Unable to read metadata YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CatalogueV4MigrationError(f"Metadata YAML root must be a mapping: {path}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _catalogue_version(path: Path) -> str | None:
    try:
        with sqlite3.connect(path) as connection:
            row = connection.execute(
                "SELECT value FROM catalogue_metadata WHERE key = 'schema_version'"
            ).fetchone()
    except sqlite3.Error as exc:
        raise CatalogueV4MigrationError(f"Unable to read catalogue version: {exc}") from exc
    return None if row is None else str(row[0])


def _catalogue_columns(path: Path) -> list[str]:
    try:
        with sqlite3.connect(path) as connection:
            return [str(row[1]) for row in connection.execute("PRAGMA table_info(datasets)")]
    except sqlite3.Error as exc:
        raise CatalogueV4MigrationError(f"Unable to inspect catalogue columns: {exc}") from exc


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _catalogue_digest(path: Path, columns: list[str]) -> str:
    selected = ", ".join(_quoted_identifier(column) for column in columns)
    try:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                f"SELECT {selected} FROM datasets ORDER BY dataset_id"
            ).fetchall()
    except sqlite3.Error as exc:
        raise CatalogueV4MigrationError(f"Unable to read catalogue rows: {exc}") from exc
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _catalogue_datasets(path: Path) -> list[dict[str, Any]]:
    try:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                "SELECT dataset_id, storage_dir, file_size, sha256 "
                "FROM datasets ORDER BY dataset_id"
            ).fetchall()
    except sqlite3.Error as exc:
        raise CatalogueV4MigrationError(f"Unable to read catalogue inventory: {exc}") from exc
    return [
        {
            "dataset_id": str(dataset_id),
            "storage_dir": str(storage_dir),
            "file_size": int(file_size),
            "sha256": str(sha256),
        }
        for dataset_id, storage_dir, file_size, sha256 in rows
    ]


def _supplemental_metadata_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        {
            path
            for pattern in ("metadata.yaml", "metadata.yml", "*.metadata.yaml", "*.metadata.yml")
            for path in root.rglob(pattern)
            if path.is_file()
        }
    )


def _metadata_targets(
    settings: Settings,
    datasets: list[dict[str, Any]],
    temp_root: Path,
    exp_root: Path,
) -> list[_MetadataTarget]:
    targets = [
        _MetadataTarget(
            path=settings.data_root / dataset["storage_dir"] / "metadata.yaml",
            group="data",
            relative_path=Path(dataset["storage_dir"]) / "metadata.yaml",
            formal=True,
        )
        for dataset in datasets
    ]
    for group, root in (("temp", temp_root), ("exp", exp_root)):
        targets.extend(
            _MetadataTarget(
                path=path,
                group=group,
                relative_path=path.relative_to(root),
                formal=False,
            )
            for path in _supplemental_metadata_paths(root)
        )
    resolved = [target.path.resolve() for target in targets]
    if len(resolved) != len(set(resolved)):
        raise CatalogueV4MigrationError("Metadata migration targets overlap.")
    return sorted(targets, key=lambda target: (target.group, str(target.relative_path)))


def _license_values(value: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    if "license" in value:
        values.append(value["license"])
    database = value.get("database")
    if isinstance(database, dict) and "license" in database:
        values.append(database["license"])
    return values


def _without_license(value: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    cleaned = dict(value)
    removed = _license_values(cleaned)
    cleaned.pop("license", None)
    database = cleaned.get("database")
    if isinstance(database, dict) and "license" in database:
        cleaned["database"] = dict(database)
        cleaned["database"].pop("license", None)
    return cleaned, removed


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _h5mu_license_paths(path: Path) -> list[str]:
    matches: list[str] = []
    try:
        with h5py.File(path, "r") as handle:
            handle.visit(
                lambda name: matches.append(name)
                if any(part.lower() == "license" for part in name.split("/"))
                else None
            )
    except (OSError, RuntimeError) as exc:
        raise CatalogueV4MigrationError(f"Unable to inspect MuData file {path}: {exc}") from exc
    return matches


def _inspect_dataset_files(settings: Settings, datasets: list[dict[str, Any]]) -> None:
    indexed_directories = {dataset["storage_dir"] for dataset in datasets}
    actual_directories = {
        path.name
        for path in settings.data_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    if indexed_directories != actual_directories:
        raise CatalogueV4MigrationError(
            "Catalogue/dataset directory mismatch; "
            f"missing={sorted(indexed_directories - actual_directories)}, "
            f"unindexed={sorted(actual_directories - indexed_directories)}."
        )
    for dataset in datasets:
        directory = settings.data_root / dataset["storage_dir"]
        metadata_path = directory / "metadata.yaml"
        h5mu_path = directory / "dataset.h5mu"
        manifest_path = directory / "manifest.json"
        for path in (metadata_path, h5mu_path, manifest_path):
            if not path.is_file():
                raise CatalogueV4MigrationError(f"Dataset file is missing: {path}")
        matches = _h5mu_license_paths(h5mu_path)
        if matches:
            raise CatalogueV4MigrationError(
                f"MuData contains unsupported License fields {matches}: {h5mu_path}"
            )
        if h5mu_path.stat().st_size != dataset["file_size"]:
            raise CatalogueV4MigrationError(f"MuData size differs from catalogue: {h5mu_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CatalogueV4MigrationError(
                f"Unable to read dataset manifest {manifest_path}: {exc}"
            ) from exc
        if _contains_key(manifest, "license"):
            raise CatalogueV4MigrationError(
                f"Manifest contains an unsupported License field: {manifest_path}"
            )
        file_record = manifest.get("files", {}).get("h5mu", {})
        if (
            file_record.get("size") != dataset["file_size"]
            or file_record.get("sha256") != dataset["sha256"]
        ):
            raise CatalogueV4MigrationError(
                f"Manifest/catalogue MuData record mismatch: {manifest_path}"
            )


def inspect_catalogue_v3(
    settings: Settings,
    *,
    temp_root: Path = PROJECT_ROOT / "temp",
    exp_root: Path = PROJECT_ROOT / "exp",
) -> CatalogueV4Inventory:
    if not settings.database_path.is_file():
        raise CatalogueV4MigrationError(f"Catalogue does not exist: {settings.database_path}")
    if not settings.data_root.is_dir():
        raise CatalogueV4MigrationError(f"Dataset root does not exist: {settings.data_root}")
    version = _catalogue_version(settings.database_path)
    if version != SOURCE_CATALOGUE_VERSION:
        raise CatalogueV4MigrationError(
            f"Expected catalogue version {SOURCE_CATALOGUE_VERSION}, found {version!r}."
        )
    columns = _catalogue_columns(settings.database_path)
    if "license" not in columns:
        raise CatalogueV4MigrationError(
            "Catalogue v3 does not contain the expected license column."
        )

    datasets = _catalogue_datasets(settings.database_path)
    _inspect_dataset_files(settings, datasets)
    targets = _metadata_targets(settings, datasets, temp_root, exp_root)
    license_values: list[Any] = []
    for target in targets:
        if not target.path.is_file():
            raise CatalogueV4MigrationError(f"Metadata file is missing: {target.path}")
        value = _read_yaml(target.path)
        cleaned, removed = _without_license(value)
        license_values.extend(removed)
        if target.formal:
            try:
                _validate_license_free_v4_metadata(cleaned)
            except ValidationError as exc:
                raise CatalogueV4MigrationError(
                    f"Metadata will not satisfy the License-free schema: {target.path}: {exc}"
                ) from exc

    return CatalogueV4Inventory(
        dataset_count=len(datasets),
        formal_metadata_count=sum(target.formal for target in targets),
        supplemental_metadata_count=sum(not target.formal for target in targets),
        license_field_count=len(license_values),
        non_null_license_count=sum(value is not None for value in license_values),
        h5mu_count=len(datasets),
        manifest_count=len(datasets),
    )


def _sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sqlite3.connect(source) as source_connection, sqlite3.connect(
            destination
        ) as destination_connection:
            source_connection.backup(destination_connection)
    except sqlite3.Error as exc:
        raise CatalogueV4MigrationError(f"Unable to back up catalogue: {exc}") from exc


def _migrate_catalogue_copy(path: Path) -> None:
    try:
        with sqlite3.connect(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("ALTER TABLE datasets DROP COLUMN license")
            connection.execute(
                "UPDATE catalogue_metadata SET value = ? WHERE key = 'schema_version'",
                (TARGET_CATALOGUE_VERSION,),
            )
            connection.commit()
    except sqlite3.Error as exc:
        raise CatalogueV4MigrationError(f"Unable to migrate catalogue copy: {exc}") from exc


def _atomic_copy_replace(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _verify_active_catalogue(
    settings: Settings,
    report: dict[str, Any],
    *,
    verify_h5mu_sha256: bool,
) -> None:
    if _catalogue_version(settings.database_path) != TARGET_CATALOGUE_VERSION:
        raise CatalogueV4MigrationError("Active catalogue is not version 4.")
    columns = _catalogue_columns(settings.database_path)
    if "license" in columns:
        raise CatalogueV4MigrationError("Active catalogue still contains the license column.")
    if _catalogue_digest(settings.database_path, columns) != report["catalogue_content_sha256"]:
        raise CatalogueV4MigrationError("Active catalogue content changed during migration.")

    for record in report["metadata_files"]:
        path = Path(record["path"])
        if not path.is_file() or _sha256(path) != record["new_sha256"]:
            raise CatalogueV4MigrationError(f"Migrated metadata does not match report: {path}")
        value = _read_yaml(path)
        if _license_values(value):
            raise CatalogueV4MigrationError(f"Migrated metadata still contains License: {path}")
        if record["formal"]:
            try:
                _validate_license_free_v4_metadata(value)
            except ValidationError as exc:
                raise CatalogueV4MigrationError(
                    f"Migrated metadata is invalid: {path}: {exc}"
                ) from exc

    datasets = _catalogue_datasets(settings.database_path)
    expected_h5mu = {record["dataset_id"]: record for record in report["h5mu_files"]}
    if {dataset["dataset_id"] for dataset in datasets} != set(expected_h5mu):
        raise CatalogueV4MigrationError("Active catalogue dataset IDs changed during migration.")
    for dataset in datasets:
        record = expected_h5mu[dataset["dataset_id"]]
        path = settings.data_root / dataset["storage_dir"] / "dataset.h5mu"
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise CatalogueV4MigrationError(f"MuData file changed during migration: {path}")
        if verify_h5mu_sha256 and _sha256(path) != record["sha256"]:
            raise CatalogueV4MigrationError(f"MuData checksum changed during migration: {path}")


def migrate_catalogue_v4(
    settings: Settings,
    *,
    temp_root: Path = PROJECT_ROOT / "temp",
    exp_root: Path = PROJECT_ROOT / "exp",
    dry_run: bool = False,
) -> CatalogueV4Inventory | CatalogueV4MigrationResult:
    temp_root = temp_root.resolve()
    exp_root = exp_root.resolve()
    inventory = inspect_catalogue_v3(settings, temp_root=temp_root, exp_root=exp_root)
    if dry_run:
        return inventory

    started_at = datetime.now(UTC)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    data_parent = settings.database_path.parent
    stage_root = data_parent / f".catalogue_v4_staging_{stamp}"
    backup_path = data_parent / "backups" / f"catalogue_v3_{stamp}"
    report_path = data_parent / "migrations" / f"catalogue_v4_{stamp}.json"
    for path in (stage_root, backup_path, report_path):
        if path.exists():
            raise CatalogueV4MigrationError(f"Migration target already exists: {path}")

    datasets = _catalogue_datasets(settings.database_path)
    old_columns = _catalogue_columns(settings.database_path)
    retained_columns = [column for column in old_columns if column != "license"]
    old_digest = _catalogue_digest(settings.database_path, retained_columns)
    targets = _metadata_targets(settings, datasets, temp_root, exp_root)
    report: dict[str, Any] = {
        "migration": "catalogue_v4_remove_license",
        "source_catalogue_version": SOURCE_CATALOGUE_VERSION,
        "target_catalogue_version": TARGET_CATALOGUE_VERSION,
        "started_at": started_at.isoformat(),
        "completed_at": None,
        "backup_deleted_at": None,
        "status": "staging",
        "catalogue_path": str(settings.database_path.resolve()),
        "data_root": str(settings.data_root.resolve()),
        "temp_root": str(temp_root),
        "exp_root": str(exp_root),
        "backup_path": str(backup_path.resolve()),
        "inventory": inventory.as_dict(),
        "catalogue_columns": retained_columns,
        "catalogue_content_sha256": old_digest,
        "metadata_files": [],
        "h5mu_files": [
            {
                "dataset_id": dataset["dataset_id"],
                "size": dataset["file_size"],
                "sha256": dataset["sha256"],
            }
            for dataset in datasets
        ],
    }
    activated_metadata: list[dict[str, Any]] = []
    catalogue_activated = False
    try:
        stage_root.mkdir(parents=True)
        backup_path.mkdir(parents=True)
        backup_catalogue = backup_path / "catalog.db"
        stage_catalogue = stage_root / "catalog.db"
        _sqlite_backup(settings.database_path, backup_catalogue)
        _sqlite_backup(settings.database_path, stage_catalogue)
        _migrate_catalogue_copy(stage_catalogue)
        if _catalogue_columns(stage_catalogue) != retained_columns:
            raise CatalogueV4MigrationError("Migrated catalogue columns differ unexpectedly.")
        if _catalogue_digest(stage_catalogue, retained_columns) != old_digest:
            raise CatalogueV4MigrationError("Migrated catalogue rows differ unexpectedly.")

        for target in targets:
            value = _read_yaml(target.path)
            cleaned, removed = _without_license(value)
            if target.formal:
                try:
                    _validate_license_free_v4_metadata(cleaned)
                except ValidationError as exc:
                    raise CatalogueV4MigrationError(
                        f"Metadata will not satisfy the License-free schema: {target.path}: {exc}"
                    ) from exc
            relative = Path("metadata") / target.group / target.relative_path
            backup_file = backup_path / relative
            stage_file = stage_root / relative
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target.path, backup_file)
            _write_yaml(stage_file, cleaned)
            report["metadata_files"].append(
                {
                    "path": str(target.path.resolve()),
                    "backup": str(backup_file.resolve()),
                    "formal": target.formal,
                    "removed_field_count": len(removed),
                    "removed_non_null_count": sum(item is not None for item in removed),
                    "old_sha256": _sha256(target.path),
                    "new_sha256": _sha256(stage_file),
                    "staged_path": str(stage_file.resolve()),
                }
            )

        report["status"] = "staged"
        _write_json_atomic(report_path, report)
        for record in report["metadata_files"]:
            _atomic_copy_replace(Path(record["staged_path"]), Path(record["path"]))
            activated_metadata.append(record)
        _atomic_copy_replace(stage_catalogue, settings.database_path)
        catalogue_activated = True

        _verify_active_catalogue(settings, report, verify_h5mu_sha256=False)
        report["status"] = "complete"
        report["completed_at"] = datetime.now(UTC).isoformat()
        _write_json_atomic(report_path, report)
        shutil.rmtree(stage_root)
        return CatalogueV4MigrationResult(
            report_path=report_path,
            backup_path=backup_path,
            inventory=inventory,
        )
    except Exception as exc:
        rollback_errors: list[str] = []
        if catalogue_activated:
            try:
                _atomic_copy_replace(backup_path / "catalog.db", settings.database_path)
            except Exception as rollback_exc:  # pragma: no cover - catastrophic I/O failure
                rollback_errors.append(f"catalogue rollback failed: {rollback_exc}")
        for record in reversed(activated_metadata):
            try:
                _atomic_copy_replace(Path(record["backup"]), Path(record["path"]))
            except Exception as rollback_exc:  # pragma: no cover - catastrophic I/O failure
                rollback_errors.append(
                    f"metadata rollback failed for {record['path']}: {rollback_exc}"
                )
        report["status"] = "rollback_failed" if rollback_errors else "rolled_back"
        report["error"] = str(exc)
        report["rollback_errors"] = rollback_errors
        try:
            _write_json_atomic(report_path, report)
        except OSError:
            pass
        details = f"Catalogue v4 migration failed and was rolled back: {exc}"
        if rollback_errors:
            details += "; " + "; ".join(rollback_errors)
        raise CatalogueV4MigrationError(details) from exc


def finalize_catalogue_v4_migration(report_path: Path, settings: Settings) -> Path:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogueV4MigrationError(f"Unable to read migration report: {exc}") from exc
    if report.get("migration") != "catalogue_v4_remove_license":
        raise CatalogueV4MigrationError("Report does not describe a catalogue v4 migration.")
    if report.get("status") != "complete":
        raise CatalogueV4MigrationError("Migration is not complete and cannot be finalized.")
    if report.get("backup_deleted_at") is not None:
        raise CatalogueV4MigrationError("The catalogue v4 backup was already finalized.")

    backup_path = Path(report["backup_path"]).resolve()
    backups_root = (settings.database_path.parent / "backups").resolve()
    if backup_path.parent != backups_root or not backup_path.name.startswith("catalogue_v3_"):
        raise CatalogueV4MigrationError(f"Unsafe backup path in migration report: {backup_path}")
    if not backup_path.is_dir() or not (backup_path / "catalog.db").is_file():
        raise CatalogueV4MigrationError(f"Migration backup is incomplete: {backup_path}")

    _verify_active_catalogue(settings, report, verify_h5mu_sha256=True)
    shutil.rmtree(backup_path)
    report["backup_deleted_at"] = datetime.now(UTC).isoformat()
    _write_json_atomic(report_path, report)
    return backup_path
