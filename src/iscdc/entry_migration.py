from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import yaml
from pydantic import ValidationError

from .config import Settings
from .database import CATALOGUE_SCHEMA_VERSION
from .schemas import SAFE_IDENTIFIER_PATTERN, MetadataDocument

SOURCE_CATALOGUE_VERSION = "4"
TARGET_CATALOGUE_VERSION = "5"
MIGRATION_NAME = "catalogue_v5_entry_id"
ARTIFACT_NAMES = ("dataset.h5mu", "metadata.yaml", "manifest.json", "checksum.sha256")


class EntryIdMigrationError(RuntimeError):
    """Raised when the entry-ID migration cannot be completed safely."""


@dataclass(frozen=True)
class EntryIdMigrationInventory:
    dataset_count: int
    full_count: int
    train_count: int
    test_count: int
    unique_entry_count: int
    h5mu_count: int
    h5mu_bytes: int
    metadata_count: int
    manifest_count: int
    checksum_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "dataset_count": self.dataset_count,
            "full_count": self.full_count,
            "train_count": self.train_count,
            "test_count": self.test_count,
            "unique_entry_count": self.unique_entry_count,
            "h5mu_count": self.h5mu_count,
            "h5mu_bytes": self.h5mu_bytes,
            "metadata_count": self.metadata_count,
            "manifest_count": self.manifest_count,
            "checksum_count": self.checksum_count,
        }


@dataclass(frozen=True)
class EntryIdMigrationResult:
    report_path: Path
    backup_path: Path
    inventory: EntryIdMigrationInventory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EntryIdMigrationError(f"Unable to read metadata YAML {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("database"), dict):
        raise EntryIdMigrationError(f"Metadata database must be a mapping: {path}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EntryIdMigrationError(f"Unable to read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EntryIdMigrationError(f"JSON root must be an object: {path}")
    return value


def _catalogue_version(path: Path) -> str | None:
    try:
        with sqlite3.connect(path) as connection:
            row = connection.execute(
                "SELECT value FROM catalogue_metadata WHERE key = 'schema_version'"
            ).fetchone()
    except sqlite3.Error as exc:
        raise EntryIdMigrationError(f"Unable to read catalogue version: {exc}") from exc
    return None if row is None else str(row[0])


def _catalogue_columns(path: Path) -> list[str]:
    try:
        with sqlite3.connect(path) as connection:
            return [str(row[1]) for row in connection.execute("PRAGMA table_info(datasets)")]
    except sqlite3.Error as exc:
        raise EntryIdMigrationError(f"Unable to inspect catalogue columns: {exc}") from exc


def _catalogue_datasets(path: Path) -> list[dict[str, Any]]:
    try:
        with sqlite3.connect(path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT dataset_id, dataset_type, split_id, storage_dir, file_size, sha256 "
                "FROM datasets ORDER BY dataset_id"
            ).fetchall()
    except sqlite3.Error as exc:
        raise EntryIdMigrationError(f"Unable to read catalogue datasets: {exc}") from exc
    return [dict(row) for row in rows]


def _catalogue_digest(path: Path) -> str:
    columns = _catalogue_columns(path)
    selected = ", ".join('"' + column.replace('"', '""') + '"' for column in columns)
    try:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                f"SELECT {selected} FROM datasets ORDER BY dataset_id"
            ).fetchall()
    except sqlite3.Error as exc:
        raise EntryIdMigrationError(f"Unable to read catalogue rows: {exc}") from exc
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _entry_id(dataset: dict[str, Any]) -> str:
    dataset_id = str(dataset["dataset_id"])
    split_id = dataset["split_id"]
    if dataset["dataset_type"] in {"train", "test"} and split_id is not None:
        candidate = str(split_id).strip() or dataset_id
    else:
        candidate = dataset_id
    if not SAFE_IDENTIFIER_PATTERN.fullmatch(candidate):
        raise EntryIdMigrationError(
            f"Unsafe entry_id {candidate!r} derived for dataset {dataset_id!r}."
        )
    return candidate


def _dataset_directory(settings: Settings, dataset: dict[str, Any]) -> Path:
    storage_dir = Path(str(dataset["storage_dir"]))
    if storage_dir.name != str(storage_dir) or storage_dir in {Path("."), Path("..")}:
        raise EntryIdMigrationError(f"Unsafe dataset storage_dir {str(storage_dir)!r}.")
    root = settings.data_root.resolve()
    directory = (root / storage_dir).resolve()
    if directory.parent != root:
        raise EntryIdMigrationError(f"Dataset directory escapes data root: {directory}")
    return directory


def _h5_scalar_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _inspect_h5mu(path: Path, dataset_id: str, dataset_type: str) -> None:
    try:
        with h5py.File(path, "r") as handle:
            database = handle.get("uns/database")
            if not isinstance(database, h5py.Group):
                raise EntryIdMigrationError(f"MuData database metadata is missing: {path}")
            if "entry_id" in database:
                raise EntryIdMigrationError(
                    f"Catalogue v4 MuData unexpectedly already contains entry_id: {path}"
                )
            for key, expected in (("dataset_id", dataset_id), ("dataset_type", dataset_type)):
                if key not in database or _h5_scalar_text(database[key][()]) != expected:
                    raise EntryIdMigrationError(
                        f"MuData {key} does not match catalogue for {dataset_id!r}: {path}"
                    )
    except EntryIdMigrationError:
        raise
    except (OSError, RuntimeError, UnicodeError) as exc:
        raise EntryIdMigrationError(f"Unable to inspect MuData {path}: {exc}") from exc


def _inspect_dataset(settings: Settings, dataset: dict[str, Any]) -> None:
    dataset_id = str(dataset["dataset_id"])
    dataset_type = str(dataset["dataset_type"])
    if dataset_type not in {"full", "train", "test"}:
        raise EntryIdMigrationError(
            f"Unsupported dataset_type {dataset_type!r} for {dataset_id!r}."
        )
    _entry_id(dataset)
    directory = _dataset_directory(settings, dataset)
    paths = {name: directory / name for name in ARTIFACT_NAMES}
    for path in paths.values():
        if not path.is_file():
            raise EntryIdMigrationError(f"Dataset artifact is missing: {path}")
    if paths["dataset.h5mu"].stat().st_size != int(dataset["file_size"]):
        raise EntryIdMigrationError(f"MuData size differs from catalogue: {paths['dataset.h5mu']}")

    metadata = _read_yaml(paths["metadata.yaml"])
    metadata_database = metadata["database"]
    if "entry_id" in metadata_database:
        raise EntryIdMigrationError(
            "Catalogue v4 metadata unexpectedly already contains entry_id: "
            f"{paths['metadata.yaml']}"
        )
    for key, expected in (("dataset_id", dataset_id), ("dataset_type", dataset_type)):
        if metadata_database.get(key) != expected:
            raise EntryIdMigrationError(
                f"Metadata {key} does not match catalogue for {dataset_id!r}: "
                f"{paths['metadata.yaml']}"
            )

    manifest = _read_json(paths["manifest.json"])
    manifest_database = manifest.get("database")
    if not isinstance(manifest_database, dict):
        raise EntryIdMigrationError(f"Manifest database is missing: {paths['manifest.json']}")
    if "entry_id" in manifest_database:
        raise EntryIdMigrationError(
            "Catalogue v4 manifest unexpectedly already contains entry_id: "
            f"{paths['manifest.json']}"
        )
    if manifest.get("dataset_id") != dataset_id:
        raise EntryIdMigrationError(f"Manifest dataset_id mismatch: {paths['manifest.json']}")
    h5mu_record = manifest.get("files", {}).get("h5mu", {})
    if h5mu_record.get("size") != int(dataset["file_size"]) or h5mu_record.get("sha256") != str(
        dataset["sha256"]
    ):
        raise EntryIdMigrationError(
            f"Manifest/catalogue MuData record mismatch: {paths['manifest.json']}"
        )
    expected_checksum = f"{dataset['sha256']}  dataset.h5mu\n"
    if paths["checksum.sha256"].read_text(encoding="utf-8") != expected_checksum:
        raise EntryIdMigrationError(
            f"Checksum file differs from catalogue: {paths['checksum.sha256']}"
        )
    _inspect_h5mu(paths["dataset.h5mu"], dataset_id, dataset_type)


def inspect_entry_id_migration(settings: Settings) -> EntryIdMigrationInventory:
    if not settings.database_path.is_file():
        raise EntryIdMigrationError(f"Catalogue does not exist: {settings.database_path}")
    if not settings.data_root.is_dir():
        raise EntryIdMigrationError(f"Dataset root does not exist: {settings.data_root}")
    version = _catalogue_version(settings.database_path)
    if version != SOURCE_CATALOGUE_VERSION:
        raise EntryIdMigrationError(
            f"Expected catalogue version {SOURCE_CATALOGUE_VERSION}, found {version!r}."
        )
    if "entry_id" in _catalogue_columns(settings.database_path):
        raise EntryIdMigrationError("Catalogue v4 unexpectedly already contains entry_id.")
    datasets = _catalogue_datasets(settings.database_path)
    for dataset in datasets:
        _inspect_dataset(settings, dataset)
    types = [str(dataset["dataset_type"]) for dataset in datasets]
    return EntryIdMigrationInventory(
        dataset_count=len(datasets),
        full_count=types.count("full"),
        train_count=types.count("train"),
        test_count=types.count("test"),
        unique_entry_count=len({_entry_id(dataset) for dataset in datasets}),
        h5mu_count=len(datasets),
        h5mu_bytes=sum(int(dataset["file_size"]) for dataset in datasets),
        metadata_count=len(datasets),
        manifest_count=len(datasets),
        checksum_count=len(datasets),
    )


def _sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            sqlite3.connect(source) as source_connection,
            sqlite3.connect(destination) as destination_connection,
        ):
            source_connection.backup(destination_connection)
    except sqlite3.Error as exc:
        raise EntryIdMigrationError(f"Unable to back up catalogue: {exc}") from exc


def _migrate_catalogue_copy(path: Path, records: list[dict[str, Any]]) -> None:
    try:
        with sqlite3.connect(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "ALTER TABLE datasets ADD COLUMN entry_id VARCHAR(128) NOT NULL DEFAULT 'legacy'"
            )
            for record in records:
                connection.execute(
                    "UPDATE datasets SET entry_id = ?, file_size = ?, sha256 = ? "
                    "WHERE dataset_id = ?",
                    (
                        record["entry_id"],
                        record["new_size"],
                        record["new_sha256"],
                        record["dataset_id"],
                    ),
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_datasets_entry_id ON datasets (entry_id)"
            )
            connection.execute(
                "UPDATE catalogue_metadata SET value = ? WHERE key = 'schema_version'",
                (TARGET_CATALOGUE_VERSION,),
            )
            connection.commit()
    except sqlite3.Error as exc:
        raise EntryIdMigrationError(f"Unable to migrate catalogue copy: {exc}") from exc


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _atomic_restore(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".restore.tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        _link_or_copy(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_h5mu_entry_id(path: Path, entry_id: str) -> None:
    original_mode = stat.S_IMODE(path.stat().st_mode)
    path.chmod(original_mode | stat.S_IWUSR)
    try:
        with h5py.File(path, "r+") as handle:
            database = handle["uns/database"]
            entry = database.create_dataset(
                "entry_id", data=entry_id, dtype=h5py.string_dtype(encoding="utf-8")
            )
            entry.attrs["encoding-type"] = "string"
            entry.attrs["encoding-version"] = "0.2.0"
            handle.flush()
    finally:
        path.chmod(original_mode)


def _stage_dataset(
    settings: Settings,
    dataset: dict[str, Any],
    stage_root: Path,
) -> dict[str, Any]:
    dataset_id = str(dataset["dataset_id"])
    directory = _dataset_directory(settings, dataset)
    stage_directory = stage_root / "datasets" / str(dataset["storage_dir"])
    stage_directory.mkdir(parents=True)
    source_h5mu = directory / "dataset.h5mu"
    staged_h5mu = stage_directory / "dataset.h5mu"
    source_stat = source_h5mu.stat()
    shutil.copy2(source_h5mu, staged_h5mu)
    if _sha256(staged_h5mu) != str(dataset["sha256"]):
        raise EntryIdMigrationError(f"MuData checksum differs from catalogue: {source_h5mu}")
    entry_id = _entry_id(dataset)
    _write_h5mu_entry_id(staged_h5mu, entry_id)
    new_size = staged_h5mu.stat().st_size
    new_sha256 = _sha256(staged_h5mu)

    metadata = _read_yaml(directory / "metadata.yaml")
    metadata["database"]["entry_id"] = entry_id
    try:
        MetadataDocument.model_validate(metadata)
    except ValidationError as exc:
        raise EntryIdMigrationError(
            f"Migrated metadata is invalid for {dataset_id!r}: {exc}"
        ) from exc
    (stage_directory / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    manifest = _read_json(directory / "manifest.json")
    manifest["database"]["entry_id"] = entry_id
    manifest["files"]["h5mu"]["size"] = new_size
    manifest["files"]["h5mu"]["sha256"] = new_sha256
    (stage_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (stage_directory / "checksum.sha256").write_text(
        f"{new_sha256}  dataset.h5mu\n", encoding="utf-8"
    )
    return {
        "dataset_id": dataset_id,
        "dataset_type": str(dataset["dataset_type"]),
        "split_id": dataset["split_id"],
        "storage_dir": str(dataset["storage_dir"]),
        "entry_id": entry_id,
        "old_size": int(dataset["file_size"]),
        "old_sha256": str(dataset["sha256"]),
        "new_size": new_size,
        "new_sha256": new_sha256,
        "source_h5mu_stat": {
            "size": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
            "inode": source_stat.st_ino,
        },
        "files": [
            {
                "name": name,
                "path": str((directory / name).resolve()),
                "staged_path": str((stage_directory / name).resolve()),
                "old_sha256": (
                    str(dataset["sha256"]) if name == "dataset.h5mu" else _sha256(directory / name)
                ),
                "new_sha256": _sha256(stage_directory / name),
            }
            for name in ARTIFACT_NAMES
        ],
    }


def _verify_source_unchanged(record: dict[str, Any]) -> None:
    h5mu_file = record["files"][0]
    path = Path(h5mu_file["path"])
    current = path.stat()
    expected = record["source_h5mu_stat"]
    if (
        current.st_size != expected["size"]
        or current.st_mtime_ns != expected["mtime_ns"]
        or current.st_ino != expected["inode"]
    ):
        raise EntryIdMigrationError(f"Source MuData changed during staging: {path}")
    for file_record in record["files"][1:]:
        path = Path(file_record["path"])
        if _sha256(path) != file_record["old_sha256"]:
            raise EntryIdMigrationError(f"Source artifact changed during staging: {path}")


def _verify_active(settings: Settings, report: dict[str, Any], *, hash_h5mu: bool) -> None:
    if _catalogue_version(settings.database_path) != TARGET_CATALOGUE_VERSION:
        raise EntryIdMigrationError("Active catalogue is not version 5.")
    columns = _catalogue_columns(settings.database_path)
    if "entry_id" not in columns:
        raise EntryIdMigrationError("Active catalogue is missing datasets.entry_id.")
    try:
        with sqlite3.connect(settings.database_path) as connection:
            rows = {
                str(row[0]): row[1:]
                for row in connection.execute(
                    "SELECT dataset_id, entry_id, file_size, sha256 FROM datasets"
                )
            }
    except sqlite3.Error as exc:
        raise EntryIdMigrationError(f"Unable to verify active catalogue: {exc}") from exc
    if set(rows) != {record["dataset_id"] for record in report["datasets"]}:
        raise EntryIdMigrationError("Active catalogue dataset IDs changed during migration.")
    for record in report["datasets"]:
        expected_row = (record["entry_id"], record["new_size"], record["new_sha256"])
        if rows[record["dataset_id"]] != expected_row:
            raise EntryIdMigrationError(
                f"Active catalogue row is inconsistent for {record['dataset_id']!r}."
            )
        for file_record in record["files"]:
            path = Path(file_record["path"])
            if not path.is_file():
                raise EntryIdMigrationError(f"Migrated artifact is missing: {path}")
            if file_record["name"] != "dataset.h5mu" or hash_h5mu:
                if _sha256(path) != file_record["new_sha256"]:
                    raise EntryIdMigrationError(f"Migrated artifact checksum mismatch: {path}")
        _inspect_migrated_fields(record)


def _inspect_migrated_fields(record: dict[str, Any]) -> None:
    directory = Path(record["files"][0]["path"]).parent
    metadata = _read_yaml(directory / "metadata.yaml")
    manifest = _read_json(directory / "manifest.json")
    if metadata["database"].get("entry_id") != record["entry_id"]:
        raise EntryIdMigrationError(f"Metadata entry_id mismatch: {directory}")
    if manifest.get("database", {}).get("entry_id") != record["entry_id"]:
        raise EntryIdMigrationError(f"Manifest entry_id mismatch: {directory}")
    h5mu_record = manifest.get("files", {}).get("h5mu", {})
    if (
        h5mu_record.get("size") != record["new_size"]
        or h5mu_record.get("sha256") != record["new_sha256"]
    ):
        raise EntryIdMigrationError(f"Manifest MuData record mismatch: {directory}")
    try:
        with h5py.File(directory / "dataset.h5mu", "r") as handle:
            value = _h5_scalar_text(handle["uns/database/entry_id"][()])
    except (KeyError, OSError, RuntimeError, UnicodeError) as exc:
        raise EntryIdMigrationError(
            f"Unable to verify MuData entry_id: {directory}: {exc}"
        ) from exc
    if value != record["entry_id"]:
        raise EntryIdMigrationError(f"MuData entry_id mismatch: {directory}")


def migrate_entry_ids(
    settings: Settings, *, dry_run: bool = False
) -> EntryIdMigrationInventory | EntryIdMigrationResult:
    inventory = inspect_entry_id_migration(settings)
    if dry_run:
        return inventory
    if CATALOGUE_SCHEMA_VERSION != TARGET_CATALOGUE_VERSION:
        raise EntryIdMigrationError(
            f"Code expects catalogue version {CATALOGUE_SCHEMA_VERSION}, not version 5."
        )

    started_at = datetime.now(UTC)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    parent = settings.database_path.parent
    stage_root = parent / f".catalogue_v5_staging_{stamp}"
    backup_path = parent / "backups" / f"catalogue_v4_{stamp}"
    report_path = parent / "migrations" / f"catalogue_v5_{stamp}.json"
    for path in (stage_root, backup_path, report_path):
        if path.exists():
            raise EntryIdMigrationError(f"Migration target already exists: {path}")

    source_digest = _catalogue_digest(settings.database_path)
    datasets = _catalogue_datasets(settings.database_path)
    report: dict[str, Any] = {
        "migration": MIGRATION_NAME,
        "source_catalogue_version": SOURCE_CATALOGUE_VERSION,
        "target_catalogue_version": TARGET_CATALOGUE_VERSION,
        "started_at": started_at.isoformat(),
        "completed_at": None,
        "backup_deleted_at": None,
        "status": "staging",
        "catalogue_path": str(settings.database_path.resolve()),
        "data_root": str(settings.data_root.resolve()),
        "backup_path": str(backup_path.resolve()),
        "inventory": inventory.as_dict(),
        "source_catalogue_content_sha256": source_digest,
        "datasets": [],
    }
    activated: list[dict[str, Any]] = []
    catalogue_activated = False
    try:
        stage_root.mkdir(parents=True)
        backup_path.mkdir(parents=True)
        stage_catalogue = stage_root / "catalog.db"
        _sqlite_backup(settings.database_path, stage_catalogue)
        for dataset in datasets:
            report["datasets"].append(_stage_dataset(settings, dataset, stage_root))
        _migrate_catalogue_copy(stage_catalogue, report["datasets"])

        if (
            _catalogue_version(settings.database_path) != SOURCE_CATALOGUE_VERSION
            or _catalogue_digest(settings.database_path) != source_digest
        ):
            raise EntryIdMigrationError("Catalogue changed during staging.")
        for record in report["datasets"]:
            _verify_source_unchanged(record)

        _sqlite_backup(settings.database_path, backup_path / "catalog.db")
        for record in report["datasets"]:
            backup_directory = backup_path / "datasets" / record["storage_dir"]
            for file_record in record["files"]:
                backup_file = backup_directory / file_record["name"]
                _link_or_copy(Path(file_record["path"]), backup_file)
                file_record["backup_path"] = str(backup_file.resolve())

        report["status"] = "staged"
        _write_json_atomic(report_path, report)
        for record in report["datasets"]:
            for file_record in record["files"]:
                os.replace(file_record["staged_path"], file_record["path"])
                activated.append(file_record)
        os.replace(stage_catalogue, settings.database_path)
        catalogue_activated = True

        _verify_active(settings, report, hash_h5mu=False)
        report["status"] = "complete"
        report["completed_at"] = datetime.now(UTC).isoformat()
        _write_json_atomic(report_path, report)
        shutil.rmtree(stage_root)
        return EntryIdMigrationResult(report_path, backup_path, inventory)
    except Exception as exc:
        rollback_errors: list[str] = []
        if catalogue_activated:
            try:
                _atomic_restore(backup_path / "catalog.db", settings.database_path)
            except Exception as rollback_exc:  # pragma: no cover - catastrophic I/O failure
                rollback_errors.append(f"catalogue rollback failed: {rollback_exc}")
        for file_record in reversed(activated):
            try:
                _atomic_restore(Path(file_record["backup_path"]), Path(file_record["path"]))
            except Exception as rollback_exc:  # pragma: no cover - catastrophic I/O failure
                rollback_errors.append(
                    f"artifact rollback failed for {file_record['path']}: {rollback_exc}"
                )
        report["status"] = "rollback_failed" if rollback_errors else "rolled_back"
        report["error"] = str(exc)
        report["rollback_errors"] = rollback_errors
        try:
            _write_json_atomic(report_path, report)
        except OSError:
            pass
        if stage_root.exists():
            shutil.rmtree(stage_root, ignore_errors=True)
        details = f"Entry-ID migration failed and was rolled back: {exc}"
        if rollback_errors:
            details += "; " + "; ".join(rollback_errors)
        raise EntryIdMigrationError(details) from exc


def finalize_entry_id_migration(report_path: Path, settings: Settings) -> Path:
    report = _read_json(report_path)
    if report.get("migration") != MIGRATION_NAME:
        raise EntryIdMigrationError("Report does not describe an entry-ID migration.")
    if report.get("status") != "complete":
        raise EntryIdMigrationError("Migration is not complete and cannot be finalized.")
    if report.get("backup_deleted_at") is not None:
        raise EntryIdMigrationError("The entry-ID migration backup was already finalized.")
    if Path(report.get("catalogue_path", "")).resolve() != settings.database_path.resolve():
        raise EntryIdMigrationError("Migration report belongs to a different catalogue.")
    if Path(report.get("data_root", "")).resolve() != settings.data_root.resolve():
        raise EntryIdMigrationError("Migration report belongs to a different data root.")

    backup_path = Path(report["backup_path"]).resolve()
    backups_root = (settings.database_path.parent / "backups").resolve()
    if backup_path.parent != backups_root or not backup_path.name.startswith("catalogue_v4_"):
        raise EntryIdMigrationError(f"Unsafe backup path in migration report: {backup_path}")
    if not backup_path.is_dir() or not (backup_path / "catalog.db").is_file():
        raise EntryIdMigrationError(f"Migration backup is incomplete: {backup_path}")
    for record in report["datasets"]:
        for file_record in record["files"]:
            backup = Path(file_record.get("backup_path", ""))
            if not backup.is_file() or backup.resolve().is_relative_to(backup_path) is False:
                raise EntryIdMigrationError(f"Migration backup is incomplete: {backup}")

    _verify_active(settings, report, hash_h5mu=True)
    shutil.rmtree(backup_path)
    report["backup_deleted_at"] = datetime.now(UTC).isoformat()
    _write_json_atomic(report_path, report)
    return backup_path
