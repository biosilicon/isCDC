from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import mudata
import yaml
from sqlalchemy import select

from .artifacts import write_dataset_artifacts, write_json
from .config import PROJECT_ROOT, Settings
from .database import (
    CATALOGUE_SCHEMA_VERSION,
    create_database_engine,
    create_session_factory,
    initialize_database,
)
from .importer import build_dataset_record
from .models import Dataset
from .schemas import MetadataDocument, load_metadata
from .validation import ValidationOutcome, validate_h5mu

SOURCE_SCHEMA_VERSION = "1.1"
TARGET_SCHEMA_VERSION = "1.2"
SOURCE_CATALOGUE_VERSION = "2"
EXPECTED_DATASET_TYPES = {"full": 35, "train": 27, "test": 27}
EXPECTED_PARTIAL_DATASET_IDS = {
    "GSE205055_mouse_brain_p21_20um_atac_rna",
    "GSE205055_mouse_brain_p21_20um_atac_rna_right20_train",
    "GSE205055_mouse_embryo_e13_25um_atac_rna",
    "GSE205055_mouse_embryo_e13_25um_atac_rna_right20_train",
}


class SchemaMigrationError(RuntimeError):
    """Raised when the schema migration cannot proceed or be finalized safely."""


@dataclass(frozen=True)
class MigrationInventory:
    dataset_count: int
    dataset_types: dict[str, int]
    challenge_count: int
    partial_dataset_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_count": self.dataset_count,
            "dataset_types": self.dataset_types,
            "challenge_count": self.challenge_count,
            "partial_dataset_ids": list(self.partial_dataset_ids),
        }


@dataclass(frozen=True)
class MigrationResult:
    report_path: Path
    backup_path: Path
    inventory: MigrationInventory


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SchemaMigrationError(f"Unable to read YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaMigrationError(f"YAML root must be a mapping: {path}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _catalogue_version(path: Path) -> str | None:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "select value from catalogue_metadata where key = 'schema_version'"
        ).fetchone()
    except sqlite3.Error as exc:
        raise SchemaMigrationError(f"Unable to read catalogue metadata: {exc}") from exc
    finally:
        connection.close()
    return None if row is None else str(row[0])


def _read_catalogue(settings: Settings) -> list[Dataset]:
    engine = create_database_engine(settings.database_path)
    session_factory = create_session_factory(engine)
    try:
        with session_factory() as session:
            return list(session.scalars(select(Dataset).order_by(Dataset.dataset_id)).all())
    finally:
        engine.dispose()


def inspect_schema_1_1(settings: Settings) -> MigrationInventory:
    if not settings.database_path.is_file():
        raise SchemaMigrationError(f"Catalogue does not exist: {settings.database_path}")
    if not settings.data_root.is_dir():
        raise SchemaMigrationError(f"Dataset root does not exist: {settings.data_root}")
    version = _catalogue_version(settings.database_path)
    if version != SOURCE_CATALOGUE_VERSION:
        raise SchemaMigrationError(
            f"Expected catalogue version {SOURCE_CATALOGUE_VERSION}, found {version!r}."
        )

    datasets = _read_catalogue(settings)
    indexed_ids = {dataset.dataset_id for dataset in datasets}
    directory_ids = {path.name for path in settings.data_root.iterdir() if path.is_dir()}
    directory_ids.discard(".staging")
    if indexed_ids != directory_ids:
        missing = sorted(indexed_ids - directory_ids)
        unindexed = sorted(directory_ids - indexed_ids)
        raise SchemaMigrationError(
            f"Catalogue/directory mismatch; missing={missing}, unindexed={unindexed}."
        )

    type_counts = {name: 0 for name in EXPECTED_DATASET_TYPES}
    split_sides: dict[str, set[str]] = {}
    partial_ids: list[str] = []
    for dataset in datasets:
        if dataset.schema_version != SOURCE_SCHEMA_VERSION:
            raise SchemaMigrationError(
                f"Dataset {dataset.dataset_id!r} is not schema {SOURCE_SCHEMA_VERSION}."
            )
        if dataset.dataset_type not in type_counts:
            raise SchemaMigrationError(
                f"Dataset {dataset.dataset_id!r} has unknown type {dataset.dataset_type!r}."
            )
        type_counts[dataset.dataset_type] += 1
        if dataset.dataset_type in {"train", "test"}:
            if not dataset.split_id:
                raise SchemaMigrationError(
                    f"Derived dataset {dataset.dataset_id!r} lacks split_id."
                )
            split_sides.setdefault(dataset.split_id, set()).add(dataset.dataset_type)
        if dataset.pairing_type == "partially_shared":
            partial_ids.append(dataset.dataset_id)
        elif dataset.pairing_type != "same_unit":
            raise SchemaMigrationError(
                f"Dataset {dataset.dataset_id!r} has ineligible pairing type "
                f"{dataset.pairing_type!r}."
            )

        directory = settings.data_root / dataset.storage_dir
        metadata_path = directory / "metadata.yaml"
        h5mu_path = directory / "dataset.h5mu"
        if not metadata_path.is_file() or not h5mu_path.is_file():
            raise SchemaMigrationError(f"Dataset files are incomplete: {directory}")
        raw = _load_yaml(metadata_path)
        if raw.get("database", {}).get("schema_version") != SOURCE_SCHEMA_VERSION:
            raise SchemaMigrationError(
                f"Metadata is not schema {SOURCE_SCHEMA_VERSION}: {metadata_path}"
            )
        with h5py.File(h5mu_path, "r") as handle:
            node = handle.get("uns/database/schema_version")
            if node is None or node.asstr()[()] != SOURCE_SCHEMA_VERSION:
                raise SchemaMigrationError(
                    f"Embedded metadata is not schema {SOURCE_SCHEMA_VERSION}: {h5mu_path}"
                )
        if h5mu_path.stat().st_size != dataset.file_size:
            raise SchemaMigrationError(f"File size differs from catalogue: {h5mu_path}")
        checksum_path = directory / "checksum.sha256"
        expected_checksum = f"{dataset.sha256}  dataset.h5mu"
        if (
            not checksum_path.is_file()
            or checksum_path.read_text(encoding="utf-8").strip() != expected_checksum
        ):
            raise SchemaMigrationError(f"Checksum artifact differs from catalogue: {directory}")
        manifest_path = directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SchemaMigrationError(f"Unable to read manifest {manifest_path}: {exc}") from exc
        h5mu_manifest = manifest.get("files", {}).get("h5mu", {})
        if (
            manifest.get("dataset_id") != dataset.dataset_id
            or manifest.get("database", {}).get("schema_version") != SOURCE_SCHEMA_VERSION
            or h5mu_manifest.get("sha256") != dataset.sha256
            or h5mu_manifest.get("size") != dataset.file_size
        ):
            raise SchemaMigrationError(f"Manifest differs from catalogue: {manifest_path}")
        if _sha256(h5mu_path) != dataset.sha256:
            raise SchemaMigrationError(f"Stored file hash differs from catalogue: {h5mu_path}")

    if len(datasets) != sum(EXPECTED_DATASET_TYPES.values()):
        raise SchemaMigrationError(
            f"Expected {sum(EXPECTED_DATASET_TYPES.values())} datasets, found {len(datasets)}."
        )
    if type_counts != EXPECTED_DATASET_TYPES:
        raise SchemaMigrationError(
            f"Expected dataset type counts {EXPECTED_DATASET_TYPES}, found {type_counts}."
        )
    incomplete = sorted(
        split_id for split_id, sides in split_sides.items() if sides != {"train", "test"}
    )
    if incomplete:
        raise SchemaMigrationError(f"Incomplete challenges: {incomplete}")
    if set(partial_ids) != EXPECTED_PARTIAL_DATASET_IDS:
        raise SchemaMigrationError(
            "Unexpected partially shared dataset set; "
            f"expected={sorted(EXPECTED_PARTIAL_DATASET_IDS)}, found={sorted(partial_ids)}."
        )

    return MigrationInventory(
        dataset_count=len(datasets),
        dataset_types=type_counts,
        challenge_count=len(split_sides),
        partial_dataset_ids=tuple(sorted(partial_ids)),
    )


def _set_embedded_schema_version(path: Path) -> None:
    with h5py.File(path, "r+") as handle:
        node = handle.get("uns/database/schema_version")
        if node is None or node.asstr()[()] != SOURCE_SCHEMA_VERSION:
            raise SchemaMigrationError(f"Cannot migrate embedded schema version: {path}")
        node[()] = TARGET_SCHEMA_VERSION
        database = handle.get("uns/database")
        dataset_id = handle.get("uns/database/dataset_id")
        if database is not None:
            if dataset_id is None:
                raise SchemaMigrationError(f"Cannot infer entry_id for {path}")
            if "entry_id" in database:
                del database["entry_id"]
            entry_node = database.create_dataset(
                "entry_id",
                data=dataset_id.asstr()[()],
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            entry_node.attrs["encoding-type"] = "string"
            entry_node.attrs["encoding-version"] = "0.2.0"
        if database is not None and "license" in database:
            del database["license"]


def _trim_two_modality_file(path: Path) -> list[str]:
    temporary = path.with_name(f".{path.name}.schema-1.2.tmp")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
        mdata = mudata.read_h5mu(path)
    try:
        if len(mdata.mod) != 2:
            raise SchemaMigrationError(f"Expected exactly two modalities in {path}")
        memberships = [set(map(str, adata.obs_names)) for adata in mdata.mod.values()]
        common = set.intersection(*memberships)
        ordered = [str(name) for name in mdata.obs_names if str(name) in common]
        removed = [str(name) for name in mdata.obs_names if str(name) not in common]
        if not ordered or not removed:
            raise SchemaMigrationError(
                f"Expected a non-empty paired intersection and removals in {path}"
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
            migrated = mdata[ordered, :].copy()
        migrated.uns["database"]["schema_version"] = TARGET_SCHEMA_VERSION
        migrated.uns["database"]["entry_id"] = migrated.uns["database"]["dataset_id"]
        migrated.uns["database"]["pairing_type"] = "same_unit"
        migrated.uns["database"].pop("license", None)
        migrated.write_h5mu(temporary)
    finally:
        mdata.file.close()
    shutil.copystat(path, temporary)
    os.replace(temporary, path)
    return removed


def _migrate_dataset_directory(directory: Path) -> list[str]:
    metadata_path = directory / "metadata.yaml"
    h5mu_path = directory / "dataset.h5mu"
    raw = _load_yaml(metadata_path)
    database = raw.get("database")
    modalities = raw.get("modalities")
    if not isinstance(database, dict) or not isinstance(modalities, dict):
        raise SchemaMigrationError(f"Invalid metadata structure: {metadata_path}")
    if database.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise SchemaMigrationError(f"Unexpected source schema in {metadata_path}")

    removed: list[str] = []
    if len(modalities) == 2 and database.get("pairing_type") == "partially_shared":
        removed = _trim_two_modality_file(h5mu_path)
        database["pairing_type"] = "same_unit"
    else:
        _set_embedded_schema_version(h5mu_path)
    database["schema_version"] = TARGET_SCHEMA_VERSION
    database["entry_id"] = database["dataset_id"]
    database.pop("license", None)
    raw.pop("license", None)
    _write_yaml(metadata_path, raw)
    return removed


def _migrate_exp(stage_exp: Path) -> None:
    for path in sorted(stage_exp.rglob("*.h5mu")):
        _set_embedded_schema_version(path)
    for path in sorted((*stage_exp.rglob("*.yaml"), *stage_exp.rglob("*.yml"))):
        raw = _load_yaml(path)
        if isinstance(raw.get("database"), dict):
            database = raw["database"]
            if database.get("schema_version") == SOURCE_SCHEMA_VERSION:
                database["schema_version"] = TARGET_SCHEMA_VERSION
            database.pop("license", None)
        elif raw.get("schema_version") == SOURCE_SCHEMA_VERSION:
            raw["schema_version"] = TARGET_SCHEMA_VERSION
        raw.pop("license", None)
        _write_yaml(path, raw)


def _inspect_exp_1_1(exp_root: Path) -> None:
    required = {
        exp_root / "xenium_human_rcc_ffpe_rna_protein.h5mu",
        exp_root / "xenium_human_rcc_ffpe_rna_protein_vertical_split.yaml",
    }
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        raise SchemaMigrationError(f"Required experiment fixtures are missing: {missing}")
    for path in sorted(exp_root.rglob("*.h5mu")):
        with h5py.File(path, "r") as handle:
            node = handle.get("uns/database/schema_version")
            if node is None or node.asstr()[()] != SOURCE_SCHEMA_VERSION:
                raise SchemaMigrationError(
                    f"Experiment file is not schema {SOURCE_SCHEMA_VERSION}: {path}"
                )
    for path in sorted((*exp_root.rglob("*.yaml"), *exp_root.rglob("*.yml"))):
        raw = _load_yaml(path)
        version = (
            raw.get("database", {}).get("schema_version")
            if isinstance(raw.get("database"), dict)
            else raw.get("schema_version")
        )
        if version is not None and version != SOURCE_SCHEMA_VERSION:
            raise SchemaMigrationError(
                f"Experiment YAML is not schema {SOURCE_SCHEMA_VERSION}: {path}"
            )


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _validate_and_index(
    stage_root: Path,
    source_data_root: Path,
    legacy_by_id: dict[str, Dataset],
    migration_started_at: datetime,
) -> tuple[list[dict[str, Any]], dict[str, ValidationOutcome]]:
    stage_data = stage_root / "datasets"
    metadata_by_id: dict[str, MetadataDocument] = {}
    for directory in sorted(path for path in stage_data.iterdir() if path.is_dir()):
        metadata = load_metadata(directory / "metadata.yaml")
        metadata_by_id[metadata.database.dataset_id] = metadata
    source_paths = {
        dataset_id: stage_data / dataset_id / "dataset.h5mu"
        for dataset_id, metadata in metadata_by_id.items()
        if metadata.database.dataset_type == "full"
    }
    peers: dict[str, list[Path]] = {}
    for dataset_id, metadata in metadata_by_id.items():
        derivation = metadata.database.derivation
        if derivation is not None:
            peers.setdefault(derivation.split_id, []).append(
                stage_data / dataset_id / "dataset.h5mu"
            )

    outcomes: dict[str, ValidationOutcome] = {}
    records: list[dict[str, Any]] = []
    for dataset_id, metadata in sorted(metadata_by_id.items()):
        directory = stage_data / dataset_id
        path = directory / "dataset.h5mu"
        derivation = metadata.database.derivation
        peer_paths = []
        if derivation is not None:
            peer_paths = [value for value in peers[derivation.split_id] if value != path]
        outcome = validate_h5mu(
            path,
            metadata,
            source_paths=source_paths if derivation is not None else None,
            peer_paths=peer_paths,
        )
        if not outcome.valid:
            details = "; ".join(f"{issue.code}: {issue.message}" for issue in outcome.errors)
            raise SchemaMigrationError(f"Validation failed for {dataset_id}: {details}")
        outcomes[dataset_id] = outcome
        sha256 = _sha256(path)
        file_size = path.stat().st_size
        legacy_directory = source_data_root / legacy_by_id[dataset_id].storage_dir
        old_manifest = json.loads(
            (legacy_directory / "manifest.json").read_text(encoding="utf-8")
        )
        imported_at = datetime.fromisoformat(old_manifest["imported_at"])
        write_dataset_artifacts(
            directory,
            metadata,
            outcome,
            file_size,
            sha256,
            imported_at,
            migration_started_at,
        )
        records.append(
            {
                "metadata": metadata,
                "outcome": outcome,
                "file_size": file_size,
                "sha256": sha256,
                "imported_at": imported_at,
            }
        )

    catalog_path = stage_root / "catalog.db"
    engine = create_database_engine(catalog_path)
    initialize_database(engine)
    session_factory = create_session_factory(engine)
    try:
        with session_factory() as session:
            for record in records:
                session.add(build_dataset_record(**record))
            session.commit()
    finally:
        engine.dispose()
    return records, outcomes


def _validate_exp(stage_exp: Path) -> None:
    full_sources: dict[str, Path] = {}
    for path in sorted(stage_exp.rglob("*.h5mu")):
        with h5py.File(path, "r") as handle:
            database_group = handle.get("uns/database")
            if database_group is None:
                raise SchemaMigrationError(f"Experiment file lacks database metadata: {path}")
            dataset_id = database_group["dataset_id"].asstr()[()]
            dataset_type = database_group["dataset_type"].asstr()[()]
        if dataset_type == "full":
            full_sources[str(dataset_id)] = path
    for path in sorted(stage_exp.rglob("*.h5mu")):
        with h5py.File(path, "r") as handle:
            dataset_type = handle["uns/database/dataset_type"].asstr()[()]
        outcome = validate_h5mu(
            path,
            source_paths=full_sources if dataset_type in {"train", "test"} else None,
        )
        if not outcome.valid:
            details = "; ".join(f"{issue.code}: {issue.message}" for issue in outcome.errors)
            raise SchemaMigrationError(f"Experiment validation failed for {path}: {details}")


def _cutover(
    settings: Settings,
    exp_root: Path,
    stage_root: Path,
    backup_path: Path,
) -> None:
    backup_path.mkdir(parents=True)
    moved_old: list[tuple[Path, Path]] = []
    moved_new: list[tuple[Path, Path]] = []
    operations = (
        (settings.data_root, backup_path / "datasets", stage_root / "datasets"),
        (settings.database_path, backup_path / "catalog.db", stage_root / "catalog.db"),
        (exp_root, backup_path / "exp", stage_root / "exp"),
    )
    try:
        for active, backup, staged in operations:
            os.replace(active, backup)
            moved_old.append((backup, active))
            os.replace(staged, active)
            moved_new.append((active, staged))
    except Exception:
        for active, staged in reversed(moved_new):
            if active.exists():
                os.replace(active, staged)
        for backup, active in reversed(moved_old):
            if backup.exists():
                os.replace(backup, active)
        raise


def migrate_schema_1_2(
    settings: Settings,
    *,
    exp_root: Path = PROJECT_ROOT / "exp",
    dry_run: bool = False,
) -> MigrationInventory | MigrationResult:
    inventory = inspect_schema_1_1(settings)
    if not exp_root.is_dir():
        raise SchemaMigrationError(f"Experiment fixture directory does not exist: {exp_root}")
    _inspect_exp_1_1(exp_root)
    if dry_run:
        return inventory

    started_at = datetime.now(UTC)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    data_parent = settings.data_root.parent
    stage_root = data_parent / f".schema_1_2_staging_{stamp}"
    backup_path = data_parent / "backups" / f"schema_1_1_{stamp}"
    report_path = data_parent / "migrations" / f"schema_1_2_{stamp}.json"
    for path in (stage_root, backup_path, report_path):
        if path.exists():
            raise SchemaMigrationError(f"Migration target already exists: {path}")
    if settings.database_path.parent != data_parent:
        raise SchemaMigrationError("Catalogue and dataset root must share a parent for cutover.")

    legacy = _read_catalogue(settings)
    legacy_by_id = {dataset.dataset_id: dataset for dataset in legacy}
    old_values = {
        dataset.dataset_id: {
            "sha256": dataset.sha256,
            "file_size": dataset.file_size,
            "n_obs": dataset.n_obs,
            "pairing_type": dataset.pairing_type,
        }
        for dataset in legacy
    }
    removed_by_id: dict[str, list[str]] = {}
    old_exp_hashes = _file_hashes(exp_root)
    try:
        stage_root.mkdir(parents=True)
        shutil.copytree(settings.data_root, stage_root / "datasets")
        shutil.copytree(exp_root, stage_root / "exp")
        for directory in sorted((stage_root / "datasets").iterdir()):
            if directory.is_dir() and not directory.name.startswith("."):
                removed_by_id[directory.name] = _migrate_dataset_directory(directory)
        _migrate_exp(stage_root / "exp")
        records, _outcomes = _validate_and_index(
            stage_root, settings.data_root, legacy_by_id, started_at
        )
        _validate_exp(stage_root / "exp")

        new_values = {
            record["metadata"].database.dataset_id: {
                "sha256": record["sha256"],
                "file_size": record["file_size"],
                "n_obs": record["outcome"].n_obs,
                "pairing_type": record["metadata"].database.pairing_type,
            }
            for record in records
        }
        new_exp_hashes = _file_hashes(stage_root / "exp")
        report = {
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "target_schema_version": TARGET_SCHEMA_VERSION,
            "source_catalogue_version": SOURCE_CATALOGUE_VERSION,
            "target_catalogue_version": CATALOGUE_SCHEMA_VERSION,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "inventory": inventory.as_dict(),
            "backup_path": str(backup_path.resolve()),
            "exp_path": str(exp_root.resolve()),
            "backup_deleted_at": None,
            "exp_files": {
                name: {
                    "old_sha256": sha256,
                    "new_sha256": new_exp_hashes[name],
                }
                for name, sha256 in old_exp_hashes.items()
            },
            "datasets": {
                dataset_id: {
                    "old": old_values[dataset_id],
                    "new": new_values[dataset_id],
                    "removed_observation_ids": removed_by_id[dataset_id],
                }
                for dataset_id in sorted(old_values)
            },
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(report_path, report)
        _cutover(settings, exp_root, stage_root, backup_path)
        if stage_root.exists():
            try:
                shutil.rmtree(stage_root)
            except OSError:
                pass
        return MigrationResult(report_path, backup_path, inventory)
    except Exception:
        if stage_root.exists():
            shutil.rmtree(stage_root)
        if report_path.exists() and not backup_path.exists():
            report_path.unlink()
        raise


def finalize_schema_1_2_migration(report_path: Path, settings: Settings) -> Path:
    report_path = report_path.expanduser().resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaMigrationError(f"Unable to read migration report: {exc}") from exc
    if report.get("target_schema_version") != TARGET_SCHEMA_VERSION:
        raise SchemaMigrationError("Report does not describe a schema 1.2 migration.")
    if report.get("backup_deleted_at") is not None:
        raise SchemaMigrationError("The migration backup was already finalized.")

    backup_path = Path(report.get("backup_path", "")).resolve()
    allowed_parent = (settings.data_root.parent / "backups").resolve()
    if backup_path.parent != allowed_parent or not backup_path.name.startswith("schema_1_1_"):
        raise SchemaMigrationError(f"Unsafe backup path in migration report: {backup_path}")
    if not backup_path.is_dir():
        raise SchemaMigrationError(f"Migration backup does not exist: {backup_path}")
    if _catalogue_version(settings.database_path) != CATALOGUE_SCHEMA_VERSION:
        raise SchemaMigrationError("Active catalogue is not at the target catalogue version.")

    datasets = _read_catalogue(settings)
    expected = report.get("datasets", {})
    if {dataset.dataset_id for dataset in datasets} != set(expected):
        raise SchemaMigrationError("Active catalogue does not match the migration report.")

    metadata_by_id: dict[str, MetadataDocument] = {}
    paths_by_id: dict[str, Path] = {}
    for dataset in datasets:
        target = expected[dataset.dataset_id]["new"]
        if dataset.schema_version != TARGET_SCHEMA_VERSION or dataset.sha256 != target["sha256"]:
            raise SchemaMigrationError(
                f"Active dataset {dataset.dataset_id!r} does not match the migration report."
            )
        directory = (settings.data_root / dataset.storage_dir).resolve()
        if directory.parent != settings.data_root.resolve() or not directory.is_dir():
            raise SchemaMigrationError(f"Unsafe or missing active dataset directory: {directory}")
        h5mu_path = directory / "dataset.h5mu"
        if not h5mu_path.is_file() or h5mu_path.stat().st_size != target["file_size"]:
            raise SchemaMigrationError(f"Active dataset file size mismatch: {h5mu_path}")
        if _sha256(h5mu_path) != target["sha256"]:
            raise SchemaMigrationError(f"Active dataset checksum mismatch: {h5mu_path}")

        metadata = load_metadata(directory / "metadata.yaml")
        if (
            metadata.database.dataset_id != dataset.dataset_id
            or metadata.database.schema_version != TARGET_SCHEMA_VERSION
            or metadata.database.pairing_type != target["pairing_type"]
        ):
            raise SchemaMigrationError(f"Active metadata mismatch: {directory / 'metadata.yaml'}")
        metadata_by_id[dataset.dataset_id] = metadata
        paths_by_id[dataset.dataset_id] = h5mu_path

        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            validation_report = json.loads(
                (directory / "validation_report.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SchemaMigrationError(
                f"Unable to read active artifacts in {directory}: {exc}"
            ) from exc
        manifest_h5mu = manifest.get("files", {}).get("h5mu", {})
        if (
            manifest.get("dataset_id") != dataset.dataset_id
            or manifest.get("database") != metadata.database_values()
            or manifest_h5mu.get("sha256") != target["sha256"]
            or manifest_h5mu.get("size") != target["file_size"]
            or validation_report.get("valid") is not True
            or validation_report.get("errors") != []
        ):
            raise SchemaMigrationError(f"Active artifact mismatch: {directory}")

        checksum = directory / "checksum.sha256"
        expected_line = f"{target['sha256']}  dataset.h5mu"
        if checksum.read_text(encoding="utf-8").strip() != expected_line:
            raise SchemaMigrationError(f"Checksum artifact mismatch: {checksum}")

    source_paths = {
        dataset_id: paths_by_id[dataset_id]
        for dataset_id, metadata in metadata_by_id.items()
        if metadata.database.dataset_type == "full"
    }
    peers: dict[str, list[Path]] = {}
    for dataset_id, metadata in metadata_by_id.items():
        derivation = metadata.database.derivation
        if derivation is not None:
            peers.setdefault(derivation.split_id, []).append(paths_by_id[dataset_id])
    for dataset in datasets:
        metadata = metadata_by_id[dataset.dataset_id]
        path = paths_by_id[dataset.dataset_id]
        derivation = metadata.database.derivation
        peer_paths = [] if derivation is None else [
            peer for peer in peers[derivation.split_id] if peer != path
        ]
        outcome = validate_h5mu(
            path,
            metadata,
            source_paths=source_paths if derivation is not None else None,
            peer_paths=peer_paths,
        )
        if not outcome.valid or outcome.n_obs != dataset.n_obs:
            details = "; ".join(f"{issue.code}: {issue.message}" for issue in outcome.errors)
            raise SchemaMigrationError(
                f"Active dataset validation failed for {dataset.dataset_id}: {details}"
            )

    exp_path_value = report.get("exp_path")
    exp_path = (
        Path(exp_path_value).resolve()
        if isinstance(exp_path_value, str)
        else (settings.data_root.parent.parent / "exp").resolve()
    )
    if not exp_path.is_dir():
        raise SchemaMigrationError(f"Active experiment directory does not exist: {exp_path}")
    _validate_exp(exp_path)
    active_exp_hashes = _file_hashes(exp_path)
    expected_exp_hashes = {
        name: values["new_sha256"] for name, values in report.get("exp_files", {}).items()
    }
    if active_exp_hashes != expected_exp_hashes:
        raise SchemaMigrationError("Active experiment files do not match the migration report.")

    shutil.rmtree(backup_path)
    report["backup_deleted_at"] = datetime.now(UTC).isoformat()
    write_json(report_path, report)
    return backup_path
