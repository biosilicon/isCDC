from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from .artifacts import write_dataset_artifacts
from .auxiliary import AuxiliaryFileError, load_auxiliary_files
from .config import Settings
from .database import create_database_engine, create_session_factory, initialize_database
from .models import Dataset, Modality
from .schemas import MetadataDocument, load_metadata
from .validation import ValidationOutcome, validate_h5mu


class DatasetImportError(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class ImportResult:
    dataset_id: str
    destination: Path
    file_size: int
    sha256: str
    warning_count: int


def _copy_and_hash(source: Path, destination: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        while chunk := input_stream.read(8 * 1024 * 1024):
            output_stream.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    shutil.copystat(source, destination)
    return size, digest.hexdigest()


def _replacement_identity(value: Dataset | MetadataDocument) -> dict[str, Any]:
    if isinstance(value, Dataset):
        dataset_type = value.dataset_type
        derivation = value.derivation
    else:
        dataset_type = value.database.dataset_type
        derivation = (
            value.database.derivation.model_dump(mode="python")
            if value.database.derivation
            else None
        )
    identity: dict[str, Any] = {"dataset_type": dataset_type}
    if derivation is not None:
        identity.update(
            {
                "construction_type": derivation.get("construction_type"),
                "source_dataset_ids": list(derivation.get("source_dataset_ids", [])),
                "split_id": derivation.get("split_id"),
                "challenge_type": derivation.get("challenge_type"),
            }
        )
    return identity


def _stage_auxiliary_files(
    existing_dir: Path, stage_dir: Path, dataset_id: str
) -> list[dict]:
    try:
        files = load_auxiliary_files(existing_dir, dataset_id)
    except AuxiliaryFileError as exc:
        raise DatasetImportError(
            f"Cannot safely preserve auxiliary files for {dataset_id!r}: {exc}"
        ) from exc
    if not files:
        return []

    auxiliary_dir = stage_dir / "auxiliary"
    auxiliary_dir.mkdir()
    entries: list[dict] = []
    for auxiliary in files:
        destination = auxiliary_dir / auxiliary.filename
        size, sha256 = _copy_and_hash(auxiliary.path, destination)
        if size != auxiliary.size or sha256 != auxiliary.sha256:
            raise DatasetImportError(
                f"Auxiliary file changed or failed checksum validation: {auxiliary.path}"
            )
        entries.append(
            {
                "id": auxiliary.auxiliary_id,
                "label": auxiliary.label,
                "media_type": auxiliary.media_type,
                "name": auxiliary.name,
                "retrieved_at": auxiliary.retrieved_at.isoformat(),
                "sha256": auxiliary.sha256,
                "size": auxiliary.size,
                "source_url": auxiliary.source_url,
            }
        )
    return entries


def build_dataset_record(
    metadata: MetadataDocument,
    outcome: ValidationOutcome,
    file_size: int,
    sha256: str,
    imported_at: datetime,
) -> Dataset:
    database = metadata.database
    dataset = Dataset(
        dataset_id=database.dataset_id,
        schema_version=database.schema_version,
        dataset_type=database.dataset_type,
        title=metadata.title,
        description=metadata.description,
        source=database.source,
        organism=database.organism,
        tissue=database.tissue,
        spatial_unit=database.spatial_unit,
        coordinate_unit=database.coordinate_unit,
        pairing_type=database.pairing_type,
        derivation=database.derivation.model_dump(mode="python") if database.derivation else None,
        split_id=database.derivation.split_id if database.derivation else None,
        sample_ids=metadata.sample_ids,
        keywords=metadata.keywords,
        license=metadata.license.model_dump() if metadata.license else None,
        publication=metadata.publication.model_dump() if metadata.publication else None,
        additional_metadata=metadata.additional_database_values(),
        n_obs=outcome.n_obs,
        coordinate_dimensions=outcome.coordinate_dimensions,
        file_size=file_size,
        sha256=sha256,
        storage_dir=database.dataset_id,
        validation_warning_count=len(outcome.warnings),
        imported_at=imported_at,
    )
    dataset.modalities = [
        Modality(
            name=summary.name,
            technology=summary.technology,
            value_type=summary.value_type,
            n_obs=summary.n_obs,
            n_vars=summary.n_vars,
        )
        for summary in sorted(outcome.modalities.values(), key=lambda item: item.name)
    ]
    return dataset


def import_dataset(
    h5mu_path: Path,
    metadata_path: Path,
    settings: Settings,
    *,
    replace: bool = False,
) -> ImportResult:
    h5mu_path = h5mu_path.expanduser().resolve()
    metadata_path = metadata_path.expanduser().resolve()
    if not h5mu_path.is_file():
        raise DatasetImportError(f"MuData file does not exist: {h5mu_path}")
    if h5mu_path.suffix.lower() != ".h5mu":
        raise DatasetImportError("The data file must use the .h5mu extension.")
    if not metadata_path.is_file():
        raise DatasetImportError(f"Metadata file does not exist: {metadata_path}")

    metadata = load_metadata(metadata_path)
    dataset_id = metadata.database.dataset_id
    settings.data_root.mkdir(parents=True, exist_ok=True)
    staging_root = settings.data_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    final_dir = settings.data_root / dataset_id

    engine = create_database_engine(settings.database_path)
    initialize_database(engine)
    session_factory = create_session_factory(engine)
    source_paths: dict[str, Path] = {}
    peer_paths: list[Path] = []
    expected_identity: dict[str, Any] | None = None
    with session_factory() as session:
        existing = session.get(Dataset, dataset_id)
        if existing is not None and not replace:
            raise DatasetImportError(f"Dataset '{dataset_id}' is already indexed.")
        if existing is None and replace:
            raise DatasetImportError(
                f"Dataset '{dataset_id}' is not indexed and cannot be replaced."
            )
        if existing is not None:
            expected_identity = _replacement_identity(existing)
            requested_identity = _replacement_identity(metadata)
            if expected_identity != requested_identity:
                raise DatasetImportError(
                    "Replacement must preserve dataset type and derivation identity; "
                    f"indexed={expected_identity!r}, requested={requested_identity!r}."
                )
        derivation = metadata.database.derivation
        if derivation is not None:
            sources = list(
                session.scalars(
                    select(Dataset).where(Dataset.dataset_id.in_(derivation.source_dataset_ids))
                ).all()
            )
            sources_by_id = {source.dataset_id: source for source in sources}
            missing_sources = [
                source_id
                for source_id in derivation.source_dataset_ids
                if source_id not in sources_by_id
            ]
            if missing_sources:
                raise DatasetImportError(
                    "Derived datasets require their full sources to be imported first; missing: "
                    + ", ".join(missing_sources)
                )
            non_full_sources = [
                source_id
                for source_id, source in sources_by_id.items()
                if source.dataset_type != "full"
            ]
            if non_full_sources:
                raise DatasetImportError(
                    "Derived source_dataset_ids must identify full datasets: "
                    + ", ".join(non_full_sources)
                )
            source_paths = {
                source_id: settings.data_root / source.storage_dir / "dataset.h5mu"
                for source_id, source in sources_by_id.items()
            }
            opposite_type = "test" if metadata.database.dataset_type == "train" else "train"
            peers = list(
                session.scalars(
                    select(Dataset).where(
                        Dataset.dataset_type == opposite_type,
                        Dataset.split_id == derivation.split_id,
                    )
                ).all()
            )
            peer_paths = [settings.data_root / peer.storage_dir / "dataset.h5mu" for peer in peers]
    if final_dir.exists() and not replace:
        raise DatasetImportError(f"Dataset directory already exists: {final_dir}")
    if replace and not final_dir.is_dir():
        raise DatasetImportError(f"Indexed dataset directory is missing: {final_dir}")

    stage_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_id}-", dir=staging_root))
    backup_dir: Path | None = None
    old_moved = False
    new_moved = False
    try:
        staged_h5mu = stage_dir / "dataset.h5mu"
        file_size, sha256 = _copy_and_hash(h5mu_path, staged_h5mu)
        outcome = validate_h5mu(
            staged_h5mu,
            metadata,
            source_paths=source_paths or None,
            peer_paths=peer_paths,
        )
        checked_at = datetime.now(timezone.utc)
        report = outcome.report(checked_at.isoformat(), "dataset.h5mu")
        if not outcome.valid:
            raise DatasetImportError("MuData validation failed.", report=report)

        shutil.copy2(metadata_path, stage_dir / "metadata.yaml")
        imported_at = checked_at
        auxiliary_files = (
            _stage_auxiliary_files(final_dir, stage_dir, dataset_id) if replace else []
        )
        write_dataset_artifacts(
            stage_dir,
            metadata,
            outcome,
            file_size,
            sha256,
            imported_at,
            checked_at,
            auxiliary_files=auxiliary_files,
        )

        with session_factory() as session:
            try:
                current = session.get(Dataset, dataset_id)
                if replace:
                    if current is None or _replacement_identity(current) != expected_identity:
                        raise DatasetImportError(
                            f"Dataset '{dataset_id}' changed while replacement was staged."
                        )
                    session.delete(current)
                    session.flush()
                elif current is not None:
                    raise DatasetImportError(f"Dataset '{dataset_id}' is already indexed.")
                session.add(
                    build_dataset_record(metadata, outcome, file_size, sha256, imported_at)
                )
                session.flush()
                if replace:
                    backup_dir = Path(
                        tempfile.mkdtemp(prefix=f"{dataset_id}-backup-", dir=staging_root)
                    )
                    backup_dir.rmdir()
                    os.replace(final_dir, backup_dir)
                    old_moved = True
                os.replace(stage_dir, final_dir)
                new_moved = True
                session.commit()
            except Exception:
                session.rollback()
                if new_moved and final_dir.exists():
                    os.replace(final_dir, stage_dir)
                    new_moved = False
                if old_moved and backup_dir is not None and backup_dir.exists():
                    os.replace(backup_dir, final_dir)
                    old_moved = False
                raise

        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir)

        return ImportResult(
            dataset_id=dataset_id,
            destination=final_dir,
            file_size=file_size,
            sha256=sha256,
            warning_count=len(outcome.warnings),
        )
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        if backup_dir is not None and backup_dir.exists() and not final_dir.exists():
            os.replace(backup_dir, final_dir)
        try:
            staging_root.rmdir()
        except OSError:
            pass
        engine.dispose()
