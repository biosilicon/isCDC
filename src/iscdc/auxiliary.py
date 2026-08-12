from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from sqlalchemy import select

from .config import Settings
from .database import create_database_engine, create_session_factory, initialize_database
from .models import Dataset

MANIFEST_VERSION = "1.1"
SUPPORTED_MANIFEST_VERSIONS = {"1.0", MANIFEST_VERSION}
AUXILIARY_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AuxiliaryFileError(RuntimeError):
    """Raised when an auxiliary file cannot be safely read or registered."""


@dataclass(frozen=True)
class AuxiliaryFile:
    auxiliary_id: str
    label: str
    name: str
    media_type: str
    size: int
    sha256: str
    source_url: str
    retrieved_at: datetime
    path: Path

    @property
    def filename(self) -> str:
        return PurePosixPath(self.name).name


@dataclass(frozen=True)
class AuxiliaryRegistrationResult:
    dataset_id: str
    auxiliary_id: str
    destination: Path
    size: int
    sha256: str


def _load_manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuxiliaryFileError(f"Unable to read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuxiliaryFileError(f"Manifest must contain a JSON object: {path}")
    version = value.get("manifest_version")
    if version not in SUPPORTED_MANIFEST_VERSIONS:
        raise AuxiliaryFileError(f"Unsupported manifest version {version!r}: {path}")
    return value


def _validate_auxiliary_id(value: str) -> str:
    value = value.strip()
    if not AUXILIARY_ID_PATTERN.fullmatch(value):
        raise AuxiliaryFileError(
            "Auxiliary file ID must contain 1-64 lowercase letters, digits, underscores, "
            "or hyphens, and must start with a letter or digit."
        )
    return value


def _validate_source_url(value: str) -> str:
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise AuxiliaryFileError("Source URL must be an absolute HTTP or HTTPS URL.")
    return value


def _safe_auxiliary_path(dataset_dir: Path, name: str) -> Path:
    if "\\" in name:
        raise AuxiliaryFileError(f"Unsafe auxiliary file path: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != "auxiliary":
        raise AuxiliaryFileError(f"Unsafe auxiliary file path: {name!r}")
    if relative.parts[1] in {"", ".", ".."}:
        raise AuxiliaryFileError(f"Unsafe auxiliary file path: {name!r}")
    path = dataset_dir / relative.parts[0] / relative.parts[1]
    auxiliary_dir = dataset_dir / "auxiliary"
    if auxiliary_dir.is_symlink() or auxiliary_dir.resolve().parent != dataset_dir.resolve():
        raise AuxiliaryFileError(f"Unsafe auxiliary directory: {auxiliary_dir}")
    if path.parent.resolve() != auxiliary_dir.resolve():
        raise AuxiliaryFileError(f"Unsafe auxiliary file path: {name!r}")
    return path


def load_auxiliary_files(dataset_dir: Path, dataset_id: str) -> tuple[AuxiliaryFile, ...]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    if manifest.get("dataset_id") != dataset_id:
        raise AuxiliaryFileError(f"Manifest dataset ID differs from {dataset_id!r}.")
    entries = manifest.get("auxiliary_files", [])
    if not isinstance(entries, list):
        raise AuxiliaryFileError("Manifest auxiliary_files must be a list.")

    files: list[AuxiliaryFile] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise AuxiliaryFileError("Each auxiliary_files entry must be an object.")
        try:
            auxiliary_id = _validate_auxiliary_id(str(entry["id"]))
            label = str(entry["label"]).strip()
            name = str(entry["name"])
            media_type = str(entry["media_type"]).strip()
            size = int(entry["size"])
            sha256 = str(entry["sha256"])
            source_url = _validate_source_url(str(entry["source_url"]))
            retrieved_at = datetime.fromisoformat(str(entry["retrieved_at"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise AuxiliaryFileError(f"Invalid auxiliary file metadata: {entry!r}") from exc
        if not label or len(label) > 200:
            raise AuxiliaryFileError("Auxiliary file label must contain 1-200 characters.")
        if not media_type or len(media_type) > 200 or "/" not in media_type:
            raise AuxiliaryFileError(f"Invalid auxiliary media type: {media_type!r}")
        if size < 0:
            raise AuxiliaryFileError("Auxiliary file size must not be negative.")
        if not SHA256_PATTERN.fullmatch(sha256):
            raise AuxiliaryFileError("Auxiliary file SHA-256 must be lowercase hexadecimal.")
        if retrieved_at.tzinfo is None:
            raise AuxiliaryFileError("Auxiliary file retrieved_at must include a timezone.")
        if auxiliary_id in seen_ids or name in seen_names:
            raise AuxiliaryFileError("Auxiliary file IDs and paths must be unique.")
        seen_ids.add(auxiliary_id)
        seen_names.add(name)

        path = _safe_auxiliary_path(dataset_dir, name)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise AuxiliaryFileError(f"Auxiliary file is missing or has the wrong size: {path}")
        files.append(
            AuxiliaryFile(
                auxiliary_id=auxiliary_id,
                label=label,
                name=name,
                media_type=media_type,
                size=size,
                sha256=sha256,
                source_url=source_url,
                retrieved_at=retrieved_at,
                path=path,
            )
        )
    return tuple(sorted(files, key=lambda item: item.auxiliary_id))


def _copy_and_hash(source: Path, destination: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        while chunk := input_stream.read(1024 * 1024):
            output_stream.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    shutil.copystat(source, destination)
    return size, digest.hexdigest()


def _write_manifest_atomically(path: Path, manifest: dict) -> None:
    original_mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".manifest-", suffix=".json.tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, original_mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def register_auxiliary_file(
    dataset_id: str,
    source_path: Path,
    settings: Settings,
    *,
    auxiliary_id: str,
    label: str,
    source_url: str,
    media_type: str,
) -> AuxiliaryRegistrationResult:
    auxiliary_id = _validate_auxiliary_id(auxiliary_id)
    label = label.strip()
    if not label or len(label) > 200:
        raise AuxiliaryFileError("Auxiliary file label must contain 1-200 characters.")
    media_type = media_type.strip()
    if not media_type or len(media_type) > 200 or "/" not in media_type:
        raise AuxiliaryFileError(f"Invalid auxiliary media type: {media_type!r}")
    source_url = _validate_source_url(source_url)
    if source_path.is_symlink() or not source_path.is_file():
        raise AuxiliaryFileError(f"Auxiliary source must be a regular file: {source_path}")
    if source_path.name in {"", ".", ".."} or "/" in source_path.name or "\\" in source_path.name:
        raise AuxiliaryFileError(f"Unsafe auxiliary source filename: {source_path.name!r}")

    engine = create_database_engine(settings.database_path)
    initialize_database(engine)
    session_factory = create_session_factory(engine)
    try:
        with session_factory() as session:
            storage_dir = session.scalar(
                select(Dataset.storage_dir).where(Dataset.dataset_id == dataset_id)
            )
    finally:
        engine.dispose()
    if storage_dir is None:
        raise AuxiliaryFileError(f"Dataset {dataset_id!r} is not indexed.")

    data_root = settings.data_root.resolve()
    dataset_dir = (settings.data_root / storage_dir).resolve()
    if dataset_dir.parent != data_root or not dataset_dir.is_dir():
        raise AuxiliaryFileError(f"Unsafe or missing dataset directory: {dataset_dir}")
    manifest_path = dataset_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    if manifest.get("dataset_id") != dataset_id:
        raise AuxiliaryFileError(f"Manifest dataset ID differs from {dataset_id!r}.")
    entries = manifest.get("auxiliary_files", [])
    if not isinstance(entries, list):
        raise AuxiliaryFileError("Manifest auxiliary_files must be a list.")
    if any(
        isinstance(entry, dict)
        and (
            entry.get("id") == auxiliary_id
            or entry.get("name") == f"auxiliary/{source_path.name}"
        )
        for entry in entries
    ):
        raise AuxiliaryFileError(
            f"Auxiliary ID {auxiliary_id!r} or filename {source_path.name!r} is already registered."
        )

    auxiliary_dir = dataset_dir / "auxiliary"
    auxiliary_dir.mkdir(exist_ok=True)
    destination = _safe_auxiliary_path(dataset_dir, f"auxiliary/{source_path.name}")
    if destination.exists() or destination.is_symlink():
        raise AuxiliaryFileError(f"Auxiliary destination already exists: {destination}")
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{source_path.name}-", suffix=".tmp", dir=auxiliary_dir
    )
    os.close(descriptor)
    staging_path = Path(staging_name)
    staging_path.unlink()
    copied = False
    try:
        size, sha256 = _copy_and_hash(source_path, staging_path)
        if size != source_path.stat().st_size:
            raise AuxiliaryFileError("Auxiliary source changed while it was being copied.")
        os.replace(staging_path, destination)
        copied = True
        manifest["manifest_version"] = MANIFEST_VERSION
        manifest["auxiliary_files"] = sorted(
            [
                *entries,
                {
                    "id": auxiliary_id,
                    "label": label,
                    "media_type": media_type,
                    "name": f"auxiliary/{source_path.name}",
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "sha256": sha256,
                    "size": size,
                    "source_url": source_url,
                },
            ],
            key=lambda entry: entry["id"],
        )
        _write_manifest_atomically(manifest_path, manifest)
        return AuxiliaryRegistrationResult(
            dataset_id=dataset_id,
            auxiliary_id=auxiliary_id,
            destination=destination,
            size=size,
            sha256=sha256,
        )
    except Exception:
        if copied and destination.exists():
            destination.unlink()
        raise
    finally:
        if staging_path.exists():
            staging_path.unlink()
        try:
            auxiliary_dir.rmdir()
        except OSError:
            pass
