from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from sqlalchemy import select
from tifffile import TiffFile, TiffFileError, TiffPageSeries

from .auxiliary import AuxiliaryFile, AuxiliaryFileError, load_auxiliary_files
from .config import Settings
from .database import create_database_engine, create_session_factory, initialize_database
from .models import Dataset

HE_WSI_AUXILIARY_ID = "he_wsi"
THUMBNAIL_DIRECTORY = "database_thumbnails"
THUMBNAIL_MAX_DIMENSION = 640
THUMBNAIL_QUALITY = 85
THUMBNAIL_METHOD = 6


class ThumbnailGenerationError(RuntimeError):
    """Raised when a Database thumbnail cannot be generated safely."""


@dataclass(frozen=True)
class ThumbnailGenerationResult:
    dataset_id: str
    auxiliary_id: str
    source: Path
    destination: Path
    source_dimensions: tuple[int, int]
    thumbnail_dimensions: tuple[int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "auxiliary_id": self.auxiliary_id,
            "source": str(self.source),
            "destination": str(self.destination),
            "source_dimensions": list(self.source_dimensions),
            "thumbnail_dimensions": list(self.thumbnail_dimensions),
        }


def _dataset_record(dataset_id: str, settings: Settings) -> tuple[str, str]:
    engine = create_database_engine(settings.database_path)
    initialize_database(engine)
    session_factory = create_session_factory(engine)
    try:
        with session_factory() as session:
            record = session.execute(
                select(Dataset.dataset_type, Dataset.storage_dir).where(
                    Dataset.dataset_id == dataset_id
                )
            ).one_or_none()
    finally:
        engine.dispose()
    if record is None:
        raise ThumbnailGenerationError(f"Dataset {dataset_id!r} is not indexed.")
    dataset_type, storage_dir = record
    if dataset_type != "full":
        raise ThumbnailGenerationError(
            f"Dataset {dataset_id!r} is not a Database (dataset_type must be 'full')."
        )
    return dataset_type, storage_dir


def list_database_ids(settings: Settings) -> tuple[str, ...]:
    engine = create_database_engine(settings.database_path)
    initialize_database(engine)
    session_factory = create_session_factory(engine)
    try:
        with session_factory() as session:
            return tuple(
                session.scalars(
                    select(Dataset.dataset_id)
                    .where(Dataset.dataset_type == "full")
                    .order_by(Dataset.dataset_id)
                )
            )
    finally:
        engine.dispose()


def _dataset_directory(dataset_id: str, storage_dir: str, settings: Settings) -> Path:
    data_root = settings.data_root.resolve()
    dataset_dir = (settings.data_root / storage_dir).resolve()
    if dataset_dir.parent != data_root or not dataset_dir.is_dir():
        raise ThumbnailGenerationError(
            f"Unsafe or missing dataset directory for {dataset_id!r}: {dataset_dir}"
        )
    return dataset_dir


def _database_auxiliary_files(
    dataset_id: str, settings: Settings
) -> tuple[AuxiliaryFile, ...]:
    _, storage_dir = _dataset_record(dataset_id, settings)
    dataset_dir = _dataset_directory(dataset_id, storage_dir, settings)
    try:
        return load_auxiliary_files(dataset_dir, dataset_id)
    except AuxiliaryFileError as exc:
        raise ThumbnailGenerationError(str(exc)) from exc


def _registered_he_wsi(dataset_id: str, settings: Settings) -> AuxiliaryFile | None:
    for auxiliary_file in _database_auxiliary_files(dataset_id, settings):
        if auxiliary_file.auxiliary_id == HE_WSI_AUXILIARY_ID:
            return auxiliary_file
    return None


def _he_wsi_file(dataset_id: str, settings: Settings) -> AuxiliaryFile:
    auxiliary_file = _registered_he_wsi(dataset_id, settings)
    if auxiliary_file is not None:
        return auxiliary_file
    raise ThumbnailGenerationError(
        f"Database {dataset_id!r} does not have a registered {HE_WSI_AUXILIARY_ID!r} file."
    )


def database_has_he_wsi(dataset_id: str, settings: Settings) -> bool:
    return _registered_he_wsi(dataset_id, settings) is not None


def _thumbnail_destination(dataset_id: str, settings: Settings) -> Path:
    static_dir = settings.static_dir.resolve()
    thumbnail_dir = settings.static_dir / THUMBNAIL_DIRECTORY
    if thumbnail_dir.is_symlink():
        raise ThumbnailGenerationError(
            f"Thumbnail directory must not be a symlink: {thumbnail_dir}"
        )
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    if not thumbnail_dir.is_dir() or thumbnail_dir.resolve().parent != static_dir:
        raise ThumbnailGenerationError(f"Unsafe thumbnail directory: {thumbnail_dir}")
    destination = thumbnail_dir / f"{dataset_id}.webp"
    if destination.is_symlink():
        raise ThumbnailGenerationError(
            f"Thumbnail destination must not be a symlink: {destination}"
        )
    return destination


def _rgb_thumbnail(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    icc_profile = image.info.get("icc_profile")
    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
    else:
        rgb = image.convert("RGB")
    if icc_profile is not None:
        rgb.info["icc_profile"] = icc_profile
    return rgb


def _level_dimensions(level: TiffPageSeries) -> tuple[int, int]:
    if level.axes != "YXS" or len(level.shape) != 3 or level.shape[2] not in {3, 4}:
        raise ThumbnailGenerationError(
            f"WSI level must be an RGB or RGBA YXS image, got {level.axes} {level.shape!r}."
        )
    height, width, _ = level.shape
    if width < 1 or height < 1:
        raise ThumbnailGenerationError(f"WSI level has invalid dimensions: {(width, height)!r}")
    return int(width), int(height)


def _select_thumbnail_level(series: TiffPageSeries) -> TiffPageSeries:
    levels = list(series.levels)
    if not levels:
        raise ThumbnailGenerationError("WSI does not contain an image level.")
    eligible = [
        level
        for level in levels
        if max(_level_dimensions(level)) >= THUMBNAIL_MAX_DIMENSION
    ]
    if eligible:
        return min(eligible, key=lambda level: max(_level_dimensions(level)))
    return max(levels, key=lambda level: max(_level_dimensions(level)))


def _apply_orientation(image: Image.Image, orientation: int) -> Image.Image:
    transpose = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    return image.transpose(transpose) if transpose is not None else image


def _read_wsi_thumbnail(path: Path) -> tuple[tuple[int, int], Image.Image]:
    try:
        with TiffFile(path) as tif:
            if not tif.series:
                raise ThumbnailGenerationError("WSI does not contain an image series.")
            series = tif.series[0]
            source_dimensions = _level_dimensions(series)
            level = _select_thumbnail_level(series)
            page = level.pages[0]
            orientation_tag = page.tags.get(274)
            orientation = int(orientation_tag.value) if orientation_tag is not None else 1
            icc_tag = page.tags.get(34675)
            icc_profile = icc_tag.value if icc_tag is not None else None
            array = level.asarray()
    except (TiffFileError, MemoryError, OSError, ValueError) as exc:
        raise ThumbnailGenerationError(f"Unable to decode TIFF WSI {path}: {exc}") from exc

    try:
        image = Image.fromarray(array)
    except (TypeError, ValueError) as exc:
        raise ThumbnailGenerationError(f"Unsupported TIFF pixel data in {path}: {exc}") from exc
    oriented = _apply_orientation(image, orientation)
    if oriented is not image:
        image.close()
    thumbnail = _rgb_thumbnail(oriented)
    if thumbnail is not oriented:
        oriented.close()
    if isinstance(icc_profile, bytes):
        thumbnail.info["icc_profile"] = icc_profile
    thumbnail.thumbnail(
        (THUMBNAIL_MAX_DIMENSION, THUMBNAIL_MAX_DIMENSION),
        Image.Resampling.LANCZOS,
    )
    return source_dimensions, thumbnail


def _write_thumbnail_atomically(
    image: Image.Image, destination: Path, *, force: bool
) -> None:
    if destination.exists() and not force:
        raise ThumbnailGenerationError(
            f"Thumbnail already exists: {destination}; use --force to replace it."
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=".webp.tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        image.save(
            temporary_path,
            format="WEBP",
            quality=THUMBNAIL_QUALITY,
            method=THUMBNAIL_METHOD,
            icc_profile=image.info.get("icc_profile"),
        )
        with Image.open(temporary_path) as generated:
            generated.load()
            if (
                generated.format != "WEBP"
                or generated.mode != "RGB"
                or generated.size != image.size
            ):
                raise ThumbnailGenerationError(
                    f"Generated thumbnail failed validation: {temporary_path}"
                )
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        if destination.is_symlink():
            raise ThumbnailGenerationError(
                f"Thumbnail destination must not be a symlink: {destination}"
            )
        if destination.exists() and not force:
            raise ThumbnailGenerationError(
                f"Thumbnail already exists: {destination}; use --force to replace it."
            )
        os.replace(temporary_path, destination)
    except OSError as exc:
        raise ThumbnailGenerationError(
            f"Unable to write thumbnail {destination}: {exc}"
        ) from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def generate_wsi_thumbnail(
    dataset_id: str, settings: Settings, *, force: bool = False
) -> ThumbnailGenerationResult:
    auxiliary_file = _he_wsi_file(dataset_id, settings)
    destination = _thumbnail_destination(dataset_id, settings)
    if destination.exists() and not force:
        raise ThumbnailGenerationError(
            f"Thumbnail already exists: {destination}; use --force to replace it."
        )

    source_dimensions, thumbnail = _read_wsi_thumbnail(auxiliary_file.path)
    thumbnail_dimensions = thumbnail.size
    try:
        if min(thumbnail_dimensions) < 1 or max(thumbnail_dimensions) > THUMBNAIL_MAX_DIMENSION:
            raise ThumbnailGenerationError(
                f"Generated invalid thumbnail dimensions: {thumbnail_dimensions!r}"
            )
        _write_thumbnail_atomically(thumbnail, destination, force=force)
    finally:
        thumbnail.close()
    return ThumbnailGenerationResult(
        dataset_id=dataset_id,
        auxiliary_id=auxiliary_file.auxiliary_id,
        source=auxiliary_file.path,
        destination=destination,
        source_dimensions=source_dimensions,
        thumbnail_dimensions=thumbnail_dimensions,
    )
