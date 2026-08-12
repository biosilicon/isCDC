from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tifffile import TiffWriter, imwrite

import iscdc.cli as cli_module
from iscdc.auxiliary import register_auxiliary_file
from iscdc.cli import build_parser, main
from iscdc.database import create_database_engine, create_session_factory
from iscdc.importer import import_dataset
from iscdc.models import Dataset
from iscdc.thumbnails import (
    ThumbnailGenerationError,
    ThumbnailGenerationResult,
    database_has_he_wsi,
    generate_wsi_thumbnail,
)


def _local_static_settings(settings, tmp_path):  # noqa: ANN001, ANN202
    return replace(settings, static_dir=tmp_path / "static")


def _register_wsi(settings, source: Path, dataset_id: str = "test_rna_protein") -> None:
    register_auxiliary_file(
        dataset_id,
        source,
        settings,
        auxiliary_id="he_wsi",
        label="H&E whole-slide image",
        source_url=f"https://example.org/{source.name}",
        media_type="image/tiff",
    )


def _write_pyramidal_wsi(path: Path) -> None:
    full = np.full((1200, 1600, 3), (220, 20, 20), dtype=np.uint8)
    middle = np.full((600, 800, 3), (20, 220, 20), dtype=np.uint8)
    lowest = np.full((300, 400, 3), (20, 20, 220), dtype=np.uint8)
    with TiffWriter(path) as tif:
        tif.write(
            full,
            tile=(128, 128),
            photometric="rgb",
            subifds=2,
            metadata=None,
        )
        tif.write(
            middle,
            tile=(128, 128),
            photometric="rgb",
            subfiletype=1,
            metadata=None,
        )
        tif.write(
            lowest,
            tile=(128, 128),
            photometric="rgb",
            subfiletype=1,
            metadata=None,
        )


def _prepare_database_with_wsi(
    tmp_path, settings, write_h5mu, write_metadata
):  # noqa: ANN001, ANN202
    local_settings = _local_static_settings(settings, tmp_path)
    import_dataset(write_h5mu(), write_metadata(), local_settings)
    source = tmp_path / "test.ome.tif"
    _write_pyramidal_wsi(source)
    _register_wsi(local_settings, source)
    return local_settings, source


def test_generate_wsi_thumbnail_uses_smallest_sufficient_pyramid_level(
    tmp_path, settings, write_h5mu, write_metadata
):
    local_settings, source = _prepare_database_with_wsi(
        tmp_path, settings, write_h5mu, write_metadata
    )

    result = generate_wsi_thumbnail("test_rna_protein", local_settings)

    assert result.source_dimensions == (1600, 1200)
    assert result.thumbnail_dimensions == (640, 480)
    assert result.source != source
    assert result.destination == (
        local_settings.static_dir / "database_thumbnails" / "test_rna_protein.webp"
    )
    assert database_has_he_wsi("test_rna_protein", local_settings)
    with Image.open(result.destination) as thumbnail:
        thumbnail.load()
        assert thumbnail.format == "WEBP"
        assert thumbnail.mode == "RGB"
        assert thumbnail.size == (640, 480)
        red, green, blue = thumbnail.getpixel((320, 240))
        assert green > 180
        assert red < 60
        assert blue < 60


def test_generate_wsi_thumbnail_requires_force_and_preserves_old_file_on_failure(
    tmp_path, settings, write_h5mu, write_metadata, monkeypatch
):
    local_settings, _ = _prepare_database_with_wsi(
        tmp_path, settings, write_h5mu, write_metadata
    )
    destination = generate_wsi_thumbnail("test_rna_protein", local_settings).destination
    original = destination.read_bytes()

    with pytest.raises(ThumbnailGenerationError, match="use --force"):
        generate_wsi_thumbnail("test_rna_protein", local_settings)
    assert destination.read_bytes() == original

    generate_wsi_thumbnail("test_rna_protein", local_settings, force=True)
    replacement = destination.read_bytes()

    def fail_save(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise OSError("simulated WebP write failure")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    with pytest.raises(ThumbnailGenerationError, match="simulated WebP write failure"):
        generate_wsi_thumbnail("test_rna_protein", local_settings, force=True)
    assert destination.read_bytes() == replacement
    assert not list(destination.parent.glob(".*.webp.tmp"))


def test_generate_wsi_thumbnail_reports_unknown_missing_and_non_database_entries(
    tmp_path, settings, write_h5mu, write_metadata
):
    local_settings = _local_static_settings(settings, tmp_path)
    import_dataset(write_h5mu(), write_metadata(), local_settings)

    with pytest.raises(ThumbnailGenerationError, match="not indexed"):
        generate_wsi_thumbnail("unknown", local_settings)
    with pytest.raises(ThumbnailGenerationError, match="does not have a registered"):
        generate_wsi_thumbnail("test_rna_protein", local_settings)
    assert not database_has_he_wsi("test_rna_protein", local_settings)

    engine = create_database_engine(local_settings.database_path)
    try:
        with create_session_factory(engine)() as session:
            dataset = session.get(Dataset, "test_rna_protein")
            assert dataset is not None
            dataset.dataset_type = "train"
            session.commit()
    finally:
        engine.dispose()
    with pytest.raises(ThumbnailGenerationError, match="not a Database"):
        generate_wsi_thumbnail("test_rna_protein", local_settings)


def test_generate_wsi_thumbnail_rejects_non_tiff_auxiliary_file(
    tmp_path, settings, write_h5mu, write_metadata
):
    local_settings = _local_static_settings(settings, tmp_path)
    import_dataset(write_h5mu(), write_metadata(), local_settings)
    source = tmp_path / "invalid.tif"
    source.write_bytes(b"not a TIFF")
    _register_wsi(local_settings, source)

    with pytest.raises(ThumbnailGenerationError, match="Unable to decode TIFF WSI"):
        generate_wsi_thumbnail("test_rna_protein", local_settings)
    thumbnail_dir = local_settings.static_dir / "database_thumbnails"
    assert not list(thumbnail_dir.iterdir())


def test_generate_wsi_thumbnails_cli_processes_all_registered_databases(
    tmp_path, settings, write_h5mu, write_metadata, metadata_values, monkeypatch, capsys
):
    local_settings = _local_static_settings(settings, tmp_path)
    import_dataset(write_h5mu(), write_metadata(), local_settings)

    metadata_values["database"]["dataset_id"] = "with_wsi"
    second_h5mu = write_h5mu(name="with_wsi.h5mu")
    second_metadata = write_metadata(metadata_values, name="with_wsi.metadata.yaml")
    import_dataset(second_h5mu, second_metadata, local_settings)
    source = tmp_path / "with_wsi.tif"
    image = np.full((640, 960, 3), (80, 100, 120), dtype=np.uint8)
    imwrite(source, image, tile=(128, 128), photometric="rgb", metadata=None)
    _register_wsi(local_settings, source, dataset_id="with_wsi")

    monkeypatch.setattr(
        cli_module.Settings,
        "from_environment",
        classmethod(lambda _cls: local_settings),
    )
    exit_code = main(["generate-wsi-thumbnails", "--all"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["skipped_without_wsi"] == 1
    assert payload["failures"] == []
    assert [item["dataset_id"] for item in payload["generated"]] == ["with_wsi"]
    assert payload["generated"][0]["thumbnail_dimensions"] == [640, 427]


def test_generate_wsi_thumbnails_cli_processes_one_database_with_force(
    tmp_path, settings, monkeypatch, capsys
):
    local_settings = _local_static_settings(settings, tmp_path)
    monkeypatch.setattr(
        cli_module.Settings,
        "from_environment",
        classmethod(lambda _cls: local_settings),
    )

    def generate(dataset_id, selected_settings, *, force):  # noqa: ANN001, ANN202
        assert dataset_id == "one_database"
        assert selected_settings == local_settings
        assert force
        return ThumbnailGenerationResult(
            dataset_id=dataset_id,
            auxiliary_id="he_wsi",
            source=tmp_path / "source.tif",
            destination=tmp_path / "one_database.webp",
            source_dimensions=(1000, 500),
            thumbnail_dimensions=(640, 320),
        )

    monkeypatch.setattr(cli_module, "generate_wsi_thumbnail", generate)
    exit_code = main(["generate-wsi-thumbnails", "one_database", "--force"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["skipped_without_wsi"] == 0
    assert payload["failures"] == []
    assert [item["dataset_id"] for item in payload["generated"]] == ["one_database"]


def test_generate_wsi_thumbnails_cli_reports_partial_batch_failure(
    tmp_path, settings, monkeypatch, capsys
):
    local_settings = _local_static_settings(settings, tmp_path)
    monkeypatch.setattr(
        cli_module.Settings,
        "from_environment",
        classmethod(lambda _cls: local_settings),
    )
    monkeypatch.setattr(cli_module, "list_database_ids", lambda _settings: ("bad", "good"))
    monkeypatch.setattr(
        cli_module, "database_has_he_wsi", lambda _dataset_id, _settings: True
    )

    def generate(dataset_id, _settings, *, force):  # noqa: ANN001, ANN202
        assert not force
        if dataset_id == "bad":
            raise ThumbnailGenerationError("simulated failure")
        return ThumbnailGenerationResult(
            dataset_id=dataset_id,
            auxiliary_id="he_wsi",
            source=tmp_path / "source.tif",
            destination=tmp_path / "good.webp",
            source_dimensions=(1000, 500),
            thumbnail_dimensions=(640, 320),
        )

    monkeypatch.setattr(cli_module, "generate_wsi_thumbnail", generate)
    exit_code = main(["generate-wsi-thumbnails", "--all"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert [item["dataset_id"] for item in payload["generated"]] == ["good"]
    assert payload["failures"] == [
        {"dataset_id": "bad", "error": "simulated failure"}
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        ["generate-wsi-thumbnails"],
        ["generate-wsi-thumbnails", "test_rna_protein", "--all"],
    ],
)
def test_generate_wsi_thumbnails_cli_requires_exactly_one_scope(arguments):
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)
