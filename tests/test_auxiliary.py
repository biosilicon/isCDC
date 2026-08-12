from __future__ import annotations

import hashlib
import json

import httpx
import pytest

import iscdc.auxiliary as auxiliary_module
from iscdc.app import create_app
from iscdc.auxiliary import AuxiliaryFileError, load_auxiliary_files, register_auxiliary_file
from iscdc.cli import main
from iscdc.importer import import_dataset


def _register_test_file(settings, source):  # noqa: ANN001, ANN202
    return register_auxiliary_file(
        "test_rna_protein",
        source,
        settings,
        auxiliary_id="he_wsi",
        label="H&E whole-slide image",
        source_url="https://example.org/test.ome.tif",
        media_type="image/tiff",
    )


def test_register_auxiliary_file_updates_manifest_and_preserves_source(
    tmp_path, settings, write_h5mu, write_metadata
):
    import_dataset(write_h5mu(), write_metadata(), settings)
    source = tmp_path / "test.ome.tif"
    content = b"II*\x00" + bytes(range(64))
    source.write_bytes(content)
    manifest_path = settings.data_root / "test_rna_protein" / "manifest.json"
    manifest_mode = manifest_path.stat().st_mode

    result = _register_test_file(settings, source)

    assert source.read_bytes() == content
    assert result.destination.read_bytes() == content
    assert result.size == len(content)
    assert result.sha256 == hashlib.sha256(content).hexdigest()
    assert manifest_path.stat().st_mode == manifest_mode
    manifest = json.loads(manifest_path.read_text())
    assert manifest["manifest_version"] == "1.1"
    assert manifest["auxiliary_files"] == [
        {
            "id": "he_wsi",
            "label": "H&E whole-slide image",
            "media_type": "image/tiff",
            "name": "auxiliary/test.ome.tif",
            "retrieved_at": manifest["auxiliary_files"][0]["retrieved_at"],
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
            "source_url": "https://example.org/test.ome.tif",
        }
    ]
    loaded = load_auxiliary_files(
        settings.data_root / "test_rna_protein", "test_rna_protein"
    )
    assert len(loaded) == 1
    assert loaded[0].filename == "test.ome.tif"
    assert loaded[0].path == result.destination

    with pytest.raises(AuxiliaryFileError, match="already registered"):
        _register_test_file(settings, source)


def test_registration_failure_removes_copied_file(
    tmp_path, settings, write_h5mu, write_metadata, monkeypatch
):
    import_dataset(write_h5mu(), write_metadata(), settings)
    source = tmp_path / "failed.tif"
    source.write_bytes(b"MM\x00*test")
    manifest_path = settings.data_root / "test_rna_protein" / "manifest.json"
    original_manifest = manifest_path.read_bytes()

    def fail_manifest_write(_path, _manifest):  # noqa: ANN001, ANN202
        raise OSError("simulated manifest failure")

    monkeypatch.setattr(auxiliary_module, "_write_manifest_atomically", fail_manifest_write)
    with pytest.raises(OSError, match="simulated manifest failure"):
        _register_test_file(settings, source)

    assert not (
        settings.data_root / "test_rna_protein" / "auxiliary" / source.name
    ).exists()
    assert manifest_path.read_bytes() == original_manifest


@pytest.mark.parametrize("auxiliary_id", ["../wsi", "WSI", "", "a" * 65])
def test_registration_rejects_unsafe_auxiliary_ids(
    tmp_path, settings, write_h5mu, write_metadata, auxiliary_id
):
    import_dataset(write_h5mu(), write_metadata(), settings)
    source = tmp_path / "test.tif"
    source.write_bytes(b"II*\x00test")

    with pytest.raises(AuxiliaryFileError, match="Auxiliary file ID"):
        register_auxiliary_file(
            "test_rna_protein",
            source,
            settings,
            auxiliary_id=auxiliary_id,
            label="H&E whole-slide image",
            source_url="https://example.org/test.tif",
            media_type="image/tiff",
        )


def test_registration_rejects_symlink_source(
    tmp_path, settings, write_h5mu, write_metadata
):
    import_dataset(write_h5mu(), write_metadata(), settings)
    target = tmp_path / "target.tif"
    target.write_bytes(b"II*\x00test")
    source = tmp_path / "linked.tif"
    source.symlink_to(target)

    with pytest.raises(AuxiliaryFileError, match="regular file"):
        _register_test_file(settings, source)


def test_auxiliary_cli_reports_unknown_dataset(tmp_path, settings, monkeypatch, capsys):
    source = tmp_path / "test.tif"
    source.write_bytes(b"II*\x00test")
    monkeypatch.setenv("ISCDC_DATABASE_PATH", str(settings.database_path))
    monkeypatch.setenv("ISCDC_DATA_ROOT", str(settings.data_root))

    result = main(
        [
            "add-auxiliary-file",
            "unknown",
            str(source),
            "--id",
            "he_wsi",
            "--label",
            "H&E whole-slide image",
            "--source-url",
            "https://example.org/test.tif",
            "--media-type",
            "image/tiff",
        ]
    )

    assert result == 1
    assert "not indexed" in capsys.readouterr().err


@pytest.mark.anyio
async def test_auxiliary_file_appears_on_detail_and_supports_ranges(
    tmp_path, settings, write_h5mu, write_metadata
):
    import_dataset(write_h5mu(), write_metadata(), settings)
    source = tmp_path / "test.ome.tif"
    content = b"II*\x00" + bytes(range(128))
    source.write_bytes(content)
    _register_test_file(settings, source)
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listing = await client.get("/databases")
        detail = await client.get("/databases/test_rna_protein")
        api = await client.get("/api/databases/test_rna_protein")
        head = await client.head("/downloads/test_rna_protein/auxiliary/he_wsi")
        partial = await client.get(
            "/downloads/test_rna_protein/auxiliary/he_wsi",
            headers={"Range": "bytes=4-11"},
        )
        invalid_range = await client.get(
            "/downloads/test_rna_protein/auxiliary/he_wsi",
            headers={"Range": "bytes=999-1000"},
        )
        missing = await client.get(
            "/downloads/test_rna_protein/auxiliary/unknown"
        )

    assert "H&amp;E whole-slide image" not in listing.text
    assert "H&amp;E whole-slide image" in detail.text
    payload = api.json()["auxiliary_files"]
    assert payload == [
        {
            "id": "he_wsi",
            "label": "H&E whole-slide image",
            "filename": "test.ome.tif",
            "media_type": "image/tiff",
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "source_url": "https://example.org/test.ome.tif",
            "download_url": "http://test/downloads/test_rna_protein/auxiliary/he_wsi",
        }
    ]
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-length"] == str(len(content))
    assert head.headers["accept-ranges"] == "bytes"
    assert partial.status_code == 206
    assert partial.content == content[4:12]
    assert partial.headers["accept-ranges"] == "bytes"
    assert partial.headers["content-range"] == f"bytes 4-11/{len(content)}"
    assert "test.ome.tif" in partial.headers["content-disposition"]
    assert invalid_range.status_code == 416
    assert missing.status_code == 404


@pytest.mark.anyio
async def test_invalid_auxiliary_manifest_fails_open(
    settings, write_h5mu, write_metadata
):
    import_dataset(write_h5mu(), write_metadata(), settings)
    manifest_path = settings.data_root / "test_rna_protein" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["auxiliary_files"] = {"not": "a list"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        detail = await client.get("/databases/test_rna_protein")
        api = await client.get("/api/databases/test_rna_protein")
        missing = await client.get(
            "/downloads/test_rna_protein/auxiliary/he_wsi"
        )

    assert detail.status_code == 200
    assert api.status_code == 200
    assert api.json()["auxiliary_files"] == []
    assert missing.status_code == 404
