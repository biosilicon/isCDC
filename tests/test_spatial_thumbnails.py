from __future__ import annotations

import json
from dataclasses import replace

import h5py
import httpx
import mudata as md
import numpy as np
import pandas as pd
import pytest
from anndata.io import write_elem
from PIL import Image
from scipy import sparse

import iscdc.spatial_thumbnails as previews
from iscdc.app import create_app
from iscdc.auxiliary import register_auxiliary_file
from iscdc.cli import build_parser, main
from iscdc.importer import import_dataset
from iscdc.thumbnails import ThumbnailGenerationError


@pytest.mark.parametrize("encoding", ["dense", "csr", "csc"])
def test_row_totals_preserve_zero_rows_without_densifying(tmp_path, encoding):
    values = np.array([[0, 0, 0], [2, 0, 4], [0, 0, 0], [0, 7, 0]], dtype=float)
    matrix = {"dense": values, "csr": sparse.csr_matrix(values), "csc": sparse.csc_matrix(values)}[
        encoding
    ]
    with h5py.File(tmp_path / "matrix.h5", "w") as handle:
        write_elem(handle, "X", matrix)
        np.testing.assert_array_equal(previews.row_totals(handle["X"]), [0, 6, 0, 7])


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, 0.5])
def test_illegal_counts_fail(tmp_path, bad):
    with h5py.File(tmp_path / "bad.h5", "w") as handle:
        write_elem(handle, "X", np.array([[bad, 2.0]]))
        with pytest.raises(ThumbnailGenerationError):
            previews.row_totals(handle["X"])


def _raw_input(path, *, array=False, samples=None, foreign=False):
    with h5py.File(path, "w") as handle:
        write_elem(
            handle,
            "obs",
            pd.DataFrame({"sample_id": samples or ["s", "s", "s"]}, index=["a", "b", "c"]),
        )
        write_elem(handle, "obsm", {"spatial": np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])})
        write_elem(
            handle,
            "uns",
            {
                "database": {
                    "coordinate_unit": "array_index" if array else "pixel",
                    "entry_id": "S065",
                    "dataset_id": "example",
                }
            },
        )
        mod = handle.create_group("mod").create_group("rna")
        write_elem(mod, "obs", pd.DataFrame(index=["foreign" if foreign else "c", "a"]))
        write_elem(mod, "var", pd.DataFrame(index=["g1", "g2"]))
        write_elem(mod, "uns", {"assay": {"value_type": "counts"}})
        write_elem(mod, "X", sparse.csr_matrix([[3, 4], [1, 2]]))


def test_rna_membership_and_order_align_by_ids(tmp_path):
    path = tmp_path / "input.h5mu"
    _raw_input(path)
    xy, signal, info, audit = previews.read_inputs(path)
    np.testing.assert_array_equal(xy, [[1, 1], [0, 0]])
    np.testing.assert_array_equal(signal, [7, 3])
    assert list(audit["obs_ids"]) == ["c", "a"]
    assert list(audit["top_level_positions"]) == [2, 0]
    assert info["plotted_n_obs"] == 2


@pytest.mark.parametrize("kwargs", [{"samples": ["a", "b", "b"]}, {"foreign": True}])
def test_unaligned_or_multiple_samples_fail(tmp_path, kwargs):
    path = tmp_path / "input.h5mu"
    _raw_input(path, **kwargs)
    with pytest.raises(ThumbnailGenerationError):
        previews.read_inputs(path)


def test_visium_geometry_is_display_only_and_hexagonal(tmp_path):
    path = tmp_path / "input.h5mu"
    _raw_input(path, array=True)
    before = previews.sha256_file(path)
    xy, _, info, audit = previews.read_inputs(path)
    assert np.linalg.norm(xy[0] - xy[1]) == pytest.approx(2)
    np.testing.assert_array_equal(audit["source_xy"], [[1, 1], [0, 0]])
    assert info["coordinate_scale"] == [1.0, np.sqrt(3)]
    assert previews.sha256_file(path) == before


def test_background_fades_continuously_and_constant_signal_survives():
    values, _ = previews.scale_signal(np.arange(101))
    assert values[0] == 0 and values[-1] == 1
    assert np.all(np.diff(values) >= 0)
    assert len(np.unique(values)) > 90
    np.testing.assert_array_equal(previews.scale_signal(np.ones(3))[0], [0.5] * 3)
    np.testing.assert_array_equal(previews.scale_signal(np.zeros(3))[0], [0] * 3)


def test_render_retains_holes_disconnected_patches_and_full_extent():
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    ring = np.column_stack([np.cos(theta), np.sin(theta)])
    xy = np.concatenate([ring, ring + [5, 0]])
    image, info, _ = previews.render_preview(xy, np.ones(len(xy)), "rna_signal")
    pixels = np.asarray(image)
    assert image.mode == "RGB" and max(image.size) == 640
    assert image.width > 2 * image.height
    assert (pixels[:, image.width // 2] == 255).all()  # gap between islands
    origin = xy.min(0)
    for center in [[0, 0], [5, 0]]:
        x, y = np.rint(
            (np.array(center) - origin + info["padding"]) * info["pixels_per_coordinate_unit"]
        ).astype(int)
        assert (pixels[y, x] == 255).all()  # no filled hole
    assert np.any(pixels < 200)
    image.close()


def test_render_overlap_is_order_independent_and_density_handles_duplicates():
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    signal = np.array([3, 1, 8, 2, 20])
    first, _, _ = previews.render_preview(xy, signal, "rna_signal")
    second, _, _ = previews.render_preview(xy[::-1], signal[::-1], "rna_signal")
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    density, _, arrays = previews.render_preview(xy, np.empty(0), "point_density")
    assert np.isfinite(arrays["signal"]).all()
    assert arrays["signal"][0] == arrays["signal"][1]
    first.close()
    second.close()
    density.close()


@pytest.mark.parametrize(
    "xy", [np.array([[0, 0], [1, np.nan]]), np.zeros((3, 2)), np.zeros((3, 3))]
)
def test_bad_coordinates_fail(xy):
    with pytest.raises(ThumbnailGenerationError):
        previews.render_preview(xy, np.ones(len(xy)), "rna_signal")


def _prepared(settings, tmp_path, write_h5mu, write_metadata):
    local = replace(settings, static_dir=tmp_path / "static", analytics_enabled=False)
    local.static_dir.mkdir()
    import_dataset(write_h5mu(), write_metadata(), local)
    record = previews.database_records(local)[0]
    return local, record


def test_generation_manifest_audit_and_existing_output_protection(
    settings,
    tmp_path,
    write_h5mu,
    write_metadata,
):
    local, record = _prepared(settings, tmp_path, write_h5mu, write_metadata)
    result = previews.generate_spatial_thumbnail(record, local)
    assert result["kind"] == "rna_signal"
    assert (
        previews.discover_spatial_thumbnails(local, [record])[record["dataset_id"]]["label"]
        == "RNA signal"
    )
    with np.load(result["audit"], allow_pickle=False) as audit:
        np.testing.assert_array_equal(audit["signal"], [1, 2])
        assert len(audit["obs_ids"]) == 2
    with pytest.raises(ThumbnailGenerationError, match="exists"):
        previews.generate_spatial_thumbnail(record, local)
    regenerated = previews.generate_spatial_thumbnail(record, local, force=True)
    assert result["image_sha256"] == regenerated["image_sha256"]


def test_publication_rolls_back_pair_when_manifest_switch_fails(tmp_path, monkeypatch):
    image = Image.new("RGB", (640, 320), "red")
    previews._publish(image, {"dataset_id": "test"}, tmp_path, force=False)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    original = previews.os.replace

    def fail_manifest(source, destination):
        if source.name == "test.json":
            raise OSError("injected switch failure")
        original(source, destination)

    monkeypatch.setattr(previews.os, "replace", fail_manifest)
    with pytest.raises(OSError, match="injected"):
        previews._publish(
            Image.new("RGB", (640, 320), "blue"), {"dataset_id": "test"}, tmp_path, force=True
        )
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    image.close()


@pytest.mark.parametrize(
    "damage", ["json", "missing", "image", "source", "method", "kind", "oversized"]
)
def test_discovery_fails_open_for_invalid_or_stale_previews(
    settings,
    tmp_path,
    write_h5mu,
    write_metadata,
    damage,
):
    local, record = _prepared(settings, tmp_path, write_h5mu, write_metadata)
    previews.generate_spatial_thumbnail(record, local)
    directory = local.static_dir / previews.DIRECTORY
    sidecar = directory / f"{record['dataset_id']}.json"
    if damage == "json":
        sidecar.write_text("{")
    elif damage == "missing":
        sidecar.unlink()
    elif damage == "image":
        (directory / f"{record['dataset_id']}.webp").write_bytes(b"broken")
    elif damage == "oversized":
        path = directory / f"{record['dataset_id']}.webp"
        Image.new("RGB", (641, 100), "red").save(path)
        manifest = json.loads(sidecar.read_text())
        manifest.update(dimensions=[641, 100], image_sha256=previews.sha256_file(path))
        sidecar.write_text(json.dumps(manifest))
    else:
        manifest = json.loads(sidecar.read_text())
        manifest[{"source": "source_sha256", "method": "method", "kind": "kind"}[damage]] = "bad"
        sidecar.write_text(json.dumps(manifest))
    assert previews.discover_spatial_thumbnails(local, [record]) == {}


def test_actual_source_checksum_mismatch_is_rejected(
    settings,
    tmp_path,
    write_h5mu,
    write_metadata,
):
    local, record = _prepared(settings, tmp_path, write_h5mu, write_metadata)
    record["sha256"] = "0" * 64
    with pytest.raises(ThumbnailGenerationError, match="checksum"):
        previews.generate_spatial_thumbnail(record, local)


def test_cli_batch_is_idempotent_and_single_existing_fails(
    settings,
    tmp_path,
    write_h5mu,
    write_metadata,
    monkeypatch,
    capsys,
):
    local, record = _prepared(settings, tmp_path, write_h5mu, write_metadata)
    monkeypatch.setattr("iscdc.cli.Settings.from_environment", lambda: local)
    assert main(["generate-spatial-thumbnails", "--all"]) == 0
    assert len(json.loads(capsys.readouterr().out)["generated"]) == 1
    assert main(["generate-spatial-thumbnails", "--all"]) == 0
    assert json.loads(capsys.readouterr().out)["skipped"][0]["reason"] == "existing_spatial_preview"
    assert main(["generate-spatial-thumbnails", record["dataset_id"]]) == 1
    assert json.loads(capsys.readouterr().out)["failures"]
    assert main(["generate-spatial-thumbnails", "missing"]) == 1


@pytest.mark.parametrize("args", [[], ["one", "--all"]])
def test_cli_requires_exactly_one_scope(args):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["generate-spatial-thumbnails", *args])


@pytest.mark.anyio
async def test_pages_label_spatial_previews_and_load_only_at_startup(
    settings,
    tmp_path,
    write_h5mu,
    write_metadata,
):
    local, record = _prepared(settings, tmp_path, write_h5mu, write_metadata)
    before = create_app(local)
    previews.generate_spatial_thumbnail(record, local)
    assert not before.state.templates.env.globals["spatial_thumbnail_info"]
    app = create_app(local)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for route in [
            "/databases",
            "/databases?view=datasets",
            "/databases/entries/TEST001",
            "/databases/test_rna_protein",
        ]:
            response = await client.get(route)
            assert response.status_code == 200, route
            assert "RNA signal" in response.text
        detail = await client.get("/databases/test_rna_protein")
        assert "not a histology image" in detail.text
        assert "Relative signal color scale" in detail.text
        api = await client.get("/api/databases/test_rna_protein")
        assert "rna_signal" not in api.text
        image = await client.get("/static/database_thumbnails/spatial/test_rna_protein.webp")
        assert image.status_code == 200 and image.headers["content-type"] == "image/webp"
    native = local.static_dir / "database_thumbnails/test_rna_protein.webp"
    Image.new("RGB", (10, 10), "green").save(native)
    prioritized = create_app(local)
    assert prioritized.state.templates.env.globals["spatial_thumbnail_info"] == {}
    assert prioritized.state.templates.env.globals["database_thumbnail_paths"][
        record["dataset_id"]
    ] == ("database_thumbnails/test_rna_protein.webp")


def test_registered_wsi_never_uses_spatial_fallback(
    settings,
    tmp_path,
    write_h5mu,
    write_metadata,
):
    local, record = _prepared(settings, tmp_path, write_h5mu, write_metadata)
    previews.generate_spatial_thumbnail(record, local)
    source = tmp_path / "wsi.tif"
    Image.new("RGB", (20, 20), "red").save(source)
    register_auxiliary_file(
        record["dataset_id"],
        source,
        local,
        auxiliary_id="he_wsi",
        label="H&E",
        source_url="https://example.org/wsi.tif",
        media_type="image/tiff",
    )
    with pytest.raises(ThumbnailGenerationError, match="registered_he_wsi"):
        previews.generate_spatial_thumbnail(record, local, force=True)
    assert create_app(local).state.templates.env.globals["spatial_thumbnail_info"] == {}


def test_batch_failure_preserves_successful_previews(
    settings,
    tmp_path,
    write_h5mu,
    write_metadata,
    monkeypatch,
    capsys,
):
    local, record = _prepared(settings, tmp_path, write_h5mu, write_metadata)
    monkeypatch.setattr("iscdc.cli.Settings.from_environment", lambda: local)
    monkeypatch.setattr(
        previews,
        "database_records",
        lambda _: [
            record,
            {
                **record,
                "dataset_id": "missing",
                "storage_dir": "missing",
            },
        ],
    )
    assert main(["generate-spatial-thumbnails", "--all"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert len(result["generated"]) == len(result["failures"]) == 1
    assert previews.discover_spatial_thumbnails(local, [record])


def test_publication_rejects_symlink_and_unsafe_id(tmp_path):
    target = tmp_path / "real.webp"
    target.write_bytes(b"protected")
    (tmp_path / "test.webp").symlink_to(target)
    image = Image.new("RGB", (640, 320), "red")
    for dataset_id in ["test", "../escape"]:
        with pytest.raises(ThumbnailGenerationError):
            previews._publish(image, {"dataset_id": dataset_id}, tmp_path, force=True)
    assert target.read_bytes() == b"protected"
    image.close()


@pytest.mark.anyio
async def test_non_rna_local_field_is_labelled_without_expression_claims(
    settings,
    tmp_path,
    write_h5mu,
    write_metadata,
    metadata_values,
):
    local = replace(settings, static_dir=tmp_path / "static", analytics_enabled=False)
    local.static_dir.mkdir()
    source = md.read_h5mu(write_h5mu())
    mdata = md.MuData({"metabolite": source.mod["rna"], "protein": source.mod["protein"]})
    mdata.obs = source.obs.copy()
    mdata.obsm["spatial"] = source.obsm["spatial"].copy()
    mdata.uns["database"] = dict(source.uns["database"], entry_id="S051")
    metadata_values["database"]["entry_id"] = "S051"
    metadata_values["modalities"]["metabolite"] = metadata_values["modalities"].pop("rna")
    path = tmp_path / "non_rna.h5mu"
    mdata.write_h5mu(path)
    import_dataset(path, write_metadata(metadata_values), local)
    record = previews.database_records(local)[0]
    result = previews.generate_spatial_thumbnail(record, local)
    assert result["kind"] == "point_density" and result["local_field"] is True
    app = create_app(local)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/databases/test_rna_protein")
    assert response.status_code == 200
    assert "Spatial point density · Local field" in response.text
    assert "Colors show relative point density" in response.text
    assert "RNA total counts" not in response.text
