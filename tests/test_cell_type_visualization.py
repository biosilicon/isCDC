from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from iscdc.cell_type_visualization import (
    POINT_HEADER,
    POINT_HEADER_SIZE,
    POINT_MAGIC,
    CellTypeVisualizationError,
    build_point_representations,
    decode_points,
    encode_points,
    load_cell_type_visualization,
    load_cell_type_visualizations,
    publish_failure,
    publish_generation,
    validate_point_header,
)

SOURCE_SHA = "a" * 64
OBS_ORDER_SHA = "b" * 64


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file(path: str, payload: bytes) -> dict[str, object]:
    return {"path": path, "size": len(payload), "sha256": _sha(payload)}


def _representation(path: str, encoding: str, payload: bytes, identity: bytes) -> dict:
    return {
        **_file(path, payload),
        "encoding": encoding,
        "content_size": len(identity),
        "content_sha256": _sha(identity),
    }


def _dataset(dataset_id: str = "example") -> SimpleNamespace:
    return SimpleNamespace(
        dataset_id=dataset_id,
        dataset_type="full",
        sha256=SOURCE_SHA,
        obs_order_sha256=OBS_ORDER_SHA,
        n_obs=3,
        coordinate_dimensions=2,
        sample_ids=["section A"],
    )


def _generation(
    dataset_id: str = "example",
    generation_id: str = "20260818T120000-0123456789",
    *,
    inferred: bool = True,
) -> tuple[dict, dict[str, bytes]]:
    confidence = [0.9, 0.4, 0.8] if inferred else None
    identity = encode_points([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0, 1, 0], confidence)
    report_document = {
        "report_version": 1,
        "dataset_id": dataset_id,
        "generation_id": generation_id,
        "source_sha256": SOURCE_SHA,
        "status": "passed",
        "quality_control": {"aligned": True},
        "thresholds": {"minimum_score": 0.5},
        "warnings": [],
    }
    report = json.dumps(report_document, sort_keys=True).encode() + b"\n"
    files = {"report.json": report}
    representations = {}
    suffixes = {"identity": ".bin", "gzip": ".bin.gz", "br": ".bin.br"}
    for encoding, representation_payload in build_point_representations(identity).items():
        name = f"samples/section{suffixes[encoding]}"
        files[name] = representation_payload
        representations[encoding] = _representation(
            name, encoding, representation_payload, identity
        )
    categories = [
        {
            "type_id": 0,
            "label": "T cell",
            "color": "#112233",
            "count": 2,
            "state": "biological",
        },
        {
            "type_id": 1,
            "label": "Uncertain" if inferred else "Unclassified by source",
            "color": "#AABBCC",
            "count": 1,
            "state": "uncertain" if inferred else "biological",
        },
    ]
    if inferred:
        categories[0]["cell_ontology_id"] = "CL:0000084"
        inference = b"\x89HDF\r\n\x1a\nopaque inference output"
        files["inference.h5"] = inference
    manifest = {
        "manifest_version": 1,
        "dataset_id": dataset_id,
        "generation_id": generation_id,
        "generated_at": "2026-08-18T12:00:00+00:00",
        "source": {
            "sha256": SOURCE_SHA,
            "obs_order_sha256": OBS_ORDER_SHA,
            "observation_count": 3,
            "coordinate_dimensions": 2,
            "sample_ids": ["section A"],
        },
        "annotation": {
            "kind": "inferred" if inferred else "source",
            "method": "SingleR" if inferred else "mdata.obs[cell_type]",
        },
        "coordinates": {"system": "global", "unit": "micrometre", "y_axis": "up"},
        "categories": categories,
        "samples": [
            {
                "key": "section",
                "id": "section A",
                "count": 3,
                "bounds": [1.0, 4.0, 3.0, 6.0],
                "category_counts": {"0": 2, "1": 1},
                "representations": representations,
            }
        ],
        "report": _file("report.json", report),
        "provenance": {
            "environment_lock_sha256": "c" * 64,
            "references": [{"id": "reference", "version": "1", "sha256": "d" * 64}],
            "parameters": {"seed": 7},
        },
    }
    if inferred:
        manifest["inference"] = _file("inference.h5", files["inference.h5"])
    return manifest, files


def _rewrite_status_manifest_digest(root: Path, dataset_id: str, generation_id: str) -> None:
    manifest_path = root / dataset_id / "generations" / generation_id / "manifest.json"
    status_path = root / dataset_id / "status.json"
    status = json.loads(status_path.read_text())
    status["manifest_sha256"] = _sha(manifest_path.read_bytes())
    status_path.write_text(json.dumps(status), encoding="utf-8")


def test_binary_v1_golden_vectors_are_little_endian_soa() -> None:
    without_confidence = encode_points([1.5], [-2.0], [513])
    assert without_confidence.hex() == (
        "4953434443435400"  # ISCDCCT\0
        "0100"  # version
        "0000"  # flags
        "01000000"  # count
        "2000000000000000"  # payload offset
        "2a00000000000000"  # total size
        "0000c03f"  # x Float32
        "000000c0"  # y Float32
        "0102"  # type Uint16
    )
    with_confidence = encode_points([1.5], [-2.0], [513], [0.25])
    assert with_confidence.hex() == (
        "4953434443435400"
        "0100"
        "0100"
        "01000000"
        "2000000000000000"
        "2e00000000000000"
        "0000c03f"
        "000000c0"
        "0000803e"
        "0102"
    )
    assert decode_points(with_confidence).confidence == pytest.approx((0.25,))


@pytest.mark.parametrize(
    "offset,value",
    [
        (0, b"BADMAGIC"),
        (8, struct.pack("<H", 2)),
        (10, struct.pack("<H", 0x8000)),
        (16, struct.pack("<Q", 31)),
        (24, struct.pack("<Q", 999)),
    ],
)
def test_binary_header_rejects_every_noncanonical_field(offset: int, value: bytes) -> None:
    payload = bytearray(encode_points([1], [2], [0]))
    payload[offset : offset + len(value)] = value
    with pytest.raises(CellTypeVisualizationError):
        validate_point_header(payload)


def test_binary_decode_rejects_nonfinite_and_bad_array_inputs() -> None:
    payload = bytearray(encode_points([1], [2], [0]))
    payload[POINT_HEADER_SIZE : POINT_HEADER_SIZE + 4] = struct.pack("<f", float("nan"))
    with pytest.raises(CellTypeVisualizationError, match="finite"):
        decode_points(payload)
    with pytest.raises(CellTypeVisualizationError, match="same number"):
        encode_points([1], [], [0])
    with pytest.raises(CellTypeVisualizationError, match="between 0 and 65535"):
        encode_points([1], [2], [65536])
    with pytest.raises(CellTypeVisualizationError, match="between 0 and 1"):
        encode_points([1], [2], [0], [1.1])
    with pytest.raises(CellTypeVisualizationError, match="Float32 range"):
        encode_points([10**1000], [2], [0])


def test_canonical_transport_representations_are_deterministic() -> None:
    payload = encode_points([1], [2], [0])
    first = build_point_representations(payload)
    second = build_point_representations(payload)
    assert first == second
    assert first["identity"] is payload
    assert first["gzip"][0:3] == b"\x1f\x8b\x08"
    assert first["br"] != payload


def test_publish_and_load_inferred_generation_with_resolved_encodings(tmp_path: Path) -> None:
    manifest, files = _generation()
    published = publish_generation(tmp_path, manifest, files)
    loaded = load_cell_type_visualization(tmp_path, _dataset())

    assert published.generation_id == loaded.generation_id == manifest["generation_id"]
    assert loaded.coordinate_system == "global"
    assert loaded.y_axis == "up"
    assert loaded.has_confidence
    assert [category.cell_ontology_id for category in loaded.categories] == [
        "CL:0000084",
        None,
    ]
    sample = loaded.samples["section"]
    assert (sample.key, sample.id, sample.count) == ("section", "section A", 3)
    assert set(sample.representations) == {"identity", "gzip", "br"}
    assert sample.resolve("gzip").path.name == "section.bin.gz"
    assert sample.resolve("br").path.name == "section.bin.br"
    with pytest.raises(CellTypeVisualizationError, match="no 'zstd'"):
        sample.resolve("zstd")

    status = json.loads((tmp_path / "example" / "status.json").read_text())
    manifest_bytes = (
        tmp_path / "example" / "generations" / manifest["generation_id"] / "manifest.json"
    ).read_bytes()
    assert status["manifest_sha256"] == _sha(manifest_bytes)


def test_source_labels_may_omit_cl_ids_and_have_no_confidence(tmp_path: Path) -> None:
    manifest, files = _generation(inferred=False)
    publish_generation(tmp_path, manifest, files)
    loaded = load_cell_type_visualization(tmp_path, _dataset())

    assert loaded.annotation_kind == "source"
    assert not loaded.has_confidence
    assert loaded.inference_path is None
    assert all(category.cell_ontology_id is None for category in loaded.categories)


@pytest.mark.parametrize("label", ["Mixed", "Uncertain"])
def test_inferred_prediction_states_must_not_have_cl_ids(tmp_path: Path, label: str) -> None:
    manifest, files = _generation()
    category = manifest["categories"][1]
    category["label"] = label
    category["state"] = label.lower()
    category["cell_ontology_id"] = "CL:0000000"
    with pytest.raises(CellTypeVisualizationError, match="must not have"):
        publish_generation(tmp_path, manifest, files)


def test_inferred_biological_types_require_stable_cl_ids(tmp_path: Path) -> None:
    manifest, files = _generation()
    del manifest["categories"][0]["cell_ontology_id"]
    with pytest.raises(CellTypeVisualizationError, match="require stable"):
        publish_generation(tmp_path, manifest, files)


def test_failure_status_hides_an_older_success(tmp_path: Path) -> None:
    manifest, files = _generation()
    publish_generation(tmp_path, manifest, files)
    publish_failure(
        tmp_path,
        "example",
        "quality gate failed",
        stage="validation",
        category="insufficient_marker_coverage",
        details={"observed": 0.2, "minimum": 0.5},
        failure_id="failed-run",
    )

    with pytest.raises(CellTypeVisualizationError, match="latest.*failed"):
        load_cell_type_visualization(tmp_path, _dataset())
    assert load_cell_type_visualizations(tmp_path, [_dataset()]) == {}
    assert (tmp_path / "example" / "generations" / manifest["generation_id"]).is_dir()
    status = json.loads((tmp_path / "example" / "status.json").read_text())
    assert "generation_id" not in status
    assert status["failure_id"] == "failed-run"
    report_path = tmp_path / "example" / "failures" / "failed-run" / "report.json"
    report = json.loads(report_path.read_text())
    assert report["details"] == {"observed": 0.2, "minimum": 0.5}
    assert status["failure_report_sha256"] == _sha(report_path.read_bytes())


def test_failure_status_rejects_a_tampered_audit_report(tmp_path: Path) -> None:
    publish_failure(tmp_path, "example", "failed", failure_id="failed-run")
    report_path = tmp_path / "example" / "failures" / "failed-run" / "report.json"
    report_path.write_bytes(report_path.read_bytes() + b" ")
    with pytest.raises(CellTypeVisualizationError, match="failure report SHA"):
        load_cell_type_visualization(tmp_path, _dataset())


def test_success_status_is_bound_to_exact_manifest_bytes(tmp_path: Path) -> None:
    manifest, files = _generation()
    publish_generation(tmp_path, manifest, files)
    manifest_path = (
        tmp_path / "example" / "generations" / manifest["generation_id"] / "manifest.json"
    )
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    with pytest.raises(CellTypeVisualizationError, match="binding"):
        load_cell_type_visualization(tmp_path, _dataset())


def test_loader_rejects_stale_source_and_tampered_encoded_file(tmp_path: Path) -> None:
    manifest, files = _generation()
    publish_generation(tmp_path, manifest, files)
    stale = _dataset()
    stale.sha256 = "b" * 64
    with pytest.raises(CellTypeVisualizationError, match="stale"):
        load_cell_type_visualization(tmp_path, stale)

    compressed_path = (
        tmp_path
        / "example"
        / "generations"
        / manifest["generation_id"]
        / "samples"
        / "section.bin.gz"
    )
    compressed_path.write_bytes(compressed_path.read_bytes() + b"corrupt")
    with pytest.raises(CellTypeVisualizationError, match="size or SHA"):
        load_cell_type_visualization(tmp_path, _dataset())


def test_contract_requires_order_digest_all_encodings_and_strict_reports(
    tmp_path: Path,
) -> None:
    manifest, files = _generation()
    del manifest["source"]["obs_order_sha256"]
    with pytest.raises(CellTypeVisualizationError, match="obs_order_sha256"):
        publish_generation(tmp_path, manifest, files)

    manifest, files = _generation(generation_id="without-br")
    del manifest["samples"][0]["representations"]["br"]
    del files["samples/section.bin.br"]
    with pytest.raises(CellTypeVisualizationError, match="identity, gzip, and br"):
        publish_generation(tmp_path, manifest, files)

    manifest, files = _generation(generation_id="bad-report")
    report = json.loads(files["report.json"])
    del report["thresholds"]
    report_payload = json.dumps(report).encode()
    files["report.json"] = report_payload
    manifest["report"] = _file("report.json", report_payload)
    with pytest.raises(CellTypeVisualizationError, match="thresholds"):
        publish_generation(tmp_path, manifest, files)

    manifest, files = _generation(generation_id="bad-provenance")
    del manifest["provenance"]["environment_lock_sha256"]
    with pytest.raises(CellTypeVisualizationError, match="environment_lock_sha256"):
        publish_generation(tmp_path, manifest, files)


def test_loader_rejects_unsafe_paths_even_with_rebound_manifest(tmp_path: Path) -> None:
    manifest, files = _generation()
    publish_generation(tmp_path, manifest, files)
    generation_id = manifest["generation_id"]
    manifest_path = tmp_path / "example" / "generations" / generation_id / "manifest.json"
    document = json.loads(manifest_path.read_text())
    document["samples"][0]["representations"]["identity"]["path"] = "../report.json"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    _rewrite_status_manifest_digest(tmp_path, "example", generation_id)
    with pytest.raises(CellTypeVisualizationError, match="safe relative path"):
        load_cell_type_visualization(tmp_path, _dataset())


def test_index_fails_open_per_dataset_and_accepts_mapping_records(tmp_path: Path) -> None:
    first_manifest, first_files = _generation("first", "g-first")
    publish_generation(tmp_path, first_manifest, first_files)
    second_manifest, second_files = _generation("second", "g-second")
    publish_generation(tmp_path, second_manifest, second_files)
    (tmp_path / "second" / "status.json").write_text("not json", encoding="utf-8")

    records = {
        "first": {
            "dataset_type": "full",
            "sha256": SOURCE_SHA,
            "obs_order_sha256": OBS_ORDER_SHA,
            "n_obs": 3,
            "coordinate_dimensions": 2,
            "sample_ids": ["section A"],
        },
        "second": {
            "dataset_type": "full",
            "sha256": SOURCE_SHA,
            "obs_order_sha256": OBS_ORDER_SHA,
            "n_obs": 3,
            "coordinate_dimensions": 2,
            "sample_ids": ["section A"],
        },
    }
    assert set(load_cell_type_visualizations(tmp_path, records)) == {"first"}


def test_generation_is_immutable_and_failed_publish_does_not_replace_status(
    tmp_path: Path,
) -> None:
    manifest, files = _generation()
    publish_generation(tmp_path, manifest, files)
    old_status = (tmp_path / "example" / "status.json").read_bytes()
    with pytest.raises(CellTypeVisualizationError, match="immutable"):
        publish_generation(tmp_path, manifest, files)
    assert (tmp_path / "example" / "status.json").read_bytes() == old_status


def test_header_layout_is_exactly_the_documented_32_bytes() -> None:
    assert POINT_HEADER.format == "<8sHHIQQ"
    assert POINT_HEADER.size == 32
    assert POINT_MAGIC == b"ISCDCCT\0"
