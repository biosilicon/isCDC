from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from dataclasses import replace

import httpx
import pytest
from test_spatial_domain_visualization import generation
from test_spatialglue import combination_generation, glue_generation

from iscdc.spatial_domain_color_prepare import (
    build_domain_color_snapshot,
    match_domain_colors,
    write_domain_color_snapshot,
)
from iscdc.spatial_domain_colors import SNAPSHOT_NAME, load_domain_color_overrides
from iscdc.spatial_domain_visualization import (
    DomainVisualization,
    assignment_bytes,
    build_point_representations,
    domain_directory,
    encode_points,
    file_record,
    obs_order_sha256,
    publish_generation,
)


def categories(colors):
    return [{"code": code, "color": color} for code, color in colors.items()]


def test_global_assignment_maximizes_unchanged_cells_not_greedy_pairs():
    rna = categories({1: "#112233", 2: "#445566"})
    glue = categories({1: "#112233", 2: "#445566"})
    # Greedy would keep 5 cells; the crossed assignment keeps 4 + 4.
    colors = match_domain_colors(rna, glue, {(1, 1): 5, (1, 2): 4, (2, 1): 4})
    assert colors == {1: "#445566", 2: "#112233"}


@pytest.mark.parametrize("rna_count,glue_count", [(2, 3), (3, 2), (0, 2), (2, 0)])
def test_unmatched_colors_are_distinct_and_reserve_all_rna_and_zero_colors(rna_count, glue_count):
    rna = categories({0: "#8A929B", **{i: f"#{i:06x}" for i in range(1, rna_count + 1)}})
    glue = categories({0: "#889999", **{i: "#000001" for i in range(1, glue_count + 1)}})
    # Excluded observations cannot force an assignment, even with huge overlap.
    overlaps = {(1, 1): 2, (0, 2): 1000, (2, 0): 1000, (2, 2): 0}
    colors = match_domain_colors(rna, glue, overlaps)
    assert 0 not in colors
    assert len(colors) == len(set(colors.values())) == glue_count
    matched = rna_count > 0 and glue_count > 0
    if matched:
        assert colors[1] == "#000001"
    reserved = {c["color"].upper() for c in rna} | {"#889999"}
    for code, color in colors.items():
        if matched and code == 1:
            continue
        assert color.upper() not in reserved


def test_ties_and_category_order_are_stable():
    rna = categories({9: "#112233", 2: "#445566"})
    glue = categories({7: "#223344", 4: "#667788", 12: "#889900"})
    overlaps = {(r, g): 3 for r in (9, 2) for g in (7, 4, 12)}
    expected = match_domain_colors(rna, glue, overlaps)
    assert match_domain_colors(rna[::-1], glue[::-1], dict(reversed(overlaps.items()))) == expected
    assert len(set(expected.values())) == 3
    assert {"#112233", "#445566"} <= set(expected.values())


def snapshot(root, family, rows, sample_colors, *, combination="rna__protein", version=3):
    """Small assignment-only snapshots for testing the post-validation alignment layer."""
    manifest_version = 1 if family == "rna" else version
    provenance = {"combination_id": combination} if manifest_version == 3 else {}
    ids, samples, codes = zip(*rows)
    payload = assignment_bytes(ids, samples, codes, ["" if c else "zero_counts" for c in codes])
    manifest = {
        "manifest_version": manifest_version,
        "provenance": provenance,
        "source": {
            "sha256": "a" * 64,
            "obs_order_sha256": obs_order_sha256(ids),
            "observation_count": len(rows),
        },
        "assignments": file_record("assignments.tsv.gz", payload),
        "samples": [
            {
                "id": sample_id,
                "key": f"{family}_{index}",
                "count": samples.count(sample_id),
                "categories": categories(colors),
            }
            for index, (sample_id, colors) in enumerate(sample_colors.items())
        ],
    }
    directory = (
        domain_directory(
            root, "example", family, combination_id=combination if manifest_version == 3 else None
        )
        / "generations"
        / "run-1"
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "assignments.tsv.gz").write_bytes(payload)
    return DomainVisualization("example", "run-1", manifest, {}, {})


def test_samples_and_combinations_align_independently_and_ignore_local_sample_keys(tmp_path):
    palette = {"A": {1: "#112233", 2: "#445566"}, "B": {1: "#778899", 2: "#AABBCC"}}
    rows = [("a", "A", 1), ("b", "B", 1), ("c", "A", 2), ("d", "B", 2)]
    rna = snapshot(tmp_path, "rna", rows, palette)
    combinations = {}
    for index, combination in enumerate(
        (
            "rna__protein__atac",
            "rna__protein__histone",
            "rna__atac__histone",
            "protein__atac__histone",
        )
    ):
        glue_rows = [
            (obs, sample, 3 - code if (sample == "A") == (index % 2 == 0) else code)
            for obs, sample, code in rows
        ]
        combinations[combination] = snapshot(
            tmp_path,
            "spatialglue",
            glue_rows,
            dict(reversed(palette.items())),
            combination=combination,
        )
    path = tmp_path / SNAPSHOT_NAME
    document = build_domain_color_snapshot(tmp_path, {"example": rna}, {"example": combinations})
    write_domain_color_snapshot(path, document)
    result = load_domain_color_overrides(path, {"example": rna}, {"example": combinations})
    for index, combination in enumerate(combinations):
        colors = result["example"][combination]
        assert colors["A"][1] == ("#445566" if index % 2 == 0 else "#112233")
        assert colors["B"][1] == ("#778899" if index % 2 == 0 else "#AABBCC")


@pytest.mark.parametrize("failure", ["source", "ids", "sample", "length", "digest"])
def test_alignment_failure_preserves_previous_snapshot(tmp_path, failure):
    rows = [("a", "A", 1), ("b", "B", 2)]
    palette = {"A": {1: "#112233"}, "B": {2: "#445566"}}
    rna = snapshot(tmp_path, "rna", rows, palette)
    good = snapshot(tmp_path, "spatialglue", rows, palette)
    bad_rows = copy.deepcopy(rows)
    if failure == "ids":
        bad_rows = [("other", "A", 1), rows[1]]
    elif failure == "sample":
        bad_rows = [("a", "B", 1), ("b", "A", 2)]
    elif failure == "length":
        bad_rows = rows[:1]
    bad = snapshot(tmp_path, "spatialglue", bad_rows, palette, combination="rna__atac")
    bad.manifest["source"] = copy.deepcopy(rna.manifest["source"])
    if failure == "source":
        bad.manifest["source"]["sha256"] = "b" * 64
    elif failure == "digest":
        bad.manifest["assignments"]["sha256"] = "b" * 64
    document = build_domain_color_snapshot(
        tmp_path, {"example": rna}, {"example": {"rna__protein": good, "rna__atac": bad}}
    )
    assert [entry["combination_id"] for entry in document["entries"]] == ["rna__protein"]
    assert document["failures"][0]["combination_id"] == "rna__atac"
    path = tmp_path / SNAPSHOT_NAME
    path.write_bytes(b"previous snapshot")
    with pytest.raises(ValueError, match="keeping previous snapshot"):
        write_domain_color_snapshot(path, document)
    assert path.read_bytes() == b"previous snapshot"


def test_missing_rna_does_not_read_assignments(tmp_path):
    assert load_domain_color_overrides(tmp_path, {}, {"example": {"rna__protein": None}}) == {}


def relabel_generation(manifest, files, codes, colors):
    """Replace labels consistently in a complete, single-sample published fixture."""
    sample = manifest["samples"][0]
    counts = Counter(codes)
    files["assignments.tsv.gz"] = assignment_bytes(
        [f"cell_{i + 1}" for i in range(len(codes))],
        [sample["id"]] * len(codes),
        codes,
        ["" if c else "zero_counts" for c in codes],
    )
    manifest["assignments"] = file_record("assignments.tsv.gz", files["assignments.tsv.gz"])
    sample["categories"] = [
        {
            "code": code,
            "label": f"Domain {code}" if code else "Not analyzed",
            "color": colors[code],
            "count": count,
        }
        for code, count in sorted(counts.items())
    ]
    payload = encode_points(range(len(codes)), range(len(codes)), codes)
    for encoding, content in build_point_representations(payload).items():
        representation = sample["representations"][encoding]
        path = representation["path"]
        files[path] = content
        representation.update(
            **file_record(path, content),
            content_size=len(payload),
            content_sha256=hashlib.sha256(payload).hexdigest(),
        )
    report = json.loads(files["report.json"])
    diagnostic = report["samples"][sample["id"]]
    diagnostic.update(
        n_domains=len(counts.keys() - {0}),
        analyzed=len(codes) - counts[0],
        not_analyzed=counts[0],
        domains={str(c): {"count": n, "spatial_components": 1} for c, n in counts.items() if c},
    )
    if "coverage" in diagnostic:
        diagnostic["coverage"]["zero_counts"] = counts[0]
    files["report.json"] = json.dumps(report).encode()
    manifest["report"] = file_record("report.json", files["report.json"])


@pytest.mark.anyio
@pytest.mark.parametrize("version", [2, 3])
async def test_page_colors_use_overlap_without_mutating_snapshots_or_points(
    settings, write_h5mu, write_metadata, metadata_values, monkeypatch, version
):
    import mudata

    from iscdc.app import create_app
    from iscdc.importer import import_dataset
    from iscdc.spatial_domain_annotation import catalogue_records

    path = write_h5mu()
    data = mudata.read_h5mu(path)
    data.mod["protein"].uns["assay"]["value_type"] = "counts"
    data.write_h5mu(path)
    metadata_values["modalities"]["protein"]["value_type"] = "counts"
    settings = replace(
        settings, spatial_domain_visualization_root=settings.data_root.parent / "domains"
    )
    import_dataset(path, write_metadata(), settings)
    source = catalogue_records(settings)[0]
    root = settings.spatial_domain_visualization_root
    _, rna_manifest, rna_files = generation(source)
    rna_palette = {1: "#112233", 2: "#445566"}
    relabel_generation(rna_manifest, rna_files, [1, 2], rna_palette)
    rna_snapshot = publish_generation(root, source, rna_manifest, rna_files)
    _, glue_manifest, glue_files = (
        glue_generation(source)
        if version == 2
        else combination_generation(source, ["rna", "protein"])
    )
    relabel_generation(glue_manifest, glue_files, [2, 1], rna_palette)
    glue_snapshot = publish_generation(
        root,
        source,
        glue_manifest,
        glue_files,
        method_family="spatialglue",
        combination_id="rna__protein" if version == 3 else None,
    )
    color_path = settings.database_path.parent / SNAPSHOT_NAME
    document = build_domain_color_snapshot(
        root,
        {source["dataset_id"]: rna_snapshot},
        {source["dataset_id"]: {"rna__protein": glue_snapshot}},
    )
    write_domain_color_snapshot(color_path, document)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    # Neither startup nor page requests may compute overlaps or reopen assignments for colors.
    monkeypatch.setattr(
        "iscdc.spatial_domain_color_prepare._snapshot_colors",
        lambda *args: pytest.fail("Website recomputed domain colors"),
    )
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for _ in range(2):
            page = await client.get(f"/databases/{source['dataset_id']}")
            assert page.status_code == 200
            config = json.loads(
                re.search(
                    r'<script id="cell-type-visualization-config"[^>]*>(.*?)</script>',
                    page.text,
                    re.S,
                )[1]
            )
            rna, glue = config["views"]
            assert rna["samples"][0]["categories"] == rna_manifest["samples"][0]["categories"]
            expected = [
                {**c, "color": rna_palette[3 - c["code"]]}
                for c in glue_manifest["samples"][0]["categories"]
            ]
            assert glue["samples"][0]["categories"] == glue["categories"] == expected
            response = await client.get(
                glue["samples"][0]["url"], headers={"Accept-Encoding": "identity"}
            )
            identity = glue_manifest["samples"][0]["representations"]["identity"]["path"]
            assert response.content == glue_files[identity]
        public = await client.get(f"/api/databases/{source['dataset_id']}")
        assert "SpatialGlue" not in public.text
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    # Missing or unreadable mappings fall back without computation.
    rna_views = {source["dataset_id"]: rna_snapshot}
    glue_views = {source["dataset_id"]: {"rna__protein": glue_snapshot}}
    color_path.unlink()
    assert load_domain_color_overrides(color_path, rna_views, glue_views) == {}
    color_path.write_text("broken snapshot")
    assert load_domain_color_overrides(color_path, rna_views, glue_views) == {}
