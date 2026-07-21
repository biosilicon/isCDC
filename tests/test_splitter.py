from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
import yaml

from iscdc.splitter import (
    FEATURE_MASK_KEY,
    SplitterError,
    compose_split,
    coordinate_ranges,
    load_compose_config,
    main,
    spatial_split,
)


def _pairing_type(modality_obs: dict[str, list[str]]) -> str:
    values = [set(names) for names in modality_obs.values()]
    if all(value == values[0] for value in values[1:]):
        return "same_unit"
    if any(
        left.intersection(right)
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    ):
        return "partially_shared"
    return "unpaired"


def _write_full(
    directory: Path,
    dataset_id: str,
    *,
    obs_names: list[str] | None = None,
    samples: list[str] | None = None,
    coordinates: np.ndarray | None = None,
    features: dict[str, list[str]] | None = None,
    modality_obs: dict[str, list[str]] | None = None,
    matrices: dict[str, np.ndarray] | None = None,
    schema_version: str = "1.1",
    spatial_unit: str = "cell",
    coordinate_unit: str = "micrometer",
    value_types: dict[str, str] | None = None,
    technologies: dict[str, str] | None = None,
) -> Path:
    obs_names = obs_names or ["cell_1", "cell_2"]
    samples = samples or ["sample_1"] * len(obs_names)
    if coordinates is None:
        coordinates = np.arange(len(obs_names) * 2, dtype=np.float32).reshape(-1, 2)
    features = features or {"rna": ["g1", "g2"], "protein": ["p1"]}
    modality_obs = modality_obs or {name: list(obs_names) for name in features}
    value_types = value_types or {"rna": "counts", "protein": "intensity", "atac": "counts"}
    technologies = technologies or {name: "test assay" for name in features}

    global_positions = {name: index for index, name in enumerate(obs_names)}
    modalities: dict[str, ad.AnnData] = {}
    for modality_index, (modality, modality_features) in enumerate(features.items()):
        names = modality_obs[modality]
        if matrices is None or modality not in matrices:
            source_rows = np.asarray([global_positions[name] for name in names])[:, None]
            feature_columns = np.arange(len(modality_features))[None, :]
            matrix = (100 * modality_index + 10 * source_rows + feature_columns + 1).astype(
                np.int32
            )
        else:
            matrix = np.asarray(matrices[modality])
        adata = ad.AnnData(
            X=matrix,
            obs=pd.DataFrame(index=pd.Index(names, dtype=str)),
            var=pd.DataFrame(index=pd.Index(modality_features, dtype=str)),
        )
        adata.uns["assay"] = {
            "technology": technologies.get(modality, "test assay"),
            "value_type": value_types.get(modality, "counts"),
        }
        modalities[modality] = adata

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
        warnings.filterwarnings("ignore", message="Cannot join columns with the same name.*")
        mdata = md.MuData(modalities)
        mdata = mdata[obs_names, :].copy()
    mdata.obs["sample_id"] = samples
    mdata.obsm["spatial"] = np.asarray(coordinates)
    mdata.uns["database"] = {
        "schema_version": schema_version,
        "dataset_id": dataset_id,
        "dataset_type": "full",
        "source": f"SOURCE-{dataset_id}",
        "organism": "Homo sapiens",
        "tissue": "kidney",
        "spatial_unit": spatial_unit,
        "coordinate_unit": coordinate_unit,
        "pairing_type": _pairing_type(modality_obs),
    }
    path = directory / f"{dataset_id}.h5mu"
    mdata.write_h5mu(path)
    return path


def _write_yaml(path: Path, values: dict) -> Path:
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _spatial_config(
    directory: Path,
    source: Path,
    regions: list[dict],
    *,
    output_name: str = "spatial_output",
) -> Path:
    return _write_yaml(
        directory / f"{output_name}.yaml",
        {
            "schema_version": "1.1",
            "split_id": f"{output_name}_split",
            "feature_merge_policy": "preserve",
            "source": source.name,
            "output_dir": output_name,
            "train": {"dataset_id": f"{output_name}_train"},
            "test": {"dataset_id": f"{output_name}_test", "regions": regions},
        },
    )


def _compose_config(
    directory: Path,
    policy: str,
    train_sources: list[Path],
    test_sources: list[Path],
    *,
    train_reference: str | None = None,
    test_reference: str | None = None,
    output_name: str | None = None,
) -> Path:
    output_name = output_name or f"compose_{policy}"
    return _write_yaml(
        directory / f"{output_name}.yaml",
        {
            "schema_version": "1.1",
            "split_id": f"{output_name}_split",
            "feature_merge_policy": policy,
            "output_dir": output_name,
            "train": {
                "dataset_id": f"{output_name}_train",
                "sources": [path.name for path in train_sources],
                "reference_dataset_id": train_reference,
            },
            "test": {
                "dataset_id": f"{output_name}_test",
                "sources": [path.name for path in test_sources],
                "reference_dataset_id": test_reference,
            },
        },
    )


def _read(path: Path) -> md.MuData:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
        return md.read_h5mu(path)


def test_coordinate_ranges_reports_global_samples_json_and_3d(tmp_path, capsys):
    source = _write_full(
        tmp_path,
        "range_source",
        obs_names=["a", "b", "c", "d"],
        samples=["s1", "s1", "s2", "s2"],
        coordinates=np.asarray(
            [[-1, 5, 100], [2, 8, 101], [10, -3, 102], [12, 4, 103]], dtype=np.float32
        ),
    )

    result = coordinate_ranges(source)

    assert result["coordinate_dimensions"] == 3
    assert result["global"] == {
        "n_obs": 4,
        "x_min": -1.0,
        "x_max": 12.0,
        "y_min": -3.0,
        "y_max": 8.0,
    }
    assert result["samples"]["s1"]["x_max"] == 2.0
    assert coordinate_ranges(source, "s2")["samples"].keys() == {"s2"}

    assert main(["range", str(source), "--sample-id", "s1", "--json"]) == 0
    cli_result = json.loads(capsys.readouterr().out)
    assert cli_result["samples"] == {"s1": result["samples"]["s1"]}


@pytest.mark.parametrize(
    ("coordinates", "message"),
    [
        (np.asarray([[0, 1], [np.nan, 2]], dtype=np.float32), "finite numeric"),
        (np.zeros((2, 4), dtype=np.float32), "two or three columns"),
    ],
)
def test_coordinate_ranges_rejects_invalid_coordinates(tmp_path, coordinates, message):
    source = _write_full(tmp_path, "bad_coordinates", coordinates=coordinates)

    with pytest.raises(SplitterError, match=message):
        coordinate_ranges(source)


def test_range_rejects_non_v11_and_unknown_sample(tmp_path):
    legacy = _write_full(tmp_path, "legacy", schema_version="1.0")
    current = _write_full(tmp_path, "current")

    with pytest.raises(SplitterError, match="schema_version must be '1.1'"):
        coordinate_ranges(legacy)
    with pytest.raises(SplitterError, match="does not exist"):
        coordinate_ranges(current, "missing")


def test_spatial_uses_closed_region_union_and_preserves_source_data(tmp_path):
    obs_names = ["c1", "c2", "c3", "c4", "c5", "c6"]
    coordinates = np.asarray(
        [[0, 0], [1, 1], [2, 2], [3, 3], [0, 0], [1, 1]], dtype=np.float32
    )
    source = _write_full(
        tmp_path,
        "spatial_full",
        obs_names=obs_names,
        samples=["s1", "s1", "s1", "s1", "s2", "s2"],
        coordinates=coordinates,
        modality_obs={"rna": obs_names, "protein": ["c2", "c3", "c4", "c5"]},
    )
    config = _spatial_config(
        tmp_path,
        source,
        [
            {"sample_id": "s1", "x_min": 1, "x_max": 1, "y_min": 1, "y_max": 1},
            {"sample_id": "s1", "x_min": 3, "x_max": 3, "y_min": 3, "y_max": 3},
        ],
    )

    train_path, test_path = spatial_split(config)
    train = _read(train_path)
    test = _read(test_path)
    source_data = _read(source)
    try:
        assert list(test.obs_names) == ["c2", "c4"]
        assert list(train.obs_names) == ["c1", "c3", "c5", "c6"]
        assert list(test.mod["protein"].obs_names) == ["c2", "c4"]
        assert list(train.mod["protein"].obs_names) == ["c3", "c5"]
        assert list(train.obs["sample_id"].astype(str))[-2:] == ["s2", "s2"]
        assert list(test.obs["source_dataset_id"].astype(str)) == [
            "spatial_full",
            "spatial_full",
        ]
        assert list(test.obs["source_obs_id"].astype(str)) == ["c2", "c4"]
        np.testing.assert_array_equal(test.obsm["spatial"], coordinates[[1, 3]])
        np.testing.assert_array_equal(
            test.mod["rna"].X,
            source_data.mod["rna"][["c2", "c4"], :].X,
        )
        assert list(test.mod["rna"].var_names) == ["g1", "g2"]
        assert set(test.obs.columns) == {"sample_id", "source_dataset_id", "source_obs_id"}
        database = test.uns["database"]
        assert database["schema_version"] == "1.1"
        assert database["dataset_type"] == "test"
        assert database["derivation"]["construction_type"] == "subset"
        assert database["derivation"]["feature_merge_policy"] == "preserve"
        assert database["derivation"]["random_seed"] is None
    finally:
        train.file.close()
        test.file.close()
        source_data.file.close()


@pytest.mark.parametrize(
    "region",
        [
            {
                "sample_id": "sample_1",
                "x_min": -1,
                "x_max": 10,
                "y_min": -1,
                "y_max": 10,
            },
            {
                "sample_id": "sample_1",
                "x_min": 20,
                "x_max": 30,
                "y_min": 20,
                "y_max": 30,
            },
    ],
)
def test_spatial_rejects_empty_side(tmp_path, region):
    source = _write_full(tmp_path, "empty_side")
    config = _spatial_config(tmp_path, source, [region])

    with pytest.raises(SplitterError, match="must both be non-empty"):
        spatial_split(config)
    assert not (tmp_path / "spatial_output").exists()


def test_spatial_rejects_unknown_sample_and_existing_output(tmp_path):
    source = _write_full(tmp_path, "spatial_errors")
    unknown = _spatial_config(
        tmp_path,
        source,
        [{"sample_id": "other", "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}],
    )
    with pytest.raises(SplitterError, match="unknown sample_id"):
        spatial_split(unknown)

    existing = tmp_path / "existing"
    existing.mkdir()
    config = _spatial_config(
        tmp_path,
        source,
        [{"sample_id": "sample_1", "x_min": 0, "x_max": 0, "y_min": 1, "y_max": 1}],
        output_name="existing",
    )
    with pytest.raises(SplitterError, match="already exists"):
        spatial_split(config)


def test_spatial_cleans_temporary_directory_when_second_write_fails(tmp_path, monkeypatch):
    source = _write_full(tmp_path, "write_failure")
    config = _spatial_config(
        tmp_path,
        source,
        [{"sample_id": "sample_1", "x_min": 0, "x_max": 0, "y_min": 1, "y_max": 1}],
    )
    original_write = md.MuData.write_h5mu
    calls = 0

    def failing_write(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated write failure")
        return original_write(self, *args, **kwargs)

    monkeypatch.setattr(md.MuData, "write_h5mu", failing_write)
    with pytest.raises(RuntimeError, match="simulated"):
        spatial_split(config)

    assert not (tmp_path / "spatial_output").exists()
    assert list(tmp_path.glob(".spatial_output.tmp-*")) == []


def test_compose_assigns_whole_sources_and_encodes_global_ids(tmp_path):
    source_a = _write_full(tmp_path, "full_a", samples=["shared", "shared"])
    source_b = _write_full(tmp_path, "full_b", samples=["shared", "shared"])
    source_c = _write_full(tmp_path, "full_c", samples=["shared", "shared"])
    config = _compose_config(
        tmp_path,
        "preserve",
        [source_a, source_b],
        [source_c],
    )

    train_path, test_path = compose_split(config)
    train = _read(train_path)
    test = _read(test_path)
    try:
        assert list(train.obs_names) == [
            "full_a::cell_1",
            "full_a::cell_2",
            "full_b::cell_1",
            "full_b::cell_2",
        ]
        assert set(train.obs["sample_id"].astype(str)) == {"full_a::shared", "full_b::shared"}
        assert list(test.obs_names) == ["full_c::cell_1", "full_c::cell_2"]
        assert train.uns["database"]["derivation"]["construction_type"] == "composite"
        assert test.uns["database"]["derivation"]["construction_type"] == "subset"
        train_pairs = set(
            zip(
                train.obs["source_dataset_id"].astype(str),
                train.obs["source_obs_id"].astype(str),
                strict=True,
            )
        )
        test_pairs = set(
            zip(
                test.obs["source_dataset_id"].astype(str),
                test.obs["source_obs_id"].astype(str),
                strict=True,
            )
        )
        assert train_pairs.isdisjoint(test_pairs)
        assert len(train_pairs | test_pairs) == 6
    finally:
        train.file.close()
        test.file.close()


def _feature_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    protein = ["p1"]
    source_a = _write_full(
        tmp_path,
        "features_a",
        features={"rna": ["g2", "g1"], "protein": protein},
        matrices={
            "rna": np.asarray([[2, 1], [20, 10]], dtype=np.int32),
            "protein": np.asarray([[1], [2]], dtype=np.int32),
        },
    )
    source_b = _write_full(
        tmp_path,
        "features_b",
        features={"rna": ["g1", "g3"], "protein": protein},
        matrices={
            "rna": np.asarray([[3, 30], [4, 40]], dtype=np.int32),
            "protein": np.asarray([[3], [4]], dtype=np.int32),
        },
    )
    source_c = _write_full(
        tmp_path,
        "features_c",
        features={"rna": ["g1", "g2", "g4"], "protein": protein},
        matrices={
            "rna": np.asarray([[5, 6, 50], [7, 8, 70]], dtype=np.int32),
            "protein": np.asarray([[5], [6]], dtype=np.int32),
        },
    )
    return source_a, source_b, source_c


def test_intersection_uses_first_source_feature_order_and_values(tmp_path):
    source_a, source_b, source_c = _feature_sources(tmp_path)
    config = _compose_config(tmp_path, "intersection", [source_a, source_b], [source_c])

    train_path, test_path = compose_split(config)
    train = _read(train_path)
    test = _read(test_path)
    try:
        assert list(train.mod["rna"].var_names) == ["g1"]
        assert list(test.mod["rna"].var_names) == ["g1"]
        np.testing.assert_array_equal(train.mod["rna"].X[:, 0], [1, 10, 3, 4])
        np.testing.assert_array_equal(test.mod["rna"].X[:, 0], [5, 7])
        assert FEATURE_MASK_KEY not in train.mod["rna"].varm
    finally:
        train.file.close()
        test.file.close()


def test_union_aligns_values_and_records_per_source_measurement_mask(tmp_path):
    source_a, source_b, source_c = _feature_sources(tmp_path)
    config = _compose_config(tmp_path, "union", [source_a, source_b], [source_c])

    train_path, test_path = compose_split(config)
    train = _read(train_path)
    test = _read(test_path)
    try:
        expected_features = ["g2", "g1", "g3", "g4"]
        assert list(train.mod["rna"].var_names) == expected_features
        assert list(test.mod["rna"].var_names) == expected_features
        np.testing.assert_array_equal(train.mod["rna"].X[0], [2, 1, 0, 0])
        np.testing.assert_array_equal(train.mod["rna"].X[2], [0, 3, 30, 0])
        np.testing.assert_array_equal(test.mod["rna"].X[0], [6, 5, 0, 50])
        np.testing.assert_array_equal(
            train.mod["rna"].varm[FEATURE_MASK_KEY],
            np.asarray(
                [[True, False], [True, True], [False, True], [False, False]], dtype=bool
            ),
        )
        metadata = train.mod["rna"].uns["feature_measurement"]
        assert list(metadata["source_dataset_ids"]) == ["features_a", "features_b"]
        assert metadata["placeholder_value"] == 0
        assert "not a true measured zero" in metadata["description"]
    finally:
        train.file.close()
        test.file.close()


def test_reference_uses_matching_reference_order_and_missing_mask(tmp_path):
    common_protein = ["p1"]
    train_other = _write_full(
        tmp_path,
        "train_other",
        features={"rna": ["r1"], "protein": common_protein},
    )
    train_reference = _write_full(
        tmp_path,
        "train_reference",
        features={"rna": ["r2", "r1"], "protein": common_protein},
    )
    test_other = _write_full(
        tmp_path,
        "test_other",
        features={"rna": ["r1", "r3"], "protein": common_protein},
    )
    test_reference = _write_full(
        tmp_path,
        "test_reference",
        features={"rna": ["r2", "r1"], "protein": common_protein},
    )
    config = _compose_config(
        tmp_path,
        "reference",
        [train_other, train_reference],
        [test_other, test_reference],
        train_reference="train_reference",
        test_reference="test_reference",
    )

    train_path, test_path = compose_split(config)
    train = _read(train_path)
    test = _read(test_path)
    try:
        assert list(train.mod["rna"].var_names) == ["r2", "r1"]
        assert list(test.mod["rna"].var_names) == ["r2", "r1"]
        np.testing.assert_array_equal(train.mod["rna"].X[0], [0, 1])
        np.testing.assert_array_equal(
            train.mod["rna"].varm[FEATURE_MASK_KEY],
            np.asarray([[False, True], [True, True]], dtype=bool),
        )
        assert (
            train.uns["database"]["derivation"]["reference_dataset_id"]
            == "train_reference"
        )
    finally:
        train.file.close()
        test.file.close()


def test_compose_does_not_fabricate_modality_for_source_that_lacks_it(tmp_path):
    train_a = _write_full(
        tmp_path,
        "modal_train_a",
        features={"rna": ["g1"], "protein": ["p1"]},
    )
    train_b = _write_full(
        tmp_path,
        "modal_train_b",
        features={"rna": ["g1"], "atac": ["a1"]},
    )
    test = _write_full(
        tmp_path,
        "modal_test",
        features={"rna": ["g1"], "protein": ["p1"], "atac": ["a1"]},
    )
    config = _compose_config(tmp_path, "union", [train_a, train_b], [test])

    train_path, _ = compose_split(config)
    train = _read(train_path)
    try:
        assert list(train.mod["protein"].obs_names) == [
            "modal_train_a::cell_1",
            "modal_train_a::cell_2",
        ]
        assert list(train.mod["atac"].obs_names) == [
            "modal_train_b::cell_1",
            "modal_train_b::cell_2",
        ]
        np.testing.assert_array_equal(
            train.mod["protein"].varm[FEATURE_MASK_KEY], [[True, False]]
        )
    finally:
        train.file.close()


def test_preserve_rejects_different_features_and_intersection_rejects_empty(tmp_path):
    source_a, source_b, source_c = _feature_sources(tmp_path)
    preserve = _compose_config(tmp_path, "preserve", [source_a], [source_b])
    with pytest.raises(SplitterError, match="preserve requires identical"):
        compose_split(preserve)

    disjoint = _write_full(
        tmp_path,
        "disjoint",
        features={"rna": ["other"], "protein": ["p1"]},
    )
    intersection = _compose_config(
        tmp_path,
        "intersection",
        [source_a],
        [disjoint],
        output_name="empty_intersection",
    )
    with pytest.raises(SplitterError, match="intersection is empty"):
        compose_split(intersection)

    assert source_c.exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("unit", "same coordinate_unit"),
        ("value_type", "share value_type"),
        ("modalities", "same final modality set"),
    ],
)
def test_compose_rejects_incomparable_sources(tmp_path, change, message):
    source_a = _write_full(tmp_path, "compare_a")
    options = {}
    if change == "unit":
        options["coordinate_unit"] = "pixel"
    elif change == "value_type":
        options["value_types"] = {"rna": "normalized", "protein": "intensity"}
    else:
        options["features"] = {"rna": ["g1", "g2"], "atac": ["a1"]}
    source_b = _write_full(tmp_path, "compare_b", **options)
    config = _compose_config(tmp_path, "intersection", [source_a], [source_b])

    with pytest.raises(SplitterError, match=message):
        compose_split(config)


def test_compose_rejects_invalid_reference_and_duplicate_source(tmp_path):
    source_a = _write_full(tmp_path, "reference_a")
    source_b = _write_full(tmp_path, "reference_b")
    invalid_reference = _compose_config(
        tmp_path,
        "reference",
        [source_a],
        [source_b],
        train_reference="not_on_train",
        test_reference="reference_b",
    )
    with pytest.raises(SplitterError, match="own side"):
        compose_split(invalid_reference)

    duplicate = _compose_config(
        tmp_path,
        "intersection",
        [source_a],
        [source_a],
        output_name="duplicate_source",
    )
    with pytest.raises(SplitterError, match="both train and test"):
        load_compose_config(duplicate)


def test_configs_require_schema_v11_and_yaml_is_only_split_interface(tmp_path, capsys):
    source = _write_full(tmp_path, "config_source")
    config = _spatial_config(
        tmp_path,
        source,
        [{"sample_id": "sample_1", "x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0}],
    )
    values = yaml.safe_load(config.read_text(encoding="utf-8"))
    values["schema_version"] = "1.0"
    _write_yaml(config, values)

    assert main(["spatial", str(config)]) == 2
    assert "schema_version must be '1.1'" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["spatial", str(config), "--split-id", "old-interface"])


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_DIR = PROJECT_ROOT / "exp"
REAL_SPATIAL_SOURCE = REAL_DATA_DIR / "xenium_human_rcc_ffpe_rna_protein.h5mu"
REAL_SPATIAL_CONFIG = (
    REAL_DATA_DIR / "xenium_human_rcc_ffpe_rna_protein_vertical_split.yaml"
)


def test_real_h5mu_spatial_split_end_to_end():
    required = (REAL_SPATIAL_SOURCE, REAL_SPATIAL_CONFIG)
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    assert not missing, (
        "make test requires the real-data fixture(s) under exp/: " + ", ".join(missing)
    )

    config_values = yaml.safe_load(REAL_SPATIAL_CONFIG.read_text(encoding="utf-8"))
    assert config_values["source"] == REAL_SPATIAL_SOURCE.name

    with tempfile.TemporaryDirectory(prefix="iscdc-real-split-") as temporary:
        work_dir = Path(temporary)
        output_dir = work_dir / "real_spatial_output"
        config_values["source"] = str(REAL_SPATIAL_SOURCE)
        config_values["output_dir"] = str(output_dir)
        config_path = _write_yaml(work_dir / "real_spatial_split.yaml", config_values)

        train_path, test_path = spatial_split(config_path)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
            source = md.read_h5mu(REAL_SPATIAL_SOURCE, backed="r")
            train = md.read_h5mu(train_path, backed="r")
            test = md.read_h5mu(test_path, backed="r")
        try:
            assert source.n_obs == 465_534
            assert train.n_obs == 259_250
            assert test.n_obs == 206_284

            source_names = set(map(str, source.obs_names))
            train_names = set(map(str, train.obs_names))
            test_names = set(map(str, test.obs_names))
            assert train_names.isdisjoint(test_names)
            assert train_names | test_names == source_names

            assert set(train.mod) == set(test.mod) == set(source.mod)
            for modality in source.mod:
                source_features = list(map(str, source.mod[modality].var_names))
                assert list(map(str, train.mod[modality].var_names)) == source_features
                assert list(map(str, test.mod[modality].var_names)) == source_features

            split_id = config_values["split_id"]
            for product, dataset_type in ((train, "train"), (test, "test")):
                database = product.uns["database"]
                assert database["schema_version"] == "1.1"
                assert database["dataset_type"] == dataset_type
                assert database["derivation"]["split_id"] == split_id
                assert database["derivation"]["feature_merge_policy"] == "preserve"
        finally:
            source.file.close()
            train.file.close()
            test.file.close()
