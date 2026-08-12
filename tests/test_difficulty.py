from __future__ import annotations

from datetime import UTC, datetime

import anndata as ad
import mudata as md
import numpy as np
import pytest

from iscdc.cli import main
from iscdc.database import create_database_engine, create_session_factory, initialize_database
from iscdc.difficulty import (
    DifficultyConfig,
    DifficultyEvaluationError,
    _fold_auroc,
    _preprocess,
    evaluate_catalogue,
    write_report_atomically,
)
from iscdc.models import Dataset, Modality


def _add_challenge_side(
    settings,
    *,
    split_id: str,
    dataset_type: str,
    matrix: np.ndarray,
    feature_names: list[str] | None = None,
    value_type: str = "counts",
    feature_mask: np.ndarray | None = None,
) -> None:
    dataset_id = f"{split_id}_{dataset_type}"
    obs_names = [f"{dataset_id}_cell_{index}" for index in range(matrix.shape[0])]
    features = feature_names or [f"gene_{index:03d}" for index in range(matrix.shape[1])]
    rna = ad.AnnData(matrix.copy())
    rna.obs_names = obs_names
    rna.var_names = features
    rna.uns["assay"] = {"technology": "Synthetic RNA", "value_type": value_type}
    if feature_mask is not None:
        rna.varm["feature_measured_by_source"] = feature_mask
        rna.uns["feature_measurement"] = {
            "mask_key": "feature_measured_by_source",
            "source_dataset_ids": [f"source_{dataset_type}"],
            "placeholder_value": 0,
        }
    mdata = md.MuData({"rna": rna})
    mdata.obs["sample_id"] = "shared_sample"
    mdata.obs["source_dataset_id"] = "shared_source"
    mdata.obs["source_obs_id"] = obs_names
    mdata.obsm["spatial"] = np.zeros((matrix.shape[0], 2), dtype=np.float32)
    mdata.uns["database"] = {
        "schema_version": "1.2",
        "dataset_id": dataset_id,
        "dataset_type": dataset_type,
        "pairing_type": "same_unit",
        "derivation": {
            "split_id": split_id,
            "challenge_type": "same_slice",
        },
    }
    directory = settings.data_root / dataset_id
    directory.mkdir(parents=True)
    mdata.write_h5mu(directory / "dataset.h5mu")

    engine = create_database_engine(settings.database_path)
    initialize_database(engine)
    now = datetime.now(UTC)
    with create_session_factory(engine)() as session:
        record = Dataset(
            dataset_id=dataset_id,
            schema_version="1.2",
            dataset_type=dataset_type,
            title=dataset_id,
            description="Synthetic difficulty fixture",
            source="synthetic",
            organism="synthetic",
            tissue="synthetic",
            spatial_unit="cell",
            coordinate_unit="pixel",
            pairing_type="same_unit",
            derivation={
                "split_id": split_id,
                "challenge_type": "same_slice",
                "construction_type": "subset",
                "source_dataset_ids": ["shared_source"],
            },
            split_id=split_id,
            sample_ids=["shared_sample"],
            keywords=[],
            license=None,
            publication=None,
            additional_metadata={},
            n_obs=matrix.shape[0],
            coordinate_dimensions=2,
            file_size=(directory / "dataset.h5mu").stat().st_size,
            sha256=f"{len(dataset_id):064x}",
            storage_dir=dataset_id,
            validation_warning_count=0,
            imported_at=now,
        )
        record.modalities = [
            Modality(
                name="rna",
                technology="Synthetic RNA",
                value_type=value_type,
                n_obs=matrix.shape[0],
                n_vars=matrix.shape[1],
            )
        ]
        session.add(record)
        session.commit()
    engine.dispose()


def _small_config(seed: int = 42) -> DifficultyConfig:
    return DifficultyConfig(
        seed=seed,
        repeats=2,
        folds=3,
        max_observations_per_domain=90,
        max_features=20,
        representation_dimensionality=5,
        minimum_observations_per_domain=30,
        low_sample_warning_threshold=50,
    )


def test_catalogue_evaluation_detects_shift_and_is_reproducible(settings):
    rng = np.random.default_rng(7)
    same_train = rng.poisson(3, size=(140, 60)).astype(np.float32)
    same_test = rng.poisson(3, size=(115, 60)).astype(np.float32)
    shifted_train = rng.poisson(3, size=(240, 60)).astype(np.float32)
    shifted_test = rng.poisson(3, size=(120, 60)).astype(np.float32)
    shifted_test[:, :20] += 12
    for split_id, train, test in (
        ("same_distribution", same_train, same_test),
        ("clear_shift", shifted_train, shifted_test),
    ):
        _add_challenge_side(
            settings, split_id=split_id, dataset_type="train", matrix=train
        )
        _add_challenge_side(settings, split_id=split_id, dataset_type="test", matrix=test)

    first = evaluate_catalogue(settings, _small_config())
    second = evaluate_catalogue(settings, _small_config())
    first_by_id = {item["split_id"]: item for item in first["challenges"]}
    second_by_id = {item["split_id"]: item for item in second["challenges"]}

    same = first_by_id["same_distribution"]
    shifted = first_by_id["clear_shift"]
    assert 0.35 <= same["mean_auroc"] <= 0.65
    assert shifted["mean_auroc"] > 0.9
    assert shifted["difficulty_rank"] > same["difficulty_rank"]
    assert shifted["difficulty_percentile"] > same["difficulty_percentile"]
    assert shifted["domain_shift_score"] > same["domain_shift_score"]
    assert shifted["train"]["observations_total"] == 240
    assert shifted["repeat_results"][0]["train_observations_used"] == 90
    assert shifted["repeat_results"][0]["test_observations_used"] == 90
    assert first_by_id == second_by_id


def test_label_swap_preserves_fold_separability():
    rng = np.random.default_rng(11)
    matrix = np.vstack(
        (rng.normal(0, 1, (100, 20)), rng.normal(2, 1, (100, 20)))
    )
    labels = np.concatenate((np.zeros(100, dtype=np.int8), np.ones(100, dtype=np.int8)))
    training = np.r_[0:80, 100:180]
    held_out = np.r_[80:100, 180:200]
    config = DifficultyConfig(
        repeats=1,
        folds=2,
        max_features=15,
        representation_dimensionality=5,
    )
    original, _ = _fold_auroc(matrix, labels, training, held_out, config, 4)
    swapped, _ = _fold_auroc(matrix, 1 - labels, training, held_out, config, 4)
    assert swapped == pytest.approx(original)


def test_normalized_values_are_not_transformed():
    matrix = np.array([[0.1, 0.4], [1.2, 3.4]], dtype=np.float64)
    result = _preprocess(matrix, "normalized", 10_000)
    assert np.array_equal(result, matrix)
    assert result is not matrix


def test_masked_features_are_excluded_and_local_failure_is_retained(settings):
    rng = np.random.default_rng(23)
    valid = rng.poisson(2, size=(100, 60)).astype(np.float32)
    mask = np.ones((60, 1), dtype=bool)
    mask[-7:] = False
    _add_challenge_side(
        settings,
        split_id="masked",
        dataset_type="train",
        matrix=valid,
        feature_mask=mask,
    )
    _add_challenge_side(
        settings, split_id="masked", dataset_type="test", matrix=valid, feature_mask=mask
    )
    too_narrow = valid[:, :5]
    _add_challenge_side(
        settings, split_id="too_narrow", dataset_type="train", matrix=too_narrow
    )
    _add_challenge_side(
        settings, split_id="too_narrow", dataset_type="test", matrix=too_narrow
    )

    report = evaluate_catalogue(settings, _small_config())
    by_id = {item["split_id"]: item for item in report["challenges"]}
    assert by_id["masked"]["common_fully_measured_features"] == 53
    assert by_id["too_narrow"]["status"] == "failed"
    assert by_id["too_narrow"]["difficulty_rank"] is None
    assert report["success_count"] == 1
    assert report["failure_count"] == 1


def test_atomic_report_requires_force(tmp_path):
    path = tmp_path / "difficulty.json"
    write_report_atomically({"version": 1}, path)
    original = path.read_text()
    with pytest.raises(DifficultyEvaluationError, match="--force"):
        write_report_atomically({"version": 2}, path)
    assert path.read_text() == original
    write_report_atomically({"version": 2}, path, force=True)
    assert '"version": 2' in path.read_text()


def test_cli_writes_report_and_reports_partial_failure(
    settings, monkeypatch, capsys
):
    rng = np.random.default_rng(31)
    matrix = rng.poisson(3, size=(110, 60)).astype(np.float32)
    _add_challenge_side(
        settings, split_id="cli_challenge", dataset_type="train", matrix=matrix
    )
    _add_challenge_side(
        settings, split_id="cli_challenge", dataset_type="test", matrix=matrix
    )
    output = settings.database_path.parent / "custom.json"
    monkeypatch.setenv("ISCDC_DATABASE_PATH", str(settings.database_path))
    monkeypatch.setenv("ISCDC_DATA_ROOT", str(settings.data_root))

    import iscdc.difficulty as difficulty

    original_config = difficulty.DifficultyConfig
    monkeypatch.setattr(
        difficulty,
        "DifficultyConfig",
        lambda **kwargs: original_config(
            **kwargs,
            repeats=1,
            folds=2,
            max_observations_per_domain=60,
            max_features=20,
            representation_dimensionality=5,
            minimum_observations_per_domain=30,
        ),
    )
    exit_code = main(
        ["evaluate-challenge-difficulty", "--output", str(output), "--seed", "123"]
    )
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert output.is_file()
    assert '"success_count": 1' in stdout

    exit_code = main(
        ["evaluate-challenge-difficulty", "--output", str(output), "--seed", "123"]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--force" in captured.err
