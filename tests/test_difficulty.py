from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256

import anndata as ad
import mudata as md
import numpy as np
import pytest
from scipy import sparse

from iscdc.cli import main
from iscdc.database import create_database_engine, create_session_factory, initialize_database
from iscdc.difficulty import (
    DifficultyConfig,
    DifficultyEvaluationError,
    _fold_auroc,
    _preprocess,
    evaluate_and_write,
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
        "entry_id": split_id,
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
            entry_id=split_id,
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
            publication=None,
            additional_metadata={},
            n_obs=matrix.shape[0],
            coordinate_dimensions=2,
            file_size=(directory / "dataset.h5mu").stat().st_size,
            sha256=sha256((directory / "dataset.h5mu").read_bytes()).hexdigest(),
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
    assert main([
        "evaluate-challenge-difficulty", "--output", str(output), "--seed", "123", "--force"
    ]) == 0
    assert '"reused_count": 1' in capsys.readouterr().out
    assert main([
        "evaluate-challenge-difficulty", "--output", str(output), "--seed", "123",
        "--force", "--recompute",
    ]) == 0
    assert '"evaluated_count": 1' in capsys.readouterr().out


@pytest.fixture
def cached_catalogue(settings):
    rng = np.random.default_rng(123)
    for split_id in ("stable", "changed"):
        for side in ("train", "test"):
            matrix = rng.poisson(3, size=(100, 60)).astype(np.float32)
            if split_id == "changed" and side == "test":
                matrix[:, :20] += 8
            _add_challenge_side(
                settings, split_id=split_id, dataset_type=side, matrix=matrix,
            )
    config = _small_config()
    return config, evaluate_catalogue(settings, config)


def _rewrite_side(settings, mutate, *, update_checksum=True, dataset_id="changed_train"):
    path = settings.data_root / dataset_id / "dataset.h5mu"
    mdata = md.read_h5mu(path)
    mutate(mdata)
    temporary = path.with_name("rewritten.h5mu")
    mdata.write_h5mu(temporary)
    temporary.replace(path)
    if update_checksum:
        engine = create_database_engine(settings.database_path)
        with create_session_factory(engine)() as session:
            dataset = session.get(Dataset, dataset_id)
            dataset.sha256 = sha256(path.read_bytes()).hexdigest()
            dataset.file_size = path.stat().st_size
            session.commit()
        engine.dispose()


def _forbid_evaluation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("verified unchanged inputs must not refit a classifier")
    monkeypatch.setattr("iscdc.difficulty._evaluate_challenge", unexpected)


@pytest.mark.parametrize("legacy", [False, True])
def test_unchanged_and_legacy_snapshots_reuse_without_fitting(
    settings, cached_catalogue, monkeypatch, legacy
):
    config, previous = cached_catalogue
    if legacy:
        for row in previous["challenges"]:
            for side in ("train", "test"):
                del row[side]["input_fingerprint"]
    _forbid_evaluation(monkeypatch)
    report = evaluate_catalogue(settings, config, previous_report=previous)
    assert report["reused_count"] == 2
    assert report["evaluated_count"] == report["failure_count"] == 0
    for old, new in zip(previous["challenges"], report["challenges"], strict=True):
        assert old["repeat_results"] == new["repeat_results"]
        assert old["difficulty_percentile"] == new["difficulty_percentile"]
        assert new["train"]["input_fingerprint"]["version"] == "1.0"


def test_metadata_only_rewrite_and_storage_encoding_rebind_without_fitting(
    settings, cached_catalogue, monkeypatch
):
    config, previous = cached_catalogue

    def metadata_only(mdata):
        mdata.uns["database"]["entry_id"] = "corrected_entry"
        mdata.uns["database"]["spatial_unit"] = "spot/bin"
        mdata.uns["database"]["description"] = "Corrected source identity"
        mdata.obs["cell_type"] = "source_label"
        mdata.obsm["spatial"][:] = 100
        mdata.mod["rna"].X = sparse.csr_matrix(mdata.mod["rna"].X)

    _rewrite_side(settings, metadata_only)
    _forbid_evaluation(monkeypatch)
    report = evaluate_catalogue(settings, config, previous_report=previous)
    old = next(row for row in previous["challenges"] if row["split_id"] == "changed")
    new = next(row for row in report["challenges"] if row["split_id"] == "changed")
    assert new["train"]["sha256"] != old["train"]["sha256"]
    assert new["train"]["input_fingerprint"] == old["train"]["input_fingerprint"]
    assert new["repeat_results"] == old["repeat_results"]
    assert report["reused_count"] == 2
    assert report["evaluated_count"] == report["failure_count"] == 0


@pytest.mark.parametrize("change", ["matrix", "features", "observations", "mask", "value_type"])
def test_effective_input_changes_refit_only_affected_challenge(
    settings, cached_catalogue, monkeypatch, change
):
    config, previous = cached_catalogue

    def mutate(mdata):
        rna = mdata.mod["rna"]
        if change == "matrix":
            rna.X[0, 0] += 100
        elif change == "features":
            rna.var_names = list(reversed(rna.var_names))
        elif change == "observations":
            rna.obs_names = list(reversed(rna.obs_names))
        elif change == "mask":
            mask = np.ones((rna.n_vars, 1), dtype=bool)
            mask[0] = False
            rna.varm["feature_measured_by_source"] = mask
        else:
            rna.uns["assay"]["value_type"] = "normalized"

    _rewrite_side(settings, mutate)
    import iscdc.difficulty as difficulty
    original = difficulty._evaluate_challenge
    evaluated = []

    def spy(challenge, *args):
        evaluated.append(challenge.split_id)
        return original(challenge, *args)

    monkeypatch.setattr(difficulty, "_evaluate_challenge", spy)
    report = evaluate_catalogue(settings, config, previous_report=previous)
    assert evaluated == ["changed"]
    assert report["reused_count"] == report["evaluated_count"] == 1
    assert report["failure_count"] == int(change == "value_type")


def test_catalogue_checksum_mismatch_is_not_rebound(settings, cached_catalogue, monkeypatch):
    config, previous = cached_catalogue
    _rewrite_side(
        settings, lambda m: m.uns["database"].update(description="changed"),
        update_checksum=False,
    )
    _forbid_evaluation(monkeypatch)
    report = evaluate_catalogue(settings, config, previous_report=previous)
    assert report["failure_count"] == report["reused_count"] == 1
    failed = next(row for row in report["challenges"] if row["status"] == "failed")
    assert "actual file checksum differs" in failed["error"]


def test_removal_reranks_remaining_results_and_clears_cohort_warnings(
    settings, cached_catalogue, monkeypatch
):
    config, previous = cached_catalogue
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        for side in ("train", "test"):
            session.delete(session.get(Dataset, f"stable_{side}"))
        session.commit()
    engine.dispose()
    _forbid_evaluation(monkeypatch)
    report = evaluate_catalogue(settings, config, previous_report=previous)
    assert report["challenge_count"] == report["reused_count"] == 1
    row = report["challenges"][0]
    assert row["difficulty_percentile"] == row["same_category_percentile"] == 0
    assert row["difficulty_rank"] == 1
    assert row["repeat_percentile_std"] is None
    assert [w for w in row["warnings"] if w["code"] == "small_category_pool"] == [
        {"code": "small_category_pool", "category_size": 1}
    ]


@pytest.mark.parametrize("invalid", ["failed", "metric", "fingerprint", "legacy_changed"])
def test_invalid_or_unproven_cached_row_is_retried(
    settings, cached_catalogue, invalid
):
    config, previous = cached_catalogue
    row = next(row for row in previous["challenges"] if row["split_id"] == "changed")
    if invalid == "failed":
        row["status"] = "failed"
    elif invalid == "metric":
        row["mean_auroc"] = 0.2
    else:
        _rewrite_side(settings, lambda m: m.uns["database"].update(description="new"))
        if invalid == "fingerprint":
            row["train"]["input_fingerprint"]["sha256"] = "0" * 64
        else:
            del row["train"]["input_fingerprint"]
    report = evaluate_catalogue(settings, config, previous_report=previous)
    assert report["reused_count"] == report["evaluated_count"] == 1
    assert report["failure_count"] == 0


@pytest.mark.parametrize("change", ["parameters", "software", "method", "explicit"])
def test_incompatible_cache_and_explicit_recompute_refit_all(settings, cached_catalogue, change):
    config, previous = cached_catalogue
    previous = deepcopy(previous)
    if change == "parameters":
        config = replace(config, seed=123)
    elif change == "software":
        previous["software"]["scikit_learn"] = "different"
    elif change == "method":
        previous["method_version"] = "different"
    output = settings.database_path.parent / "difficulty.json"
    write_report_atomically(previous, output)
    report = evaluate_and_write(
        settings, output, config=config, force=True, recompute=change == "explicit"
    )
    assert report["evaluated_count"] == 2
    assert report["reused_count"] == report["failure_count"] == 0


def test_force_only_overwrites_and_reuses_verified_results(
    settings, cached_catalogue, monkeypatch
):
    config, previous = cached_catalogue
    output = settings.database_path.parent / "difficulty.json"
    write_report_atomically(previous, output)
    _forbid_evaluation(monkeypatch)
    report = evaluate_and_write(settings, output, config=config, force=True)
    assert report["reused_count"] == 2
    assert report["evaluated_count"] == 0


def test_added_challenge_evaluates_only_new_pair(settings, cached_catalogue, monkeypatch):
    config, previous = cached_catalogue
    matrix = np.random.default_rng(42).poisson(2, size=(100, 60)).astype(np.float32)
    for side in ("train", "test"):
        _add_challenge_side(settings, split_id="new", dataset_type=side, matrix=matrix)
    import iscdc.difficulty as difficulty
    original = difficulty._evaluate_challenge
    called = []

    def spy(challenge, *args):
        called.append(challenge.split_id)
        return original(challenge, *args)

    monkeypatch.setattr(difficulty, "_evaluate_challenge", spy)
    report = evaluate_catalogue(settings, config, previous_report=previous)
    assert called == ["new"]
    assert report["challenge_count"] == 3
    assert report["reused_count"] == 2
    assert sorted(row["difficulty_percentile"] for row in report["challenges"]) == [0, 50, 100]


def test_other_modality_does_not_invalidate_rna_evaluation(
    settings, cached_catalogue, monkeypatch
):
    config, previous = cached_catalogue

    def add_protein(mdata):
        protein = ad.AnnData(np.ones((mdata.n_obs, 2)))
        protein.obs_names = mdata.obs_names
        protein.var_names = ["protein_a", "protein_b"]
        mdata.mod["protein"] = protein
        mdata.update()

    _rewrite_side(settings, add_protein)
    _forbid_evaluation(monkeypatch)
    report = evaluate_catalogue(settings, config, previous_report=previous)
    assert report["reused_count"] == 2
    assert report["evaluated_count"] == report["failure_count"] == 0


def test_challenge_type_correction_only_regroups_ranking(
    settings, cached_catalogue, monkeypatch
):
    config, previous = cached_catalogue
    for side in ("train", "test"):
        _rewrite_side(
            settings,
            lambda m: m.uns["database"]["derivation"].update(challenge_type="cross_subject"),
            dataset_id=f"changed_{side}",
        )
    engine = create_database_engine(settings.database_path)
    with create_session_factory(engine)() as session:
        for side in ("train", "test"):
            dataset = session.get(Dataset, f"changed_{side}")
            dataset.derivation = {**dataset.derivation, "challenge_type": "cross_subject"}
        session.commit()
    engine.dispose()
    _forbid_evaluation(monkeypatch)
    report = evaluate_catalogue(settings, config, previous_report=previous)
    assert report["reused_count"] == 2
    assert report["evaluated_count"] == report["failure_count"] == 0
    changed = next(row for row in report["challenges"] if row["split_id"] == "changed")
    assert changed["challenge_type"] == "cross_subject"
    assert changed["same_category_percentile"] == 0
    for row in report["challenges"]:
        assert [w for w in row["warnings"] if w["code"] == "small_category_pool"] == [
            {"code": "small_category_pool", "category_size": 1}
        ]


def test_file_changed_during_evaluation_cannot_publish_a_success(
    settings, cached_catalogue, monkeypatch
):
    config, previous = cached_catalogue
    import iscdc.difficulty as difficulty
    original = difficulty._evaluate_challenge

    def racing_change(challenge, *args):
        result = original(challenge, *args)
        if challenge.split_id == "changed":
            _rewrite_side(
                settings, lambda m: m.uns["database"].update(description="concurrent write"),
                update_checksum=False,
            )
        return result

    monkeypatch.setattr(difficulty, "_evaluate_challenge", racing_change)
    report = evaluate_catalogue(settings, config)
    assert report["failure_count"] == 1
    failed = next(row for row in report["challenges"] if row["status"] == "failed")
    assert "actual file checksum differs" in failed["error"]
