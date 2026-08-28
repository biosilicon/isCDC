from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import mudata as md
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmread, mmwrite

from iscdc.cell_type_annotation import (
    ANNOTATION_ROOT,
    DEFAULT_PLAN_PATH,
    AnnotationPrediction,
    CatalogueRecord,
    CellTypeAnnotationError,
    DatasetPlan,
    QualityGates,
    ReferenceMetadata,
    ScientificAnnotationFailure,
    build_parser,
    export_sparse_rna_exchange,
    generate_source_labels,
    load_catalogue_plan,
    load_vocabulary,
    run_r_annotation,
    validate_prediction,
    validate_reference_pack,
)
from iscdc.cell_type_visualization import load_cell_type_visualization


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_import_is_lazy_about_heavy_annotation_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import iscdc.cell_type_annotation; "
            "assert 'mudata' not in sys.modules; assert 'h5py' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_sparse_exchange_preserves_sparse_counts_and_source(tmp_path, write_h5mu, monkeypatch):
    source = write_h5mu()
    original = _sha256(source)

    def reject_dense(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("sparse matrix was densified")

    monkeypatch.setattr(sparse.spmatrix, "toarray", reject_dense, raising=False)
    exchange = tmp_path / "exchange"
    metadata = export_sparse_rna_exchange(source, exchange)

    matrix = sparse.coo_matrix(mmread(exchange / "matrix.mtx"))
    assert matrix.shape == (2, 2)
    assert matrix.nnz == 2
    assert sorted(matrix.data.tolist()) == [1, 2]
    assert metadata["matrix_format"] == "MatrixMarket coordinate"
    assert _sha256(source) == original


def test_sparse_exchange_rejects_non_count_values():
    from iscdc.cell_type_annotation import _validated_sparse_matrix

    with pytest.raises(CellTypeAnnotationError, match="integer counts"):
        _validated_sparse_matrix(sparse.csr_matrix([[1.5, 0]]))
    with pytest.raises(CellTypeAnnotationError, match="integer counts"):
        _validated_sparse_matrix(sparse.csr_matrix([[-1, 0]]))


def test_sparse_exchange_normalizes_unsigned_counts_for_r_matrix_market():
    from iscdc.cell_type_annotation import _validated_sparse_matrix

    matrix = _validated_sparse_matrix(sparse.csr_matrix(np.array([[1, 0]], dtype=np.uint32)))

    assert matrix.dtype == np.int64
    assert matrix.data.tolist() == [1]


def test_source_labels_are_complete_and_do_not_invent_confidence(tmp_path, write_h5mu):
    source = write_h5mu()
    product = md.read_h5mu(source)
    product.obs["cell_type"] = pd.Categorical(["T cell", "Myeloid"])
    labelled = tmp_path / "labelled.h5mu"
    product.write_h5mu(labelled)
    before = _sha256(labelled)
    plan = DatasetPlan(
        "source-data",
        "source",
        None,
        True,
        True,
        source_evidence="verified public source",
        qc=QualityGates(require_calibration=False),
    )

    prediction = generate_source_labels(labelled, plan)

    assert prediction.labels == ("T cell", "Myeloid")
    assert prediction.statuses == ("Source", "Source")
    assert prediction.confidence is None
    assert _sha256(labelled) == before


def test_source_prediction_publishes_strict_sidecar(tmp_path, write_h5mu):
    from iscdc.cell_type_annotation import _sidecar_publish_generation

    source = write_h5mu()
    product = md.read_h5mu(source)
    product.obs["cell_type"] = pd.Categorical(["T cell", "Myeloid"])
    labelled = tmp_path / "source.h5mu"
    product.write_h5mu(labelled)
    plan = DatasetPlan(
        "source-data",
        "source",
        None,
        True,
        True,
        source_evidence="verified public source",
        qc=QualityGates(require_calibration=False),
    )
    prediction = generate_source_labels(labelled, plan)
    root = tmp_path / "visualizations"
    settings = SimpleNamespace(cell_type_visualization_root=root)
    record = CatalogueRecord(
        "source-data", labelled, _sha256(labelled), 2, 2, ("sample_01",), "pixel", "cell"
    )

    _sidecar_publish_generation(
        settings, record, plan, prediction, {"n_observations": 2}, None, None
    )

    snapshot = load_cell_type_visualization(
        root,
        {
            "dataset_id": "source-data",
            "sha256": record.sha256,
            "n_obs": 2,
            "coordinate_dimensions": 2,
            "sample_ids": ["sample_01"],
            "dataset_type": "full",
        },
    )
    assert snapshot.annotation_kind == "source"
    assert snapshot.inference_path is None
    assert not snapshot.has_confidence


def test_inferred_publication_preserves_full_h5_diagnostics(tmp_path):
    from iscdc.cell_type_annotation import _sidecar_publish_generation

    result = tmp_path / "r-result"
    result.mkdir()
    (result / "predictions.tsv").write_text(
        "observation_id\tlabel\tontology_id\tstatus\tconfidence\t"
        "top_weight\tsecond_weight\tdelta\tentropy\teffective_types\tconverged\n"
        "a\tT cell\tCL:0000084\tPredicted\t0.91\t0.8\t0.2\t0.6\t0.4\t1.5\tTRUE\n"
        "b\tB cell\tCL:0000236\tMixed\t0.62\t0.55\t0.45\t0.1\t0.9\t1.9\tTRUE\n",
        encoding="utf-8",
    )
    mmwrite(
        result / "cell_type_weights.mtx",
        sparse.csr_matrix([[0.8, 0.2], [0.45, 0.55]]),
    )
    (result / "weight_labels.tsv").write_text("label\nT cell\nB cell\n", encoding="utf-8")
    diagnostics = {
        "qc": {"shared_genes": 1500, "balanced_accuracy": 0.9},
        "calibration": {
            "completed": True,
            "method": "logistic",
            "confidence_definition": "pseudo-spot calibrated dominant-type reliability",
            "calibration_id": "pseudo-spots-v1",
        },
    }
    prediction = AnnotationPrediction(
        observation_ids=("a", "b"),
        labels=("T cell", "B cell"),
        ontology_ids=("CL:0000084", None),
        statuses=("Predicted", "Mixed"),
        sample_ids=("sample", "sample"),
        x=(0.0, 1.0),
        y=(0.0, 1.0),
        confidence=(0.91, 0.62),
        method="rctd",
        diagnostics=diagnostics,
    )
    plan = DatasetPlan(
        "inferred-data",
        "rctd",
        "reference-v1",
        True,
        True,
        parameters={"rctd_mode": "full", "cores": 12},
    )
    root = tmp_path / "visualizations"
    reference_dir = root / "references" / "reference-v1"
    reference_dir.mkdir(parents=True)
    (reference_dir / "reference.json").write_text("{}\n", encoding="utf-8")
    reference = ReferenceMetadata("reference-v1", "Homo sapiens", "tonsil", "v1", (), {})
    settings = SimpleNamespace(cell_type_visualization_root=root)
    record = CatalogueRecord(
        "inferred-data",
        tmp_path / "unused.h5mu",
        "a" * 64,
        2,
        2,
        ("sample",),
        "pixel",
        "spot",
    )

    _sidecar_publish_generation(
        settings,
        record,
        plan,
        prediction,
        {"n_observations": 2, "calibrated": True},
        reference,
        result,
    )

    snapshot = load_cell_type_visualization(
        root,
        {
            "dataset_id": "inferred-data",
            "sha256": "a" * 64,
            "n_obs": 2,
            "coordinate_dimensions": 2,
            "sample_ids": ["sample"],
            "dataset_type": "full",
        },
    )
    assert snapshot.annotation_kind == "inferred"
    assert snapshot.inference_path is not None
    with h5py.File(snapshot.inference_path, "r") as inference:
        assert "cell_type_weights" in inference
        assert {
            "top_weight",
            "second_weight",
            "delta",
            "entropy",
            "converged",
        } <= set(inference.keys())
        assert inference["cell_type_weights"].attrs["shape"].tolist() == [2, 2]
        assert bool(inference["calibration"].attrs["completed"])
        assert inference["calibration"].attrs["calibration_id"] == "pseudo-spots-v1"


def _inferred_prediction(**diagnostic_overrides) -> AnnotationPrediction:
    diagnostics = {
        "qc": {
            "shared_genes": 1000,
            "balanced_accuracy": 0.9,
            "macro_f1": 0.88,
            "ece": 0.04,
            "marker_agreement": 0.8,
        },
        "calibration": {
            "completed": True,
            "confidence_definition": "held-out isotonic reliability",
        },
    }
    diagnostics.update(diagnostic_overrides)
    return AnnotationPrediction(
        observation_ids=("a", "b"),
        labels=("T cell", "T cell"),
        ontology_ids=("CL:0000084", None),
        statuses=("Predicted", "Uncertain"),
        sample_ids=("sample", "sample"),
        x=(0.0, 1.0),
        y=(0.0, 1.0),
        confidence=(0.9, 0.2),
        method="singler",
        diagnostics=diagnostics,
    )


def test_inferred_validation_rejects_missing_calibration_and_failed_qc():
    plan = DatasetPlan(
        "inferred",
        "singler",
        "reference",
        True,
        True,
        qc=QualityGates(min_balanced_accuracy=0.8, max_ece=0.1),
    )
    uncalibrated = _inferred_prediction(calibration={"completed": False})
    with pytest.raises(CellTypeAnnotationError, match="calibration is incomplete"):
        validate_prediction(uncalibrated, plan, expected_observation_ids=("a", "b"))

    failed = _inferred_prediction()
    failed.diagnostics["qc"]["balanced_accuracy"] = 0.5
    with pytest.raises(ScientificAnnotationFailure, match="balanced_accuracy"):
        validate_prediction(failed, plan, expected_observation_ids=("a", "b"))


def test_prediction_order_and_cell_ontology_are_strict():
    plan = DatasetPlan("x", "singler", "ref", True, True)
    prediction = _inferred_prediction()
    with pytest.raises(CellTypeAnnotationError, match="source order"):
        validate_prediction(prediction, plan, expected_observation_ids=("b", "a"))
    invalid = AnnotationPrediction(**{**prediction.__dict__, "ontology_ids": (None, None)})
    with pytest.raises(CellTypeAnnotationError, match="Cell Ontology"):
        validate_prediction(invalid, plan, expected_observation_ids=("a", "b"))
    state_with_ontology = AnnotationPrediction(
        **{**prediction.__dict__, "ontology_ids": ("CL:0000084", "CL:0000084")}
    )
    with pytest.raises(CellTypeAnnotationError, match="must not carry"):
        validate_prediction(state_with_ontology, plan, expected_observation_ids=("a", "b"))


def test_reference_metadata_checks_every_file(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    payload = reference / "reference.rds"
    payload.write_bytes(b"reference")
    metadata = {
        "schema_version": "1.0",
        "reference_id": "human-kidney-v1",
        "species": "Homo sapiens",
        "tissue": "kidney",
        "version": "2026-08-18",
        "files": [
            {"name": payload.name, "size": payload.stat().st_size, "sha256": _sha256(payload)}
        ],
    }
    (reference / "reference.json").write_text(__import__("json").dumps(metadata), encoding="utf-8")
    assert validate_reference_pack(reference).reference_id == "human-kidney-v1"
    payload.write_bytes(b"changed")
    with pytest.raises(CellTypeAnnotationError, match="size mismatch|checksum mismatch"):
        validate_reference_pack(reference)


def test_rscript_dispatch_uses_full_adapters_and_worker_caps(tmp_path):
    calls = []
    environments = []

    def runner(command, **kwargs):  # noqa: ANN001, ANN003
        calls.append(command)
        environments.append(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, "ok", "")

    run_r_annotation(
        "rctd",
        tmp_path / "exchange",
        tmp_path / "reference",
        tmp_path / "result",
        DEFAULT_PLAN_PATH,
        workers=3,
        parameters={"uncertain_min_confidence": 0.2},
        runner=runner,
    )
    assert calls[0][0:3] == ["Rscript", "--vanilla", str(ANNOTATION_ROOT / "run_rctd.R")]
    assert calls[0][-2:] == ["--workers", "3"]
    assert environments[0]["OPENBLAS_NUM_THREADS"] == "1"
    assert environments[0]["OMP_THREAD_LIMIT"] == "1"
    runtime_config = json.loads((tmp_path / "result" / "runtime_config.json").read_text())
    assert runtime_config["parameters"]["uncertain_min_confidence"] == 0.2
    with pytest.raises(CellTypeAnnotationError, match="between 1 and 12"):
        run_r_annotation(
            "rctd",
            tmp_path / "e",
            tmp_path / "r",
            tmp_path / "another",
            DEFAULT_PLAN_PATH,
            workers=13,
            runner=runner,
        )
    with pytest.raises(CellTypeAnnotationError, match="between 1 and 30"):
        run_r_annotation(
            "singler",
            tmp_path / "single-e",
            tmp_path / "single-r",
            tmp_path / "single-result",
            DEFAULT_PLAN_PATH,
            workers=31,
            runner=runner,
        )


def test_rscript_failure_survives_atomic_staging_cleanup(tmp_path):
    from iscdc.cell_type_annotation import _run_rscript

    def runner(command, **kwargs):  # noqa: ANN001, ANN003
        return subprocess.CompletedProcess(command, 2, "", "scientific calibration failed")

    with pytest.raises(CellTypeAnnotationError, match="scientific calibration failed"):
        _run_rscript(Path("builder.R"), (), tmp_path / "rscript.log", runner=runner)


def test_failure_publication_bounds_summary_but_preserves_diagnostics(tmp_path):
    from iscdc.cell_type_annotation import _publish_failure

    root = tmp_path / "visualizations"
    detail = "diagnostic " * 400
    _publish_failure(
        SimpleNamespace(cell_type_visualization_root=root),
        "dataset",
        {"detail": detail, "exit_code": 1},
    )

    status = json.loads((root / "dataset" / "status.json").read_text(encoding="utf-8"))
    report = json.loads(
        (root / "dataset" / "failures" / status["failure_id"] / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(report["error"]) == 2000
    assert report["details"]["detail"] == detail


def test_current_quality_gate_failure_is_a_terminal_scientific_state(tmp_path, monkeypatch):
    import iscdc.cell_type_annotation as annotation

    root = tmp_path / "visualizations"
    settings = SimpleNamespace(cell_type_visualization_root=root)
    plan = DatasetPlan(
        "dataset",
        "source",
        None,
        True,
        True,
        source_evidence="verified public source",
        qc=QualityGates(require_calibration=False),
    )
    record = CatalogueRecord(
        "dataset", tmp_path / "source.h5mu", "a" * 64, 1, 2, ("sample",), "pixel", "cell"
    )
    report = annotation._failure_report(
        "dataset", plan, record.sha256, "scientific_failure", "quality gate failed"
    )
    annotation._publish_failure(settings, "dataset", report, category="scientific_failure")
    monkeypatch.setattr(annotation, "_catalogue_record", lambda dataset_id, resolved: record)

    assert annotation._published_states(settings, ["dataset"], {"dataset": plan}) == {
        "dataset": "scientific_failure"
    }


@pytest.mark.parametrize("publication_fails", [False, True])
def test_generation_removes_work_directory(tmp_path, monkeypatch, publication_fails):
    import iscdc.cell_type_annotation as annotation

    source = tmp_path / "source.h5mu"
    source.write_bytes(b"immutable source")
    plan = DatasetPlan(
        "source-data",
        "source",
        None,
        True,
        True,
        source_evidence="verified public source",
        qc=QualityGates(require_calibration=False),
    )
    record = CatalogueRecord(
        "source-data", source, "a" * 64, 1, 2, ("sample",), "pixel", "cell"
    )
    prediction = AnnotationPrediction(
        observation_ids=("cell",),
        labels=("T cell",),
        ontology_ids=(None,),
        statuses=("Source",),
        sample_ids=("sample",),
        x=(0.0,),
        y=(0.0,),
        confidence=None,
        method="source",
    )
    root = tmp_path / "visualizations"
    settings = SimpleNamespace(cell_type_visualization_root=root)

    monkeypatch.setattr(annotation, "_require_annotation_environment", lambda: None)
    monkeypatch.setattr(annotation, "load_catalogue_plan", lambda path: {"source-data": plan})
    monkeypatch.setattr(annotation, "_catalogue_record", lambda dataset_id, resolved: record)
    monkeypatch.setattr(annotation, "sha256_file", lambda path: record.sha256)
    monkeypatch.setattr(annotation, "generate_source_labels", lambda path, selected: prediction)
    monkeypatch.setattr(annotation, "validate_prediction", lambda *args, **kwargs: {})
    monkeypatch.setattr(annotation, "_publish_failure", lambda *args, **kwargs: None)

    def publish(*args, **kwargs):  # noqa: ANN002, ANN003
        if publication_fails:
            raise RuntimeError("publication failed")

    monkeypatch.setattr(annotation, "_sidecar_publish_generation", publish)

    if publication_fails:
        with pytest.raises(CellTypeAnnotationError, match="publication failed"):
            annotation.generate_cell_type_visualization("source-data", settings, force=True)
    else:
        annotation.generate_cell_type_visualization("source-data", settings, force=True)

    assert list((root / "work").iterdir()) == []


def test_checked_in_plan_covers_catalogue_and_environment_is_cpu_only():
    plans = load_catalogue_plan()
    assert len(plans) == 36
    assert sum(plan.pilot for plan in plans.values()) == 4
    assert {plan.method for plan in plans.values()} == {"source", "singler", "rctd"}
    assert plans["xenium_human_ccrcc_ffpe_rna_protein"].method == "source"
    assert not plans["xenium_human_ccrcc_ffpe_rna_protein"].qc.require_calibration
    assert {
        plan.parameters.get("cores") for plan in plans.values() if plan.method == "rctd"
    } == {3, 5, 12}
    assert plans["xenium_human_rcc_ffpe_rna_protein"].parameters["cores"] == 30
    assert plans["xenium_human_rcc_ffpe_rna_protein"].parameters["exclusive"] is False
    five_core_repair_ids = {
        "GSE213264_human_gbm_spatial_citeseq",
        "GSE213264_mouse_colon_spatial_citeseq",
        "GSE213264_mouse_intestine_spatial_citeseq",
        "GSE213264_mouse_kidney_spatial_citeseq",
        "MISAR_seq_mouse_brain_E15_5_S1",
        "MISAR_seq_mouse_brain_E18_5_S1",
    }
    assert {
        plans[dataset_id].parameters["cores"] for dataset_id in five_core_repair_ids
    } == {5}
    assert plans["MISAR_seq_mouse_brain_E13_5_S1"].parameters["cores"] == 12
    assert "Mixed" not in load_vocabulary()
    environment = (ANNOTATION_ROOT / "environment.yml").read_text(encoding="utf-8").lower()
    assert "name: iscdc-cell-annotation" in environment
    assert "python=3.13.15" in environment
    assert "r-base=4.6.0" in environment
    assert "bioconductor-" not in environment
    assert "r-spacexr" not in environment
    assert "pytorch" not in environment
    assert "cuda" not in environment
    lock = json.loads((ANNOTATION_ROOT / "renv.lock").read_text(encoding="utf-8"))
    assert lock["R"]["Version"] == "4.6.0"
    assert lock["Bioconductor"]["Version"] == "3.23"
    assert lock["Packages"]["SingleR"]["Version"] == "2.14.1"
    assert lock["Packages"]["spacexr"]["Version"] == "1.4.0"
    assert lock["Packages"]["renv"]["Version"] == "1.2.4"
    assert len(lock["Packages"]) >= 90


def test_resource_batches_enforce_job_and_40_logical_core_limits():
    from iscdc.cell_type_annotation import _resource_batches

    plans = {
        f"dataset-{index:02d}": DatasetPlan(
            f"dataset-{index:02d}",
            "rctd",
            "reference",
            False,
            True,
            parameters={"cores": 5},
        )
        for index in range(21)
    }
    batches = _resource_batches(tuple(plans), plans, max_jobs=20)
    assert tuple(map(len, batches)) == (8, 8, 5)
    assert all(len(batch) <= 20 for batch in batches)
    assert all(
        sum(plans[dataset_id].parameters["cores"] for dataset_id in batch) <= 40
        for batch in batches
    )


def test_cli_parser_exposes_offline_workflow():
    parser = build_parser()
    assert parser.parse_args(["build-cell-type-reference", "ref"]).command == (
        "build-cell-type-reference"
    )
    assert parser.parse_args(["generate-cell-type-visualization", "dataset"]).dataset_id == (
        "dataset"
    )
    audit = parser.parse_args(["audit-cell-type-visualizations", "--all", "--jobs", "20"])
    assert audit.all_datasets and audit.jobs == 20
