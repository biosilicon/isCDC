from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from iscdc.difficulty_snapshot import (
    DifficultySnapshotError,
    load_difficulty_snapshot,
)
from iscdc.repository import Challenge


def _challenge() -> Challenge:
    derivation = {"challenge_type": "same_slice"}
    return Challenge(
        split_id="split_v1",
        train=SimpleNamespace(
            dataset_id="train_v1", sha256="a" * 64, derivation=derivation
        ),
        test=SimpleNamespace(
            dataset_id="test_v1", sha256="b" * 64, derivation=derivation
        ),
    )


def _report() -> dict:
    return {
        "report_version": "1.0",
        "method_version": "1.0",
        "generated_at": "2026-08-12T12:00:00+00:00",
        "parameters": {"input_modality": "rna"},
        "challenge_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "challenges": [
            {
                "split_id": "split_v1",
                "challenge_type": "same_slice",
                "status": "success",
                "input_modality": "rna",
                "train": {"dataset_id": "train_v1", "sha256": "a" * 64},
                "test": {"dataset_id": "test_v1", "sha256": "b" * 64},
                "mean_auroc": 0.8,
                "domain_shift_score": 0.6,
                "difficulty_percentile": 0.0,
            }
        ],
    }


def _write(path, report):  # noqa: ANN001, ANN202
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_load_difficulty_snapshot_validates_and_exposes_only_public_summary(tmp_path):
    snapshot = load_difficulty_snapshot(
        _write(tmp_path / "difficulty.json", _report()), [_challenge()]
    )

    difficulty = snapshot.by_split_id["split_v1"]
    assert snapshot.input_modality == "rna"
    assert difficulty.mean_auroc == 0.8
    assert difficulty.domain_shift_score == 0.6
    assert difficulty.difficulty_percentile == 0.0


def test_failed_difficulty_result_remains_available_as_an_empty_entry(tmp_path):
    report = _report()
    report["challenges"][0] = {
        "split_id": "split_v1",
        "challenge_type": "same_slice",
        "status": "failed",
        "input_modality": "rna",
    }
    report["success_count"] = 0
    report["failure_count"] = 1

    snapshot = load_difficulty_snapshot(
        _write(tmp_path / "difficulty.json", report), [_challenge()]
    )
    assert snapshot.by_split_id == {}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report.update(report_version="2.0"), "report version"),
        (
            lambda report: report["challenges"][0].update(split_id="other"),
            "Challenge set",
        ),
        (
            lambda report: report["challenges"][0].update(challenge_type="cross_subject"),
            "challenge_type differs",
        ),
        (
            lambda report: report["challenges"][0]["train"].update(sha256="c" * 64),
            "checksum differs",
        ),
        (
            lambda report: report["challenges"][0].update(domain_shift_score=0.1),
            "shift score is inconsistent",
        ),
        (
            lambda report: report["challenges"][0].update(difficulty_percentile=50.0),
            "percentile is inconsistent",
        ),
        (
            lambda report: report.update(success_count=0),
            "success_count",
        ),
    ],
)
def test_invalid_difficulty_snapshots_are_rejected(tmp_path, mutation, message):
    report = deepcopy(_report())
    mutation(report)
    path = _write(tmp_path / "difficulty.json", report)

    with pytest.raises(DifficultySnapshotError, match=message):
        load_difficulty_snapshot(path, [_challenge()])


@pytest.mark.parametrize("content", ["not json", "[]"])
def test_malformed_difficulty_snapshots_are_rejected(tmp_path, content):
    path = tmp_path / "difficulty.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(DifficultySnapshotError):
        load_difficulty_snapshot(path, [_challenge()])
