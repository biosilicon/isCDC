from __future__ import annotations

from copy import deepcopy

import pytest

from iscdc.entry_reconciliation import (
    EntryIdReconciliationError,
    rebind_difficulty_snapshot_sha256,
)


def _snapshot() -> dict:
    return {
        "report_version": "1.0",
        "method_version": "1.0",
        "generated_at": "2026-08-31T00:00:00+00:00",
        "parameters": {"input_modality": "rna", "repeats": 5},
        "challenge_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "challenges": [
            {
                "split_id": "split_1",
                "status": "success",
                "mean_auroc": 0.72,
                "difficulty_percentile": 0.0,
                "repeat_results": [{"mean_auroc": 0.71}],
                "train": {"dataset_id": "train_1", "sha256": "a" * 64},
                "test": {"dataset_id": "test_1", "sha256": "b" * 64},
            }
        ],
    }


def test_rebind_difficulty_snapshot_changes_only_sha256_fields():
    original = _snapshot()
    rebound, count = rebind_difficulty_snapshot_sha256(
        original,
        {
            "train_1": ("a" * 64, "c" * 64),
            "test_1": ("b" * 64, "d" * 64),
        },
    )

    assert count == 2
    assert original == _snapshot()
    assert rebound["challenges"][0]["train"]["sha256"] == "c" * 64
    assert rebound["challenges"][0]["test"]["sha256"] == "d" * 64
    comparable = deepcopy(rebound)
    comparable["challenges"][0]["train"]["sha256"] = "a" * 64
    comparable["challenges"][0]["test"]["sha256"] = "b" * 64
    assert comparable == original


def test_rebind_difficulty_snapshot_rejects_stale_or_missing_rows():
    with pytest.raises(EntryIdReconciliationError, match="unexpected SHA-256"):
        rebind_difficulty_snapshot_sha256(
            _snapshot(), {"train_1": ("e" * 64, "f" * 64)}
        )

    with pytest.raises(EntryIdReconciliationError, match="does not contain"):
        rebind_difficulty_snapshot_sha256(
            _snapshot(), {"unknown": ("e" * 64, "f" * 64)}
        )
