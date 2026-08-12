"""Read-only publication of offline Challenge difficulty results."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .repository import Challenge

SUPPORTED_REPORT_VERSIONS = {"1.0"}
SUPPORTED_METHOD_VERSIONS = {"1.0"}


class DifficultySnapshotError(RuntimeError):
    """Raised when a difficulty snapshot is unavailable or inconsistent."""


@dataclass(frozen=True)
class ChallengeDifficulty:
    mean_auroc: float
    domain_shift_score: float
    difficulty_percentile: float


@dataclass(frozen=True)
class DifficultySnapshot:
    report_version: str
    method_version: str
    generated_at: datetime
    input_modality: str
    by_split_id: Mapping[str, ChallengeDifficulty]


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DifficultySnapshotError(f"{field} must be an object")
    return value


def _finite_number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DifficultySnapshotError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise DifficultySnapshotError(f"{field} must be between {minimum} and {maximum}")
    return result


def _parse_generated_at(value: Any) -> datetime:
    if not isinstance(value, str):
        raise DifficultySnapshotError("generated_at must be an ISO 8601 timestamp")
    try:
        generated_at = datetime.fromisoformat(value)
    except ValueError as exc:
        raise DifficultySnapshotError("generated_at must be an ISO 8601 timestamp") from exc
    if generated_at.tzinfo is None:
        raise DifficultySnapshotError("generated_at must include a timezone")
    return generated_at


def _validate_side(
    value: Any,
    challenge: Challenge,
    side: str,
) -> None:
    expected = getattr(challenge, side)
    if expected is None:
        raise DifficultySnapshotError(
            f"successful result for {challenge.split_id!r} has no {side} catalogue side"
        )
    record = _object(value, f"{challenge.split_id}.{side}")
    if record.get("dataset_id") != expected.dataset_id:
        raise DifficultySnapshotError(
            f"{side} dataset ID differs for Challenge {challenge.split_id!r}"
        )
    if record.get("sha256") != expected.sha256:
        raise DifficultySnapshotError(
            f"{side} checksum differs for Challenge {challenge.split_id!r}"
        )


def load_difficulty_snapshot(
    path: Path,
    challenges: Sequence[Challenge],
) -> DifficultySnapshot:
    """Load and validate one complete report against the current catalogue."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DifficultySnapshotError(f"difficulty report is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DifficultySnapshotError(f"unable to read difficulty report: {exc}") from exc
    report = _object(document, "difficulty report")

    report_version = report.get("report_version")
    if report_version not in SUPPORTED_REPORT_VERSIONS:
        raise DifficultySnapshotError(
            f"unsupported difficulty report version: {report_version!r}"
        )
    method_version = report.get("method_version")
    if method_version not in SUPPORTED_METHOD_VERSIONS:
        raise DifficultySnapshotError(
            f"unsupported difficulty method version: {method_version!r}"
        )
    generated_at = _parse_generated_at(report.get("generated_at"))
    parameters = _object(report.get("parameters"), "parameters")
    input_modality = parameters.get("input_modality")
    if not isinstance(input_modality, str) or not input_modality.strip():
        raise DifficultySnapshotError("parameters.input_modality must be non-empty")

    rows = report.get("challenges")
    if not isinstance(rows, list):
        raise DifficultySnapshotError("challenges must be a list")
    if report.get("challenge_count") != len(rows):
        raise DifficultySnapshotError("challenge_count does not match challenges")

    catalogue_by_id = {challenge.split_id: challenge for challenge in challenges}
    if len(catalogue_by_id) != len(challenges):
        raise DifficultySnapshotError("catalogue contains duplicate Challenge split IDs")
    rows_by_id: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(rows):
        row = _object(value, f"challenges[{index}]")
        split_id = row.get("split_id")
        if not isinstance(split_id, str) or not split_id:
            raise DifficultySnapshotError(f"challenges[{index}].split_id must be non-empty")
        if split_id in rows_by_id:
            raise DifficultySnapshotError(f"duplicate difficulty result for {split_id!r}")
        rows_by_id[split_id] = row
    if set(rows_by_id) != set(catalogue_by_id):
        raise DifficultySnapshotError(
            "difficulty report Challenge set differs from the current catalogue"
        )

    by_split_id: dict[str, ChallengeDifficulty] = {}
    success_count = 0
    failure_count = 0
    for split_id, challenge in catalogue_by_id.items():
        row = rows_by_id[split_id]
        if row.get("challenge_type") != challenge.challenge_type:
            raise DifficultySnapshotError(
                f"challenge_type differs for Challenge {challenge.split_id!r}"
            )
        if row.get("input_modality") != input_modality:
            raise DifficultySnapshotError(
                f"input modality differs for Challenge {challenge.split_id!r}"
            )
        status = row.get("status")
        if status == "failed":
            failure_count += 1
            continue
        if status != "success":
            raise DifficultySnapshotError(
                f"invalid difficulty status for Challenge {challenge.split_id!r}"
            )
        success_count += 1
        _validate_side(row.get("train"), challenge, "train")
        _validate_side(row.get("test"), challenge, "test")
        mean_auroc = _finite_number(
            row.get("mean_auroc"), f"{split_id}.mean_auroc", 0.0, 1.0
        )
        domain_shift_score = _finite_number(
            row.get("domain_shift_score"), f"{split_id}.domain_shift_score", 0.0, 1.0
        )
        expected_score = min(1.0, max(0.0, 2 * (mean_auroc - 0.5)))
        if not math.isclose(domain_shift_score, expected_score, abs_tol=1e-9):
            raise DifficultySnapshotError(
                f"domain shift score is inconsistent for Challenge {split_id!r}"
            )
        percentile = _finite_number(
            row.get("difficulty_percentile"),
            f"{split_id}.difficulty_percentile",
            0.0,
            100.0,
        )
        by_split_id[split_id] = ChallengeDifficulty(
            mean_auroc=mean_auroc,
            domain_shift_score=domain_shift_score,
            difficulty_percentile=percentile,
        )

    if report.get("success_count") != success_count:
        raise DifficultySnapshotError("success_count does not match successful results")
    if report.get("failure_count") != failure_count:
        raise DifficultySnapshotError("failure_count does not match failed results")
    successful = list(by_split_id.items())
    for split_id, difficulty in successful:
        if len(successful) == 1:
            expected_percentile = 0.0
        else:
            lower_count = sum(
                other.mean_auroc < difficulty.mean_auroc for _, other in successful
            )
            equal_count = sum(
                other.mean_auroc == difficulty.mean_auroc for _, other in successful
            )
            expected_percentile = (
                100 * (lower_count + (equal_count - 1) / 2) / (len(successful) - 1)
            )
        if not math.isclose(
            difficulty.difficulty_percentile, expected_percentile, abs_tol=1e-9
        ):
            raise DifficultySnapshotError(
                f"difficulty percentile is inconsistent for Challenge {split_id!r}"
            )
    return DifficultySnapshot(
        report_version=report_version,
        method_version=method_version,
        generated_at=generated_at,
        input_modality=input_modality,
        by_split_id=by_split_id,
    )
