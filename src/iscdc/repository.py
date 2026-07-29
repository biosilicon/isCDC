from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import String, cast, exists, func, or_, select
from sqlalchemy.orm import Session

from .models import Dataset, Modality
from .schemas import ChallengeType

DERIVED_DATASET_TYPES = ("train", "test")
CHALLENGE_TYPES = ("same_slice", "cross_slice_same_subject", "cross_subject")


class CatalogueIntegrityError(RuntimeError):
    """Raised when catalogue records cannot form unambiguous challenges."""


@dataclass(frozen=True)
class CatalogueFilters:
    query: str | None = None
    organism: str | None = None
    tissue: str | None = None
    modality: str | None = None
    technology: str | None = None
    spatial_unit: str | None = None
    challenge_type: ChallengeType | None = None


@dataclass(frozen=True)
class Challenge:
    split_id: str
    train: Dataset | None
    test: Dataset | None

    @property
    def status(self) -> Literal["complete", "missing_train", "missing_test"]:
        if self.train is None:
            return "missing_train"
        if self.test is None:
            return "missing_test"
        return "complete"

    @property
    def challenge_type(self) -> ChallengeType:
        dataset = self.train or self.test
        if dataset is None or dataset.derivation is None:
            raise CatalogueIntegrityError(
                f"Challenge {self.split_id!r} does not define a challenge_type."
            )
        return dataset.derivation["challenge_type"]

    @property
    def datasets(self) -> list[Dataset]:
        return [dataset for dataset in (self.train, self.test) if dataset is not None]


def _escaped_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _json_contains(column, value):  # noqa: ANN001, ANN202
    entries = func.json_each(column).table_valued("value").alias()
    return exists(select(1).select_from(entries).where(entries.c.value == value))


def _conditions(filters: CatalogueFilters, *, include_split_id: bool = False):  # noqa: ANN202
    conditions = []
    if filters.query:
        pattern = _escaped_pattern(filters.query.strip())
        query_fields = [
            Dataset.dataset_id.ilike(pattern, escape="\\"),
            Dataset.title.ilike(pattern, escape="\\"),
            Dataset.description.ilike(pattern, escape="\\"),
            Dataset.source.ilike(pattern, escape="\\"),
            Dataset.organism.ilike(pattern, escape="\\"),
            Dataset.tissue.ilike(pattern, escape="\\"),
            cast(Dataset.keywords, String).ilike(pattern, escape="\\"),
            Dataset.modalities.any(Modality.technology.ilike(pattern, escape="\\")),
        ]
        if include_split_id:
            query_fields.append(Dataset.split_id.ilike(pattern, escape="\\"))
        conditions.append(or_(*query_fields))
    if filters.organism:
        conditions.append(_json_contains(Dataset.organism, filters.organism))
    if filters.tissue:
        conditions.append(_json_contains(Dataset.tissue, filters.tissue))
    if filters.spatial_unit:
        conditions.append(Dataset.spatial_unit == filters.spatial_unit)
    if filters.modality:
        conditions.append(Dataset.modalities.any(Modality.name == filters.modality))
    if filters.technology:
        conditions.append(
            Dataset.modalities.any(_json_contains(Modality.technology, filters.technology))
        )
    if filters.challenge_type:
        conditions.append(
            func.json_extract(Dataset.derivation, "$.challenge_type")
            == filters.challenge_type
        )
    return conditions


def count_databases(session: Session) -> int:
    value = session.scalar(
        select(func.count()).select_from(Dataset).where(Dataset.dataset_type == "full")
    )
    return int(value or 0)


def count_challenges(session: Session) -> int:
    value = session.scalar(
        select(func.count(func.distinct(Dataset.split_id))).where(
            Dataset.dataset_type.in_(DERIVED_DATASET_TYPES),
            Dataset.split_id.is_not(None),
        )
    )
    return int(value or 0)


def list_databases(
    session: Session, filters: CatalogueFilters, offset: int, limit: int
) -> tuple[list[Dataset], int]:
    conditions = [Dataset.dataset_type == "full", *_conditions(filters)]
    total = session.scalar(select(func.count()).select_from(Dataset).where(*conditions)) or 0
    databases = list(
        session.scalars(
            select(Dataset)
            .where(*conditions)
            .order_by(Dataset.imported_at.desc(), Dataset.dataset_id)
            .offset(offset)
            .limit(limit)
        ).all()
    )
    return databases, int(total)


def get_database(session: Session, dataset_id: str) -> Dataset | None:
    return session.scalar(
        select(Dataset).where(
            Dataset.dataset_id == dataset_id,
            Dataset.dataset_type == "full",
        )
    )


def _validate_challenge_integrity(session: Session) -> None:
    missing_split_id = session.scalar(
        select(Dataset.dataset_id)
        .where(
            Dataset.dataset_type.in_(DERIVED_DATASET_TYPES),
            Dataset.split_id.is_(None),
        )
        .limit(1)
    )
    if missing_split_id is not None:
        raise CatalogueIntegrityError(
            f"Derived dataset {missing_split_id!r} does not define a split_id."
        )

    derived = session.execute(
        select(Dataset.dataset_id, Dataset.split_id, Dataset.derivation).where(
            Dataset.dataset_type.in_(DERIVED_DATASET_TYPES)
        )
    ).all()
    challenge_types_by_split: dict[str, set[str]] = {}
    for dataset_id, split_id, derivation in derived:
        challenge_type = derivation.get("challenge_type") if isinstance(derivation, dict) else None
        if challenge_type not in CHALLENGE_TYPES:
            raise CatalogueIntegrityError(
                f"Derived dataset {dataset_id!r} does not define a valid challenge_type."
            )
        challenge_types_by_split.setdefault(split_id, set()).add(challenge_type)
    for split_id, challenge_types in challenge_types_by_split.items():
        if len(challenge_types) > 1:
            raise CatalogueIntegrityError(
                f"Challenge {split_id!r} contains inconsistent challenge_type values."
            )

    duplicate = session.execute(
        select(Dataset.split_id, Dataset.dataset_type, func.count())
        .where(Dataset.dataset_type.in_(DERIVED_DATASET_TYPES))
        .group_by(Dataset.split_id, Dataset.dataset_type)
        .having(func.count() > 1)
        .limit(1)
    ).first()
    if duplicate is not None:
        split_id, dataset_type, _count = duplicate
        raise CatalogueIntegrityError(
            f"Challenge {split_id!r} contains multiple {dataset_type} datasets."
        )


def _build_challenges(datasets: list[Dataset], split_ids: list[str]) -> list[Challenge]:
    grouped: dict[str, dict[str, Dataset]] = {split_id: {} for split_id in split_ids}
    for dataset in datasets:
        if dataset.split_id is not None:
            grouped[dataset.split_id][dataset.dataset_type] = dataset
    return [
        Challenge(
            split_id=split_id,
            train=grouped[split_id].get("train"),
            test=grouped[split_id].get("test"),
        )
        for split_id in split_ids
    ]


def list_challenges(
    session: Session, filters: CatalogueFilters, offset: int, limit: int
) -> tuple[list[Challenge], int]:
    _validate_challenge_integrity(session)
    conditions = [
        Dataset.dataset_type.in_(DERIVED_DATASET_TYPES),
        Dataset.split_id.is_not(None),
        *_conditions(filters, include_split_id=True),
    ]
    matching_ids = (
        select(Dataset.split_id.label("split_id"))
        .where(*conditions)
        .group_by(Dataset.split_id)
        .subquery()
    )
    total = session.scalar(select(func.count()).select_from(matching_ids)) or 0
    split_ids = list(
        session.scalars(
            select(Dataset.split_id)
            .where(*conditions)
            .group_by(Dataset.split_id)
            .order_by(func.max(Dataset.imported_at).desc(), Dataset.split_id)
            .offset(offset)
            .limit(limit)
        ).all()
    )
    if not split_ids:
        return [], int(total)

    datasets = list(
        session.scalars(
            select(Dataset).where(
                Dataset.dataset_type.in_(DERIVED_DATASET_TYPES),
                Dataset.split_id.in_(split_ids),
            )
        ).all()
    )
    return _build_challenges(datasets, split_ids), int(total)


def get_challenge(session: Session, split_id: str) -> Challenge | None:
    _validate_challenge_integrity(session)
    datasets = list(
        session.scalars(
            select(Dataset).where(
                Dataset.dataset_type.in_(DERIVED_DATASET_TYPES),
                Dataset.split_id == split_id,
            )
        ).all()
    )
    if not datasets:
        return None
    return _build_challenges(datasets, [split_id])[0]


def get_facets(session: Session, dataset_types: tuple[str, ...]) -> dict[str, list[str]]:
    def distinct_dataset_values(column) -> list[str]:  # noqa: ANN001
        return list(
            session.scalars(
                select(column)
                .where(Dataset.dataset_type.in_(dataset_types))
                .distinct()
                .order_by(column)
            ).all()
        )

    def flattened_dataset_values(column) -> list[str]:  # noqa: ANN001
        values = session.scalars(
            select(column).where(Dataset.dataset_type.in_(dataset_types))
        ).all()
        return sorted(
            {
                str(item)
                for value in values
                for item in (value if isinstance(value, list) else [value])
            }
        )

    def distinct_modality_values(column) -> list[str]:  # noqa: ANN001
        return list(
            session.scalars(
                select(column)
                .join(Dataset, Modality.dataset_id == Dataset.dataset_id)
                .where(Dataset.dataset_type.in_(dataset_types))
                .distinct()
                .order_by(column)
            ).all()
        )

    technologies = session.scalars(
        select(Modality.technology)
        .join(Dataset, Modality.dataset_id == Dataset.dataset_id)
        .where(Dataset.dataset_type.in_(dataset_types))
    ).all()
    challenge_types = list(
        session.scalars(
            select(func.json_extract(Dataset.derivation, "$.challenge_type"))
            .where(
                Dataset.dataset_type.in_(dataset_types),
                Dataset.derivation.is_not(None),
            )
            .distinct()
            .order_by(func.json_extract(Dataset.derivation, "$.challenge_type"))
        ).all()
    )
    return {
        "organisms": flattened_dataset_values(Dataset.organism),
        "tissues": flattened_dataset_values(Dataset.tissue),
        "spatial_units": distinct_dataset_values(Dataset.spatial_unit),
        "modalities": distinct_modality_values(Modality.name),
        "technologies": sorted(
            {
                str(item)
                for value in technologies
                for item in (value if isinstance(value, list) else [value])
            }
        ),
        "challenge_types": [value for value in challenge_types if value in CHALLENGE_TYPES],
    }
