from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import String, cast, exists, func, or_, select
from sqlalchemy.orm import Session

from .models import Dataset, Modality


@dataclass(frozen=True)
class DatasetFilters:
    query: str | None = None
    organism: str | None = None
    tissue: str | None = None
    modality: str | None = None
    technology: str | None = None
    spatial_unit: str | None = None
    dataset_type: str | None = None


def _escaped_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _conditions(filters: DatasetFilters):  # noqa: ANN202
    def json_contains(column, value):  # noqa: ANN001, ANN202
        entries = func.json_each(column).table_valued("value").alias()
        return exists(select(1).select_from(entries).where(entries.c.value == value))

    conditions = []
    if filters.query:
        pattern = _escaped_pattern(filters.query.strip())
        conditions.append(
            or_(
                Dataset.dataset_id.ilike(pattern, escape="\\"),
                Dataset.title.ilike(pattern, escape="\\"),
                Dataset.description.ilike(pattern, escape="\\"),
                Dataset.source.ilike(pattern, escape="\\"),
                Dataset.organism.ilike(pattern, escape="\\"),
                Dataset.tissue.ilike(pattern, escape="\\"),
                cast(Dataset.keywords, String).ilike(pattern, escape="\\"),
                Dataset.modalities.any(Modality.technology.ilike(pattern, escape="\\")),
            )
        )
    if filters.organism:
        conditions.append(json_contains(Dataset.organism, filters.organism))
    if filters.tissue:
        conditions.append(json_contains(Dataset.tissue, filters.tissue))
    if filters.spatial_unit:
        conditions.append(Dataset.spatial_unit == filters.spatial_unit)
    if filters.modality:
        conditions.append(Dataset.modalities.any(Modality.name == filters.modality))
    if filters.technology:
        conditions.append(
            Dataset.modalities.any(json_contains(Modality.technology, filters.technology))
        )
    if filters.dataset_type:
        conditions.append(Dataset.dataset_type == filters.dataset_type)
    return conditions


def list_datasets(
    session: Session, filters: DatasetFilters, offset: int, limit: int
) -> tuple[list[Dataset], int]:
    conditions = _conditions(filters)
    total = session.scalar(select(func.count()).select_from(Dataset).where(*conditions)) or 0
    datasets = list(
        session.scalars(
            select(Dataset)
            .where(*conditions)
            .order_by(Dataset.imported_at.desc(), Dataset.dataset_id)
            .offset(offset)
            .limit(limit)
        ).all()
    )
    return datasets, int(total)


def get_facets(session: Session) -> dict[str, list[str]]:
    def distinct_values(column) -> list[str]:  # noqa: ANN001
        return list(session.scalars(select(column).distinct().order_by(column)).all())

    def flattened_values(column) -> list[str]:  # noqa: ANN001
        values = session.scalars(select(column)).all()
        return sorted(
            {
                str(item)
                for value in values
                for item in (value if isinstance(value, list) else [value])
            }
        )

    return {
        "organisms": flattened_values(Dataset.organism),
        "tissues": flattened_values(Dataset.tissue),
        "spatial_units": distinct_values(Dataset.spatial_unit),
        "dataset_types": distinct_values(Dataset.dataset_type),
        "modalities": distinct_values(Modality.name),
        "technologies": flattened_values(Modality.technology),
    }
