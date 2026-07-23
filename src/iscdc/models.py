from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class CatalogueMetadata(Base):
    __tablename__ = "catalogue_metadata"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(64), nullable=False)


class Dataset(Base):
    __tablename__ = "datasets"

    dataset_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | list[str]] = mapped_column(JSON, nullable=False)
    organism: Mapped[str | list[str]] = mapped_column(JSON, nullable=False)
    tissue: Mapped[str | list[str]] = mapped_column(JSON, nullable=False)
    spatial_unit: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    coordinate_unit: Mapped[str] = mapped_column(String(64), nullable=False)
    pairing_type: Mapped[str] = mapped_column(String(64), nullable=False)
    derivation: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    split_id: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    sample_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    keywords: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    license: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    publication: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    additional_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    n_obs: Mapped[int] = mapped_column(Integer, nullable=False)
    coordinate_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_dir: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    validation_warning_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    modalities: Mapped[list[Modality]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="Modality.name",
    )


class Modality(Base):
    __tablename__ = "modalities"
    __table_args__ = (UniqueConstraint("dataset_id", "name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    technology: Mapped[str | list[str]] = mapped_column(JSON, nullable=False)
    value_type: Mapped[str] = mapped_column(String(100), nullable=False)
    n_obs: Mapped[int] = mapped_column(Integer, nullable=False)
    n_vars: Mapped[int] = mapped_column(Integer, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="modalities")
