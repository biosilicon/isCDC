from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class DatabaseMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1)
    organism: str = Field(min_length=1)
    tissue: str = Field(min_length=1)
    spatial_unit: str = Field(min_length=1)
    coordinate_unit: str = Field(min_length=1)
    pairing_type: str = Field(min_length=1)

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        if not DATASET_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "must start with an alphanumeric character and contain only letters, numbers, "
                "dots, underscores, or hyphens"
            )
        return value


class ModalityMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    technology: str = Field(min_length=1)
    value_type: str = Field(min_length=1)


class LicenseMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    identifier: str | None = None
    url: str | None = None


class PublicationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation: str | None = None
    doi: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def require_one_value(self) -> PublicationMetadata:
        if not any((self.citation, self.doi, self.url)):
            raise ValueError("at least one publication field must be populated")
        return self


class MetadataDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database: DatabaseMetadata
    sample_ids: list[str] = Field(min_length=1)
    modalities: dict[str, ModalityMetadata] = Field(min_length=2)
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1)
    keywords: list[str]
    license: LicenseMetadata | None
    publication: PublicationMetadata | None

    @field_validator("sample_ids", "keywords")
    @classmethod
    def validate_string_lists(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("entries must not be blank")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("entries must be unique")
        return cleaned

    @field_validator("modalities")
    @classmethod
    def validate_modality_names(
        cls, values: dict[str, ModalityMetadata]
    ) -> dict[str, ModalityMetadata]:
        if any(not name.strip() for name in values):
            raise ValueError("modality names must not be blank")
        return values

    def database_values(self) -> dict[str, Any]:
        return self.database.model_dump(mode="python")

    def additional_database_values(self) -> dict[str, Any]:
        standard_fields = set(DatabaseMetadata.model_fields)
        return {
            key: value
            for key, value in self.database_values().items()
            if key not in standard_fields
        }


class MetadataLoadError(ValueError):
    pass


def load_metadata(path) -> MetadataDocument:  # noqa: ANN001
    try:
        with open(path, encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise MetadataLoadError(f"Unable to read metadata YAML: {exc}") from exc

    try:
        return MetadataDocument.model_validate(raw)
    except ValidationError as exc:
        raise MetadataLoadError(f"Invalid metadata YAML:\n{exc}") from exc


class ModalityResponse(BaseModel):
    name: str
    technology: str
    value_type: str
    n_obs: int
    n_vars: int


class DatasetResponse(BaseModel):
    dataset_id: str
    schema_version: str
    title: str
    description: str
    source: str
    organism: str
    tissue: str
    spatial_unit: str
    coordinate_unit: str
    pairing_type: str
    sample_ids: list[str]
    keywords: list[str]
    license: dict[str, Any] | None
    publication: dict[str, Any] | None
    additional_metadata: dict[str, Any]
    n_obs: int
    coordinate_dimensions: int
    modalities: list[ModalityResponse]
    file_size: int
    sha256: str
    validation_warning_count: int
    imported_at: datetime
    downloads: dict[str, str]


class DatasetListResponse(BaseModel):
    items: list[DatasetResponse]
    total: int
    limit: int
    offset: int
