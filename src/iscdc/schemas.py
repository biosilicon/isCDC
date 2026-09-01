from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .technology import technology_vocabulary_message, unsupported_technologies

SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DATASET_ID_PATTERN = SAFE_IDENTIFIER_PATTERN
ScalarOrList = str | list[str]
ChallengeType = Literal["same_slice", "cross_slice_same_subject", "cross_subject"]
PairingType = Literal["same_unit", "partially_shared"]


def _clean_scalar_or_list(value: ScalarOrList) -> ScalarOrList:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned
    cleaned = [item.strip() for item in value]
    if not cleaned or any(not item for item in cleaned):
        raise ValueError("must contain non-blank strings")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("entries must be unique")
    return cleaned


class FeatureHarmonizationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["1.0"]
    scope: Literal["all_challenge_sources"]
    aggregation: Literal["sum"]
    source_dataset_ids: list[str] = Field(min_length=2)
    modalities: dict[str, str] = Field(min_length=2)

    @field_validator("source_dataset_ids")
    @classmethod
    def validate_source_ids(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("entries must not be blank")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("entries must be unique")
        return cleaned

    @field_validator("modalities")
    @classmethod
    def validate_namespaces(cls, values: dict[str, str]) -> dict[str, str]:
        cleaned = {name.strip(): namespace.strip() for name, namespace in values.items()}
        if any(not name or not namespace for name, namespace in cleaned.items()):
            raise ValueError("modality names and namespaces must not be blank")
        return cleaned


class CoordinateHarmonizationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["1.0"]
    spatial_unit: str = Field(min_length=1)
    coordinate_unit: str = Field(min_length=1)
    source_dataset_ids: list[str] = Field(min_length=2)

    @field_validator("spatial_unit", "coordinate_unit")
    @classmethod
    def strip_units(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("source_dataset_ids")
    @classmethod
    def validate_source_ids(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("entries must not be blank")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("entries must be unique")
        return cleaned


class DerivationMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    construction_type: Literal["subset", "composite"]
    source_dataset_ids: list[str] = Field(min_length=1)
    split_id: str = Field(min_length=1)
    challenge_type: ChallengeType
    selection_description: str = Field(min_length=1)
    feature_merge_policy: Literal["preserve", "intersection", "union", "reference"]
    processing_description: str = Field(min_length=1)
    reference_dataset_id: str | None = None
    random_seed: int | None = None
    feature_harmonization: FeatureHarmonizationMetadata | None = None
    coordinate_harmonization: CoordinateHarmonizationMetadata | None = None

    @field_validator(
        "source_dataset_ids",
        mode="after",
    )
    @classmethod
    def validate_source_dataset_ids(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("entries must not be blank")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("entries must be unique")
        return cleaned

    @field_validator("split_id", "selection_description", "processing_description")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_derivation_relationships(self) -> DerivationMetadata:
        source_count = len(self.source_dataset_ids)
        if self.construction_type == "subset" and source_count != 1:
            raise ValueError("subset requires exactly one source_dataset_id")
        if self.construction_type == "composite" and source_count < 2:
            raise ValueError("composite requires at least two source_dataset_ids")
        if self.feature_merge_policy == "reference":
            if self.reference_dataset_id not in self.source_dataset_ids:
                raise ValueError("reference_dataset_id must identify one of the sources")
        elif self.reference_dataset_id is not None:
            raise ValueError("reference_dataset_id is only valid for the reference policy")
        if self.feature_harmonization is not None:
            if self.feature_merge_policy != "intersection":
                raise ValueError("feature harmonization requires the intersection policy")
            if not set(self.source_dataset_ids).issubset(
                self.feature_harmonization.source_dataset_ids
            ):
                raise ValueError(
                    "feature harmonization sources must include every contributing source"
                )
        if self.coordinate_harmonization is not None:
            if not set(self.source_dataset_ids).issubset(
                self.coordinate_harmonization.source_dataset_ids
            ):
                raise ValueError(
                    "coordinate harmonization sources must include every contributing source"
                )
        return self


class DatabaseMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal["1.2"]
    dataset_id: str = Field(min_length=1, max_length=128)
    entry_id: str = Field(min_length=1, max_length=128)
    dataset_type: Literal["full", "train", "test"]
    source: ScalarOrList
    organism: ScalarOrList
    tissue: ScalarOrList
    spatial_unit: str = Field(min_length=1)
    coordinate_unit: str = Field(min_length=1)
    pairing_type: PairingType
    derivation: DerivationMetadata | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_empty_full_derivation(cls, value: Any) -> Any:
        if isinstance(value, dict) and "license" in value:
            raise ValueError("dataset license metadata is not supported")
        if isinstance(value, dict) and value.get("dataset_type") == "full":
            if not value.get("derivation"):
                value = dict(value)
                value["derivation"] = None
        return value

    @field_validator("dataset_id", "entry_id")
    @classmethod
    def validate_safe_identifier(cls, value: str) -> str:
        value = value.strip()
        if not SAFE_IDENTIFIER_PATTERN.fullmatch(value):
            raise ValueError(
                "must start with an alphanumeric character and contain only letters, numbers, "
                "dots, underscores, or hyphens"
            )
        return value

    @field_validator("source", "organism", "tissue")
    @classmethod
    def validate_scalar_or_list(cls, value: ScalarOrList) -> ScalarOrList:
        return _clean_scalar_or_list(value)

    @field_validator("spatial_unit", "coordinate_unit")
    @classmethod
    def validate_required_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_dataset_derivation(self) -> DatabaseMetadata:
        if self.dataset_type == "full" and self.derivation is not None:
            raise ValueError("full datasets must not define derivation metadata")
        if self.dataset_type in {"train", "test"} and self.derivation is None:
            raise ValueError("train and test datasets require derivation metadata")
        return self


class ModalityMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    technology: ScalarOrList
    value_type: str = Field(min_length=1)

    @field_validator("technology")
    @classmethod
    def validate_technology(cls, value: ScalarOrList) -> ScalarOrList:
        cleaned = _clean_scalar_or_list(value)
        unsupported = unsupported_technologies(cleaned)
        if unsupported:
            raise ValueError(technology_vocabulary_message(unsupported))
        return cleaned

    @field_validator("value_type")
    @classmethod
    def validate_value_type(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


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

    @model_validator(mode="after")
    def validate_pairing_policy(self) -> MetadataDocument:
        if len(self.modalities) == 2 and self.database.pairing_type != "same_unit":
            raise ValueError("two-modality datasets require pairing_type 'same_unit'")
        return self

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
    technology: ScalarOrList
    value_type: str
    n_obs: int
    n_vars: int


class AuxiliaryFileResponse(BaseModel):
    id: str
    label: str
    filename: str
    media_type: str
    size: int
    sha256: str
    source_url: str
    download_url: str


class SampleSourceResponse(BaseModel):
    sample_id: str
    source_sample_id: str
    source_database_id: str
    source_database_title: str
    source: ScalarOrList


class DataFileResponse(BaseModel):
    dataset_id: str
    entry_id: str
    schema_version: str
    dataset_type: Literal["full", "train", "test"]
    title: str
    description: str
    source: ScalarOrList
    organism: ScalarOrList
    tissue: ScalarOrList
    spatial_unit: str
    coordinate_unit: str
    pairing_type: PairingType
    derivation: DerivationMetadata | None
    sample_ids: list[str]
    sample_sources: list[SampleSourceResponse]
    keywords: list[str]
    publication: dict[str, Any] | None
    additional_metadata: dict[str, Any]
    n_obs: int
    coordinate_dimensions: int
    modalities: list[ModalityResponse]
    modality_count: int
    file_size: int
    sha256: str
    validation_warning_count: int
    imported_at: date
    downloads: dict[str, str]
    auxiliary_files: list[AuxiliaryFileResponse]


class DatabaseListResponse(BaseModel):
    items: list[DataFileResponse]
    total: int
    limit: int
    offset: int


class ChallengeDifficultyResponse(BaseModel):
    mean_auroc: float
    domain_shift_score: float
    difficulty_percentile: float


class ChallengeResponse(BaseModel):
    split_id: str
    challenge_type: ChallengeType
    status: Literal["complete", "missing_train", "missing_test"]
    difficulty: ChallengeDifficultyResponse | None
    train: DataFileResponse | None
    test: DataFileResponse | None


class ChallengeListResponse(BaseModel):
    items: list[ChallengeResponse]
    total: int
    limit: int
    offset: int
