from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pytest
import yaml

from iscdc.config import PROJECT_ROOT, Settings


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=tmp_path / "catalog.db",
        data_root=tmp_path / "datasets",
        templates_dir=PROJECT_ROOT / "assets" / "templates",
        static_dir=PROJECT_ROOT / "assets" / "static",
        analytics_database_path=tmp_path / "analytics.db",
        analytics_enabled=True,
        analytics_retention_days=30,
        analytics_cookie_secure=False,
    )


@pytest.fixture
def metadata_values() -> dict:
    return {
        "database": {
            "schema_version": "1.2",
            "dataset_id": "test_rna_protein",
            "dataset_type": "full",
            "source": "TEST001",
            "organism": "Homo sapiens",
            "tissue": "kidney",
            "spatial_unit": "cell",
            "coordinate_unit": "micrometer",
            "pairing_type": "same_unit",
        },
        "sample_ids": ["sample_01"],
        "modalities": {
            "rna": {"technology": "Xenium", "value_type": "counts"},
            "protein": {"technology": "Xenium", "value_type": "intensity"},
        },
        "title": "Test RNA and protein dataset",
        "description": "A deterministic spatial multi-omics test dataset.",
        "keywords": ["kidney", "test"],
        "publication": None,
    }


@pytest.fixture
def write_metadata(tmp_path: Path, metadata_values: dict):
    def writer(values: dict | None = None, name: str = "metadata.yaml") -> Path:
        path = tmp_path / name
        path.write_text(
            yaml.safe_dump(values or metadata_values, sort_keys=False), encoding="utf-8"
        )
        return path

    return writer


@pytest.fixture
def write_h5mu(tmp_path: Path, metadata_values: dict):
    def writer(
        pairing_type: str = "same_unit",
        *,
        declared_pairing_type: str | None = None,
        include_assay: bool = True,
        include_protein: bool = True,
        include_spatial: bool = True,
        missing_rna_x: bool = False,
        duplicate_rna_features: bool = False,
        second_modality_name: str = "protein",
        include_third_modality: bool = False,
        name: str = "dataset.h5mu",
    ) -> Path:
        if pairing_type == "same_unit":
            rna_obs = ["cell_1", "cell_2"]
            protein_obs = ["cell_1", "cell_2"]
        elif pairing_type == "partially_shared":
            rna_obs = ["cell_1", "cell_2"]
            protein_obs = ["cell_2", "cell_3"]
        elif pairing_type == "unpaired":
            rna_obs = ["cell_1", "cell_2"]
            protein_obs = ["cell_3", "cell_4"]
        else:
            raise ValueError(pairing_type)

        rna = ad.AnnData(
            X=np.array([[1, 0], [0, 2]], dtype=np.uint32),
            obs={"obs_id": rna_obs},
            var={"feature_id": ["gene_1", "gene_2"]},
        )
        rna.obs_names = rna_obs
        rna.var_names = ["gene_1", "gene_1"] if duplicate_rna_features else ["gene_1", "gene_2"]
        if missing_rna_x:
            rna.X = None
        protein = ad.AnnData(
            X=np.array([[1], [3]], dtype=np.uint32),
            obs={"obs_id": protein_obs},
            var={"feature_id": ["CD3"]},
        )
        protein.obs_names = protein_obs
        protein.var_names = ["CD3"]
        if include_assay:
            rna.uns["assay"] = {"technology": "Xenium", "value_type": "counts"}
            protein.uns["assay"] = {
                "technology": "Xenium",
                "value_type": "intensity",
            }

        modalities = {"rna": rna}
        if include_protein:
            modalities[second_modality_name] = protein
        if include_third_modality:
            if pairing_type == "same_unit":
                metabolite_obs = ["cell_1", "cell_2"]
            elif pairing_type == "partially_shared":
                metabolite_obs = ["cell_3", "cell_4"]
            else:
                metabolite_obs = ["cell_5", "cell_6"]
            metabolite = ad.AnnData(
                X=np.arange(1, len(metabolite_obs) + 1, dtype=np.uint32)[:, None],
                obs={"obs_id": metabolite_obs},
                var={"feature_id": ["metabolite_1"]},
            )
            metabolite.obs_names = metabolite_obs
            metabolite.var_names = ["metabolite_1"]
            if include_assay:
                metabolite.uns["assay"] = {
                    "technology": "Xenium",
                    "value_type": "intensity",
                }
            modalities["metabolite"] = metabolite
        mdata = md.MuData(modalities)
        mdata.obs["sample_id"] = "sample_01"
        if include_spatial:
            mdata.obsm["spatial"] = np.arange(mdata.n_obs * 2, dtype=np.float32).reshape(-1, 2)
        database = deepcopy(metadata_values["database"])
        database["pairing_type"] = declared_pairing_type or pairing_type
        mdata.uns["database"] = database
        path = tmp_path / name
        mdata.write_h5mu(path)
        return path

    return writer
