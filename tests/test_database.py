from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from iscdc.database import (
    CatalogueSchemaError,
    create_database_engine,
    initialize_database,
)


def test_empty_legacy_catalogue_is_rebuilt(tmp_path):
    engine = create_database_engine(tmp_path / "legacy-empty.db")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE datasets (dataset_id VARCHAR(128) PRIMARY KEY)"))

    initialize_database(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("datasets")}
    assert {"dataset_type", "derivation", "split_id"}.issubset(columns)
    assert "catalogue_metadata" in inspect(engine).get_table_names()
    engine.dispose()


def test_nonempty_legacy_catalogue_requires_explicit_reimport(tmp_path):
    engine = create_database_engine(tmp_path / "legacy-data.db")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE datasets (dataset_id VARCHAR(128) PRIMARY KEY)"))
        connection.execute(text("INSERT INTO datasets VALUES ('legacy')"))

    with pytest.raises(CatalogueSchemaError, match="contains data"):
        initialize_database(engine)
    engine.dispose()


def test_catalogue_version_two_requires_schema_migration(tmp_path):
    engine = create_database_engine(tmp_path / "catalogue-v2.db")
    initialize_database(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE catalogue_metadata SET value = '2' "
                "WHERE key = 'schema_version'"
            )
        )

    with pytest.raises(CatalogueSchemaError, match="Unsupported catalogue schema version"):
        initialize_database(engine)
    engine.dispose()
