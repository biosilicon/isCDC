from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event, func, inspect, select
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, CatalogueMetadata, Dataset

CATALOGUE_SCHEMA_VERSION = "4"


class CatalogueSchemaError(RuntimeError):
    """Raised when a non-empty legacy catalogue cannot be upgraded safely."""


def create_database_engine(database_path: Path) -> Engine:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def initialize_database(engine: Engine) -> None:
    inspector = inspect(engine)
    if "datasets" in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns("datasets")}
        required_columns = {"dataset_type", "derivation", "split_id"}
        if not required_columns.issubset(columns):
            with engine.connect() as connection:
                dataset_count = connection.scalar(
                    select(func.count()).select_from(Dataset.__table__)
                )
            if dataset_count:
                raise CatalogueSchemaError(
                    "The catalogue uses the legacy schema and contains data. Back it up, remove "
                    "the old catalogue, and re-import schema 1.2 datasets."
                )
            Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        version = connection.scalar(
            select(CatalogueMetadata.value).where(CatalogueMetadata.key == "schema_version")
        )
        if version is None:
            connection.execute(
                CatalogueMetadata.__table__.insert().values(
                    key="schema_version", value=CATALOGUE_SCHEMA_VERSION
                )
            )
        elif version != CATALOGUE_SCHEMA_VERSION:
            raise CatalogueSchemaError(
                f"Unsupported catalogue schema version {version!r}; "
                f"expected {CATALOGUE_SCHEMA_VERSION!r}."
            )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
