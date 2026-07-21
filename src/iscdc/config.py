from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configured_path(variable: str, default: Path, project_root: Path) -> Path:
    value = os.getenv(variable)
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    database_path: Path
    data_root: Path
    templates_dir: Path
    static_dir: Path

    @classmethod
    def from_environment(cls, project_root: Path = PROJECT_ROOT) -> Settings:
        project_root = project_root.resolve()
        return cls(
            database_path=_configured_path(
                "ISCDC_DATABASE_PATH", project_root / "data" / "catalog.db", project_root
            ),
            data_root=_configured_path(
                "ISCDC_DATA_ROOT", project_root / "data" / "datasets", project_root
            ),
            templates_dir=project_root / "assets" / "templates",
            static_dir=project_root / "assets" / "static",
        )
