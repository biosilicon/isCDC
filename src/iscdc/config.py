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


def _configured_bool(variable: str, default: bool) -> bool:
    value = os.getenv(variable)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{variable} must be a boolean value")


def _configured_positive_int(variable: str, default: int) -> int:
    value = os.getenv(variable)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{variable} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{variable} must be a positive integer")
    return parsed


@dataclass(frozen=True)
class Settings:
    database_path: Path
    data_root: Path
    templates_dir: Path
    static_dir: Path
    analytics_database_path: Path
    analytics_enabled: bool
    analytics_retention_days: int
    analytics_cookie_secure: bool

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
            analytics_database_path=_configured_path(
                "ISCDC_ANALYTICS_DATABASE_PATH",
                project_root / "data" / "analytics.db",
                project_root,
            ),
            analytics_enabled=_configured_bool("ISCDC_ANALYTICS_ENABLED", True),
            analytics_retention_days=_configured_positive_int(
                "ISCDC_ANALYTICS_RETENTION_DAYS", 30
            ),
            analytics_cookie_secure=_configured_bool(
                "ISCDC_ANALYTICS_COOKIE_SECURE", False
            ),
        )
