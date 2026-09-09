"""Presentation names, independent of persistent catalogue and file identities."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import yaml

from .config import PROJECT_ROOT
from .models import Dataset
from .schemas import SAFE_IDENTIFIER_PATTERN

logger = logging.getLogger(__name__)
DEFAULT_ENTRY_NAMES_PATH = PROJECT_ROOT / "assets" / "database_entry_names.yaml"


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):  # noqa: ANN001, ANN202
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate entry ID")
        return super().construct_mapping(node, deep=deep)


def load_entry_names(path: Path = DEFAULT_ENTRY_NAMES_PATH) -> dict[str, str]:
    """Load valid rows; unavailable or malformed configuration fails open."""
    try:
        values = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        if not isinstance(values, dict):
            raise ValueError("expected an entry ID to display name mapping")
    except (OSError, UnicodeError, yaml.YAMLError, ValueError, TypeError) as exc:
        logger.warning("Entry names unavailable at %s; using fallback names: %s", path, exc)
        return {}
    names = {}
    for entry_id, name in values.items():
        if (
            not isinstance(entry_id, str)
            or not SAFE_IDENTIFIER_PATTERN.fullmatch(entry_id)
            or len(entry_id) > 128
            or not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            or len(name) > 300
            or any(ord(character) < 32 for character in name)
        ):
            logger.warning("Ignoring invalid Entry name for %r", entry_id)
            continue
        names[entry_id] = name
    return names


def fallback_entry_name(datasets: Sequence[Dataset]) -> str:
    if len(datasets) == 1:
        return datasets[0].title
    technologies: set[str] = set()
    organisms: set[str] = set()
    tissues: set[str] = set()
    for dataset in datasets:
        for target, value in [(organisms, dataset.organism), (tissues, dataset.tissue)]:
            target.update(value if isinstance(value, list) else [value])
        for modality in dataset.modalities:
            value = modality.technology
            technologies.update(value if isinstance(value, list) else [value])
    return " — ".join(
        ", ".join(sorted(values)) for values in (technologies, organisms, tissues) if values
    )


def resolve_entry_names(
    datasets: Sequence[Dataset], path: Path = DEFAULT_ENTRY_NAMES_PATH
) -> dict[str, str]:
    """Resolve names from complete full-file groups once at application startup."""
    configured = load_entry_names(path)
    grouped: dict[str, list[Dataset]] = defaultdict(list)
    for dataset in datasets:
        if dataset.dataset_type == "full":
            grouped[dataset.entry_id].append(dataset)
    resolved = {}
    for entry_id, members in grouped.items():
        if entry_id not in configured:
            logger.warning("No configured name for Entry %s; using fallback name", entry_id)
        resolved[entry_id] = configured.get(entry_id) or fallback_entry_name(members)
    return resolved
