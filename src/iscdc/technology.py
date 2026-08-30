from __future__ import annotations

from collections.abc import Iterable

TECHNOLOGIES: tuple[str, ...] = (
    "Immunofluorescence",
    "MISAR-seq",
    "SPOTS",
    "STARmap PLUS",
    "Spatial ATAC-RNA-seq",
    "Spatial CUT&Tag-RNA-seq",
    "Spatial-CITE-seq",
    "Stereo-CITE-seq",
    "Visium CytAssist",
    "Xenium",
)
TECHNOLOGY_SET = frozenset(TECHNOLOGIES)


def unsupported_technologies(value: str | Iterable[str]) -> list[str]:
    """Return unique technology values outside the catalogue vocabulary."""
    values = [value] if isinstance(value, str) else value
    return list(dict.fromkeys(item for item in values if item not in TECHNOLOGY_SET))


def technology_vocabulary_message(values: Iterable[str]) -> str:
    unsupported = ", ".join(repr(value) for value in values)
    supported = ", ".join(TECHNOLOGIES)
    return f"unsupported technology value(s): {unsupported}. Allowed values: {supported}."
