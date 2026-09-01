from __future__ import annotations

from collections.abc import Iterable

TECHNOLOGIES: tuple[str, ...] = (
    "CODEX",
    "CosMx SMI",
    "DBiT-seq",
    "DBiTplus",
    "DESI-MSI",
    "GeoMx DSP",
    "Immunofluorescence",
    "MISAR-seq",
    "RIBOmap",
    "SPOTS",
    "STARmap",
    "STARmap PLUS",
    "Spatial metatranscriptomics",
    "Spatial ATAC-RNA-seq",
    "Spatial CUT&Tag-RNA-seq",
    "Spatial VDJ",
    "Spatial-CITE-seq",
    "Spatial-DMT",
    "Stereo-CITE-seq",
    "Visium CytAssist",
    "Visium",
    "Xenium",
    "circVDJ-seq",
    "microSTRS",
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
