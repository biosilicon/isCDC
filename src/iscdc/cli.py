from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .database import CatalogueSchemaError
from .importer import DatasetImportError, import_dataset
from .schemas import MetadataLoadError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iscdc", description="Manage the isCDC dataset catalogue."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    import_parser = subparsers.add_parser(
        "import-dataset", help="Validate and atomically import a dataset."
    )
    import_parser.add_argument("h5mu", type=Path)
    import_parser.add_argument("metadata", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "import-dataset":
        try:
            result = import_dataset(args.h5mu, args.metadata, Settings.from_environment())
        except (CatalogueSchemaError, DatasetImportError, MetadataLoadError) as exc:
            print(str(exc), file=sys.stderr)
            if isinstance(exc, DatasetImportError) and exc.report:
                print(json.dumps(exc.report, ensure_ascii=False, indent=2), file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "dataset_id": result.dataset_id,
                    "destination": str(result.destination),
                    "file_size": result.file_size,
                    "sha256": result.sha256,
                    "warning_count": result.warning_count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
