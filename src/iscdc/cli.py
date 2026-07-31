from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TextIO

from sqlalchemy.exc import SQLAlchemyError

from .analytics import AnalyticsSchemaError, create_analytics_service
from .config import Settings
from .database import CatalogueSchemaError
from .importer import DatasetImportError, import_dataset
from .schemas import MetadataLoadError

ANALYTICS_EXPORT_FIELDS = (
    "id",
    "occurred_at",
    "session_id",
    "event_type",
    "route_name",
    "path",
    "details",
    "ip_address",
    "user_agent",
    "referrer",
    "status_code",
    "duration_ms",
    "automated",
)


def _date_argument(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a date in YYYY-MM-DD format") from exc


def _add_date_range(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--from", dest="start", type=_date_argument)
    parser.add_argument("--to", dest="end", type=_date_argument)


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

    analytics_parser = subparsers.add_parser(
        "analytics", help="Summarize or export visitor analytics."
    )
    analytics_subparsers = analytics_parser.add_subparsers(
        dest="analytics_command", required=True
    )
    summary_parser = analytics_subparsers.add_parser(
        "summary", help="Print daily and aggregate analytics as JSON."
    )
    _add_date_range(summary_parser)
    export_parser = analytics_subparsers.add_parser(
        "export", help="Export retained event details, including client IP addresses."
    )
    _add_date_range(export_parser)
    export_parser.add_argument("--format", choices=("csv", "jsonl"), required=True)
    export_parser.add_argument("--output", type=Path)
    export_parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing output file."
    )
    return parser


def _analytics_dates(args: argparse.Namespace) -> tuple[date, date]:
    today = datetime.now(UTC).date()
    end = args.end or today
    start = args.start or (end - timedelta(days=29))
    return start, end


def _export_analytics(
    records, output: TextIO, export_format: str  # noqa: ANN001
) -> None:
    if export_format == "csv":
        writer = csv.DictWriter(output, fieldnames=ANALYTICS_EXPORT_FIELDS)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["details"] = json.dumps(row["details"], ensure_ascii=False, separators=(",", ":"))
            writer.writerow(row)
        return
    for record in records:
        output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _run_analytics(args: argparse.Namespace) -> int:
    start, end = _analytics_dates(args)
    if start > end:
        print("--from must not be later than --to", file=sys.stderr)
        return 2
    settings = Settings.from_environment()
    try:
        service = create_analytics_service(
            settings.analytics_database_path, settings.analytics_retention_days
        )
    except (AnalyticsSchemaError, OSError, SQLAlchemyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        if args.analytics_command == "summary":
            print(json.dumps(service.summary(start, end), ensure_ascii=False, indent=2))
            return 0

        try:
            output_context = nullcontext(sys.stdout)
            if args.output is not None:
                mode = "w" if args.force else "x"
                output_context = args.output.open(mode, encoding="utf-8", newline="")
            with output_context as output:
                _export_analytics(service.event_records(start, end), output, args.format)
        except FileExistsError:
            print(
                f"output file already exists: {args.output}; use --force to overwrite it",
                file=sys.stderr,
            )
            return 1
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0
    except (SQLAlchemyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        service.engine.dispose()


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
    if args.command == "analytics":
        return _run_analytics(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
