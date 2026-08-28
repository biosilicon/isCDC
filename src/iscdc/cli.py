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
from .auxiliary import AuxiliaryFileError, register_auxiliary_file
from .catalogue_migration import (
    CatalogueV4Inventory,
    CatalogueV4MigrationError,
    CatalogueV4MigrationResult,
    finalize_catalogue_v4_migration,
    migrate_catalogue_v4,
)
from .config import Settings
from .database import CatalogueSchemaError
from .importer import DatasetImportError, import_dataset
from .schema_migration import (
    MigrationInventory,
    MigrationResult,
    SchemaMigrationError,
    finalize_schema_1_2_migration,
    migrate_schema_1_2,
)
from .schemas import MetadataLoadError
from .thumbnails import (
    ThumbnailGenerationError,
    database_has_he_wsi,
    generate_wsi_thumbnail,
    list_database_ids,
)

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
    import_parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Atomically replace the indexed dataset while preserving its identity and "
            "auxiliaries."
        ),
    )

    auxiliary_parser = subparsers.add_parser(
        "add-auxiliary-file",
        help="Copy and register an auxiliary file for an indexed dataset.",
    )
    auxiliary_parser.add_argument("dataset_id")
    auxiliary_parser.add_argument("file", type=Path)
    auxiliary_parser.add_argument("--id", dest="auxiliary_id", required=True)
    auxiliary_parser.add_argument("--label", required=True)
    auxiliary_parser.add_argument("--source-url", required=True)
    auxiliary_parser.add_argument("--media-type", required=True)

    thumbnail_parser = subparsers.add_parser(
        "generate-wsi-thumbnails",
        help="Generate local Database WebP thumbnails from registered H&E WSI files.",
    )
    thumbnail_scope = thumbnail_parser.add_mutually_exclusive_group(required=True)
    thumbnail_scope.add_argument("dataset_id", nargs="?")
    thumbnail_scope.add_argument(
        "--all",
        action="store_true",
        help="Generate thumbnails for every Database with a registered he_wsi file.",
    )
    thumbnail_parser.add_argument(
        "--force", action="store_true", help="Atomically replace existing thumbnails."
    )

    difficulty_parser = subparsers.add_parser(
        "evaluate-challenge-difficulty",
        help="Evaluate and rank Challenge train-test separability offline.",
    )
    difficulty_parser.add_argument(
        "--input-modality",
        default="rna",
        help="Use one input modality consistently across the generated ranking (default: rna).",
    )
    difficulty_parser.add_argument(
        "--seed", type=int, default=42, help="Base seed for deterministic repeated evaluation."
    )
    difficulty_parser.add_argument(
        "--output",
        type=Path,
        help="JSON destination (default: beside catalog.db as challenge_difficulty.json).",
    )
    difficulty_parser.add_argument(
        "--force", action="store_true", help="Atomically replace an existing report."
    )

    reference_parser = subparsers.add_parser(
        "build-cell-type-reference",
        help="Build a frozen SingleR/RCTD reference in the annotation environment.",
    )
    reference_parser.add_argument("reference_id")

    visualization_parser = subparsers.add_parser(
        "generate-cell-type-visualization",
        help="Generate and atomically publish one offline cell type visualization.",
    )
    visualization_parser.add_argument("dataset_id")
    visualization_parser.add_argument(
        "--force", action="store_true", help="Replace the current successful generation."
    )

    annotation_audit_parser = subparsers.add_parser(
        "audit-cell-type-visualizations",
        help="Audit configured visualization artifacts and run eligible annotation jobs.",
    )
    annotation_audit_parser.add_argument("dataset_ids", nargs="*")
    annotation_audit_parser.add_argument(
        "--all", action="store_true", help="Audit the complete configured Database catalogue."
    )
    annotation_audit_parser.add_argument(
        "--jobs", type=int, default=1, help="Maximum concurrent dataset jobs (default: 1)."
    )

    catalogue_v4_parser = subparsers.add_parser(
        "migrate-catalogue-v4",
        help="Back up the catalogue and remove dataset License metadata.",
    )
    catalogue_v4_parser.add_argument("--dry-run", action="store_true")
    catalogue_v4_parser.add_argument("--temp-root", type=Path)
    catalogue_v4_parser.add_argument("--exp-root", type=Path)

    finalize_catalogue_v4_parser = subparsers.add_parser(
        "finalize-catalogue-v4",
        help="Verify a catalogue v4 migration and delete its v3 backup.",
    )
    finalize_catalogue_v4_parser.add_argument("report", type=Path)

    migration_parser = subparsers.add_parser(
        "migrate-schema-1-2",
        help="Stage, validate, back up, and activate the schema 1.2 catalogue.",
    )
    migration_parser.add_argument("--dry-run", action="store_true")
    migration_parser.add_argument("--exp-root", type=Path)

    finalize_parser = subparsers.add_parser(
        "finalize-schema-1-2",
        help="Verify a migrated catalogue and delete its schema 1.1 backup.",
    )
    finalize_parser.add_argument("report", type=Path)

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


def _run_wsi_thumbnail_generation(args: argparse.Namespace) -> int:
    settings = Settings.from_environment()
    generated: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    skipped_without_wsi = 0

    try:
        dataset_ids = list_database_ids(settings) if args.all else (args.dataset_id,)
    except (CatalogueSchemaError, OSError, SQLAlchemyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for dataset_id in dataset_ids:
        assert dataset_id is not None
        if args.all:
            try:
                if not database_has_he_wsi(dataset_id, settings):
                    skipped_without_wsi += 1
                    continue
            except (
                CatalogueSchemaError,
                OSError,
                SQLAlchemyError,
                ThumbnailGenerationError,
            ) as exc:
                failures.append({"dataset_id": dataset_id, "error": str(exc)})
                continue
        try:
            result = generate_wsi_thumbnail(dataset_id, settings, force=args.force)
        except (CatalogueSchemaError, OSError, SQLAlchemyError, ThumbnailGenerationError) as exc:
            failures.append({"dataset_id": dataset_id, "error": str(exc)})
            continue
        generated.append(result.as_dict())

    print(
        json.dumps(
            {
                "generated": generated,
                "skipped_without_wsi": skipped_without_wsi,
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if generated and not failures else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "import-dataset":
        try:
            result = import_dataset(
                args.h5mu,
                args.metadata,
                Settings.from_environment(),
                replace=args.replace,
            )
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
    if args.command == "add-auxiliary-file":
        try:
            result = register_auxiliary_file(
                args.dataset_id,
                args.file,
                Settings.from_environment(),
                auxiliary_id=args.auxiliary_id,
                label=args.label,
                source_url=args.source_url,
                media_type=args.media_type,
            )
        except (AuxiliaryFileError, CatalogueSchemaError, OSError, SQLAlchemyError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "dataset_id": result.dataset_id,
                    "auxiliary_id": result.auxiliary_id,
                    "destination": str(result.destination),
                    "size": result.size,
                    "sha256": result.sha256,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "generate-wsi-thumbnails":
        return _run_wsi_thumbnail_generation(args)
    if args.command == "evaluate-challenge-difficulty":
        try:
            from .difficulty import (
                DifficultyConfig,
                DifficultyEvaluationError,
                evaluate_and_write,
            )
        except ModuleNotFoundError:
            print(
                "challenge difficulty evaluation requires optional analysis dependencies; "
                "install requirements-difficulty.txt",
                file=sys.stderr,
            )
            return 1
        settings = Settings.from_environment()
        output = args.output or settings.database_path.parent / "challenge_difficulty.json"
        try:
            report = evaluate_and_write(
                settings,
                output,
                config=DifficultyConfig(input_modality=args.input_modality, seed=args.seed),
                force=args.force,
            )
        except (DifficultyEvaluationError, OSError, SQLAlchemyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "output": str(output.expanduser().resolve()),
                    "challenge_count": report["challenge_count"],
                    "success_count": report["success_count"],
                    "failure_count": report["failure_count"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if report["failure_count"] == 0 else 1
    if args.command in {
        "build-cell-type-reference",
        "generate-cell-type-visualization",
        "audit-cell-type-visualizations",
    }:
        try:
            from .cell_type_annotation import (
                CellTypeAnnotationError,
                audit_cell_type_visualizations,
                build_cell_type_reference,
                generate_cell_type_visualization,
            )
        except ModuleNotFoundError:
            print(
                "cell type annotation tooling is unavailable; run this command through "
                "conda run -n iscdc-cell-annotation",
                file=sys.stderr,
            )
            return 1
        settings = Settings.from_environment()
        try:
            if args.command == "build-cell-type-reference":
                payload = build_cell_type_reference(args.reference_id, settings)
            elif args.command == "generate-cell-type-visualization":
                payload = generate_cell_type_visualization(
                    args.dataset_id, settings, force=args.force
                )
            else:
                if args.jobs < 1:
                    print("--jobs must be a positive integer", file=sys.stderr)
                    return 2
                if args.all == bool(args.dataset_ids):
                    print(
                        "choose exactly one of --all or one or more DATASET_ID values",
                        file=sys.stderr,
                    )
                    return 2
                payload = audit_cell_type_visualizations(
                    None if args.all else args.dataset_ids,
                    settings,
                    all_datasets=args.all,
                    jobs=args.jobs,
                )
        except (CellTypeAnnotationError, OSError, SQLAlchemyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if hasattr(payload, "as_dict"):
            payload = payload.as_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if isinstance(payload, dict) and payload.get("failure_count", 0):
            return 1
        return 0
    if args.command == "migrate-catalogue-v4":
        settings = Settings.from_environment()
        options = {"dry_run": args.dry_run}
        if args.temp_root is not None:
            options["temp_root"] = args.temp_root
        if args.exp_root is not None:
            options["exp_root"] = args.exp_root
        try:
            result = migrate_catalogue_v4(settings, **options)
        except (CatalogueV4MigrationError, OSError, SQLAlchemyError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if isinstance(result, CatalogueV4Inventory):
            payload = {"dry_run": True, "inventory": result.as_dict()}
        else:
            assert isinstance(result, CatalogueV4MigrationResult)
            payload = {
                "dry_run": False,
                "inventory": result.inventory.as_dict(),
                "report": str(result.report_path),
                "backup": str(result.backup_path),
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "finalize-catalogue-v4":
        try:
            removed = finalize_catalogue_v4_migration(
                args.report, Settings.from_environment()
            )
        except (CatalogueV4MigrationError, OSError, SQLAlchemyError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps({"deleted_backup": str(removed)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "migrate-schema-1-2":
        settings = Settings.from_environment()
        options = {"dry_run": args.dry_run}
        if args.exp_root is not None:
            options["exp_root"] = args.exp_root
        try:
            result = migrate_schema_1_2(settings, **options)
        except (OSError, SQLAlchemyError, SchemaMigrationError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if isinstance(result, MigrationInventory):
            payload = {"dry_run": True, "inventory": result.as_dict()}
        else:
            assert isinstance(result, MigrationResult)
            payload = {
                "dry_run": False,
                "inventory": result.inventory.as_dict(),
                "report": str(result.report_path),
                "backup": str(result.backup_path),
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "finalize-schema-1-2":
        try:
            removed = finalize_schema_1_2_migration(
                args.report, Settings.from_environment()
            )
        except (OSError, SQLAlchemyError, SchemaMigrationError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps({"deleted_backup": str(removed)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "analytics":
        return _run_analytics(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
