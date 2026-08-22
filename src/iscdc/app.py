import hashlib
import logging
import math
import mimetypes
from collections.abc import AsyncGenerator
from http.cookies import SimpleCookie
from pathlib import Path
from time import perf_counter
from typing import Annotated
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BeforeValidator
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.requests import Request as StarletteRequest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .analytics import (
    SESSION_COOKIE_NAME,
    AnalyticsService,
    create_analytics_service,
    is_automated_user_agent,
)
from .auxiliary import AuxiliaryFile, AuxiliaryFileError, load_auxiliary_files
from .cell_type_visualization import (
    POINT_MEDIA_TYPE,
    CellTypeVisualization,
    load_cell_type_visualizations,
)
from .config import Settings
from .database import create_database_engine, create_session_factory, initialize_database
from .difficulty_snapshot import (
    ChallengeDifficulty,
    DifficultySnapshotError,
    load_difficulty_snapshot,
)
from .models import Dataset
from .repository import (
    DERIVED_DATASET_TYPES,
    CatalogueFilters,
    CatalogueIntegrityError,
    Challenge,
    ChallengeSort,
    SampleSource,
    count_challenges,
    count_databases,
    get_challenge,
    get_database,
    get_facets,
    list_challenges,
    list_databases,
    resolve_sample_sources,
)
from .schemas import (
    ChallengeDifficultyResponse,
    ChallengeListResponse,
    ChallengeResponse,
    ChallengeType,
    DatabaseListResponse,
    DataFileResponse,
    SampleSourceResponse,
)

CHALLENGE_TYPE_LABELS = {
    "same_slice": "Same slice",
    "cross_slice_same_subject": "Cross-slice, same subject",
    "cross_subject": "Cross-subject (including biological replicates)",
}

OptionalChallengeTypeQuery = Annotated[
    ChallengeType | None,
    BeforeValidator(lambda value: None if value == "" else value),
]

DOWNLOAD_FILES = {
    "h5mu": ("dataset.h5mu", "application/x-hdf5", ".h5mu"),
    "metadata": ("metadata.yaml", "application/yaml", "_metadata.yaml"),
    "manifest": ("manifest.json", "application/json", "_manifest.json"),
    "validation": (
        "validation_report.json",
        "application/json",
        "_validation_report.json",
    ),
    "checksum": ("checksum.sha256", "text/plain", "_checksum.sha256"),
}

logger = logging.getLogger(__name__)

DATABASE_THUMBNAIL_DIRECTORY = "database_thumbnails"


def _preferred_content_encoding(
    value: str | None, available: set[str]
) -> str | None:
    """Select a validated representation without deriving a filesystem path."""
    if value is None:
        return "identity" if "identity" in available else None
    qualities: dict[str, float] = {}
    for item in value.split(","):
        parts = [part.strip() for part in item.split(";")]
        encoding = parts[0].lower()
        quality = 1.0
        for parameter in parts[1:]:
            if parameter.lower().startswith("q="):
                try:
                    quality = float(parameter[2:])
                except ValueError:
                    quality = 0.0
        if 0 <= quality <= 1:
            qualities[encoding] = quality
    candidates: list[tuple[float, int, str]] = []
    for priority, encoding in enumerate(("br", "gzip", "identity"), start=1):
        if encoding not in available:
            continue
        if encoding in qualities:
            quality = qualities[encoding]
        elif encoding == "identity":
            quality = qualities.get("*", 1.0)
        else:
            quality = qualities.get("*", 0.0)
        if quality > 0:
            candidates.append((quality, -priority, encoding))
    return max(candidates)[2] if candidates else None


def _static_asset_version(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except OSError:
        return ""


def _cell_type_method_details(
    visualization: CellTypeVisualization,
) -> dict[str, object]:
    details: dict[str, object] = {
        "kind": visualization.annotation_kind,
        "method": visualization.method,
    }
    if visualization.annotation_kind == "source":
        return details

    references = visualization.provenance["references"]
    parameters = visualization.provenance["parameters"]
    thresholds = visualization.report["thresholds"]
    quality_control = visualization.report["quality_control"]
    details.update(
        {
            "references": tuple(
                {"id": reference["id"], "version": reference["version"]}
                for reference in references
            ),
            "parameters": tuple(sorted(parameters.items())),
            "thresholds": tuple(sorted(thresholds.items())),
            "quality_control": tuple(sorted(quality_control.items())),
            "qc_status": visualization.report["status"],
        }
    )
    return details


def _discover_database_thumbnails(static_dir: Path) -> dict[str, str]:
    thumbnail_dir = static_dir / DATABASE_THUMBNAIL_DIRECTORY
    if not thumbnail_dir.is_dir():
        return {}
    return {
        path.stem: f"{DATABASE_THUMBNAIL_DIRECTORY}/{path.name}"
        for path in sorted(thumbnail_dir.glob("*.webp"))
        if path.is_file()
    }


class AnalyticsMiddleware:
    def __init__(
        self, app: ASGIApp, analytics: AnalyticsService, cookie_secure: bool
    ) -> None:
        self.app = app
        self.analytics = analytics
        self.cookie_secure = cookie_secure

    def _cookie_header(self, session_id: str) -> bytes:
        cookie = SimpleCookie()
        cookie[SESSION_COOKIE_NAME] = session_id
        cookie[SESSION_COOKIE_NAME]["httponly"] = True
        cookie[SESSION_COOKIE_NAME]["samesite"] = "lax"
        cookie[SESSION_COOKIE_NAME]["path"] = "/"
        if self.cookie_secure:
            cookie[SESSION_COOKIE_NAME]["secure"] = True
        return cookie.output(header="").strip().encode("latin-1")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started_at = perf_counter()
        request = StarletteRequest(scope)

        async def send_with_analytics(message: Message) -> None:
            if message["type"] == "http.response.start":
                event_context = scope.get("state", {}).get("analytics_event")
                status_code = message["status"]
                if event_context is not None and status_code < 400:
                    if event_context["set_cookie"]:
                        message["headers"] = list(message.get("headers", []))
                        message["headers"].append(
                            (
                                b"set-cookie",
                                self._cookie_header(event_context["session_id"]),
                            )
                        )
                    try:
                        self.analytics.record_event(
                            session_id=event_context["session_id"],
                            event_type=event_context["event_type"],
                            route_name=event_context["route_name"],
                            path=request.url.path,
                            details=event_context["details"],
                            ip_address=request.client.host if request.client else None,
                            user_agent=event_context["user_agent"],
                            referrer=request.headers.get("referer"),
                            status_code=status_code,
                            duration_ms=round((perf_counter() - started_at) * 1000),
                            automated=event_context["automated"],
                        )
                    except Exception:  # analytics must not alter the successful response
                        logger.exception("Analytics event write failed")
            await send(message)

        await self.app(scope, receive, send_with_analytics)


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _as_list(value):  # noqa: ANN001, ANN202
    return value if isinstance(value, list) else [value]


def _format_metadata(value):  # noqa: ANN001, ANN202
    return ", ".join(map(str, _as_list(value)))


def _bounded_analytics_value(value):  # noqa: ANN001, ANN202
    if isinstance(value, dict):
        return {
            str(key)[:100]: _bounded_analytics_value(item)
            for key, item in list(value.items())[:20]
        }
    if isinstance(value, list):
        return [_bounded_analytics_value(item) for item in value[:20]]
    if isinstance(value, str):
        return value[:500]
    return value


def _filters(
    q: str | None,
    organism: str | None,
    tissue: str | None,
    modality: str | None,
    technology: str | None,
    spatial_unit: str | None,
    challenge_type: ChallengeType | None = None,
) -> CatalogueFilters:
    return CatalogueFilters(
        q, organism, tissue, modality, technology, spatial_unit, challenge_type
    )


def _selected_filters(
    q: str | None,
    organism: str | None,
    tissue: str | None,
    modality: str | None,
    technology: str | None,
    spatial_unit: str | None,
    challenge_type: ChallengeType | None = None,
) -> dict[str, str]:
    return {
        key: value
        for key, value in {
            "q": q,
            "organism": organism,
            "tissue": tissue,
            "modality": modality,
            "technology": technology,
            "spatial_unit": spatial_unit,
            "challenge_type": challenge_type,
        }.items()
        if value
    }


def _pagination_context(
    request: Request, selected: dict[str, str], page: int, total: int, per_page: int
) -> dict[str, object]:
    page_count = max(1, math.ceil(total / per_page))

    def page_url(number: int) -> str:
        return f"{request.url.path}?{urlencode({**selected, 'page': number})}"

    return {
        "page": page,
        "page_count": page_count,
        "previous_url": page_url(page - 1) if page > 1 else None,
        "next_url": page_url(page + 1) if page < page_count else None,
    }


def _data_file_response(
    dataset: Dataset,
    request: Request,
    auxiliary_files_by_dataset: dict[str, tuple[AuxiliaryFile, ...]],
    sample_sources_by_dataset_id: dict[str, list[SampleSource]],
) -> DataFileResponse:
    downloads = {
        kind: str(request.url_for("download_file", dataset_id=dataset.dataset_id, kind=kind))
        for kind in DOWNLOAD_FILES
    }
    return DataFileResponse(
        dataset_id=dataset.dataset_id,
        schema_version=dataset.schema_version,
        dataset_type=dataset.dataset_type,
        title=dataset.title,
        description=dataset.description,
        source=dataset.source,
        organism=dataset.organism,
        tissue=dataset.tissue,
        spatial_unit=dataset.spatial_unit,
        coordinate_unit=dataset.coordinate_unit,
        pairing_type=dataset.pairing_type,
        derivation=dataset.derivation,
        sample_ids=dataset.sample_ids,
        sample_sources=[
            SampleSourceResponse(
                sample_id=item.sample_id,
                source_sample_id=item.source_sample_id,
                source_database_id=item.source_database_id,
                source_database_title=item.source_database_title,
                source=item.source,
            )
            for item in sample_sources_by_dataset_id.get(dataset.dataset_id, ())
        ],
        keywords=dataset.keywords,
        license=dataset.license,
        publication=dataset.publication,
        additional_metadata=dataset.additional_metadata,
        n_obs=dataset.n_obs,
        coordinate_dimensions=dataset.coordinate_dimensions,
        modalities=[
            {
                "name": modality.name,
                "technology": modality.technology,
                "value_type": modality.value_type,
                "n_obs": modality.n_obs,
                "n_vars": modality.n_vars,
            }
            for modality in dataset.modalities
        ],
        modality_count=len(dataset.modalities),
        file_size=dataset.file_size,
        sha256=dataset.sha256,
        validation_warning_count=dataset.validation_warning_count,
        imported_at=dataset.imported_at,
        downloads=downloads,
        auxiliary_files=[
            {
                "id": auxiliary_file.auxiliary_id,
                "label": auxiliary_file.label,
                "filename": auxiliary_file.filename,
                "media_type": auxiliary_file.media_type,
                "size": auxiliary_file.size,
                "sha256": auxiliary_file.sha256,
                "source_url": auxiliary_file.source_url,
                "download_url": str(
                    request.url_for(
                        "download_auxiliary_file",
                        dataset_id=dataset.dataset_id,
                        auxiliary_id=auxiliary_file.auxiliary_id,
                    )
                ),
            }
            for auxiliary_file in auxiliary_files_by_dataset.get(dataset.dataset_id, ())
        ],
    )


def _challenge_response(
    challenge: Challenge,
    request: Request,
    auxiliary_files_by_dataset: dict[str, tuple[AuxiliaryFile, ...]],
    sample_sources_by_dataset_id: dict[str, list[SampleSource]],
    difficulty: ChallengeDifficulty | None,
) -> ChallengeResponse:
    return ChallengeResponse(
        split_id=challenge.split_id,
        challenge_type=challenge.challenge_type,
        status=challenge.status,
        difficulty=(
            ChallengeDifficultyResponse(
                mean_auroc=difficulty.mean_auroc,
                domain_shift_score=difficulty.domain_shift_score,
                difficulty_percentile=difficulty.difficulty_percentile,
            )
            if difficulty is not None
            else None
        ),
        train=(
            _data_file_response(
                challenge.train,
                request,
                auxiliary_files_by_dataset,
                sample_sources_by_dataset_id,
            )
            if challenge.train is not None
            else None
        ),
        test=(
            _data_file_response(
                challenge.test,
                request,
                auxiliary_files_by_dataset,
                sample_sources_by_dataset_id,
            )
            if challenge.test is not None
            else None
        ),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    mimetypes.add_type("image/webp", ".webp")
    engine = create_database_engine(settings.database_path)
    initialize_database(engine)
    session_factory = create_session_factory(engine)
    auxiliary_files_by_dataset: dict[str, tuple[AuxiliaryFile, ...]] = {}
    with session_factory() as session:
        catalogue_datasets = session.scalars(select(Dataset)).all()
    for dataset in catalogue_datasets:
        dataset_id = dataset.dataset_id
        storage_dir = dataset.storage_dir
        try:
            auxiliary_files_by_dataset[dataset_id] = load_auxiliary_files(
                settings.data_root / storage_dir, dataset_id
            )
        except AuxiliaryFileError:
            logger.warning(
                "Ignoring invalid auxiliary file manifest for dataset %s",
                dataset_id,
                exc_info=True,
            )

    cell_type_visualizations: dict[str, CellTypeVisualization] = {}
    if (
        settings.cell_type_visualization_root is not None
        and settings.cell_type_visualization_root.is_dir()
    ):
        try:
            cell_type_visualizations = load_cell_type_visualizations(
                settings.cell_type_visualization_root,
                (
                    {
                        "dataset_id": dataset.dataset_id,
                        "dataset_type": dataset.dataset_type,
                        "sha256": dataset.sha256,
                        "n_obs": dataset.n_obs,
                        "coordinate_dimensions": dataset.coordinate_dimensions,
                        "sample_ids": dataset.sample_ids,
                    }
                    for dataset in catalogue_datasets
                    if dataset.dataset_type == "full"
                ),
            )
        except Exception:
            logger.exception(
                "Cell type visualizations are unavailable; continuing without them"
            )

    difficulty_by_split_id: dict[str, ChallengeDifficulty] = {}
    difficulty_snapshot = None
    difficulty_path = settings.database_path.parent / "challenge_difficulty.json"
    try:
        with session_factory() as session:
            catalogue_challenges, challenge_total = list_challenges(
                session, CatalogueFilters(), offset=0, limit=1_000_000
            )
        if len(catalogue_challenges) != challenge_total:
            raise DifficultySnapshotError(
                "unable to load the complete Challenge catalogue for difficulty validation"
            )
        difficulty_snapshot = load_difficulty_snapshot(
            difficulty_path, catalogue_challenges
        )
        difficulty_by_split_id = dict(difficulty_snapshot.by_split_id)
    except (CatalogueIntegrityError, DifficultySnapshotError) as exc:
        logger.warning(
            "Challenge difficulty information is unavailable; continuing without it: %s",
            exc,
        )

    analytics: AnalyticsService | None = None
    if settings.analytics_enabled:
        try:
            analytics = create_analytics_service(
                settings.analytics_database_path, settings.analytics_retention_days
            )
        except Exception:  # analytics must not prevent the catalogue from starting
            logger.exception("Analytics initialization failed; visitor tracking is disabled")

    def analytics_template_context(_request: Request) -> dict[str, int | None]:
        if analytics is None:
            return {"visit_count": None}
        try:
            return {"visit_count": analytics.total_visits()}
        except Exception:  # analytics must not prevent page rendering
            logger.exception("Analytics counter read failed")
            return {"visit_count": None}

    templates = Jinja2Templates(
        directory=settings.templates_dir,
        context_processors=[analytics_template_context],
    )
    templates.env.filters["filesize"] = _format_bytes
    templates.env.filters["as_list"] = _as_list
    templates.env.filters["metadata_values"] = _format_metadata
    templates.env.filters["challenge_type_label"] = CHALLENGE_TYPE_LABELS.__getitem__
    templates.env.globals["static_styles_version"] = _static_asset_version(
        settings.static_dir / "styles.css"
    )
    templates.env.globals["cell_type_bundle_version"] = _static_asset_version(
        settings.static_dir / "cell_type_visualization.js"
    )
    templates.env.globals["database_thumbnail_paths"] = _discover_database_thumbnails(
        settings.static_dir
    )
    templates.env.globals["auxiliary_files_by_dataset"] = auxiliary_files_by_dataset

    application = FastAPI(
        title="isCDC Spatial Multi-omics Database",
        version="0.3.0",
        description="A read-only catalogue of databases and benchmark challenges.",
    )
    application.state.settings = settings
    application.state.engine = engine
    application.state.session_factory = session_factory
    application.state.analytics = analytics
    application.state.difficulty_snapshot = difficulty_snapshot
    application.state.difficulty_path = difficulty_path
    application.state.cell_type_visualizations = cell_type_visualizations
    application.state.templates = templates
    application.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    if analytics is not None:
        application.add_middleware(
            AnalyticsMiddleware,
            analytics=analytics,
            cookie_secure=settings.analytics_cookie_secure,
        )

    def prepare_analytics_event(
        request: Request,
        event_type: str,
        route_name: str,
        details: dict[str, object],
    ) -> None:
        if analytics is None:
            return
        user_agent = request.headers.get("user-agent")
        automated = is_automated_user_agent(user_agent)
        try:
            result = analytics.start_session(
                request.cookies.get(SESSION_COOKIE_NAME), automated=automated
            )
        except Exception:  # analytics must not prevent request handling
            logger.exception("Analytics session update failed")
            return
        request.state.analytics_event = {
            "session_id": result.session_id,
            "set_cookie": result.set_cookie,
            "event_type": event_type,
            "route_name": route_name,
            "details": _bounded_analytics_value(details),
            "automated": automated,
            "user_agent": user_agent,
        }

    async def get_session() -> AsyncGenerator[Session, None]:
        with session_factory() as session:
            yield session

    SessionDependency = Annotated[Session, Depends(get_session)]

    @application.exception_handler(CatalogueIntegrityError)
    async def catalogue_integrity_error(
        request: Request, exc: CatalogueIntegrityError
    ) -> HTMLResponse | JSONResponse:
        if request.url.path.startswith("/api/"):
            return JSONResponse(status_code=500, content={"detail": str(exc)})
        return templates.TemplateResponse(
            request=request,
            name="catalogue_error.html",
            context={"message": str(exc)},
            status_code=500,
        )

    @application.get("/healthz", name="health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/", response_class=HTMLResponse, name="home")
    async def home(request: Request, session: SessionDependency):
        database_count = count_databases(session)
        challenge_count = count_challenges(session)
        prepare_analytics_event(request, "page_view", "home", {"page": "home"})
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "database_count": database_count,
                "challenge_count": challenge_count,
            },
        )

    @application.get("/databases", response_class=HTMLResponse, name="database_list")
    async def database_list(
        request: Request,
        session: SessionDependency,
        q: str | None = None,
        organism: str | None = None,
        tissue: str | None = None,
        modality: str | None = None,
        technology: str | None = None,
        spatial_unit: str | None = None,
        page: Annotated[int, Query(ge=1)] = 1,
    ):
        filters = _filters(q, organism, tissue, modality, technology, spatial_unit)
        selected = _selected_filters(q, organism, tissue, modality, technology, spatial_unit)
        per_page = 20
        databases, total = list_databases(session, filters, (page - 1) * per_page, per_page)
        facets = get_facets(session, ("full",))
        event_type = "catalogue_search" if selected else "page_view"
        prepare_analytics_event(
            request,
            event_type,
            "database_list",
            {"catalogue": "databases", "filters": selected, "page": page},
        )
        return templates.TemplateResponse(
            request=request,
            name="databases.html",
            context={
                "databases": databases,
                "total": total,
                "facets": facets,
                "selected": selected,
                **_pagination_context(request, selected, page, total, per_page),
            },
        )

    @application.get(
        "/databases/{dataset_id}", response_class=HTMLResponse, name="database_detail"
    )
    async def database_detail(request: Request, dataset_id: str, session: SessionDependency):
        database = get_database(session, dataset_id)
        if database is None:
            return templates.TemplateResponse(
                request=request,
                name="404.html",
                context={"resource": dataset_id},
                status_code=404,
            )
        prepare_analytics_event(
            request,
            "database_detail_view",
            "database_detail",
            {"dataset_id": database.dataset_id},
        )
        visualization = cell_type_visualizations.get(database.dataset_id)
        visualization_config = None
        if visualization is not None:
            samples = [
                {
                    "key": sample.key,
                    "id": sample.id,
                    "count": sample.count,
                    "url": str(
                        request.url_for(
                            "cell_type_visualization_points",
                            dataset_id=database.dataset_id,
                            generation_id=visualization.generation_id,
                            sample_key=sample.key,
                        )
                    ),
                }
                for sample in visualization.samples.values()
            ]
            visualization_config = {
                "datasetId": database.dataset_id,
                "generationId": visualization.generation_id,
                "annotationKind": visualization.annotation_kind,
                "yAxis": visualization.y_axis,
                "categories": [
                    {
                        "code": category.type_id,
                        "label": category.label,
                        "color": category.color,
                        "count": category.count,
                        "cellOntologyId": category.cell_ontology_id,
                    }
                    for category in visualization.categories
                ],
                "samples": samples,
                "initialSampleKey": samples[0]["key"],
            }
        return templates.TemplateResponse(
            request=request,
            name="database_detail.html",
            context={
                "database": database,
                "download_kinds": DOWNLOAD_FILES,
                "cell_type_visualization": visualization_config,
                "cell_type_annotation_method": (
                    _cell_type_method_details(visualization)
                    if visualization is not None
                    else None
                ),
            },
        )

    @application.api_route(
        "/databases/{dataset_id}/cell-type-visualization/{generation_id}/{sample_key}",
        methods=["GET", "HEAD"],
        name="cell_type_visualization_points",
        include_in_schema=False,
    )
    async def cell_type_visualization_points(
        request: Request,
        dataset_id: str,
        generation_id: str,
        sample_key: str,
    ):
        visualization = cell_type_visualizations.get(dataset_id)
        if visualization is None or visualization.generation_id != generation_id:
            raise HTTPException(status_code=404, detail="Visualization not found")
        sample = visualization.samples.get(sample_key)
        if sample is None:
            raise HTTPException(status_code=404, detail="Visualization sample not found")
        encoding = _preferred_content_encoding(
            request.headers.get("accept-encoding"), set(sample.representations)
        )
        if encoding is None:
            raise HTTPException(
                status_code=406, detail="No acceptable visualization encoding"
            )
        representation = sample.resolve(encoding)
        if (
            representation.path.is_symlink()
            or not representation.path.is_file()
            or representation.path.stat().st_size != representation.size
        ):
            raise HTTPException(status_code=404, detail="Visualization file not found")
        headers = {
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": f'"{representation.sha256}"',
            "Vary": "Accept-Encoding",
            "X-Content-Type-Options": "nosniff",
        }
        if encoding != "identity":
            headers["Content-Encoding"] = encoding
        return FileResponse(
            representation.path,
            media_type=POINT_MEDIA_TYPE,
            headers=headers,
        )

    @application.get("/challenges", response_class=HTMLResponse, name="challenge_list")
    async def challenge_list(
        request: Request,
        session: SessionDependency,
        q: str | None = None,
        organism: str | None = None,
        tissue: str | None = None,
        modality: str | None = None,
        technology: str | None = None,
        spatial_unit: str | None = None,
        challenge_type: OptionalChallengeTypeQuery = None,
        sort: ChallengeSort = "newest",
        page: Annotated[int, Query(ge=1)] = 1,
    ):
        filters = _filters(
            q, organism, tissue, modality, technology, spatial_unit, challenge_type
        )
        selected = _selected_filters(
            q, organism, tissue, modality, technology, spatial_unit, challenge_type
        )
        if sort != "newest":
            selected["sort"] = sort
        per_page = 20
        challenges, total = list_challenges(
            session,
            filters,
            (page - 1) * per_page,
            per_page,
            sort=sort,
            difficulty_aurocs={
                split_id: difficulty.mean_auroc
                for split_id, difficulty in difficulty_by_split_id.items()
            },
        )
        facets = get_facets(session, DERIVED_DATASET_TYPES)
        event_type = "catalogue_search" if selected else "page_view"
        prepare_analytics_event(
            request,
            event_type,
            "challenge_list",
            {"catalogue": "challenges", "filters": selected, "page": page},
        )
        return templates.TemplateResponse(
            request=request,
            name="challenges.html",
            context={
                "challenges": challenges,
                "total": total,
                "facets": facets,
                "selected": selected,
                "selected_sort": sort,
                "difficulty_by_split_id": difficulty_by_split_id,
                "difficulty_sort_options": (
                    ("newest", "Newest"),
                    ("difficulty_asc", "Difficulty: low to high"),
                    ("difficulty_desc", "Difficulty: high to low"),
                ),
                **_pagination_context(request, selected, page, total, per_page),
            },
        )

    @application.get(
        "/challenges/{split_id:path}", response_class=HTMLResponse, name="challenge_detail"
    )
    async def challenge_detail(request: Request, split_id: str, session: SessionDependency):
        challenge = get_challenge(session, split_id)
        if challenge is None:
            return templates.TemplateResponse(
                request=request,
                name="404.html",
                context={"resource": split_id},
                status_code=404,
            )
        prepare_analytics_event(
            request,
            "challenge_detail_view",
            "challenge_detail",
            {"split_id": challenge.split_id},
        )
        sample_sources_by_dataset_id = resolve_sample_sources(
            session, challenge.datasets
        )
        return templates.TemplateResponse(
            request=request,
            name="challenge_detail.html",
            context={
                "challenge": challenge,
                "difficulty": difficulty_by_split_id.get(challenge.split_id),
                "download_kinds": DOWNLOAD_FILES,
                "sample_sources_by_dataset_id": sample_sources_by_dataset_id,
            },
        )

    @application.get("/downloads/{dataset_id}/{kind}", name="download_file")
    async def download_file(
        request: Request, dataset_id: str, kind: str, session: SessionDependency
    ):
        file_spec = DOWNLOAD_FILES.get(kind)
        if file_spec is None:
            raise HTTPException(status_code=404, detail="Unknown download type")
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Data file not found")
        stored_name, media_type, download_suffix = file_spec
        path = settings.data_root / dataset.storage_dir / stored_name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Stored file not found")

        prepare_analytics_event(
            request,
            "download",
            "download_file",
            {"dataset_id": dataset.dataset_id, "kind": kind},
        )

        async def chunks():  # noqa: ANN202
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    yield chunk

        filename = f"{dataset.dataset_id}{download_suffix}"
        return StreamingResponse(
            chunks(),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(path.stat().st_size),
            },
        )

    @application.api_route(
        "/downloads/{dataset_id}/auxiliary/{auxiliary_id}",
        methods=["GET", "HEAD"],
        name="download_auxiliary_file",
    )
    async def download_auxiliary_file(
        request: Request,
        dataset_id: str,
        auxiliary_id: str,
        session: SessionDependency,
    ):
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Data file not found")
        auxiliary_file = next(
            (
                item
                for item in auxiliary_files_by_dataset.get(dataset_id, ())
                if item.auxiliary_id == auxiliary_id
            ),
            None,
        )
        if auxiliary_file is None:
            raise HTTPException(status_code=404, detail="Auxiliary file not found")
        if (
            auxiliary_file.path.is_symlink()
            or not auxiliary_file.path.is_file()
            or auxiliary_file.path.stat().st_size != auxiliary_file.size
        ):
            raise HTTPException(status_code=404, detail="Stored auxiliary file not found")

        prepare_analytics_event(
            request,
            "download",
            "download_auxiliary_file",
            {
                "dataset_id": dataset.dataset_id,
                "kind": "auxiliary",
                "auxiliary_id": auxiliary_file.auxiliary_id,
            },
        )
        return FileResponse(
            auxiliary_file.path,
            media_type=auxiliary_file.media_type,
            filename=auxiliary_file.filename,
        )

    @application.get(
        "/api/databases", response_model=DatabaseListResponse, name="api_database_list"
    )
    async def api_database_list(
        request: Request,
        session: SessionDependency,
        q: str | None = None,
        organism: str | None = None,
        tissue: str | None = None,
        modality: str | None = None,
        technology: str | None = None,
        spatial_unit: str | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> DatabaseListResponse:
        databases, total = list_databases(
            session,
            _filters(q, organism, tissue, modality, technology, spatial_unit),
            offset,
            limit,
        )
        return DatabaseListResponse(
            items=[
                _data_file_response(database, request, auxiliary_files_by_dataset, {})
                for database in databases
            ],
            total=total,
            limit=limit,
            offset=offset,
        )

    @application.get(
        "/api/databases/{dataset_id}",
        response_model=DataFileResponse,
        name="api_database_detail",
    )
    async def api_database_detail(
        request: Request, dataset_id: str, session: SessionDependency
    ) -> DataFileResponse:
        database = get_database(session, dataset_id)
        if database is None:
            raise HTTPException(status_code=404, detail="Database not found")
        return _data_file_response(database, request, auxiliary_files_by_dataset, {})

    @application.get(
        "/api/challenges", response_model=ChallengeListResponse, name="api_challenge_list"
    )
    async def api_challenge_list(
        request: Request,
        session: SessionDependency,
        q: str | None = None,
        organism: str | None = None,
        tissue: str | None = None,
        modality: str | None = None,
        technology: str | None = None,
        spatial_unit: str | None = None,
        challenge_type: OptionalChallengeTypeQuery = None,
        sort: ChallengeSort = "newest",
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> ChallengeListResponse:
        challenges, total = list_challenges(
            session,
            _filters(q, organism, tissue, modality, technology, spatial_unit, challenge_type),
            offset,
            limit,
            sort=sort,
            difficulty_aurocs={
                split_id: difficulty.mean_auroc
                for split_id, difficulty in difficulty_by_split_id.items()
            },
        )
        sample_sources_by_dataset_id = resolve_sample_sources(
            session,
            [dataset for challenge in challenges for dataset in challenge.datasets],
        )
        return ChallengeListResponse(
            items=[
                _challenge_response(
                    challenge,
                    request,
                    auxiliary_files_by_dataset,
                    sample_sources_by_dataset_id,
                    difficulty_by_split_id.get(challenge.split_id),
                )
                for challenge in challenges
            ],
            total=total,
            limit=limit,
            offset=offset,
        )

    @application.get(
        "/api/challenges/{split_id:path}",
        response_model=ChallengeResponse,
        name="api_challenge_detail",
    )
    async def api_challenge_detail(
        request: Request, split_id: str, session: SessionDependency
    ) -> ChallengeResponse:
        challenge = get_challenge(session, split_id)
        if challenge is None:
            raise HTTPException(status_code=404, detail="Challenge not found")
        sample_sources_by_dataset_id = resolve_sample_sources(
            session, challenge.datasets
        )
        return _challenge_response(
            challenge,
            request,
            auxiliary_files_by_dataset,
            sample_sources_by_dataset_id,
            difficulty_by_split_id.get(challenge.split_id),
        )

    return application


app = create_app()
