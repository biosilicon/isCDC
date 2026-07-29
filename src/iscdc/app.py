import math
from collections.abc import AsyncGenerator
from typing import Annotated
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BeforeValidator
from sqlalchemy.orm import Session

from .config import Settings
from .database import create_database_engine, create_session_factory, initialize_database
from .models import Dataset
from .repository import (
    DERIVED_DATASET_TYPES,
    CatalogueFilters,
    CatalogueIntegrityError,
    Challenge,
    count_challenges,
    count_databases,
    get_challenge,
    get_database,
    get_facets,
    list_challenges,
    list_databases,
)
from .schemas import (
    ChallengeListResponse,
    ChallengeResponse,
    ChallengeType,
    DatabaseListResponse,
    DataFileResponse,
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


def _data_file_response(dataset: Dataset, request: Request) -> DataFileResponse:
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
        file_size=dataset.file_size,
        sha256=dataset.sha256,
        validation_warning_count=dataset.validation_warning_count,
        imported_at=dataset.imported_at,
        downloads=downloads,
    )


def _challenge_response(challenge: Challenge, request: Request) -> ChallengeResponse:
    return ChallengeResponse(
        split_id=challenge.split_id,
        challenge_type=challenge.challenge_type,
        status=challenge.status,
        train=(
            _data_file_response(challenge.train, request) if challenge.train is not None else None
        ),
        test=_data_file_response(challenge.test, request) if challenge.test is not None else None,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    engine = create_database_engine(settings.database_path)
    initialize_database(engine)
    session_factory = create_session_factory(engine)
    templates = Jinja2Templates(directory=settings.templates_dir)
    templates.env.filters["filesize"] = _format_bytes
    templates.env.filters["as_list"] = _as_list
    templates.env.filters["metadata_values"] = _format_metadata
    templates.env.filters["challenge_type_label"] = CHALLENGE_TYPE_LABELS.__getitem__

    application = FastAPI(
        title="isCDC Spatial Multi-omics Database",
        version="0.2.0",
        description="A read-only catalogue of databases and benchmark challenges.",
    )
    application.state.settings = settings
    application.state.engine = engine
    application.state.session_factory = session_factory
    application.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

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

    @application.get("/", response_class=HTMLResponse, name="home")
    async def home(request: Request, session: SessionDependency):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "database_count": count_databases(session),
                "challenge_count": count_challenges(session),
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
        return templates.TemplateResponse(
            request=request,
            name="databases.html",
            context={
                "databases": databases,
                "total": total,
                "facets": get_facets(session, ("full",)),
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
        return templates.TemplateResponse(
            request=request,
            name="database_detail.html",
            context={"database": database, "download_kinds": DOWNLOAD_FILES},
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
        page: Annotated[int, Query(ge=1)] = 1,
    ):
        filters = _filters(
            q, organism, tissue, modality, technology, spatial_unit, challenge_type
        )
        selected = _selected_filters(
            q, organism, tissue, modality, technology, spatial_unit, challenge_type
        )
        per_page = 20
        challenges, total = list_challenges(session, filters, (page - 1) * per_page, per_page)
        return templates.TemplateResponse(
            request=request,
            name="challenges.html",
            context={
                "challenges": challenges,
                "total": total,
                "facets": get_facets(session, DERIVED_DATASET_TYPES),
                "selected": selected,
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
        return templates.TemplateResponse(
            request=request,
            name="challenge_detail.html",
            context={"challenge": challenge, "download_kinds": DOWNLOAD_FILES},
        )

    @application.get("/downloads/{dataset_id}/{kind}", name="download_file")
    async def download_file(dataset_id: str, kind: str, session: SessionDependency):
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
            items=[_data_file_response(database, request) for database in databases],
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
        return _data_file_response(database, request)

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
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> ChallengeListResponse:
        challenges, total = list_challenges(
            session,
            _filters(q, organism, tissue, modality, technology, spatial_unit, challenge_type),
            offset,
            limit,
        )
        return ChallengeListResponse(
            items=[_challenge_response(challenge, request) for challenge in challenges],
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
        return _challenge_response(challenge, request)

    return application


app = create_app()
