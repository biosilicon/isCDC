import math
from collections.abc import AsyncGenerator
from typing import Annotated
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import Settings
from .database import create_database_engine, create_session_factory, initialize_database
from .models import Dataset
from .repository import DatasetFilters, get_facets, list_datasets
from .schemas import DatasetListResponse, DatasetResponse

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
    dataset_type: str | None,
) -> DatasetFilters:
    return DatasetFilters(q, organism, tissue, modality, technology, spatial_unit, dataset_type)


def _dataset_response(dataset: Dataset, request: Request) -> DatasetResponse:
    downloads = {
        kind: str(
            request.url_for("download_dataset_file", dataset_id=dataset.dataset_id, kind=kind)
        )
        for kind in DOWNLOAD_FILES
    }
    return DatasetResponse(
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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    engine = create_database_engine(settings.database_path)
    initialize_database(engine)
    session_factory = create_session_factory(engine)
    templates = Jinja2Templates(directory=settings.templates_dir)
    templates.env.filters["filesize"] = _format_bytes
    templates.env.filters["as_list"] = _as_list
    templates.env.filters["metadata_values"] = _format_metadata

    application = FastAPI(
        title="isCDC Spatial Multi-omics Database",
        version="0.1.0",
        description="A read-only catalogue of validated spatial multi-omics datasets.",
    )
    application.state.settings = settings
    application.state.engine = engine
    application.state.session_factory = session_factory
    application.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

    async def get_session() -> AsyncGenerator[Session, None]:
        with session_factory() as session:
            yield session

    SessionDependency = Annotated[Session, Depends(get_session)]

    @application.get("/", response_class=HTMLResponse, name="home")
    async def home(request: Request, session: SessionDependency):
        count = session.scalar(select(func.count()).select_from(Dataset)) or 0
        datasets, _ = list_datasets(session, DatasetFilters(), 0, 6)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"dataset_count": count, "datasets": datasets},
        )

    @application.get("/datasets", response_class=HTMLResponse, name="dataset_list")
    async def dataset_list(
        request: Request,
        session: SessionDependency,
        q: str | None = None,
        organism: str | None = None,
        tissue: str | None = None,
        modality: str | None = None,
        technology: str | None = None,
        spatial_unit: str | None = None,
        dataset_type: str | None = None,
        page: Annotated[int, Query(ge=1)] = 1,
    ):
        filters = _filters(q, organism, tissue, modality, technology, spatial_unit, dataset_type)
        per_page = 20
        datasets, total = list_datasets(session, filters, (page - 1) * per_page, per_page)
        page_count = max(1, math.ceil(total / per_page))
        base_params = {
            key: value
            for key, value in {
                "q": q,
                "organism": organism,
                "tissue": tissue,
                "modality": modality,
                "technology": technology,
                "spatial_unit": spatial_unit,
                "dataset_type": dataset_type,
            }.items()
            if value
        }

        def page_url(number: int) -> str:
            return f"{request.url.path}?{urlencode({**base_params, 'page': number})}"

        return templates.TemplateResponse(
            request=request,
            name="datasets.html",
            context={
                "datasets": datasets,
                "total": total,
                "page": page,
                "page_count": page_count,
                "previous_url": page_url(page - 1) if page > 1 else None,
                "next_url": page_url(page + 1) if page < page_count else None,
                "facets": get_facets(session),
                "selected": base_params,
            },
        )

    @application.get("/datasets/{dataset_id}", response_class=HTMLResponse, name="dataset_detail")
    async def dataset_detail(request: Request, dataset_id: str, session: SessionDependency):
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            return templates.TemplateResponse(
                request=request,
                name="404.html",
                context={"dataset_id": dataset_id},
                status_code=404,
            )
        return templates.TemplateResponse(
            request=request,
            name="dataset_detail.html",
            context={"dataset": dataset, "download_kinds": DOWNLOAD_FILES},
        )

    @application.get("/datasets/{dataset_id}/downloads/{kind}", name="download_dataset_file")
    async def download_dataset_file(dataset_id: str, kind: str, session: SessionDependency):
        file_spec = DOWNLOAD_FILES.get(kind)
        if file_spec is None:
            raise HTTPException(status_code=404, detail="Unknown download type")
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
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

    @application.get("/api/datasets", response_model=DatasetListResponse, name="api_dataset_list")
    async def api_dataset_list(
        request: Request,
        session: SessionDependency,
        q: str | None = None,
        organism: str | None = None,
        tissue: str | None = None,
        modality: str | None = None,
        technology: str | None = None,
        spatial_unit: str | None = None,
        dataset_type: str | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> DatasetListResponse:
        filters = _filters(q, organism, tissue, modality, technology, spatial_unit, dataset_type)
        datasets, total = list_datasets(session, filters, offset, limit)
        return DatasetListResponse(
            items=[_dataset_response(dataset, request) for dataset in datasets],
            total=total,
            limit=limit,
            offset=offset,
        )

    @application.get(
        "/api/datasets/{dataset_id}", response_model=DatasetResponse, name="api_dataset_detail"
    )
    async def api_dataset_detail(
        request: Request, dataset_id: str, session: SessionDependency
    ) -> DatasetResponse:
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return _dataset_response(dataset, request)

    return application


app = create_app()
