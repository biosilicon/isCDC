# Repository Guidelines

## Project Structure & Module Organization

Production code lives under `src/iscdc/`, automated tests under `tests/`, and templates, styles, and example metadata under `assets/`. Runtime catalogue data, visitor analytics, and imported files are written to the ignored `data/` directory. The ignored `temp/` directory stages datasets that are not yet ready for catalogue import; keep source files, their in-progress `metadata.yaml`, and dataset-specific conversion work and output directories there until they satisfy schema 1.2. The ignored `exp/` directory is the local real-data experiment area: keep required real inputs, manual test YAML, and generated experiment outputs there, and never commit its contents. Local `dataset_planner` and `dataset_worker` definitions and their concurrency settings live under the ignored `.codex/` directory; follow `原始数据处理规范.md` when using them, and do not assume those local definitions exist in a fresh checkout. Mirror source paths in the test tree where practical and keep root-level files limited to project configuration, documentation, and entry points.

The tracked `annotation/` directory contains the isolated R/Conda environment declarations and R
adapters for offline cell type work. The tracked `frontend/` directory contains the Node 24 source,
lock file, and tests used to build committed browser bundles; neither toolchain is a website runtime
dependency. The completed 35-dataset implementation and operational lessons are documented in
`plan.md` and `annotation/细胞类型注释经验总结.md`; keep those records synchronized when methods,
QC gates, scheduling limits, or sidecar contracts change.

Local Database thumbnails live under the ignored `assets/static/database_thumbnails/` directory
as WebP files named exactly `<dataset_id>.webp`; do not commit them. Keep images downloaded only
to produce thumbnails in the separately ignored `assets/he_wsi_thumbnails/` directory. A missing
thumbnail is valid and must not produce a placeholder or empty image container. When a `full`
Database has a registered `he_wsi`, generate its thumbnail directly from that WSI with
`generate-wsi-thumbnails`; do not substitute a separate preview image. Preserve the whole-slide
view and aspect ratio, do not crop to tissue, and use the fixed 640 px maximum dimension. The
output must be RGB WebP encoded with quality 85 and method 6. The command must refuse existing
output unless `--force` is supplied, and replacements must remain atomic. Store downloadable WSI
and other formal auxiliary files under the owning dataset's ignored
`data/datasets/<dataset_id>/auxiliary/` directory, never under assets, and register them through
`add-auxiliary-file` so manifest 1.1
records their stable ID, original filename, media type, size, SHA-256, source URL, and retrieval
time. Auxiliary files remain part of the owning data-file detail page and JSON record; do not give
them catalogue rows, independent detail pages, or user-controlled filesystem paths. Keep manifest
1.0 readable for datasets without auxiliary files. Invalid or missing auxiliary files must fail
open without breaking catalogue pages, JSON APIs, or primary downloads. Thumbnail and auxiliary
discovery and the stylesheet content version are computed at application startup, so restart a
running application after adding or removing thumbnails or auxiliary files, or changing
`assets/static/styles.css`.

Challenge difficulty is an offline, catalogue-wide distribution-separability snapshot stored in
the ignored `challenge_difficulty.json` beside `catalog.db`; do not add it to the catalogue schema,
train/test `.h5mu` files, metadata, or manifests. Generate it with the fixed domain-classifier
workflow and publish only mean AUROC, shift score, and global percentile on Challenge list/detail
pages and JSON responses. The website must validate the complete Challenge set, type, input
modality, train/test IDs and checksums, and metric consistency at startup. Missing, invalid, stale,
or individually failed results must fail open as unavailable without breaking the catalogue.
Difficulty ordering must be applied after filtering and before pagination, with unavailable items
last in either direction. Re-evaluate after any Challenge import, replacement, or removal, and
restart the application after replacing the snapshot.

Keep the read-only catalogue in `catalog.db` and visitor tracking in the independently versioned
`analytics.db`; do not add analytics fields to catalogue tables. Analytics initialization, reads,
writes, and retention cleanup must fail open so catalogue pages, JSON APIs, and downloads remain
available. The default `data/` location may be a network filesystem, so analytics SQLite databases
must use DELETE journal mode; do not switch them to WAL. Treat retained IP addresses, User-Agent
values, and referrers as sensitive operational data: keep raw events for 30 days by default, expose
them only through the local CLI, and never add them to public pages or APIs. Health checks, static
assets, JSON APIs, and failed requests must not create visitor sessions or behavior events.

Prefer small, focused modules with clear public interfaces. Group code by feature or domain rather than creating broad utility directories. Document any new top-level directory in `README.md` and update this guide when the layout becomes established.

The public catalogue has two presentation classes without adding another `dataset_type`
value or a separate persisted presentation-class field:

- `full` files are exposed as **Databases**.
- `train` and `test` files are grouped by `split_id` and exposed as one **Challenge**. Every
  train/test file must declare `derivation.challenge_type` as `same_slice`,
  `cross_slice_same_subject`, or `cross_subject`; both sides of one Challenge must agree.

Database and Challenge list pages and JSON APIs must apply filters within their own class.
Challenge lists must support `challenge_type` filtering, and Challenge responses must expose
the type derived from schema 1.2 derivation metadata. A Challenge response must include both
imported sides even when only one side matched the filters. Show an incomplete status when one
side is absent, and treat multiple train files, multiple test files, a missing or invalid
`challenge_type`, or conflicting types under one `split_id` as a catalogue integrity error.
Do not duplicate either the Database/Challenge presentation class or `challenge_type` in a
separate catalogue column.

Schema 1.2 is restricted to cross-omics translation data. A file with exactly two modalities
must use `pairing_type: same_unit`, and both modality `obs_names` sets must be identical. Files
with three or more modalities may use `same_unit` or `partially_shared`; partially shared files
may retain observations present in only one modality, but at least one modality pair must overlap.
Reject `unpaired` files and do not make the importer silently discard observations. Derive
`modality_count` from the modality relationship rather than adding a persisted catalogue field,
and visibly annotate files with more than two modalities in pages and JSON responses.

Top-level `mdata.obs["cell_type"]` is an optional schema 1.2 annotation. Include it only when a
public source supplies discrete labels aligned to every top-level observation; partial coverage
requires omitting the whole column. When present, it must be an unordered pandas categorical with
non-null, non-blank, whitespace-trimmed string labels and no unused categories. Preserve the
source's biological semantics and spelling rather than imposing a cross-dataset ontology; an
explicit source category such as `Unlabeled` is valid. Do not add canonical cell type metadata to
`metadata.yaml`, catalogue tables, public JSON responses, or filters. The Database detail page may
render a separately versioned, startup-validated cell type visualization sidecar; that sidecar and
its internal data endpoint must not present inferred labels as canonical dataset metadata. Spatial
splits propagate the source column.
Each composed output side includes it only when every full source assigned to that side has a
valid complete column; categories are merged in source and first-seen order, otherwise the output
omits the column.

`import-dataset` rejects existing IDs unless `--replace` is explicit. Replacement must preserve
the indexed `dataset_type` and, for derived data, the construction type, ordered source IDs,
`split_id`, and `challenge_type`. It must stage and validate the new data first, checksum and
preserve registered auxiliary files, and restore the original database record and directory if
the transaction or filesystem switch fails. Serialize catalogue writes. Rebuild dependent
Challenges after changing a full source when their propagated annotations need updating, then
re-evaluate the catalogue-wide difficulty snapshot.

## Build, Test, and Development Commands

Website, catalogue, and general development commands must be run in the Conda environment named
`iscdc`. Activate it with `conda activate iscdc` before installing dependencies, running normal
tests, linting, or starting the application. Cell type reference, annotation, calibration, artifact,
and annotation-audit commands are the sole exception: run them through the separately locked Conda
environment `iscdc-cell-annotation`, never through the website environment.

The project uses requirements files and a small Makefile command set documented in `README.md`:

- `make setup` — install development dependencies from `requirements-dev.txt`.
- `make test` — run the complete pytest suite, including the required real-data split.
- `make lint` — run Ruff static-analysis checks.
- `make run` — start the FastAPI application with Uvicorn.
- `make import-example` — import the documented example dataset into the local catalogue.

Install the optional domain-classifier dependency set through `requirements-difficulty.txt` (it is
already included by `requirements-dev.txt`) and evaluate the complete Challenge catalogue with:

```bash
PYTHONPATH=src python -m iscdc.cli evaluate-challenge-difficulty [--force]
```

The default published destination is `challenge_difficulty.json` beside `catalog.db`. A non-default
`--output` is an experiment snapshot and is not read by the website.

Invoke the standalone schema 1.2 splitter with:

```bash
PYTHONPATH=src python -m iscdc.splitter
```

Use these subcommands:

- `range FULL.h5mu` for read-only coordinate inspection.
- `spatial CONFIG.yaml` for a spatial train/test split.
- `compose CONFIG.yaml` to assign whole datasets to train or test.

All split parameters belong in the YAML configuration; paths in it are resolved
relative to the configuration file.

Do not document placeholder commands as working until their targets are implemented.

## Coding Style & Naming Conventions

Follow the standard formatter and linter for the chosen language, checked into project configuration. Use spaces rather than tabs unless the ecosystem requires otherwise. Choose descriptive names: `snake_case` for files and functions in Python, `camelCase` for JavaScript/TypeScript functions, and `PascalCase` for classes and components. Avoid unrelated formatting changes in feature commits.

## Testing Guidelines

Add tests with every behavior change and bug fix. Keep tests deterministic and independent of network services by default. Name tests after observable behavior, and place shared fixtures in the nearest appropriate test support module. Run tests with `make test` (equivalent to `PYTHONPATH=src python -m pytest`). No coverage threshold is currently enforced.

Test runs must allow local IPC sockets because the PyTorch suite exercises multi-process
`DataLoader` workers. In sandboxed environments, request the minimum additional permission needed
for local process-to-process communication before running `make test`; this does not authorize
external network access. Do not run the suite in a socket-blocking sandbox, where worker queue
failures can leave pytest waiting indefinitely.

Application, page, and API tests must use `httpx.AsyncClient` with
`httpx.ASGITransport`; do not use the synchronous `fastapi.testclient.TestClient` or
`starlette.testclient.TestClient`. Pure data imports that do not change source code, schemas,
templates, or API behavior do not require HTTP/ASGI-layer tests. Validate those imports through
the importer result, `validation_report.json`, checksum and manifest consistency, and direct
catalogue or repository reads instead. Run network-layer tests when application behavior changes,
not merely to confirm that a data file was imported.

Auxiliary-file behavior tests must cover safe manifest parsing, atomic registration rollback,
duplicate and path rejection, fail-open discovery, detail-page and JSON exposure, HEAD and byte
Range downloads, and 404/416 failures. A pure auxiliary-data registration using already-tested
code does not require a new ASGI test, but it must be verified through CLI output, source and stored
SHA-256/size equality, manifest consistency, format-specific checks, and direct endpoint reads.

WSI-thumbnail behavior tests must use small deterministic tiled TIFF fixtures and cover pyramid
level selection, aspect-ratio-preserving 640 px output, RGB WebP validation, existing-output
protection, atomic replacement rollback, invalid inputs, single-dataset and `--all` CLI modes,
batch skipping, and partial failures. Real WSI thumbnail generation additionally requires visual
inspection for completeness, orientation, tile seams, and black borders, plus direct detail-page
and static-file reads after restarting the application.

Difficulty publication tests must cover strict snapshot validation, startup-only loading,
fail-open missing/corrupt/stale reports, nullable API results, list/detail presentation, one shared
accessible method modal per page, ascending and descending ordering before pagination, and
unavailable items sorting last. Domain-classifier tests must retain same-distribution, clear-shift,
label-swap, reproducibility, seed stability, and class-imbalance sanity checks.

The complete suite requires these local real-data fixtures:

- `exp/xenium_human_rcc_ffpe_rna_protein.h5mu`
- `exp/xenium_human_rcc_ffpe_rna_protein_vertical_split.yaml`

The real-data test reruns the configured spatial split in a temporary directory and needs roughly 300 MB of temporary disk space. Missing fixtures must fail the suite with a clear message; do not skip the real-data test. Keep synthetic MuData for focused unit and edge-case coverage so those cases remain fast and reproducible.

Run splitter-only tests with:

```bash
PYTHONPATH=src python -m pytest tests/test_splitter.py
```

## Commit & Pull Request Guidelines

Use a descriptive Conventional Commit subject in the form `type(scope): imperative summary`; omit the scope only when none is useful. Every commit must include a body that explains the motivation, the principal implementation changes, and the verification performed. Record breaking changes, migration requirements, and related issues in the body or footer when applicable. Do not use an underspecified single-line message, even for a focused change.

Pull requests should explain the change, motivation, and verification performed. Link relevant issues, identify breaking changes, and include screenshots or terminal output when behavior is visual or operational. Keep each pull request focused and ensure documented checks pass before requesting review.
