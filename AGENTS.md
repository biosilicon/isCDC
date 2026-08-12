# Repository Guidelines

## Project Structure & Module Organization

Production code lives under `src/iscdc/`, automated tests under `tests/`, and templates, styles, and example metadata under `assets/`. Runtime catalogue data, visitor analytics, and imported files are written to the ignored `data/` directory. The ignored `temp/` directory stages datasets that are not yet ready for catalogue import; keep source files, their in-progress `metadata.yaml`, and dataset-specific conversion work and output directories there until they satisfy schema 1.2. The ignored `exp/` directory is the local real-data experiment area: keep required real inputs, manual test YAML, and generated experiment outputs there, and never commit its contents. Local `dataset_planner` and `dataset_worker` definitions and their concurrency settings live under the ignored `.codex/` directory; follow `原始数据处理规范.md` when using them, and do not assume those local definitions exist in a fresh checkout. Mirror source paths in the test tree where practical and keep root-level files limited to project configuration, documentation, and entry points.

Committed Database thumbnails live under `assets/static/database_thumbnails/` as WebP files named
exactly `<dataset_id>.webp`. Keep downloaded source images in the ignored
`assets/he_wsi_thumbnails/` directory; do not commit them. A missing thumbnail is valid and must
not produce a placeholder or empty image container. Thumbnail discovery and the stylesheet content
version are computed at application startup, so restart a running application after adding or
removing thumbnails or changing `assets/static/styles.css`.

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

## Build, Test, and Development Commands

All project commands must be run in the Conda environment named `iscdc`. Activate it with `conda activate iscdc` before installing dependencies, running tests, linting, or starting the application.

The project uses requirements files and a small Makefile command set documented in `README.md`:

- `make setup` — install development dependencies from `requirements-dev.txt`.
- `make test` — run the complete pytest suite, including the required real-data split.
- `make lint` — run Ruff static-analysis checks.
- `make run` — start the FastAPI application with Uvicorn.
- `make import-example` — import the documented example dataset into the local catalogue.

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
