# Repository Guidelines

## Project Structure & Module Organization

Production code lives under `src/iscdc/`, automated tests under `tests/`, and templates, styles, and example metadata under `assets/`. Runtime catalogue data is written to the ignored `data/` directory. The ignored `exp/` directory is the local real-data experiment area: keep required real inputs, manual test YAML, and generated experiment outputs there, and never commit its contents. Mirror source paths in the test tree where practical and keep root-level files limited to project configuration, documentation, and entry points.

Prefer small, focused modules with clear public interfaces. Group code by feature or domain rather than creating broad utility directories. Document any new top-level directory in `README.md` and update this guide when the layout becomes established.

The public catalogue has two presentation classes without adding another storage-level type:

- `full` files are exposed as **Databases**.
- `train` and `test` files are grouped by `split_id` and exposed as one **Challenge**.

Database and Challenge list pages and JSON APIs must apply filters within their own class. A Challenge response must include both imported sides even when only one side matched the filters. Show an incomplete status when one side is absent, and treat multiple train files or multiple test files under one `split_id` as a catalogue integrity error. Keep these presentation rules derived from schema 1.1 metadata; do not duplicate them in a new persisted classification field.

## Build, Test, and Development Commands

All project commands must be run in the Conda environment named `iscdc`. Activate it with `conda activate iscdc` before installing dependencies, running tests, linting, or starting the application.

The project uses requirements files and a small Makefile command set documented in `README.md`:

- `make setup` — install development dependencies from `requirements-dev.txt`.
- `make test` — run the complete pytest suite, including the required real-data split.
- `make lint` — run Ruff static-analysis checks.
- `make run` — start the FastAPI application with Uvicorn.
- `make import-example` — import the documented example dataset into the local catalogue.

Invoke the standalone schema 1.1 splitter with:

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
