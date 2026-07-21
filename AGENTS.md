# Repository Guidelines

## Project Structure & Module Organization

Production code lives under `src/iscdc/`, automated tests under `tests/`, and templates, styles, and example metadata under `assets/`. Runtime catalogue data is written to the ignored `data/` directory. Mirror source paths in the test tree where practical and keep root-level files limited to project configuration, documentation, and entry points.

Prefer small, focused modules with clear public interfaces. Group code by feature or domain rather than creating broad utility directories. Document any new top-level directory in `README.md` and update this guide when the layout becomes established.

## Build, Test, and Development Commands

All project commands must be run in the Conda environment named `iscdc`. Activate it with `conda activate iscdc` before installing dependencies, running tests, linting, or starting the application.

The project uses requirements files and a small Makefile command set documented in `README.md`:

- `make setup` — install development dependencies from `requirements-dev.txt`.
- `make test` — run the complete pytest suite.
- `make lint` — run Ruff static-analysis checks.
- `make run` — start the FastAPI application with Uvicorn.

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

Run splitter-only tests with:

```bash
PYTHONPATH=src python -m pytest tests/test_splitter.py
```

Synthetic MuData should remain the default; a root-level `.h5mu` may be used by
optional tests that skip when the file is not available.

## Commit & Pull Request Guidelines

No commit history is available to establish an existing convention. Until one emerges, use concise imperative subjects, optionally with a Conventional Commit prefix, such as `feat: add configuration loader` or `fix: reject invalid input`.

Pull requests should explain the change, motivation, and verification performed. Link relevant issues, identify breaking changes, and include screenshots or terminal output when behavior is visual or operational. Keep each pull request focused and ensure documented checks pass before requesting review.
