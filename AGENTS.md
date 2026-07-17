# Repository Guidelines

## Project Structure & Module Organization

This repository is currently an empty scaffold. As implementation is added, keep production code under `src/`, automated tests under `tests/`, and non-code resources under `assets/`. Mirror source paths in the test tree; for example, tests for `src/api/client.py` should live in `tests/api/test_client.py`. Keep root-level files limited to project configuration, documentation, and entry points.

Prefer small, focused modules with clear public interfaces. Group code by feature or domain rather than creating broad utility directories. Document any new top-level directory in `README.md` and update this guide when the layout becomes established.

## Build, Test, and Development Commands

No build system, dependency manifest, or task runner is configured yet. When adding one, expose a small, predictable command set and document it in `README.md`. Recommended task names are:

- `make setup` — install development dependencies.
- `make test` — run the complete automated test suite.
- `make lint` — run formatting and static-analysis checks.
- `make run` — start the project locally.

Do not document placeholder commands as working until their targets are implemented.

## Coding Style & Naming Conventions

Follow the standard formatter and linter for the chosen language, checked into project configuration. Use spaces rather than tabs unless the ecosystem requires otherwise. Choose descriptive names: `snake_case` for files and functions in Python, `camelCase` for JavaScript/TypeScript functions, and `PascalCase` for classes and components. Avoid unrelated formatting changes in feature commits.

## Testing Guidelines

Add tests with every behavior change and bug fix. Keep tests deterministic and independent of network services by default. Name tests after observable behavior, and place shared fixtures in the nearest appropriate test support module. Once a framework is selected, record the exact invocation and any coverage threshold here.

## Commit & Pull Request Guidelines

No commit history is available to establish an existing convention. Until one emerges, use concise imperative subjects, optionally with a Conventional Commit prefix, such as `feat: add configuration loader` or `fix: reject invalid input`.

Pull requests should explain the change, motivation, and verification performed. Link relevant issues, identify breaking changes, and include screenshots or terminal output when behavior is visual or operational. Keep each pull request focused and ensure documented checks pass before requesting review.
