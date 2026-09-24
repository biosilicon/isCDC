# Repository Guidelines

## Working scope and completion

Work through the requested outcome: implement the change, run relevant checks, fix failures caused
by the change, and report the result and any remaining blockers. A first implementation is not a
review gate unless the user requested one. Use existing session decisions and authorization;
routine implementation choices, reversible local edits, and affected fixture-based test reruns
do not need repeated approval.

Ask only when a missing decision materially affects scientific meaning, scope, or an action outside
existing authorization. Identify the blocked action and continue independent work. Data import,
replacement, publication/restart, and source cleanup must stay within the requested scope; a
documentation or code edit alone does not authorize them. Dataset-specific decision boundaries
are in [原始数据处理规范](doc/原始数据处理规范.md#授权与完成边界).

## Layout and environments

- `src/iscdc/`: production Python; `tests/`: automated tests; `assets/`: templates, styles,
  example metadata and committed browser bundles.
- `frontend/`: Node 24 source, lock file and tests for browser bundles; `annotation/`: isolated
  R/Conda declarations and offline annotation adapters. Neither is a website runtime dependency.
- Keep tracked documentation under `doc/`, except root `AGENTS.md` and `README.md`.
  Annotation documentation belongs under `doc/annotation/`.
- Ignored `data/` holds runtime catalogues, analytics and imported files; `temp/<dataset_name>/`
  holds original inputs, in-progress metadata, isolated conversion work and outputs until accepted;
  `exp/` holds real-data fixtures and experiments. Keep their contents out of Git.
- Ignored `.codex/` holds local dataset agent definitions and concurrency settings; do not assume
  it exists in a fresh checkout. Use the dataset workflow only for dataset preparation tasks.
- Keep root files to project configuration, entry points, `AGENTS.md` and `README.md`.
  Document new top-level directories in `README.md` and update this layout when established.

Activate `conda activate iscdc` before website, catalogue and general development commands,
including dependency installation, tests and lint. Cell type reference, annotation, calibration,
artifact and annotation-audit commands use the separately locked `iscdc-cell-annotation` environment.
RNA spatial-domain inference uses its separately locked `iscdc-spatial-domain` environment;
SpatialGLUE uses the separately locked `iscdc-spatial-domain-gpu` environment.

## Read by task

Read the relevant sections when a task touches these areas; no full documentation sweep is required.

| Task | Guidance |
| --- | --- |
| Setup, CLI usage, test deployment or complete-suite prerequisites | [README](README.md) |
| Catalogue, API grouping, schema, IDs, import/replacement or splitting | [Catalogue contracts](doc/开发约束.md#catalogue), [schema 1.2](doc/数据库存储规范_v1.2.md) |
| Raw dataset investigation, conversion, acceptance or import | [Dataset workflow](doc/原始数据处理规范.md) |
| Intake eligibility or spatial resolution | [Resolution rules](doc/空间分辨率分类.md), [scope decisions](doc/空间观测层级收录审计_2026-09-05.md) |
| Source cell types, propagation or visualization | [Cell type contracts](doc/开发约束.md#cell-type); for inference/reference/QC work, [annotation guide](doc/annotation/README.md) and [operational lessons](doc/annotation/细胞类型注释经验总结.md) |
| Spatial-domain inference or visualization | [Domain workflow](doc/annotation/空间域识别.md), [feature contracts](doc/开发约束.md#spatial-domain-visualization) |
| WSI, thumbnails, auxiliary files or stylesheet publication | [Thumbnail contracts](doc/开发约束.md#thumbnails); for spatial previews, [rendering and replay contract](doc/空间信号缩略图.md) |
| Difficulty evaluation, publication or changed Challenge files | [Difficulty contracts](doc/开发约束.md#difficulty); dated results go in [run records](doc/Challenge难度快照运行记录.md) |
| Visitor analytics | [Analytics contracts](doc/开发约束.md#analytics) |

Retain these boundaries across features: the public catalogue is read-only, catalogue writes are
serialized, and analytics remains separate. Imports require schema and collection eligibility,
consistent `entry_id`, and reproducibility evidence. Inferred labels belong in visualization
sidecars, not canonical source annotations. Feature-specific details and regression coverage live
in the linked contracts.

Keep `README.md` and `doc/annotation/细胞类型注释经验总结.md` synchronized when annotation methods,
QC gates, scheduling limits or sidecar contracts change.

## Verification

Choose checks for the changed behavior. For code changes, use the narrowest relevant pytest node
IDs or test files and add regression coverage when behavior changes. Fixture-based tests use
temporary catalogues and outputs; run and repair affected tests within the existing task without
asking at each step. Once relevant checks pass, repeat or broaden them only for a new change,
failure or unresolved concern. Documentation-only edits normally need link and consistency checks.

Run `make test` or the complete pytest suite only when explicitly requested. The complete suite
requires the two ignored Xenium fixtures, about 450 MB of temporary space, and local IPC sockets;
see [README prerequisites](README.md#快速开始). Missing fixtures must fail clearly, not be skipped.
If the sandbox blocks IPC, request the minimum required local communication permission before
running the suite; this does not authorize external network access.

Application/page/API tests use `httpx.AsyncClient` with `httpx.ASGITransport`, not synchronous
FastAPI/Starlette `TestClient`. Pure imports use importer results, `validation_report.json`, hashes,
manifests and direct catalogue reads; they do not require new ASGI tests. Relevant feature
contracts specify additional checks for data publication and browser behavior.

Useful commands after activating the environment:

```bash
PYTHONPATH=src python -m pytest tests/test_splitter.py::test_compose_assigns_whole_sources_and_encodes_global_ids
make lint
make run
```

The pytest node is an example for compose changes, not a mandatory check for unrelated work.
`make setup` installs development dependencies; `make import-example` imports the documented
example into the local catalogue. Split parameters belong in YAML and paths resolve relative to
that YAML; see README for `range`, `spatial` and `compose` usage.

## Code and contribution conventions

Prefer focused feature/domain modules and mirror source paths in tests where practical. Follow
the checked-in formatter/linter and existing naming conventions; avoid unrelated formatting.
Keep tests deterministic and independent of network services by default. Do not document placeholder
commands as working before their targets exist.

When committing, use `type(scope): imperative summary` (scope optional) with a body explaining
motivation, principal changes and verification. Include migration/breaking-change notes when
applicable. PRs should describe the problem, resulting behavior and verification, with screenshots
or operational evidence when useful. Commit and PR creation follow the user's requested scope.
