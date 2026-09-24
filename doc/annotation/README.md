# Offline cell-type annotation

The implementation records the dated 2026-08-18 baseline of 35 two-dimensional `full` Databases.
The final audit reported 35 successes, no scientific failures, and no framework
failures: 3 source-label datasets, 1 SingleR dataset, and 31 full-mode RCTD datasets.
Databases imported after that baseline are not implicitly covered by this audit and require their
own configured annotation review before sidecar publication.
The website architecture and operating commands are in the root README's
[Cell type spatial visualization](../../README.md#cell-type-空间可视化) section; methodology,
failure analysis, calibration, and scheduling lessons are in
[`细胞类型注释经验总结.md`](细胞类型注释经验总结.md); exact round outcomes are in
[`../../assets/cell_type_annotation/iteration_history.yaml`](../../assets/cell_type_annotation/iteration_history.yaml).

The [2026-09-20 visualization recheck](细胞类型可视化复核_2026-09-20.md) identified four stale
bindings and 52 source-labeled datasets without sidecars. The subsequent authorized
[integration](细胞类型可视化接入记录_2026-09-20.md) restored those four historical results and
published all 52 source-label visualizations. Current coverage is 88 full Databases: 56 source,
31 RCTD and one SingleR. The remaining 143 require separate source-label or inference review.

The repository's `annotation/` directory defines the separate `iscdc-cell-annotation`
environment and the R adapters used to create visualization sidecars. It is not an
application dependency.
The environment deliberately contains no PyTorch, CUDA, GPU runtime, scVI, CellTypist,
or cell2location package. Cell-resolution inference uses SingleR; bin/spot inference
uses full-mode RCTD (`spacexr`).

Create and lock the environment with Conda, then restore the exact R library:

```bash
conda env create -f annotation/environment.yml
conda activate iscdc-cell-annotation
Rscript -e 'options(timeout=600); install.packages("https://cloud.r-project.org/src/contrib/renv_1.2.4.tar.gz", repos=NULL, type="source")'
Rscript -e 'renv::restore(lockfile="annotation/renv.lock", library=.libPaths()[1], prompt=FALSE)'
```

R packages intentionally come from the exact `renv.lock` sources rather than Conda:
at the R 4.6 release boundary, conda-forge's prebuilt R extension packages still target
older R ABIs. The Conda solve therefore fixes the interpreter and Python sparse-I/O
stack, while `renv` fixes the R and Bioconductor package set.

The Python orchestrator reads catalogue `.h5mu` files in backed/read-only mode. RNA is
exchanged as a sparse Matrix Market file plus TSV observation, gene, sample, and spatial
metadata. It never writes annotations into `.h5mu`. An inferred result is publishable
only after observation-order, reference checksum, calibration, and configured QC gates
pass. Raw SingleR scores and RCTD weights are diagnostics, never probabilities. Source
labels omit confidence. `Mixed` and `Uncertain` are prediction statuses, not Cell
Ontology terms.

On a Database detail page, the visualization's method-details control identifies source
labels as coming from an existing annotation file with no computational inference. It
shows reference ID/version, runtime parameters, QC publication thresholds, and QC results
only for inferred sidecars. This presentation reads the already validated manifest/report
and does not change the sidecar contract or public Database API.

Offline entry points are available through `python -m iscdc.cell_type_annotation`, and
the main project CLI wires the same public functions. Run annotation work through the
isolated environment even when it already exists:

```bash
conda run -n iscdc-cell-annotation env PYTHONPATH=src \
  python -m iscdc.cell_type_annotation build-cell-type-reference REFERENCE_ID [--force]
conda run -n iscdc-cell-annotation env PYTHONPATH=src \
  python -m iscdc.cell_type_annotation generate-cell-type-visualization DATASET_ID [--force]
conda run -n iscdc-cell-annotation env PYTHONPATH=src \
  python -m iscdc.cell_type_annotation audit-cell-type-visualizations \
  [--all | DATASET_ID ...] [--jobs N]
```

Reference recipes and dataset thresholds require scientific review. An entry marked
`complete: false` fails closed for generation and is reported as a complete scientific
failure for audit purposes. Full-catalogue expansion is gated until all four pilot
datasets have either a successful result or a complete scientific-failure report.

Tracked configuration lives under `assets/cell_type_annotation/`: `configs/catalogue.yaml`
defines every dataset method, parameters, and QC gates; `configs/references/` freezes
reference selection and calibration recipes; `vocabulary.yaml` defines allowed state
semantics. Runtime references, staging, immutable generations, and failures live under
the ignored `data/cell_type_visualizations/` root. A dataset's `status.json` points only
to its latest successful generation or latest failure report.

The scheduler accepts at most 20 requested jobs and packs configured per-task cores
under a 40-logical-core declared limit. BLAS/OMP libraries are capped before R starts.
Declared workers are ceilings rather than observed utilization, so production runs
must monitor aggregate `%CPU/100` and RSS. Reference downloads may overlap unrelated
annotation work; publication always remains staged, validated, and atomic.

Run focused verification with:

```bash
conda run -n iscdc-cell-annotation env PYTHONPATH=. \
  python -m pytest annotation/tests -q
conda run -n iscdc-cell-annotation \
  Rscript --vanilla annotation/test_census_reference_contract.R
conda run -n iscdc-cell-annotation \
  Rscript --vanilla annotation/test_single_r_contract.R
conda run -n iscdc env PYTHONPATH=src \
  python -m pytest tests/test_cell_type_annotation.py -q
```

## Spatial domains

SpatialGLUE adds a separate paired multi-omics result in the locked
`iscdc-spatial-domain-gpu` environment. It uses CUDA for training and igraph Leiden for
clustering; no R/mclust dependency is added. RNA-only results continue to use the existing
CPU environment. See [空间域识别](空间域识别.md#spatialglue-多模态空间域) for inputs,
explicit modality subsets, resource calibration, CLI and sidecar compatibility.

[Spatial domain identification](空间域识别.md) adds a separate RNA-based BANKSY/GraphST
pipeline and same-canvas mode switch. Its `iscdc-spatial-domain` environment and lockfile
are under `annotation/spatial_domain/`; PyTorch remains outside `iscdc-cell-annotation`.
Domain labels are sample-local clusters, without reference labels or confidence scores.
The default CPU ceiling is 40 threads, memory budget 128 GiB, with serialized jobs,
resource preflight and runtime RSS monitoring. Source H5MU and canonical annotations
are not modified. Source/inferred cell-type sidecar contracts remain unchanged.

The [parallel continuation](空间域识别.md#并行续跑) supervisor owns the global resource lock
and leases disjoint CPU sets to workers. Its default limits are eight jobs, 64 logical CPUs
and 192 GiB aggregate reserved memory; actual concurrency follows memory availability.

SpatialGLUE 的环境、真实样本校准及隔离页面验证见 [SpatialGLUE 接入记录](SpatialGLUE接入记录_2026-09-20.md)。

SpatialGLUE 的补充多规模资源测试已完成 4 并发短测，并按用户要求暂停；更高并发和
完整轮次复核待继续。当前运行建议保持不变，详见[并行度测试记录](SpatialGLUE并行度测试_2026-09-20.md)。

SpatialGLUE 的非微生物准入、四模态四组合和 69 万观测测试见[准入扩展与调度测试](SpatialGLUE准入扩展与调度测试_2026-09-20.md)。
