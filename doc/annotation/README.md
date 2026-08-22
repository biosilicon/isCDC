# Offline cell-type annotation

The implementation is complete for the current 35 two-dimensional `full` Databases.
The final audit reported 35 successes, no scientific failures, and no framework
failures: 3 source-label datasets, 1 SingleR dataset, and 31 full-mode RCTD datasets.
The website architecture and operating commands are in the root README's
[Cell type spatial visualization](../../README.md#cell-type-空间可视化) section; methodology,
failure analysis, calibration, and scheduling lessons are in
[`细胞类型注释经验总结.md`](细胞类型注释经验总结.md); exact round outcomes are in
[`../../assets/cell_type_annotation/iteration_history.yaml`](../../assets/cell_type_annotation/iteration_history.yaml).

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
