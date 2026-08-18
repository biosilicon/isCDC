#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(BiocParallel)
  library(data.table)
  library(jsonlite)
  library(Matrix)
  library(SingleR)
})

parse_args <- function(args) {
  keys <- c("--exchange", "--reference", "--output", "--config", "--workers")
  if (length(args) != 10L || !all(args[seq(1L, 9L, 2L)] == keys)) {
    stop("usage: run_single_r.R --exchange DIR --reference DIR --output DIR --config YAML --workers N")
  }
  setNames(as.list(args[seq(2L, 10L, 2L)]), substring(keys, 3L))
}

interpolate_calibration <- function(raw, calibration) {
  if (!identical(calibration$method, "isotonic") ||
      length(calibration$x) < 2L || length(calibration$x) != length(calibration$y)) {
    stop("reference calibration.rds is not a validated isotonic calibration")
  }
  pmax(0, pmin(1, approx(calibration$x, calibration$y, raw,
                         rule = 2L, ties = "ordered")$y))
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
workers <- as.integer(args$workers)
if (is.na(workers) || workers < 1L || workers > 30L) stop("workers must be in [1, 30]")
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
runtime <- read_json(args$config, simplifyVector = FALSE)
if (!identical(runtime$schema_version, "1.0") || !identical(runtime$method, "singler") ||
    !is.list(runtime$parameters)) {
  stop("runtime config is invalid for SingleR")
}
runtime_number <- function(name, fallback) {
  value <- runtime$parameters[[name]]
  if (is.null(value)) return(fallback)
  value <- as.numeric(value)
  if (length(value) != 1L || !is.finite(value)) stop(paste0(name, " must be finite"))
  value
}
exchange <- args$exchange
reference_dir <- args$reference
output <- args$output

counts <- readMM(file.path(exchange, "matrix.mtx"))
observations <- fread(file.path(exchange, "observations.tsv"), sep = "\t")
genes <- fread(file.path(exchange, "genes.tsv"), sep = "\t")
if (nrow(counts) != nrow(observations) || ncol(counts) != nrow(genes)) {
  stop("Matrix Market dimensions do not match exchange metadata")
}
rownames(counts) <- observations$observation_id
colnames(counts) <- genes$gene_id

reference <- readRDS(file.path(reference_dir, "reference.rds"))
calibration <- readRDS(file.path(reference_dir, "calibration.rds"))
if (!all(c("expression", "labels", "ontology_ids", "validation") %in% names(reference))) {
  stop("SingleR reference.rds is incomplete")
}
shared <- intersect(colnames(counts), rownames(reference$expression))
if (length(shared) == 0L) stop("SingleR target and reference share no genes")
test <- t(counts[, shared, drop = FALSE])
ref <- reference$expression[shared, , drop = FALSE]
prediction <- SingleR(
  test = test,
  ref = ref,
  labels = reference$labels,
  BPPARAM = MulticoreParam(workers = workers)
)

scores <- as.matrix(prediction$scores)
ordered_scores <- t(apply(scores, 1L, sort, decreasing = TRUE))
best <- ordered_scores[, 1L]
second <- if (ncol(ordered_scores) > 1L) ordered_scores[, 2L] else rep(NA_real_, nrow(scores))
delta <- best - second
raw_reliability <- best + calibration$delta_weight * ifelse(is.na(delta), 0, delta)
confidence <- interpolate_calibration(raw_reliability, calibration)
pruned <- as.character(prediction$pruned.labels)
applied_thresholds <- list(
  uncertain_min_delta = runtime_number(
    "uncertain_min_delta", calibration$uncertain_min_delta
  ),
  uncertain_min_score = runtime_number(
    "uncertain_min_score", calibration$uncertain_min_score
  )
)
pruned_missing <- is.na(pruned) | pruned == ""
delta_below <- delta < applied_thresholds$uncertain_min_delta
score_below <- best < applied_thresholds$uncertain_min_score
uncertain <- pruned_missing | delta_below | score_below
labels <- ifelse(is.na(pruned) | pruned == "", as.character(prediction$labels), pruned)

result <- data.table(
  observation_id = observations$observation_id,
  label = labels,
  ontology_id = ifelse(uncertain, NA_character_, unname(reference$ontology_ids[labels])),
  status = ifelse(uncertain, "Uncertain", "Predicted"),
  confidence = confidence,
  best_score = best,
  second_score = second,
  delta = delta,
  pruned_label = ifelse(pruned_missing, NA_character_, pruned)
)
fwrite(result, file.path(output, "predictions.tsv"), sep = "\t", na = "")
writeMM(as(scores, "dgCMatrix"), file.path(output, "candidate_scores.mtx"))
fwrite(data.table(label = colnames(scores)), file.path(output, "candidate_score_labels.tsv"), sep = "\t")

diagnostics <- list(
  method = "SingleR",
  method_version = as.character(packageVersion("SingleR")),
  qc = c(list(shared_genes = length(shared)), reference$validation),
  calibration = list(
    completed = TRUE,
    method = calibration$method,
    confidence_definition = "held-out isotonic reliability from SingleR score and delta",
    calibration_id = calibration$id,
    applied_thresholds = applied_thresholds
  ),
  uncertain_count = sum(uncertain),
  status_reasons = list(
    pruned_label_missing = sum(pruned_missing),
    delta_below_threshold = sum(delta_below, na.rm = TRUE),
    score_below_threshold = sum(score_below, na.rm = TRUE)
  ),
  score_semantics = "SingleR matching scores are diagnostics and are not probabilities"
)
write_json(diagnostics, file.path(output, "diagnostics.json"), auto_unbox = TRUE, pretty = TRUE)
