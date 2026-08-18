#!/usr/bin/env Rscript

cache_user <- gsub("[^A-Za-z0-9_.-]", "_", Sys.getenv("USER", "unknown"))
shared_cache <- file.path("/tmp", paste0("iscdc-cell-annotation-cache-", cache_user))
annotation_cache <- file.path(tempdir(), "iscdc-cell-annotation-cache")
source_cache <- file.path(shared_cache, "R", "BiocFileCache")
target_cache <- file.path(annotation_cache, "R", "BiocFileCache")
dir.create(target_cache, recursive = TRUE, showWarnings = FALSE)
seed_files <- list.files(source_cache, full.names = TRUE)
seed_files <- seed_files[basename(seed_files) != "BiocFileCache.sqlite.LOCK"]
if (length(seed_files) > 0L && !all(file.copy(seed_files, target_cache))) {
  stop("Unable to seed the job-local spacexr cache")
}
Sys.setenv(R_USER_CACHE_DIR = annotation_cache)

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
  library(Matrix)
  library(SpatialExperiment)
  library(spacexr)
  library(SummarizedExperiment)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1L]
script_path <- normalizePath(sub("^--file=", "", script_arg))
source(file.path(dirname(script_path), "rctd_utils.R"))

parse_args <- function(args) {
  keys <- c("--exchange", "--reference", "--output", "--config", "--workers")
  if (length(args) != 10L || !all(args[seq(1L, 9L, 2L)] == keys)) {
    stop("usage: run_rctd.R --exchange DIR --reference DIR --output DIR --config YAML --workers N")
  }
  setNames(as.list(args[seq(2L, 10L, 2L)]), substring(keys, 3L))
}

calibrate <- function(features, calibration) {
  if (!identical(calibration$method, "logistic") || is.null(calibration$coefficients)) {
    stop("reference calibration.rds is not a validated logistic calibration")
  }
  design <- cbind(1, as.matrix(features[, names(calibration$coefficients)[-1L], drop = FALSE]))
  eta <- drop(design %*% unname(calibration$coefficients))
  probability <- plogis(eta)
  if (!is.null(calibration$probability_x) && !is.null(calibration$probability_y)) {
    if (length(calibration$probability_x) < 2L ||
        length(calibration$probability_x) != length(calibration$probability_y)) {
      stop("reference calibration.rds has invalid isotonic probability calibration")
    }
    probability <- approx(
      calibration$probability_x,
      calibration$probability_y,
      probability,
      rule = 2L,
      ties = "ordered"
    )$y
  }
  pmax(0, pmin(1, probability))
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
workers <- as.integer(args$workers)
if (is.na(workers) || workers < 1L || workers > 12L) stop("workers must be in [1, 12]")
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
runtime <- read_json(args$config, simplifyVector = FALSE)
if (!identical(runtime$schema_version, "1.0") || !identical(runtime$method, "rctd") ||
    !is.list(runtime$parameters)) {
  stop("runtime config is invalid for RCTD")
}
runtime_fraction <- function(name, fallback) {
  value <- runtime$parameters[[name]]
  if (is.null(value)) return(fallback)
  value <- as.numeric(value)
  if (length(value) != 1L || !is.finite(value) || value < 0 || value > 1) {
    stop(paste0(name, " must be a finite fraction"))
  }
  value
}
counts <- readMM(file.path(args$exchange, "matrix.mtx"))
observations <- fread(file.path(args$exchange, "observations.tsv"), sep = "\t")
genes <- fread(file.path(args$exchange, "genes.tsv"), sep = "\t")
coordinates <- fread(file.path(args$exchange, "spatial.tsv"), sep = "\t")
if (nrow(counts) != nrow(observations) || ncol(counts) != nrow(genes)) {
  stop("Matrix Market dimensions do not match exchange metadata")
}
rownames(counts) <- observations$observation_id
colnames(counts) <- genes$gene_id
rownames(coordinates) <- coordinates$observation_id

reference_data <- readRDS(file.path(args$reference, "reference.rds"))
calibration <- readRDS(file.path(args$reference, "calibration.rds"))
if (!all(c("counts", "cell_types", "ontology_ids", "validation") %in% names(reference_data))) {
  stop("RCTD reference.rds is incomplete")
}
shared <- intersect(colnames(counts), rownames(reference_data$counts))
if (length(shared) == 0L) stop("RCTD target and reference share no genes")

target_counts <- t(counts[, shared, drop = FALSE])
reference_counts <- reference_data$counts[shared, , drop = FALSE]
if (length(reference_data$cell_types) != ncol(reference_counts)) {
  stop("RCTD reference cell types are not aligned to reference count columns")
}
spatial_coordinates <- as.matrix(coordinates[, .(x, y)])
rownames(spatial_coordinates) <- observations$observation_id
spatial_spe <- SpatialExperiment(
  assays = list(counts = target_counts),
  spatialCoords = spatial_coordinates
)
reference_se <- SummarizedExperiment(
  assays = list(counts = reference_counts),
  colData = S4Vectors::DataFrame(
    cell_type = factor(reference_data$cell_types),
    row.names = colnames(reference_counts)
  )
)
rctd_data <- createRctd(spatial_spe, reference_se, cell_type_col = "cell_type", require_int = TRUE)
rctd_result <- run_rctd_full(rctd_data, workers)

weights <- aligned_rctd_weights(rctd_result, observations$observation_id)
weights[!is.finite(weights) | weights < 0] <- 0
totals <- rowSums(weights)
converged <- is.finite(totals) & totals > 0
weights[converged, ] <- weights[converged, , drop = FALSE] / totals[converged]
order_index <- t(apply(weights, 1L, order, decreasing = TRUE))
top_index <- order_index[, 1L]
second_index <- if (ncol(weights) > 1L) order_index[, 2L] else top_index
top_weight <- weights[cbind(seq_len(nrow(weights)), top_index)]
second_weight <- weights[cbind(seq_len(nrow(weights)), second_index)]
delta <- top_weight - second_weight
entropy <- -rowSums(ifelse(weights > 0, weights * log(weights), 0)) /
  ifelse(ncol(weights) > 1L, log(ncol(weights)), 1)
effective_types <- exp(-rowSums(ifelse(weights > 0, weights * log(weights), 0)))
features <- data.frame(top_weight = top_weight, delta = delta, entropy = entropy)
confidence <- calibrate(features, calibration)
applied_thresholds <- list(
  uncertain_min_confidence = runtime_fraction(
    "uncertain_min_confidence", calibration$uncertain_max_confidence
  ),
  mixed_min_top_weight = runtime_fraction(
    "mixed_min_top_weight", calibration$mixed_min_top_weight
  ),
  mixed_min_delta = runtime_fraction("mixed_min_delta", calibration$mixed_min_delta),
  mixed_max_entropy = runtime_fraction("mixed_max_entropy", calibration$mixed_max_entropy)
)
mixed_top <- top_weight < applied_thresholds$mixed_min_top_weight
mixed_delta <- delta < applied_thresholds$mixed_min_delta
mixed_entropy <- entropy > applied_thresholds$mixed_max_entropy
mixed <- mixed_top | mixed_delta | mixed_entropy
uncertain_confidence <- confidence < applied_thresholds$uncertain_min_confidence
uncertain <- !converged | uncertain_confidence
status <- ifelse(uncertain, "Uncertain", ifelse(mixed, "Mixed", "Predicted"))
labels <- colnames(weights)[top_index]

result <- data.table(
  observation_id = observations$observation_id,
  label = labels,
  ontology_id = ifelse(
    status == "Predicted", unname(reference_data$ontology_ids[labels]), NA_character_
  ),
  status = status,
  confidence = confidence,
  top_weight = top_weight,
  second_weight = second_weight,
  delta = delta,
  entropy = entropy,
  effective_types = effective_types,
  converged = converged
)
fwrite(result, file.path(args$output, "predictions.tsv"), sep = "\t", na = "")
writeMM(as(weights, "dgCMatrix"), file.path(args$output, "cell_type_weights.mtx"))
fwrite(data.table(label = colnames(weights)), file.path(args$output, "weight_labels.tsv"), sep = "\t")

diagnostics <- list(
  method = "RCTD full",
  method_version = as.character(packageVersion("spacexr")),
  qc = c(list(shared_genes = length(shared)), reference_data$validation),
  calibration = list(
    completed = TRUE,
    method = calibration$method,
    confidence_definition = "pseudo-spot calibrated dominant-type reliability",
    calibration_id = calibration$id,
    applied_thresholds = applied_thresholds
  ),
  mixed_count = sum(status == "Mixed"),
  uncertain_count = sum(status == "Uncertain"),
  status_reasons = list(
    not_converged = sum(!converged),
    confidence_below_threshold = sum(uncertain_confidence),
    top_weight_below_threshold = sum(mixed_top),
    delta_below_threshold = sum(mixed_delta),
    entropy_above_threshold = sum(mixed_entropy)
  ),
  weight_semantics = "normalized RCTD composition; not prediction probability"
)
write_json(diagnostics, file.path(args$output, "diagnostics.json"), auto_unbox = TRUE, pretty = TRUE)
