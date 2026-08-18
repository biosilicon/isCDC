stop_rctd_workers <- function() {
  parameters <- BiocParallel::registered()
  for (parameter in parameters) {
    if (inherits(parameter, "MulticoreParam")) {
      BiocParallel::bpstop(parameter)
    }
  }
  BiocParallel::register(BiocParallel::SerialParam())
  invisible(NULL)
}

run_rctd_full <- function(rctd_data, workers) {
  tryCatch(
    spacexr::runRctd(rctd_data, rctd_mode = "full", max_cores = workers),
    finally = stop_rctd_workers()
  )
}

aligned_rctd_weights <- function(result, target_ids) {
  raw <- SummarizedExperiment::assay(result, "weights")
  result_ids <- colnames(result)
  if (is.null(result_ids)) {
    result_ids <- rownames(SummarizedExperiment::colData(result))
  }
  if (is.null(result_ids) || length(result_ids) != ncol(raw) ||
      anyDuplicated(result_ids) || !all(result_ids %in% target_ids)) {
    stop("RCTD result observations are not aligned to target observations")
  }
  raw_ids <- colnames(raw)
  if (is.null(raw_ids)) {
    colnames(raw) <- result_ids
  } else if (!identical(raw_ids, result_ids)) {
    stop("RCTD assay columns differ from result observations")
  }
  if (is.null(rownames(raw)) || anyDuplicated(rownames(raw))) {
    stop("RCTD weight rows do not have unique cell-type names")
  }
  weights <- matrix(
    0,
    nrow = length(target_ids),
    ncol = nrow(raw),
    dimnames = list(target_ids, rownames(raw))
  )
  weights[result_ids, ] <- t(as.matrix(raw))
  weights
}
