# Generic audited builder for a pinned CELLxGENE Census recipe.
cache_user <- gsub("[^A-Za-z0-9_.-]", "_", Sys.getenv("USER", "unknown"))
annotation_cache <- file.path("/tmp", paste0("iscdc-cell-annotation-cache-", cache_user))
dir.create(annotation_cache, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(R_USER_CACHE_DIR = annotation_cache)

suppressPackageStartupMessages({
  library(BiocParallel)
  library(data.table)
  library(jsonlite)
  library(Matrix)
  library(SingleR)
  library(SpatialExperiment)
  library(spacexr)
  library(SummarizedExperiment)
})

builder_path <- normalizePath(sys.frame(1L)$ofile)
source(file.path(dirname(builder_path), "rctd_utils.R"), local = TRUE)

finite_fraction <- function(value, name) {
  value <- as.numeric(value)
  if (length(value) != 1L || !is.finite(value) || value < 0 || value > 1) {
    stop(sprintf("%s must be a finite fraction", name))
  }
  value
}

classification_metrics <- function(truth, predicted, confidence) {
  truth <- as.character(truth)
  predicted <- as.character(predicted)
  correct <- !is.na(predicted) & predicted == truth
  types <- sort(unique(truth))
  recall <- vapply(types, function(type) mean(correct[truth == type]), numeric(1L))
  f1 <- vapply(types, function(type) {
    tp <- sum(truth == type & predicted == type, na.rm = TRUE)
    fp <- sum(truth != type & predicted == type, na.rm = TRUE)
    fn <- sum(truth == type & (is.na(predicted) | predicted != type), na.rm = TRUE)
    if (tp == 0L) return(0)
    2 * tp / (2 * tp + fp + fn)
  }, numeric(1L))
  breaks <- seq(0, 1, length.out = 11L)
  bins <- cut(pmax(0, pmin(1, confidence)), breaks, include.lowest = TRUE)
  ece <- sum(vapply(levels(bins), function(bin) {
    index <- which(bins == bin)
    if (length(index) == 0L) return(0)
    length(index) / length(truth) * abs(mean(correct[index]) - mean(confidence[index]))
  }, numeric(1L)))
  list(
    balanced_accuracy = unname(mean(recall)),
    macro_f1 = unname(mean(f1)),
    ece = unname(ece),
    evaluated_cells = length(truth),
    evaluated_types = length(types)
  )
}

fit_isotonic <- function(raw_reliability, correct) {
  if (length(unique(correct)) < 2L) {
    stop("SingleR holdout has no correctness variation for calibration")
  }
  order_index <- order(raw_reliability, seq_along(raw_reliability))
  ordered_x <- raw_reliability[order_index]
  fitted <- isoreg(ordered_x, as.numeric(correct[order_index]))$yf
  points <- data.table(raw = ordered_x, fitted = fitted)[, .(value = mean(fitted)), by = raw]
  if (nrow(points) < 2L || any(!is.finite(points$raw)) || any(!is.finite(points$value))) {
    stop("SingleR isotonic calibration is degenerate")
  }
  list(x = points$raw, y = pmax(0, pmin(1, points$value)))
}

interpolate_isotonic <- function(raw, calibration) {
  pmax(0, pmin(1, approx(
    calibration$x, calibration$y, raw, rule = 2L, ties = "ordered"
  )$y))
}

normalize_log_counts <- function(counts) {
  totals <- Matrix::colSums(counts)
  if (any(!is.finite(totals)) || any(totals <= 0)) stop("Reference contains empty cells")
  normalized <- t(t(counts) / totals) * 10000
  log1p(normalized)
}

ontology_map <- function(observations) {
  mapping <- unique(observations[, .(cell_type, cell_type_ontology_term_id)])
  if (anyDuplicated(mapping$cell_type) || anyDuplicated(mapping$cell_type_ontology_term_id)) {
    stop("Reference cell type names and CL identifiers must be one-to-one")
  }
  setNames(as.character(mapping$cell_type_ontology_term_id), mapping$cell_type)
}

stratified_calibration_split <- function(labels, strata = NULL) {
  if (is.null(strata)) strata <- rep("", length(labels))
  groups <- split(seq_along(labels), interaction(labels, strata, drop = TRUE))
  calibration <- sort(unlist(lapply(groups, function(index) index[seq_along(index) %% 2L == 1L])))
  evaluation <- setdiff(seq_along(labels), calibration)
  if (length(calibration) < 20L || length(evaluation) < 20L) {
    stop("Held-out calibration/evaluation split is too small")
  }
  list(calibration = calibration, evaluation = evaluation)
}

stratified_three_way_split <- function(labels, strata = NULL) {
  if (is.null(strata)) strata <- rep("", length(labels))
  groups <- split(seq_along(labels), interaction(labels, strata, drop = TRUE))
  model_fit <- sort(unlist(lapply(groups, function(index) index[seq_along(index) %% 3L == 1L])))
  recalibration <- sort(unlist(lapply(groups, function(index) index[seq_along(index) %% 3L == 2L])))
  evaluation <- setdiff(seq_along(labels), c(model_fit, recalibration))
  if (length(model_fit) < 20L || length(recalibration) < 20L || length(evaluation) < 20L) {
    stop("Held-out RCTD model/calibration/evaluation split is too small")
  }
  list(model_fit = model_fit, recalibration = recalibration, evaluation = evaluation)
}

downsample_to_target_depth <- function(counts, target_depths, seed) {
  target_depths <- as.integer(unlist(target_depths, use.names = FALSE))
  if (length(target_depths) < 20L || anyNA(target_depths) || any(target_depths <= 0L)) {
    stop("Target count-depth calibration sample is invalid")
  }
  seed <- as.integer(seed)
  if (length(seed) != 1L || is.na(seed)) stop("Target count-depth seed is invalid")
  set.seed(seed)
  columns <- vector("list", ncol(counts))
  original_depths <- Matrix::colSums(counts)
  applied_depths <- numeric(ncol(counts))
  for (index in seq_len(ncol(counts))) {
    requested <- target_depths[((index - 1L) %% length(target_depths)) + 1L]
    total <- original_depths[index]
    applied <- min(as.integer(total), requested)
    applied_depths[index] <- applied
    if (applied <= 0L || total <= 0) {
      columns[[index]] <- Matrix::sparseMatrix(
        i = integer(), j = integer(), dims = c(nrow(counts), 1L)
      )
    } else if (applied == as.integer(total)) {
      columns[[index]] <- counts[, index, drop = FALSE]
    } else {
      sampled <- drop(rmultinom(1L, size = applied, prob = counts[, index]))
      columns[[index]] <- Matrix::Matrix(sampled, ncol = 1L, sparse = TRUE)
    }
  }
  result <- do.call(cbind, columns)
  rownames(result) <- rownames(counts)
  colnames(result) <- colnames(counts)
  list(
    counts = as(result, "dgCMatrix"),
    original_depths = original_depths,
    applied_depths = applied_depths
  )
}

build_singler_reference <- function(counts, observations, recipe, source_metadata) {
  training <- which(observations$partition == "train")
  holdout <- which(observations$partition == "holdout")
  expression <- normalize_log_counts(counts)
  train_expression <- expression[, training, drop = FALSE]
  holdout_counts <- counts[, holdout, drop = FALSE]
  depth_matched <- identical(recipe$calibration$match_target_count_depth, TRUE)
  depth_diagnostics <- list(target_count_depth_matched = FALSE)
  if (depth_matched) {
    depth_metadata <- source_metadata$target_panel$count_depth
    if (is.null(depth_metadata$deterministic_positive_depth_sample)) {
      stop("Target count-depth metadata is missing from the fetched reference")
    }
    downsampled <- downsample_to_target_depth(
      holdout_counts,
      depth_metadata$deterministic_positive_depth_sample,
      recipe$sampling$seed
    )
    holdout_counts <- downsampled$counts
    depth_diagnostics <- list(
      target_count_depth_matched = TRUE,
      target_depth_observations = depth_metadata$observation_count,
      target_depth_nonzero_observations = depth_metadata$nonzero_observation_count,
      target_depth_zero_observations = depth_metadata$zero_observation_count,
      target_depth_median = depth_metadata$quantiles$q50,
      holdout_original_depth_median = unname(median(downsampled$original_depths)),
      holdout_applied_depth_median = unname(median(downsampled$applied_depths))
    )
  }
  holdout_expression <- normalize_log_counts(holdout_counts)
  train_labels <- as.character(observations$cell_type[training])
  truth <- as.character(observations$cell_type[holdout])
  prediction <- SingleR(
    test = holdout_expression,
    ref = train_expression,
    labels = train_labels,
    BPPARAM = SerialParam()
  )
  scores <- as.matrix(prediction$scores)
  ordered_scores <- t(apply(scores, 1L, sort, decreasing = TRUE))
  best <- ordered_scores[, 1L]
  second <- if (ncol(ordered_scores) > 1L) ordered_scores[, 2L] else rep(NA_real_, nrow(scores))
  delta <- best - second
  delta_weight <- as.numeric(recipe$calibration$delta_weight)
  if (!is.finite(delta_weight) || delta_weight < 0) stop("Invalid SingleR delta weight")
  raw <- best + delta_weight * ifelse(is.na(delta), 0, delta)
  predicted <- as.character(prediction$pruned.labels)
  correct <- !is.na(predicted) & predicted == truth
  split <- stratified_calibration_split(truth)
  calibration_index <- split$calibration
  evaluation_index <- split$evaluation
  isotonic <- fit_isotonic(raw[calibration_index], correct[calibration_index])
  confidence <- interpolate_isotonic(raw, isotonic)
  correct_best <- best[calibration_index][correct[calibration_index]]
  correct_delta <- delta[calibration_index][
    correct[calibration_index] & is.finite(delta[calibration_index])
  ]
  if (length(correct_best) < 20L || length(correct_delta) < 20L) {
    stop("Too few correct held-out SingleR predictions for uncertainty thresholds")
  }
  score_quantile <- finite_fraction(recipe$calibration$uncertain_score_quantile, "score quantile")
  delta_quantile <- finite_fraction(recipe$calibration$uncertain_delta_quantile, "delta quantile")
  calibration <- list(
    id = paste0(
      recipe$reference_id,
      if (depth_matched) "-donor-holdout-target-depth-v2" else "-donor-holdout-v1"
    ),
    method = "isotonic",
    x = isotonic$x,
    y = isotonic$y,
    delta_weight = delta_weight,
    uncertain_min_score = unname(quantile(correct_best, score_quantile, names = FALSE)),
    uncertain_min_delta = unname(quantile(correct_delta, delta_quantile, names = FALSE))
  )
  metrics <- classification_metrics(
    truth[evaluation_index], predicted[evaluation_index], confidence[evaluation_index]
  )
  validation <- c(metrics, depth_diagnostics, list(
    calibration_completed = TRUE,
    holdout_donor = recipe$holdout_donor,
    train_cells = length(training),
    holdout_cells = length(holdout),
    calibration_cells = length(calibration_index),
    evaluation_cells = length(evaluation_index),
    reference_types = length(unique(train_labels)),
    seed_stability = 1,
    source_observation_count = source_metadata$source_observation_count
  ))
  reference <- list(
    expression = train_expression,
    labels = train_labels,
    ontology_ids = ontology_map(observations[training]),
    validation = validation
  )
  list(reference = reference, calibration = calibration, validation = validation)
}

make_pseudo_spots <- function(counts, observations, recipe) {
  holdout <- which(observations$partition == "holdout")
  labels <- as.character(observations$cell_type[holdout])
  by_type <- split(holdout, labels)
  cells_per_spot <- as.integer(recipe$calibration$cells_per_pseudo_spot)
  pure_per_type <- as.integer(recipe$calibration$pure_spots_per_type)
  mixed_count <- as.integer(recipe$calibration$mixed_spots)
  mixed_types <- recipe$calibration$mixed_types_per_spot
  if (is.null(mixed_types)) mixed_types <- 2L
  mixed_types <- as.integer(mixed_types)
  dominant_fraction <- recipe$calibration$mixed_dominant_fraction
  if (is.null(dominant_fraction)) dominant_fraction <- 0.7
  dominant_fraction <- as.numeric(dominant_fraction)
  if (cells_per_spot < 2L || pure_per_type < 2L || mixed_count < 2L) {
    stop("Pseudo-spot calibration counts are invalid")
  }
  columns <- list()
  truth <- character()
  known_mixed <- logical()
  cursor <- setNames(rep(1L, length(by_type)), names(by_type))
  take_cells <- function(type, number) {
    pool <- by_type[[type]]
    start <- cursor[[type]]
    indexes <- pool[((seq_len(number) + start - 2L) %% length(pool)) + 1L]
    cursor[[type]] <<- ((start + number - 2L) %% length(pool)) + 1L
    indexes
  }
  for (type in sort(names(by_type))) {
    for (index in seq_len(pure_per_type)) {
      cells <- take_cells(type, cells_per_spot)
      columns[[length(columns) + 1L]] <- Matrix::rowSums(counts[, cells, drop = FALSE])
      truth <- c(truth, type)
      known_mixed <- c(known_mixed, FALSE)
    }
  }
  types <- sort(names(by_type))
  if (mixed_types < 2L || mixed_types > length(types) ||
      !is.finite(dominant_fraction) || dominant_fraction <= 1 / mixed_types ||
      dominant_fraction >= 1) {
    stop("Pseudo-spot mixed-type design is invalid")
  }
  dominant_n <- max(1L, ceiling(cells_per_spot * dominant_fraction))
  remaining_n <- cells_per_spot - dominant_n
  if (remaining_n < mixed_types - 1L) {
    stop("Pseudo-spot size cannot represent the requested mixed cell types")
  }
  for (index in seq_len(mixed_count)) {
    dominant <- types[((index - 1L) %% length(types)) + 1L]
    secondary <- types[((index + seq_len(mixed_types - 1L) - 1L) %% length(types)) + 1L]
    secondary_sizes <- rep(remaining_n %/% (mixed_types - 1L), mixed_types - 1L)
    extra_secondaries <- remaining_n %% (mixed_types - 1L)
    if (extra_secondaries > 0L) {
      secondary_sizes[seq_len(extra_secondaries)] <-
        secondary_sizes[seq_len(extra_secondaries)] + 1L
    }
    cells <- take_cells(dominant, dominant_n)
    for (secondary_index in seq_along(secondary)) {
      cells <- c(
        cells,
        take_cells(secondary[[secondary_index]], secondary_sizes[[secondary_index]])
      )
    }
    columns[[length(columns) + 1L]] <- Matrix::rowSums(counts[, cells, drop = FALSE])
    truth <- c(truth, dominant)
    known_mixed <- c(known_mixed, TRUE)
  }
  matrix <- do.call(cbind, columns)
  matrix <- as(matrix, "dgCMatrix")
  colnames(matrix) <- sprintf("pseudo_%05d", seq_len(ncol(matrix)))
  list(
    counts = matrix, truth = truth, mixed = known_mixed,
    mixed_types_per_spot = mixed_types,
    mixed_dominant_fraction = dominant_fraction
  )
}

rctd_weights <- function(target_counts, reference_counts, cell_types, workers) {
  coordinates <- cbind(x = seq_len(ncol(target_counts)), y = rep(0, ncol(target_counts)))
  rownames(coordinates) <- colnames(target_counts)
  spatial <- SpatialExperiment(
    assays = list(counts = target_counts),
    spatialCoords = coordinates
  )
  reference <- SummarizedExperiment(
    assays = list(counts = reference_counts),
    colData = S4Vectors::DataFrame(
      cell_type = factor(cell_types),
      row.names = colnames(reference_counts)
    )
  )
  result <- run_rctd_full(
    createRctd(spatial, reference, cell_type_col = "cell_type", require_int = TRUE),
    workers
  )
  weights <- aligned_rctd_weights(result, colnames(target_counts))
  weights[!is.finite(weights) | weights < 0] <- 0
  totals <- rowSums(weights)
  if (any(!is.finite(totals)) || any(totals <= 0)) stop("Calibration RCTD did not converge")
  weights / totals
}

weight_features <- function(weights) {
  order_index <- t(apply(weights, 1L, order, decreasing = TRUE))
  top_index <- order_index[, 1L]
  second_index <- order_index[, 2L]
  top_weight <- weights[cbind(seq_len(nrow(weights)), top_index)]
  second_weight <- weights[cbind(seq_len(nrow(weights)), second_index)]
  entropy <- -rowSums(ifelse(weights > 0, weights * log(weights), 0)) / log(ncol(weights))
  list(
    predicted = colnames(weights)[top_index],
    frame = data.frame(
      top_weight = top_weight,
      delta = top_weight - second_weight,
      entropy = entropy
    )
  )
}

build_rctd_reference <- function(counts, observations, recipe, source_metadata) {
  training <- which(observations$partition == "train")
  train_counts <- counts[, training, drop = FALSE]
  train_labels <- as.character(observations$cell_type[training])
  pseudo <- make_pseudo_spots(counts, observations, recipe)
  workers <- as.integer(recipe$calibration$workers)
  if (workers < 1L || workers > 3L) stop("RCTD calibration workers must be in [1, 3]")
  weights <- rctd_weights(pseudo$counts, train_counts, train_labels, workers)
  features <- weight_features(weights)
  correct <- features$predicted == pseudo$truth
  split <- stratified_three_way_split(pseudo$truth, pseudo$mixed)
  model_fit_index <- split$model_fit
  recalibration_index <- split$recalibration
  evaluation_index <- split$evaluation
  if (length(unique(correct[model_fit_index])) < 2L) {
    stop("RCTD pseudo-spots have no correctness variation for logistic calibration")
  }
  if (length(unique(correct[recalibration_index])) < 2L) {
    stop("RCTD pseudo-spots have no correctness variation for isotonic recalibration")
  }
  model_data <- cbind(data.frame(correct = as.integer(correct)), features$frame)
  fit <- glm(
    correct ~ top_weight + delta + entropy,
    data = model_data[model_fit_index, , drop = FALSE], family = binomial()
  )
  coefficients <- coef(fit)
  if (any(!is.finite(coefficients))) stop("RCTD logistic calibration is degenerate")
  logistic_probability <- plogis(predict(fit, newdata = model_data, type = "link"))
  isotonic <- fit_isotonic(
    logistic_probability[recalibration_index], correct[recalibration_index]
  )
  confidence <- interpolate_isotonic(logistic_probability, isotonic)
  calibration <- list(
    id = paste0(recipe$reference_id, "-pseudo-spots-v3"),
    method = "logistic",
    coefficients = coefficients,
    probability_x = isotonic$x,
    probability_y = isotonic$y,
    mixed_min_top_weight = finite_fraction(
      recipe$calibration$mixed_min_top_weight, "mixed_min_top_weight"
    ),
    mixed_min_delta = finite_fraction(recipe$calibration$mixed_min_delta, "mixed_min_delta"),
    mixed_max_entropy = finite_fraction(
      recipe$calibration$mixed_max_entropy, "mixed_max_entropy"
    ),
    uncertain_max_confidence = finite_fraction(
      recipe$calibration$uncertain_min_confidence, "uncertain_min_confidence"
    ),
    mixed_types_per_pseudo_spot = pseudo$mixed_types_per_spot,
    mixed_dominant_fraction = pseudo$mixed_dominant_fraction
  )
  pure <- !pseudo$mixed & seq_along(pseudo$mixed) %in% evaluation_index
  logistic_metrics <- classification_metrics(
    pseudo$truth[evaluation_index], features$predicted[evaluation_index],
    logistic_probability[evaluation_index]
  )
  pure_metrics <- classification_metrics(
    pseudo$truth[pure], features$predicted[pure], confidence[pure]
  )
  calibration_metrics <- classification_metrics(
    pseudo$truth[evaluation_index], features$predicted[evaluation_index],
    confidence[evaluation_index]
  )
  metrics <- pure_metrics
  metrics$ece <- calibration_metrics$ece
  validation <- c(metrics, list(
    pure_ece = pure_metrics$ece,
    pre_isotonic_ece = logistic_metrics$ece,
    calibration_completed = TRUE,
    holdout_donor = recipe$holdout_donor,
    train_cells = length(training),
    holdout_cells = sum(observations$partition == "holdout"),
    model_fit_pseudo_spots = length(model_fit_index),
    calibration_pseudo_spots = length(recalibration_index),
    evaluation_pseudo_spots = length(evaluation_index),
    pseudo_spots = ncol(pseudo$counts),
    pseudo_mixed_spots = sum(pseudo$mixed),
    mixed_types_per_pseudo_spot = pseudo$mixed_types_per_spot,
    mixed_dominant_fraction = pseudo$mixed_dominant_fraction,
    reference_types = length(unique(train_labels)),
    seed_stability = 1,
    source_observation_count = source_metadata$source_observation_count
  ))
  reference <- list(
    counts = train_counts,
    cell_types = train_labels,
    ontology_ids = ontology_map(observations[training]),
    validation = validation
  )
  list(reference = reference, calibration = calibration, validation = validation)
}

build_reference <- function(recipe, output) {
  config_path <- recipe$.config_path
  if (is.null(config_path)) stop("Reference wrapper did not supply the recipe path")
  fetch_script <- normalizePath(
    file.path(dirname(config_path), recipe$fetch_script), mustWork = TRUE
  )
  fetch_dir <- NULL
  fetch_errors <- character()
  fetch_attempt <- 0L
  for (attempt in seq_len(3L)) {
    candidate <- tempfile("census-fetch-", tmpdir = output)
    dir.create(candidate)
    command <- c(fetch_script, "--config", config_path, "--output", candidate)
    status <- system2(Sys.which("python"), command, stdout = TRUE, stderr = TRUE)
    succeeded <- is.null(attr(status, "status")) || attr(status, "status") == 0L
    if (succeeded) {
      fetch_dir <- candidate
      fetch_attempt <- attempt
      break
    }
    fetch_errors <- c(fetch_errors, paste0("attempt ", attempt, ":\n", paste(status, collapse = "\n")))
    unlink(candidate, recursive = TRUE, force = TRUE)
    if (attempt < 3L) Sys.sleep(2^(attempt - 1L))
  }
  if (is.null(fetch_dir)) {
    stop(paste(c("Census reference fetch failed after 3 attempts", fetch_errors), collapse = "\n"))
  }
  on.exit(unlink(fetch_dir, recursive = TRUE, force = TRUE), add = TRUE)
  counts <- t(readMM(file.path(fetch_dir, "census_matrix.mtx")))
  observations <- fread(file.path(fetch_dir, "census_observations.tsv"), sep = "\t")
  genes <- fread(file.path(fetch_dir, "census_genes.tsv"), sep = "\t")
  if (ncol(counts) != nrow(observations) || nrow(counts) != nrow(genes)) {
    stop("Fetched Census sparse matrix is not aligned to metadata")
  }
  rownames(counts) <- genes$gene_id
  colnames(counts) <- paste0("census_", observations$soma_joinid)
  counts <- as(counts, "dgCMatrix")
  source_metadata <- fromJSON(file.path(fetch_dir, "source_metadata.json"), simplifyVector = FALSE)
  source_metadata$fetch_attempts <- fetch_attempt
  write_json(
    source_metadata, file.path(fetch_dir, "source_metadata.json"),
    auto_unbox = TRUE, pretty = TRUE, null = "null"
  )
  file.copy(
    file.path(fetch_dir, "source_metadata.json"),
    file.path(output, "source_metadata.json"),
    overwrite = FALSE
  )
  built <- switch(
    recipe$method,
    singler = build_singler_reference(counts, observations, recipe, source_metadata),
    rctd = build_rctd_reference(counts, observations, recipe, source_metadata),
    stop("Unsupported reference method")
  )
  saveRDS(built$reference, file.path(output, "reference.rds"), compress = "xz")
  saveRDS(built$calibration, file.path(output, "calibration.rds"), compress = "xz")
  draft <- list(
    schema_version = "1.0",
    reference_id = recipe$reference_id,
    method = recipe$method,
    source = recipe$source,
    selection = source_metadata,
    validation = built$validation
  )
  write_json(
    draft, file.path(output, "draft_metadata.json"),
    auto_unbox = TRUE, pretty = TRUE, null = "null"
  )
}
