#!/usr/bin/env Rscript

source("annotation/build_census_reference.R")

metrics <- classification_metrics(
  c("A", "A", "B", "B"),
  c("A", "B", "B", "B"),
  c(0.9, 0.4, 0.8, 0.7)
)
stopifnot(abs(metrics$balanced_accuracy - 0.75) < 1e-12)
stopifnot(metrics$macro_f1 > 0.73, metrics$macro_f1 < 0.74)

counts <- Matrix::Matrix(
  matrix(seq_len(120), nrow = 6L, ncol = 20L), sparse = TRUE
)
observations <- data.table::data.table(
  partition = rep("holdout", 20L),
  cell_type = rep(c("A", "B"), each = 10L)
)
recipe <- list(calibration = list(
  cells_per_pseudo_spot = 4L,
  pure_spots_per_type = 3L,
  mixed_spots = 4L
))
first <- make_pseudo_spots(counts, observations, recipe)
second <- make_pseudo_spots(counts, observations, recipe)
stopifnot(identical(first$truth, second$truth))
stopifnot(identical(first$mixed, second$mixed))
stopifnot(identical(as.matrix(first$counts), as.matrix(second$counts)))
stopifnot(ncol(first$counts) == 10L, sum(first$mixed) == 4L)

depth_counts <- Matrix::Matrix(
  matrix(c(10, 0, 0, 10, 3, 7), nrow = 2L), sparse = TRUE
)
depth_sample <- rep(c(4L, 5L, 6L), 7L)
depth_first <- downsample_to_target_depth(depth_counts, depth_sample, 17L)
depth_second <- downsample_to_target_depth(depth_counts, depth_sample, 17L)
stopifnot(identical(as.matrix(depth_first$counts), as.matrix(depth_second$counts)))
stopifnot(identical(as.numeric(Matrix::colSums(depth_first$counts)), c(4, 5, 6)))

weight_fixture <- SpatialExperiment::SpatialExperiment(
  assays = list(weights = Matrix::Matrix(
    matrix(c(0.8, 0.2, 0.1, 0.9), nrow = 2L), sparse = TRUE,
    dimnames = list(c("A", "B"), NULL)
  )),
  colData = S4Vectors::DataFrame(row.names = c("spot_b", "spot_a")),
  spatialCoords = matrix(0, nrow = 2L, ncol = 2L)
)
aligned <- aligned_rctd_weights(weight_fixture, c("spot_a", "filtered_spot", "spot_b"))
stopifnot(identical(rownames(aligned), c("spot_a", "filtered_spot", "spot_b")))
stopifnot(identical(colnames(aligned), c("A", "B")))
stopifnot(identical(as.numeric(aligned[1L, ]), c(0.1, 0.9)))
stopifnot(identical(as.numeric(aligned[2L, ]), c(0, 0)))
