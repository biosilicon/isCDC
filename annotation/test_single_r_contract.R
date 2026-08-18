#!/usr/bin/env Rscript

# Fast adapter contract test. It exercises the exact SingleR DataFrame fields used by
# run_single_r.R without requiring a large biological reference.
suppressPackageStartupMessages(library(S4Vectors))

prediction <- DataFrame(
  labels = c("T cell", "B cell"),
  pruned.labels = c("T cell", NA_character_)
)
prediction$scores <- matrix(
  c(0.9, 0.2, 0.4, 0.5), nrow = 2L, byrow = TRUE,
  dimnames = list(NULL, c("T cell", "B cell"))
)
scores <- as.matrix(prediction$scores)
ordered <- t(apply(scores, 1L, sort, decreasing = TRUE))
best <- ordered[, 1L]
second <- ordered[, 2L]
delta <- best - second
calibration <- list(x = c(0, 1), y = c(0.05, 0.95), delta_weight = 0.5)
confidence <- approx(
  calibration$x, calibration$y, best + calibration$delta_weight * delta,
  rule = 2L, ties = "ordered"
)$y
uncertain <- is.na(prediction$pruned.labels)

stopifnot(
  identical(dim(scores), c(2L, 2L)),
  isTRUE(all.equal(delta, c(0.7, 0.1))),
  all(is.finite(confidence) & confidence >= 0 & confidence <= 1),
  identical(uncertain, c(FALSE, TRUE))
)
