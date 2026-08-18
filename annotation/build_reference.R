#!/usr/bin/env Rscript

# Reference building is recipe-driven because species/tissue/disease matching and
# held-out calibration are scientific decisions. A recipe supplies an audited R builder
# which must write reference.rds, calibration.rds, and draft_metadata.json. This wrapper
# records exact checksums only after that builder succeeds.
suppressPackageStartupMessages({
  library(jsonlite)
  library(tools)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L || args[1L] != "--config" || args[3L] != "--output") {
  stop("usage: build_reference.R --config RECIPE --output DIR")
}
config_path <- normalizePath(args[2L], mustWork = TRUE)
output <- normalizePath(args[4L], mustWork = TRUE)
if (!requireNamespace("yaml", quietly = TRUE)) {
  stop("The pinned reference builder requires the yaml R package")
}
recipe <- yaml::read_yaml(config_path)
required <- c("reference_id", "species", "tissue", "version", "builder_script")
if (!all(required %in% names(recipe))) stop("Reference recipe is incomplete")
builder <- normalizePath(file.path(dirname(config_path), recipe$builder_script), mustWork = TRUE)
recipe$.config_path <- config_path
source(builder, local = TRUE)
if (!exists("build_reference", mode = "function")) {
  stop("Reference builder must define build_reference(recipe, output)")
}
build_reference(recipe, output)

files <- c(
  "reference.rds", "calibration.rds", "source_metadata.json", "draft_metadata.json"
)
if (!all(file.exists(file.path(output, files)))) stop("Reference builder output is incomplete")
records <- lapply(files, function(name) {
  path <- file.path(output, name)
  list(
    name = name,
    size = unname(file.info(path)$size),
    sha256 = digest::digest(path, algo = "sha256", file = TRUE, serialize = FALSE)
  )
})
metadata <- list(
  schema_version = "1.0",
  reference_id = recipe$reference_id,
  species = recipe$species,
  tissue = recipe$tissue,
  version = recipe$version,
  source = recipe$source,
  built_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  software = list(
    R = R.version.string,
    SingleR = as.character(packageVersion("SingleR")),
    spacexr = as.character(packageVersion("spacexr")),
    BiocParallel = as.character(packageVersion("BiocParallel"))
  ),
  files = records
)
write_json(metadata, file.path(output, "reference.json"), auto_unbox = TRUE, pretty = TRUE)
