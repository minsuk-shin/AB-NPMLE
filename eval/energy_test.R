#!/usr/bin/env Rscript
# Energy permutation test on saved sample CSVs. No sampling or NPZ lookup.
# Run from the repository root:
# Usage: Rscript eval/energy_test.R [--sample DATA_ID CSV ... | --sample-dir DIR]
#        --output-dir DIR [--n-perm 1000]

args <- commandArgs(trailingOnly = TRUE)
samples_arg <- list()
sample_dir <- NULL
output_dir <- NULL
nperm <- 1000L
position <- 1L
while (position <= length(args)) {
  option <- args[[position]]
  if (identical(option, "--sample") && position + 2L <= length(args)) {
    samples_arg[[length(samples_arg) + 1L]] <- list(
      data_id = suppressWarnings(as.integer(args[[position + 1L]])),
      path = args[[position + 2L]])
    position <- position + 3L
  } else if (identical(option, "--sample-dir") && position + 1L <= length(args)) {
    sample_dir <- args[[position + 1L]]
    position <- position + 2L
  } else if (identical(option, "--output-dir") && position + 1L <= length(args)) {
    output_dir <- args[[position + 1L]]
    position <- position + 2L
  } else if (identical(option, "--n-perm") && position + 1L <= length(args)) {
    nperm <- suppressWarnings(as.integer(args[[position + 1L]]))
    position <- position + 2L
  } else {
    stop("unknown or incomplete argument: ", option)
  }
}
if (!is.null(sample_dir)) {
  if (!dir.exists(sample_dir)) stop("missing sample directory: ", sample_dir)
  paths <- list.files(sample_dir, pattern = "^data[0-9]+[.]csv$", full.names = TRUE)
  ids <- as.integer(sub("^data([0-9]+)[.]csv$", "\\1", basename(paths)))
  for (index in order(ids)) {
    samples_arg[[length(samples_arg) + 1L]] <- list(data_id = ids[[index]],
                                                    path = paths[[index]])
  }
}
if (!length(samples_arg) || is.null(output_dir) || !nzchar(output_dir) ||
    is.na(nperm) || nperm < 1L) {
  stop("usage: Rscript eval/energy_test.R [--sample DATA_ID CSV ... | --sample-dir DIR] --output-dir DIR [--n-perm 1000]")
}
data_ids <- vapply(samples_arg, function(item) item$data_id, integer(1L))
if (anyNA(data_ids) || any(data_ids < 1L) || anyDuplicated(data_ids)) {
  stop("DATA_ID values must be distinct positive integers")
}
if (!requireNamespace("energy", quietly = TRUE)) stop("R package 'energy' is required")

read_samples <- function(path) {
  if (!file.exists(path)) stop("missing sample CSV: ", path)
  samples <- read.csv(path, check.names = FALSE)
  if ("AB" %in% names(samples) && !("AMEM" %in% names(samples))) {
    names(samples)[names(samples) == "AB"] <- "AMEM"
  }
  if (!(identical(names(samples), c("NPEB", "AMEM")) ||
        identical(names(samples), c("NPEB", "GB")) ||
        identical(names(samples), c("NPEB", "AMEM", "GB"))) ||
      nrow(samples) != 1000L || any(!is.finite(as.matrix(samples)))) {
    stop("expected 1000 finite NPEB/AMEM samples, with optional GB, in ", path)
  }
  samples
}

rows <- list()
for (item in samples_arg) {
  samples <- read_samples(item$path)
  pairs <- list()
  if ("AMEM" %in% names(samples)) pairs[[length(pairs) + 1L]] <- c("NPEB", "AMEM")
  if ("GB" %in% names(samples)) pairs[[length(pairs) + 1L]] <- c("NPEB", "GB")
  for (pair_index in seq_along(pairs)) {
    pair <- pairs[[pair_index]]
    pair_seed_index <- if (identical(pair[[2L]], "GB")) 2L else 1L
    x <- samples[[pair[[1L]]]]
    y <- samples[[pair[[2L]]]]
    test_seed <- 82000L + 100L * item$data_id + pair_seed_index
    set.seed(test_seed)
    result <- energy::eqdist.etest(matrix(c(x, y), ncol = 1L),
                                   sizes = c(length(x), length(y)), R = nperm)
    pvalue <- as.numeric(result$p.value)
    rows[[length(rows) + 1L]] <- data.frame(
      data_id = item$data_id, comparison = paste(pair, collapse = " vs "),
      energy_statistic = as.numeric(result$statistic), p_value = pvalue,
      reject_0_05 = as.integer(pvalue < 0.05),
      n_permutations = nperm, seed = test_seed)
  }
}

each <- do.call(rbind, rows)
summary <- do.call(rbind, lapply(unique(each$comparison), function(label) {
  subset <- each[each$comparison == label, ]
  data.frame(
    comparison = label, n_tested = nrow(subset),
    mean_energy = mean(subset$energy_statistic),
    mean_p = mean(subset$p_value), sd_p = sd(subset$p_value),
    n_reject_0_05 = sum(subset$reject_0_05),
    reject_rate_0_05 = mean(subset$reject_0_05), n_permutations = nperm)
}))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(each, file.path(output_dir, "energy_each.csv"), row.names = FALSE)
write.csv(summary, file.path(output_dir, "energy_summary.csv"), row.names = FALSE)
print(summary, row.names = FALSE)
