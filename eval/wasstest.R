library(twosamples)

DATA_NAME   <- "WCS1_Normal"
SAMPLES_DIR <- file.path("./ks_results", DATA_NAME, "samples")

N_DATA  <- 50
ALPHA   <- 0.05
N_BOOTS <- 1000

pairs <- list(
  list(name = "NPMLE vs AB", col1 = "s_npmle", col2 = "s_ab")
)

records <- list()
errors  <- c()

for (i in seq_len(N_DATA)) {

  csv_path <- file.path(SAMPLES_DIR, paste0("samples_data", i, ".csv"))

  df <- tryCatch(
    read.csv(csv_path),
    error = function(e) NULL
  )
  if (is.null(df)) {
    msg <- paste("[SKIP] csv not found:", csv_path)
    message(msg); errors <- c(errors, msg); next
  }

  for (p in pairs) {
    x1  <- df[[p$col1]]
    x2  <- df[[p$col2]]
    res <- wass_test(x1, x2, nboots = N_BOOTS)

    stat   <- res[1]
    pval   <- res[2]
    reject <- ifelse(pval < ALPHA, 1L, 0L)

    records[[length(records) + 1]] <- data.frame(
      data_id    = i,
      comparison = p$name,
      statistic  = stat,
      pvalue     = pval,
      reject     = reject
    )
    message(sprintf("[OK] data%02d | %-16s | W1=%.4f | p=%.4f | reject=%d",
                    i, p$name, stat, pval, reject))
  }
}

results_df <- do.call(rbind, records)

summary_rows <- lapply(pairs, function(p) {
  sub <- results_df[results_df$comparison == p$name, ]
  data.frame(
    comparison    = p$name,
    n_tested      = nrow(sub),
    alpha         = ALPHA,
    mean_stat     = mean(sub$statistic),
    sd_stat       = sd(sub$statistic),
    mean_pvalue   = mean(sub$pvalue),
    sd_pvalue     = sd(sub$pvalue),
    median_pvalue = median(sub$pvalue),
    n_reject      = sum(sub$reject),
    reject_rate   = mean(sub$reject)
  )
})
summary_df <- do.call(rbind, summary_rows)

write.csv(results_df, "wass_results_each.csv",    row.names = FALSE)
write.csv(summary_df, "wass_results_summary.csv", row.names = FALSE)

cat("\n========================================\n")
cat("Per-dataset results -> wass_results_each.csv\n")
cat("Summary results     -> wass_results_summary.csv\n\n")
print(summary_df, row.names = FALSE)

if (length(errors) > 0) {
  cat(sprintf("\n%d error(s):\n", length(errors)))
  for (e in errors) cat(" ", e, "\n")
}
