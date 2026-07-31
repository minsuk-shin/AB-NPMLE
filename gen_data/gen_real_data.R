# gen_real_data.R
# ----------------
# Downloads the GIANT height GWAS summary statistics, computes the
# Z-statistic Z = b/SE per SNP, QC-filters, subsamples n_target SNPs, and
# writes them to a csv (column: x) for giant_csv_to_npz.py to convert into
# the .npz format used by the rest of the pipeline.

install.packages("data.table")
install.packages("R.utils")
library(data.table)
library(R.utils)

set.seed(1)
n_target <- 1000
url  <- "https://portals.broadinstitute.org/collaboration/giant/images/0/01/GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt.gz"
dest <- "giant_height.txt.gz"
options(timeout = 600)
if (!file.exists(dest)) download.file(url, dest, mode = "wb")
dt <- fread(dest)
print(names(dt)); print(head(dt))
dt[, MAF := pmin(`Freq.Allele1.HapMapCEU`, 1 - `Freq.Allele1.HapMapCEU`)]
dt[, X   := b / SE]
dt <- dt[is.finite(X) & is.finite(SE) & SE > 0 & is.finite(MAF) & MAF >= 0.01]
cat("SNP count after QC:", nrow(dt), "\n")

idx <- sample(nrow(dt), min(n_target, nrow(dt)))
x   <- dt[idx, X]
cat(sprintf("n = %d | mean(x) = %.3f | var(x) = %.3f\n",
            length(x), mean(x), var(x)))
fwrite(data.table(x = x), "./giant_height_x.csv")
cat("save complete: giant_height_x.csv (n =", length(x), ")\n")
