import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.stats import ks_2samp

RESULTS_DIR = Path("./results_marginal")
N_DATA      = 50
ALPHA       = 0.05
N_SAMPLE    = 1000
SIGMA_EPS   = 1.0
RNG         = np.random.default_rng(42)

DATA_NAME   = "WCS1_Normal"

OUTPUT_DIR  = Path(f"./ks_results/{DATA_NAME}")
SAMPLES_DIR = OUTPUT_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

OI_BLUE = "#0072B2"


def sample_from_mixture(atoms, weights, sigma_eps, n_samples, rng):
    """Draws n_samples from the marginal x = theta + eps induced by a discrete
    mixing distribution (atoms, weights): pick theta ~ (atoms, weights), then
    add N(0, sigma_eps^2) noise."""
    w              = np.array(weights) / np.sum(weights)
    chosen_thetas  = rng.choice(atoms, size=n_samples, p=w)
    return chosen_thetas + rng.normal(0, sigma_eps, n_samples)


records = []
errors  = []

for i in range(1, N_DATA + 1):
    folder   = f"data{i}"
    npz_path = RESULTS_DIR / folder / DATA_NAME / f"{DATA_NAME}_marginal_all_agg.npz"

    if not npz_path.exists():
        msg = f"[SKIP] npz not found: {npz_path}"
        print(msg); errors.append(msg); continue

    npz     = np.load(npz_path, allow_pickle=True)
    npmle_a = npz["npmle_a"]
    npmle_w = npz["npmle_w"]
    ab_a    = np.asarray(npz["ab_atoms"][0], dtype=np.float64)
    ab_w    = np.asarray(npz["ab_weights"][0], dtype=np.float64)

    s_npmle = sample_from_mixture(npmle_a, npmle_w, SIGMA_EPS, N_SAMPLE, RNG)
    s_ab    = sample_from_mixture(ab_a,    ab_w,    SIGMA_EPS, N_SAMPLE, RNG)

    sample_df = pd.DataFrame({
        "s_npmle" : s_npmle,
        "s_ab"    : s_ab,
    })
    sample_csv_path = SAMPLES_DIR / f"samples_data{i}.csv"
    sample_df.to_csv(sample_csv_path, index=False)

    stat_na, pval_na = ks_2samp(s_npmle, s_ab)

    records.append({
        "data_id"           : i,
        "ks_npmle_ab"       : stat_na,
        "pval_npmle_ab"     : pval_na,
        "reject_npmle_ab"   : int(pval_na < ALPHA),
    })
    print(
        f"[OK] data{i:>2d} | "
        f"NPMLE-AB: KS={stat_na:.4f} p={pval_na:.4f} rej={int(pval_na<ALPHA)}"
    )

results_df = pd.DataFrame(records)
results_df.to_csv(OUTPUT_DIR / "ks_results_each.csv", index=False)

ks_vals   = results_df["ks_npmle_ab"]
pval_vals = results_df["pval_npmle_ab"]
rej_vals  = results_df["reject_npmle_ab"]

summary_df = pd.DataFrame([{
    "comparison"      : "NPMLE vs AB",
    "n_tested"        : len(results_df),
    "mean_ks"         : ks_vals.mean(),
    "std_ks"          : ks_vals.std(),
    "min_ks"          : ks_vals.min(),
    "max_ks"          : ks_vals.max(),
    "mean_pval"       : pval_vals.mean(),
    "std_pval"        : pval_vals.std(),
    "median_pval"     : pval_vals.median(),
    "n_reject"        : rej_vals.sum(),
    "reject_rate"     : rej_vals.mean(),
}])
summary_df.to_csv(OUTPUT_DIR / "ks_results_summary.csv", index=False)

print("\n" + "=" * 60)
print("[Summary]")
print(summary_df.to_string(index=False))


plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#F7F7F7",
    "axes.edgecolor"    : "#CCCCCC",
    "axes.linewidth"    : 0.8,
    "axes.grid"         : True,
    "grid.color"        : "white",
    "grid.linewidth"    : 1.0,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "font.family"       : "DejaVu Sans",
    "font.size"         : 11,
    "axes.labelsize"    : 12,
    "axes.titlesize"    : 13,
    "axes.titleweight"  : "bold",
    "legend.frameon"    : True,
    "legend.framealpha" : 0.92,
    "legend.edgecolor"  : "#CCCCCC",
})

fig, ax = plt.subplots(figsize=(5, 4.5))
fig.suptitle(
    f"KS Test p-value Distribution: NPMLE vs AB  (n={N_SAMPLE}, B=50 datasets, alpha={ALPHA})",
    fontsize=13, fontweight="bold", y=1.02
)

bins = np.linspace(0, 1, 21)

pvals       = results_df["pval_npmle_ab"]
n_reject    = (pvals < ALPHA).sum()
reject_rate = n_reject / len(pvals)

ax.hist(pvals, bins=bins, color=OI_BLUE, alpha=0.75, edgecolor="white", linewidth=0.8)
ax.axvline(ALPHA, color="#333333", lw=1.5, linestyle="--", label=f"alpha = {ALPHA}")
ax.axvspan(0, ALPHA, alpha=0.10, color="#333333")

ax.set_title("NPMLE vs AB")
ax.set_xlabel("p-value")
ax.set_ylabel("Count")
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

ax.text(
    0.97, 0.97,
    f"reject: {n_reject}/{len(pvals)} ({reject_rate:.0%})\n"
    f"median p: {pvals.median():.3f}",
    transform=ax.transAxes,
    ha="right", va="top", fontsize=9.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9)
)
ax.legend(fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ks_pvalue_histograms.png", dpi=180, bbox_inches="tight")
print("\nHistogram saved -> ks_pvalue_histograms.png")
plt.close()

print(f"\nOutput dir: ./ks_results/{DATA_NAME}/")
print(f"  samples/samples_data{{1..{N_DATA}}}.csv")
print(f"  ks_results_each.csv")
print(f"  ks_results_summary.csv")
print(f"  ks_pvalue_histograms.png")

if errors:
    print(f"\n{len(errors)} error(s):")
    for e in errors:
        print(" ", e)
