"""
Bootstrap of the Posterior (Mixture Aggregation) — NPMLE + AB-NPMLE
Model: x_i = theta_i + eps_i, eps_i ~ N(0, sigma^2)
 
For each bootstrap replicate's pihat_b, computes the conditional posterior
pi_b(theta | x_j) and averages over B replicates into a single "bootstrap of
the posterior" mixture. NPMLE and AB-NPMLE are processed identically.

Usage
-----
  python eval/posterior_bootstrap.py \
      --npmle_result ./sim_data_npmle/data1/result_npmle_WCS1_Normal.npz \
      --ab_result    ./sim_data_gms_1000/data1/WCS1_Normal/result_AB_npmle_WCS1_Normal.npz \
      --data_path    ./sim_data/data1/WCS1_Normal.npz \
      --x_values     -0 \
      --sigma        1.0 \
      --summary_plot \
      --save_dir     ./posterior_bootstrap_out
"""

import numpy as np
import os
import csv
import argparse
import warnings
warnings.filterwarnings("ignore")


def _normalize_atoms_weights(atoms_list, weights_list):
    """
    Cleans up a per-bootstrap-replicate list of (atoms, weights) into a common
    format: drops Nones, casts to float64 + ravel, renormalizes weights (sum=1).
    """
    out_atoms, out_weights = [], []
    for a, w in zip(atoms_list, weights_list):
        if a is None or w is None:
            continue
        a = np.asarray(a, dtype=np.float64).ravel()
        w = np.asarray(w, dtype=np.float64).ravel()
        w = w / (w.sum() + 1e-300)
        out_atoms.append(a)
        out_weights.append(w)
    return out_atoms, out_weights


def load_bootstrap(npz_path: str):
    """
    Reads a bootstrap result npz (from run_npmle_*.py or infer.py).
    npz["atoms"] and npz["weights"] are each a (B,) object array, where each
    element is one bootstrap replicate Ghat_b's atom locations / probability
    masses. Replicates where the fit failed (atoms=None) are skipped.
    """
    data        = np.load(npz_path, allow_pickle=True)
    atoms_raw   = data["atoms"]
    weights_raw = data["weights"]
    return _normalize_atoms_weights(list(atoms_raw), list(weights_raw))


def load_x(data_path: str) -> np.ndarray:
    raw   = np.load(data_path, allow_pickle=True)
    x_key = "x" if "x" in raw.files else ("z" if "z" in raw.files else raw.files[0])
    return np.asarray(raw[x_key], dtype=np.float64).ravel()


def aggregate_bootstrap_posterior(x_j: float,
                                   atoms_list: list,
                                   weights_list: list,
                                   sigma: float):
    """
    From the B bootstrap results (atoms_list[b], weights_list[b]), computes
    each replicate's posterior pi_b(theta|x_j) and combines them by (1/B)
    weighted averaging into a single mixture posterior ("bootstrap of the
    posterior"). Works identically whether the (atoms_list, weights_list)
    come from NPMLE or AB-NPMLE, as long as the format (a list of atoms
    arrays, weights arrays) matches.
    """
    atoms_chunks = []
    probs_chunks = []
    B = len(atoms_list)
    for a, w in zip(atoms_list, weights_list):
        a = np.asarray(a, dtype=np.float64).ravel()
        w = np.asarray(w, dtype=np.float64).ravel()
        w = w / (w.sum() + 1e-300)
        log_lik  = -0.5 * ((x_j - a) / sigma) ** 2
        log_lik -= log_lik.max()
        lik  = np.exp(log_lik) * w
        post = lik / (lik.sum() + 1e-300)
        atoms_chunks.append(a)
        probs_chunks.append(post / B)
    atoms_all = np.concatenate(atoms_chunks)
    probs_all = np.concatenate(probs_chunks)
    return atoms_all, probs_all, B


def weighted_mean_var(values: np.ndarray, weights: np.ndarray):
    """Mean and variance of a discrete distribution (values, weights). weights is auto-normalized."""
    w  = weights / weights.sum()
    m  = float((w * values).sum())
    m2 = float((w * values ** 2).sum())
    return m, m2 - m ** 2


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q):
    """
    Computes quantile q (scalar or array) of the discrete distribution
    (values[k], weights[k]). weights need not sum to 1 — it's normalized internally.
    """
    order    = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    w_sorted = w_sorted / w_sorted.sum()
    cdf      = np.cumsum(w_sorted)
    cdf[-1]  = 1.0

    q_arr  = np.atleast_1d(np.asarray(q, dtype=np.float64))
    idx    = np.searchsorted(cdf, q_arr, side="left")
    idx    = np.clip(idx, 0, len(v_sorted) - 1)
    result = v_sorted[idx]
    return result if np.ndim(q) > 0 else float(result[0])


def posterior_bootstrap_summary(x_j: float,
                                 atoms_list: list,
                                 weights_list: list,
                                 sigma: float):
    """
    Computes the bootstrap of the posterior (mixture) for a single x_j, and
    returns both the full distribution (atoms_all, probs_all) and summary
    statistics. Works identically whichever model's (atoms_list, weights_list)
    (NPMLE or AB-NPMLE) is passed in.
    """
    atoms_all, probs_all, B = aggregate_bootstrap_posterior(
        x_j, atoms_list, weights_list, sigma
    )
    mean_val, var_val = weighted_mean_var(atoms_all, probs_all)

    return {
        "x_j":       x_j,
        "B":         B,
        "atoms_all": atoms_all,
        "probs_all": probs_all,
        "mean":      mean_val,
        "std":       float(np.sqrt(max(var_val, 0.0))),
        "var":       var_val,
        "median":    weighted_quantile(atoms_all, probs_all, 0.5),
        "q25":       weighted_quantile(atoms_all, probs_all, 0.25),
        "q75":       weighted_quantile(atoms_all, probs_all, 0.75),
    }


def plot_posterior_bootstrap(summary: dict, save_path: str, n_bins: int = 80):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        atoms_all = summary["atoms_all"]
        probs_all = summary["probs_all"]

        lo_edge = atoms_all.min() - 1e-6
        hi_edge = atoms_all.max() + 1e-6
        bins = np.linspace(lo_edge, hi_edge, n_bins + 1)
        hist, edges = np.histogram(atoms_all, bins=bins, weights=probs_all)
        centers = 0.5 * (edges[:-1] + edges[1:])

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(centers, hist, width=(edges[1] - edges[0]),
               color="steelblue", alpha=0.85, edgecolor="white")
        ax.axvline(summary["mean"], color="black", lw=1.8, ls="--",
                   label=f"mean={summary['mean']:.4f}")
        ax.set_title(f"Bootstrap of the posterior  (x_j={summary['x_j']:+.4f}, "
                    f"B={summary['B']})")
        ax.set_xlabel("theta")
        ax.set_ylabel("mixture probability mass")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {save_path}")
    except Exception as e:
        print(f"  [Warning] Plot failed for x_j={summary['x_j']}: {e}")


def _weighted_range(atoms: np.ndarray, weights: np.ndarray, lo=0.005, hi=0.995):
    """Same approach as ab_npmle_iter.get_weighted_range."""
    order    = np.argsort(atoms)
    a_sorted = atoms[order]
    w_sorted = weights[order] / weights[order].sum()
    cum_w    = np.cumsum(w_sorted)
    x_min = float(a_sorted[np.searchsorted(cum_w, lo)])
    x_max = float(a_sorted[np.searchsorted(cum_w, hi)])
    return x_min, x_max


def plot_bootstrap_comparison(atoms_ab: np.ndarray, probs_ab: np.ndarray, B_ab: int,
                               atoms_orig: np.ndarray, probs_orig: np.ndarray, B_orig: int,
                               x_j: float, save_path: str):
    """
    Compares AB-NPMLE (ab) vs NPMLE (orig)'s bootstrap-of-the-posterior
    mixture pi_agg(theta|x_j) as a KDE plot.
    """
    try:
        import matplotlib
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"]  = 42
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import seaborn as sns

        C_AB     = "#DC2626"
        C_NPMLE  = "#2563EB"

        ab_xmin, ab_xmax = _weighted_range(atoms_ab, probs_ab)
        orig_xmin, orig_xmax = _weighted_range(atoms_orig, probs_orig)
        x_min = min(ab_xmin, orig_xmin)
        x_max = max(ab_xmax, orig_xmax)
        x_pad = (x_max - x_min) * 0.05
        x_lo, x_hi = x_min - x_pad, x_max + x_pad

        FONTSIZE = 23
        plt.rcParams.update({
            "figure.facecolor"  : "white",
            "axes.facecolor"    : "#F7F7F7",
            "axes.edgecolor"    : "#CCCCCC",
            "axes.linewidth"    : 0.8,
            "axes.grid"         : False,
            "axes.spines.top"   : False,
            "axes.spines.right" : False,
            "xtick.direction"   : "out",
            "ytick.direction"   : "out",
            "font.family"       : "DejaVu Sans",
            "font.size"         : FONTSIZE,
            "axes.labelsize"    : FONTSIZE,
            "xtick.labelsize"   : FONTSIZE,
            "ytick.labelsize"   : FONTSIZE,
            "axes.titlesize"    : FONTSIZE,
            "axes.titleweight"  : "bold",
            "legend.frameon"    : True,
            "legend.framealpha" : 0.92,
            "legend.edgecolor"  : "#CCCCCC",
            "legend.fontsize"   : FONTSIZE,
        })
        LEGEND_KW = dict(handlelength=1.0, handletextpad=0.4,
                          borderpad=0.3, labelspacing=0.3)

        fig, ax = plt.subplots(figsize=(4, 4))
        sns.kdeplot(x=atoms_ab, weights=probs_ab, bw_adjust=0.05,
                    fill=True, color=C_AB, alpha=0.35,
                    linewidth=2.5, linestyle='-',
                    label='AB-NPMLE', ax=ax)
        sns.kdeplot(x=atoms_orig, weights=probs_orig, bw_adjust=0.05,
                    fill=True, color=C_NPMLE, alpha=0.30,
                    linewidth=2.5, linestyle='--',
                    label='Bootstrap-NPMLE', ax=ax)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel(r'$\theta | x$', labelpad=8)
        ax.set_ylabel('Density', labelpad=8)
        ax.legend(**LEGEND_KW)

        plt.tight_layout(pad=0.3)

        base, _ = os.path.splitext(save_path)
        panel_png = f"{base}_kde.png"
        panel_pdf = f"{base}_kde.pdf"
        plt.savefig(panel_png, dpi=300, bbox_inches='tight')
        with PdfPages(panel_pdf) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
        print(f"  Comparison plot saved (kde): {panel_png}  /  {panel_pdf}")
        return [('kde', panel_png, panel_pdf)]
    except Exception as e:
        print(f"  [Warning] comparison plot failed (x_j={x_j}): {e}")


_CSV_FIELDS = ["x_j", "B", "mean", "std", "median", "q25", "q75"]


def save_summary_csv(summaries: list, csv_path: str):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            row = {k: s[k] for k in _CSV_FIELDS}
            for k, v in row.items():
                if isinstance(v, float):
                    row[k] = f"{v:.6f}"
            writer.writerow(row)
    print(f"Summary CSV saved: {csv_path}  ({len(summaries)} rows)")


def save_full_npz(x_targets, summaries, sigma, npz_path):
    np.savez(
        npz_path,
        x_values  = np.array(x_targets, dtype=np.float64),
        atoms_all = np.array([s["atoms_all"] for s in summaries], dtype=object),
        probs_all = np.array([s["probs_all"] for s in summaries], dtype=object),
        mean      = np.array([s["mean"] for s in summaries], dtype=np.float64),
        std       = np.array([s["std"] for s in summaries], dtype=np.float64),
        median    = np.array([s["median"] for s in summaries], dtype=np.float64),
        sigma     = np.float64(sigma),
        n_bootstrap = np.int64(summaries[0]["B"]),
    )
    print(f"Full distribution saved: {npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap of the posterior (mixture aggregation) — "
                     "NPMLE, and if --ab_result is given, AB-NPMLE loaded "
                     "from infer.py's output and compared via KDE"
    )
    parser.add_argument("--npmle_result", type=str, required=True,
                        help="Path to the bootstrap NPMLE result npz (needs atoms, weights keys)")
    parser.add_argument("--ab_result", type=str, required=True,
                        help="Path to the AB-NPMLE bootstrap result npz produced "
                             "by infer.py (result_AB_npmle_{data_name}.npz; needs "
                             "atoms, weights keys). No inference is run here.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to the data npz (uses the 'x' key, falling "
                             "back to 'z' or the first key). Used with --obs_idx.")
    parser.add_argument("--obs_idx", type=int, nargs="+", default=None,
                        help="Use x[obs_idx] as the conditioning value(s) (multiple allowed)")
    parser.add_argument("--x_values", type=float, nargs="+", default=None,
                        help="Directly specified x_j values, without data (used instead of obs_idx)")
    parser.add_argument("--sigma", type=float, required=True,
                        help="Noise std used in the posterior likelihood N(x_j; theta, sigma^2), "
                             "shared by both NPMLE and AB-NPMLE.")
    parser.add_argument("--tag", type=str, default="result",
                        help="Tag appended to output filenames")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--summary_plot", action="store_true",
                        help="Plot the per-x_j mixture posterior (NPMLE, AB-NPMLE, "
                             "and their KDE comparison) as PNGs")
    args = parser.parse_args()

    if args.x_values is None and (args.data_path is None or args.obs_idx is None):
        raise ValueError(
            "Either specify --x_values directly, or specify both "
            "--data_path and --obs_idx."
        )

    os.makedirs(args.save_dir, exist_ok=True)

    x_np = None
    if args.data_path is not None:
        x_np = load_x(args.data_path)
    if args.x_values is not None:
        x_targets = list(args.x_values)
        obs_labels = [f"x={v:+.4f}" for v in x_targets]
    else:
        x_targets = [float(x_np[idx]) for idx in args.obs_idx]
        obs_labels = [f"obs_idx={idx}" for idx in args.obs_idx]

    npmle_atoms_list, npmle_weights_list = load_bootstrap(args.npmle_result)
    B_npmle = len(npmle_atoms_list)
    if B_npmle == 0:
        raise RuntimeError(f"No valid bootstrap replicates: {args.npmle_result}")

    ab_atoms_list, ab_weights_list = load_bootstrap(args.ab_result)
    B_ab = len(ab_atoms_list)
    if B_ab == 0:
        raise RuntimeError(f"No valid bootstrap replicates: {args.ab_result}")

    print("=" * 70)
    print("Bootstrap of the Posterior — NPMLE + AB-NPMLE")
    print("=" * 70)
    print(f"  npmle_result : {args.npmle_result}  (B={B_npmle})")
    print(f"  ab_result    : {args.ab_result}  (B={B_ab})")
    print(f"  sigma        : {args.sigma}")
    print(f"  targets      : {len(x_targets)}  ({obs_labels})")
    print(f"  save_dir     : {args.save_dir}")
    print("=" * 70)
    print()

    npmle_summaries = []
    ab_summaries    = []

    for label, x_j in zip(obs_labels, x_targets):
        s_npmle = posterior_bootstrap_summary(
            x_j, npmle_atoms_list, npmle_weights_list, args.sigma
        )
        npmle_summaries.append(s_npmle)
        print(f"  [NPMLE   ][{label}]  mean={s_npmle['mean']:+.4f}  "
              f"std={s_npmle['std']:.4f}  median={s_npmle['median']:+.4f}")

        s_ab = posterior_bootstrap_summary(
            x_j, ab_atoms_list, ab_weights_list, args.sigma
        )
        ab_summaries.append(s_ab)
        print(f"  [AB-NPMLE][{label}]  mean={s_ab['mean']:+.4f}  "
              f"std={s_ab['std']:.4f}  median={s_ab['median']:+.4f}")

        if args.summary_plot:
            safe_label = label.replace("=", "").replace(" ", "").replace("+", "p").replace("-", "m")

            plot_posterior_bootstrap(
                s_npmle,
                os.path.join(args.save_dir,
                             f"posterior_bootstrap_npmle_{args.tag}_{safe_label}.png")
            )
            plot_posterior_bootstrap(
                s_ab,
                os.path.join(args.save_dir,
                             f"posterior_bootstrap_ab_{args.tag}_{safe_label}.png")
            )
            plot_bootstrap_comparison(
                atoms_ab=s_ab["atoms_all"], probs_ab=s_ab["probs_all"],
                B_ab=s_ab["B"],
                atoms_orig=s_npmle["atoms_all"], probs_orig=s_npmle["probs_all"],
                B_orig=s_npmle["B"],
                x_j=x_j,
                save_path=os.path.join(
                    args.save_dir,
                    f"posterior_bootstrap_compare_{args.tag}_{safe_label}.png"
                ),
            )

    print()
    save_full_npz(x_targets, npmle_summaries, args.sigma,
                  os.path.join(args.save_dir, f"posterior_bootstrap_npmle_{args.tag}.npz"))
    save_summary_csv(npmle_summaries,
                      os.path.join(args.save_dir, f"posterior_bootstrap_npmle_{args.tag}.csv"))

    save_full_npz(x_targets, ab_summaries, args.sigma,
                  os.path.join(args.save_dir, f"posterior_bootstrap_ab_{args.tag}.npz"))
    save_summary_csv(ab_summaries,
                      os.path.join(args.save_dir, f"posterior_bootstrap_ab_{args.tag}.csv"))


if __name__ == "__main__":
    main()
