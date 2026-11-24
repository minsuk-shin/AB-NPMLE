r"""No-bootstrap MDV from a saved ReLU AMEM checkpoint and observed-data NPZ.

Run from the repository root (replace the paths with your own files):

    python eval/mdv.py \
      --data_path /path/to/observed_data.npz \
      --ckpt_path /path/to/checkpoint_AMEM.pth \
      --save_dir /path/to/mdv_output \
      --sigma_min 0.1 --sigma_max 2.5 --n_grid 50

Reuse NPMLE fits from an earlier MDV result with the same data and sigma grid:

    python eval/mdv.py \
      --data_path /path/to/observed_data.npz \
      --ckpt_path /path/to/checkpoint_AMEM.pth \
      --save_dir /path/to/mdv_output \
      --sigma_min 0.1 --sigma_max 2.5 --n_grid 50 \
      --npmle_cache /path/to/no_bootstrap_result_DATA_NAME.npz

The data NPZ must contain x. Without --npmle_cache, NPMLE is fitted at each
sigma. The checkpoint architecture must match --hidden_dim, --num_layers,
--n_support, and --num_group (defaults: 256, 5, 100, and n). The old GELU
implementation is preserved in mdv_legacy.py.
"""

import os
import time
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#F7F7F7",
    "axes.edgecolor"    : "#CCCCCC",
    "axes.linewidth"    : 0.8,
    "axes.grid"         : False,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "xtick.direction"   : "out",
    "ytick.direction"   : "out",
    "xtick.major.size"  : 13,
    "ytick.major.size"  : 13,
    "xtick.labelsize"   : 23,
    "ytick.labelsize"   : 23,
    "xtick.color"       : "#555555",
    "ytick.color"       : "#555555",
    "font.family"       : "DejaVu Sans",
    "font.size"         : 18,
    "axes.labelsize"    : 25,
    "axes.titlesize"    : 18,
    "axes.titleweight"  : "bold",
    "legend.frameon"    : True,
    "legend.framealpha" : 0.92,
    "legend.edgecolor"  : "#CCCCCC",
    "legend.fontsize"   : 20,
})
import matplotlib.pyplot as plt
from matplotlib import ticker
from multiprocessing import get_context

from model.model import AMEM_NET
from npeb import GLMixture

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon GPU acceleration
else:
    device = torch.device("cpu")
print(f"Device: {device}")


def compute_ab_no_bootstrap(model, sigma_grid, x_mean, x_std_val, m, device,
                             weight_threshold=0.001):
    """
    Single forward pass with w = ones(1, 1, m) — the special case of
    Multinomial(total_count=m, probs=1/m) where every observation is drawn
    exactly once (i.e. the raw data as-is).

    Returns
    -------
    var_pi_vals : (n_sigma,) ndarray
    Mhat_vals   : (n_sigma,) ndarray
    atoms_list  : list of (K,) ndarray  — active atoms per sigma (pass threshold)
    weights_list: list of (K,) ndarray  — active weights per sigma (raw, unnormalized)
    """
    model.eval()
    w_ones = torch.ones((1, 1, m), device=device)

    var_pi_vals = []
    Mhat_vals = []
    atoms_list = []
    weights_list = []

    with torch.no_grad():
        for sig in sigma_grid:
            sigma_std = sig / x_std_val
            sigma_t = torch.full((1, 1), sigma_std, device=device)

            prob, supp = model(sigma_t, w_ones)
            supp_orig = (supp.squeeze(1) * x_std_val + x_mean).squeeze(0)
            prob_orig = prob.squeeze(1).squeeze(0)

            mask = prob_orig > weight_threshold
            a = supp_orig[mask]
            w_ = prob_orig[mask]
            w_norm = w_ / w_.sum()

            E1 = (w_norm * a).sum()
            E2 = (w_norm * a ** 2).sum()
            var_pi = (E2 - E1 ** 2).item()
            M_hat = var_pi + sig ** 2

            var_pi_vals.append(var_pi)
            Mhat_vals.append(M_hat)
            atoms_list.append(a.detach().cpu().numpy().copy())
            weights_list.append(w_.detach().cpu().numpy().copy())

    return (np.array(var_pi_vals), np.array(Mhat_vals), atoms_list, weights_list)


def fit_npmle_with_retry(x_np, sig, weight_threshold=0.001, max_retries=5,
                          verbose_prefix=""):
    """
    Wrapper that retries GLMixture.fit up to max_retries times if a mosek
    SolutionError (or other fit-related exception) occurs.

    mosek's "Solution status is Unknown" is usually intermittent interior-point
    numerical instability inside the solver, and simply retrying with the
    identical input often succeeds. Since the point of the no-bootstrap
    comparison is to compare AB and NPMLE under identical conditions on "the
    raw data x_np as-is," no perturbation (e.g. jitter) is applied to the data
    across retries — every attempt refits the same x_np.

    Returns
    -------
    atoms, weights : ndarray  (None, None on failure)
    n_retries_used : int
    """
    prec = np.array([1.0 / (sig ** 2)])
    last_err = None

    for attempt in range(max_retries + 1):
        try:
            m = GLMixture(prec_type='diagonal', homoscedastic=True)
            m.fit(x_np.reshape(-1, 1), prec, solver='mosek', numThreads=1)
            atoms = m.atoms.flatten().copy()
            weights = m.weights.flatten().copy()
            if attempt > 0:
                print(f"  {verbose_prefix}[retry {attempt}] succeeded")
            return atoms, weights, attempt
        except Exception as e:
            last_err = e
            print(f"  {verbose_prefix}[attempt {attempt+1}/{max_retries+1}] "
                  f"fit failed: {type(e).__name__}: {str(e)[:120]} -> retrying")

    print(f"  {verbose_prefix}[warning] all {max_retries+1} attempts failed. Last error: {last_err}")
    return None, None, max_retries + 1


def compute_npmle_no_bootstrap(x_np, sigma_grid, weight_threshold=0.001,
                                max_retries=5, use_subprocess=True,
                                npmle_workers=1, per_sigma_dir=None):
    """
    Fits GLMixture on x_boot = x_np as-is (np.repeat(x_np, ones) == x_np).
    If the mosek fit fails (e.g. SolutionError), fit_npmle_with_retry retries
    automatically (up to max_retries times).

    [Avoiding a memory leak]
    Calling mosek (GLMixture.fit) repeatedly within the same process leaks a
    bit of memory each time (the session/environment object is never fully
    released), so over a long sigma grid this eventually kills the process
    with an OOM (often silently) — the same issue already hit and worked
    around with multiprocessing.Pool(..., maxtasksperchild=1) in
    run_npmle_real.py / run_npmle_sim_iter.py / run_npmle_mdv_dn.py. The same
    approach is used here: after each fit, the worker process is torn down and
    restarted (maxtasksperchild=1) so the OS fully reclaims whatever memory
    mosek was holding. Set use_subprocess=False to run sequentially in the
    same process instead (for debugging only — not recommended for long grids).

    Returns
    -------
    var_pi_vals : (n_sigma,) ndarray  — np.nan for any sigma that ultimately failed
    Mhat_vals   : (n_sigma,) ndarray  — np.nan for any sigma that ultimately failed
    atoms_list  : list of (K,) ndarray or None (on failure)
    weights_list: list of (K,) ndarray or None (on failure)
    """
    var_pi_vals = []
    Mhat_vals = []
    atoms_list = []
    weights_list = []

    if per_sigma_dir is not None:
        os.makedirs(per_sigma_dir, exist_ok=True)

    def _result_file(idx, sig):
        return os.path.join(per_sigma_dir, f"npmle_sigma_{idx:03d}_{sig:.6f}.npz")

    def _load_saved(idx, sig):
        if per_sigma_dir is None:
            return None
        path = _result_file(idx, sig)
        if not os.path.isfile(path):
            return None
        with np.load(path) as saved:
            if not np.isclose(float(saved["sigma"]), sig, rtol=0, atol=1e-10):
                raise ValueError(f"Saved sigma does not match grid: {path}")
            if not bool(saved["success"]):
                return None  # A failed fit should be retried on resume.
            return (saved["atoms_raw"].copy(), saved["weights_raw"].copy(),
                    int(saved["n_retries"]))

    def _save_fit(idx, sig, atoms, weights, n_retries):
        if per_sigma_dir is None:
            return
        path = _result_file(idx, sig)
        tmp = path + ".tmp.npz"
        success = atoms is not None
        np.savez(tmp, sigma=np.float64(sig), index=np.int64(idx),
                 success=np.bool_(success), n_retries=np.int64(n_retries),
                 atoms_raw=atoms if success else np.empty(0),
                 weights_raw=weights if success else np.empty(0),
                 weight_threshold=np.float64(weight_threshold))
        os.replace(tmp, path)

    def _process_one(atoms, weights, n_retries, sig):
        if atoms is None:
            var_pi_vals.append(np.nan)
            Mhat_vals.append(np.nan)
            atoms_list.append(None)
            weights_list.append(None)
            return

        mask = weights > weight_threshold
        a = atoms[mask]
        w_ = weights[mask]
        w_norm = w_ / w_.sum()

        E1 = (w_norm * a).sum()
        E2 = (w_norm * a ** 2).sum()
        var_pi = float(E2 - E1 ** 2)
        M_hat = var_pi + sig ** 2

        var_pi_vals.append(var_pi)
        Mhat_vals.append(M_hat)
        atoms_list.append(a)
        weights_list.append(w_)
        print(f"  [NPMLE no-boot] sigma={sig:.4f}  var_pi={var_pi:.6f}  K={len(a)}"
              + (f"  (succeeded after {n_retries} retries)" if n_retries > 0 else ""))

    if use_subprocess:
        # Forking many workers after the CUDA model has run can leave MOSEK
        # workers blocked in inherited library locks. Spawn fresh processes
        # for parallel fits; retain the original fork path for one worker.
        ctx = get_context("spawn" if npmle_workers > 1 else "fork")
        with ctx.Pool(processes=npmle_workers, maxtasksperchild=1) as pool:
            jobs = []
            for idx, sig in enumerate(sigma_grid):
                saved = _load_saved(idx, sig)
                job = None if saved is not None else pool.apply_async(
                    fit_npmle_with_retry,
                    args=(x_np, sig, weight_threshold, max_retries, f"sigma={sig:.4f} "),
                )
                jobs.append((saved, job))
            for idx, (sig, (saved, job)) in enumerate(zip(sigma_grid, jobs)):
                atoms, weights, n_retries = saved if saved is not None else job.get()
                if saved is None:
                    _save_fit(idx, sig, atoms, weights, n_retries)
                _process_one(atoms, weights, n_retries, sig)
    else:
        for idx, sig in enumerate(sigma_grid):
            saved = _load_saved(idx, sig)
            if saved is not None:
                atoms, weights, n_retries = saved
            else:
                atoms, weights, n_retries = fit_npmle_with_retry(
                    x_np, sig, weight_threshold=weight_threshold,
                    max_retries=max_retries,
                    verbose_prefix=f"sigma={sig:.4f} ",
                )
                _save_fit(idx, sig, atoms, weights, n_retries)
            _process_one(atoms, weights, n_retries, sig)

    n_failed = sum(1 for a in atoms_list if a is None)
    if n_failed > 0:
        print(f"\n  [note] fit ultimately failed for {n_failed}/{len(sigma_grid)} sigma values "
              f"(saved as var_pi=NaN). Exclude these from plots/statistics.")

    return (np.array(var_pi_vals), np.array(Mhat_vals), atoms_list, weights_list)


def resolve_tau_n(args, std_x):
    """
    tau_n = args.tau_n_std_frac * std(x).

    Regardless of whether a cache (.npz) is used, this function always
    recomputes tau_n from "the args passed to this run" only — a tau_n value
    saved in an old cache is never used. Re-running with a different
    --tau_n_std_frac always takes effect, cache or not.

    std_x : std(x) reconstructed as sqrt(S2_n). Can be computed straight from
            S2_n even when using the cache, so x_np doesn't need to be reloaded.
    """
    tau_n = args.tau_n_std_frac * std_x
    print(f"  tau_n = {args.tau_n_std_frac} x std(x) "
          f"= {args.tau_n_std_frac} x {std_x:.6f} = {tau_n:.6f}")
    return tau_n


def estimate_mdv(sigma_grid, Dn_vals, tau_n):
    """
    sigma_hat^2_n = max{ sigma^2 in G_n : Dn(sigma) <= tau_n }

    If Dn_vals has NaNs (fit failed at that sigma), those points are ignored
    and the estimate uses only the remaining grid.
    """
    valid = ~np.isnan(Dn_vals)
    if not valid.any():
        print("  [warning] Dn is NaN at every sigma (all fits failed). Cannot estimate MDV.")
        return None
    sigma2_grid = sigma_grid[valid] ** 2
    Dn_valid = Dn_vals[valid]
    below = sigma2_grid[Dn_valid <= tau_n]
    if len(below) == 0:
        print(f"  [warning] No sigma has Dn below tau_n={tau_n:.4f}. Increase tau_n.")
        return None
    return float(below.max())


def plot_dn_curve_no_bootstrap(sigma_grid, Dn_ab, Dn_npmle, S2_n,
                                sigma_hat2_ab, sigma_hat2_npmle, tau_n, sigma_true,
                                save_path, data_name, dn_ymin=None, dn_ymax=None,
                                true_mdv=None, prior_var=None,
                                xlim_left=None, fig_height=6,
                                compact_layout=False):
    """
    true_mdv, prior_var
    -------------------
    When the prior (= the distribution of theta, pi_0) is Normal(., prior_var),
    the theoretical MDV of a prior with a Gaussian factor is, by Lemma
    (mdv-char):
        true_mdv = sigma_true**2 + prior_var   (= sigma_0^2 + tau_0^2)
    (computed in main() and passed in). If the prior isn't Normal or wasn't
    given, both are None and this function behaves exactly as before
    (backward compatible).

    Style/color note
    -----------------
    The red solid curve shows AMEM-Net and the blue dashed curve shows NPMLE.
    Their estimated-sigma lines can coincide; draw the thicker red solid line
    first and the thinner blue dashed line above it so both remain visible.
    """
    sigma_hat_ab = np.sqrt(sigma_hat2_ab) if sigma_hat2_ab is not None else None
    sigma_hat_npmle = np.sqrt(sigma_hat2_npmle) if sigma_hat2_npmle is not None else None
    sqrt_true_mdv = np.sqrt(true_mdv) if true_mdv is not None else None

    C_AB, C_NPMLE, C_TAU, C_TRUE, C_MDV_TRUE = (
        "#DC2626",
        "#2563EB",
        "#6B7280",
        "#27AE60",
        "#0D9488",
    )

    fig, ax = plt.subplots(figsize=(8, fig_height))

    ax.plot(sigma_grid, Dn_ab, color=C_AB,
            lw=4.2 if compact_layout else 2.2)
    ax.plot(sigma_grid, Dn_npmle, color=C_NPMLE,
            lw=3.6 if compact_layout else 1.8, ls="--")
    ax.axhline(tau_n, color=C_TAU,
               lw=2.4 if compact_layout else 1.5, ls="--")

    if sqrt_true_mdv is not None:
        ax.axvline(sqrt_true_mdv, color=C_MDV_TRUE,
                   lw=4.0 if compact_layout else 2.0,
                   ls=(0, (1.2, 2.0)) if compact_layout else "-.",
                   dash_capstyle="round",
                   label=r"$\sigma^{\star}$")
    if sigma_hat_ab is not None:
        ax.axvline(sigma_hat_ab, color=C_AB,
                   lw=6.0 if compact_layout else 4.0, ls="-", zorder=6,
                   label=r"$\hat\sigma_{\mathrm{AMEM}}$")
    if sigma_hat_npmle is not None:
        ax.axvline(sigma_hat_npmle, color=C_NPMLE,
                   lw=4.0 if compact_layout else 2.0, ls="--", zorder=7,
                   label=r"$\hat\sigma_{\mathrm{NPMLE}}$")

    ax.set_xlabel(r"$\sigma$", labelpad=8)
    ax.set_ylabel(r"$D_n(\sigma)$", labelpad=8)
    ax.tick_params(axis="y", labelrotation=90)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    if compact_layout:
        ax.legend(loc="upper left", fontsize=28, borderpad=0.3,
                  labelspacing=0.25, handlelength=2.1)
    else:
        ax.legend(loc="best")
    ax.set_xlim(left=xlim_left)
    if dn_ymin is not None or dn_ymax is not None:
        ax.set_ylim(bottom=dn_ymin, top=dn_ymax)

    if compact_layout:
        ax.set_xlabel(r"$\sigma$", labelpad=2)
        ax.set_ylabel(r"$D_n(\sigma)$", labelpad=2)
        ax.xaxis.label.set_size(29)
        ax.yaxis.label.set_size(29)
        ax.tick_params(axis="both", labelsize=27, pad=2)
        fig.subplots_adjust(left=0.15, right=0.995, bottom=0.15, top=0.99)
    else:
        plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    print(f"Saved PNG: {save_path}")
    print(f"Saved PDF: {pdf_path}")


def plot_no_bootstrap_comparison(sigma_grid, var_pi_ab, var_pi_npmle,
                                  sigma_true, save_path, data_name,
                                  true_mdv=None, prior_var=None,
                                  sigma_hat2_ab=None, sigma_hat2_npmle=None,
                                  xlim_left=None):
    C_AB, C_NPMLE, C_DIFF, C_TRUE, C_FAIL, C_MDV_TRUE = (
        "#DC2626", "#2563EB", "#8E44AD", "#27AE60", "#111111", "#0D9488"
    )
    sqrt_true_mdv = np.sqrt(true_mdv) if true_mdv is not None else None

    sigma_hat_ab = np.sqrt(sigma_hat2_ab) if sigma_hat2_ab is not None else None
    sigma_hat_npmle = np.sqrt(sigma_hat2_npmle) if sigma_hat2_npmle is not None else None

    from matplotlib.backends.backend_pdf import PdfPages

    diff = var_pi_ab - var_pi_npmle
    valid = ~(np.isnan(var_pi_ab) | np.isnan(var_pi_npmle))
    n_failed = (~valid).sum()

    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
    for panel in axes:
        panel.tick_params(labelsize=12, length=5)

    ax = axes[0]
    ax.plot(sigma_grid, var_pi_ab, color=C_AB, lw=2.2, label="AMEM-Net (no bootstrap)")
    ax.plot(sigma_grid, var_pi_npmle, color=C_NPMLE, lw=1.8, ls="--",
            label="NPMLE (no bootstrap)")
    if n_failed > 0:
        ax.scatter(sigma_grid[~valid], np.zeros(n_failed), color=C_FAIL, marker="x",
                   s=50, zorder=5, label=f"NPMLE fit failed (n={n_failed})")
    if sqrt_true_mdv is not None:
        ax.axvline(sqrt_true_mdv, color=C_MDV_TRUE, lw=1.8, ls="-.",
                   label=r"$\sigma^{\star}$")
    if sigma_hat_ab is not None:
        ax.axvline(sigma_hat_ab, color=C_AB, lw=4.0, ls="-", zorder=6,
                   label=r"$\hat\sigma_{\mathrm{AMEM}}$")
    if sigma_hat_npmle is not None:
        ax.axvline(sigma_hat_npmle, color=C_NPMLE, lw=2.0, ls="--", zorder=7,
                   label=r"$\hat\sigma_{\mathrm{NPMLE}}$")
    ax.set_ylabel(r"$var\_\pi(\sigma)$", labelpad=6, fontsize=15)
    ax.tick_params(axis="y", labelrotation=90)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.set_title(f"No-bootstrap var_pi comparison  ({data_name})", fontsize=16)
    ax.legend(loc="upper right", fontsize=9, ncol=2)

    ax2 = axes[1]
    ax2.plot(sigma_grid, diff, color=C_DIFF, lw=2.0,
              label=r"$var\_\pi_{AMEM} - var\_\pi_{NPMLE}$")
    ax2.axhline(0, color="#555555", lw=0.8)
    if sqrt_true_mdv is not None:
        ax2.axvline(sqrt_true_mdv, color=C_MDV_TRUE, lw=1.5, ls="-.")
    if sigma_hat_ab is not None:
        ax2.axvline(sigma_hat_ab, color=C_AB, lw=4.0, ls="-", zorder=6)
    if sigma_hat_npmle is not None:
        ax2.axvline(sigma_hat_npmle, color=C_NPMLE, lw=2.0, ls="--", zorder=7)
    ax2.set_xlabel(r"$\sigma$", labelpad=6, fontsize=15)
    ax2.set_ylabel(r"$\Delta\,\mathrm{Var}_{\pi}\;(\times 10^{-1})$",
                   labelpad=6, fontsize=15)
    ax2.tick_params(axis="y", labelrotation=90)
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda value, _position: f"{10 * value:.1f}"))
    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_xlim(left=xlim_left)

    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    print(f"Saved PNG: {save_path}")
    print(f"Saved PDF: {pdf_path}")

    if n_failed > 0:
        print(f"\n  [note] NPMLE fit ultimately failed for {n_failed}/{len(sigma_grid)} sigma "
              f"values and were excluded from the statistics (sigma={sigma_grid[~valid].round(4).tolist()})")

    if valid.any():
        abs_diff_valid = np.abs(diff[valid])
        print(f"\n  mean |diff| (no-bootstrap, successful sigma only): {abs_diff_valid.mean():.6f}")
        print(f"  max |diff| (no-bootstrap, successful sigma only): {abs_diff_valid.max():.6f}  "
              f"(sigma={sigma_grid[valid][np.argmax(abs_diff_valid)]:.4f})")
    else:
        print("\n  [warning] NPMLE fit failed at every sigma; nothing to compare.")


def main():
    parser = argparse.ArgumentParser(description="No-bootstrap var_pi comparison")
    parser.add_argument("--ckpt_path",   type=str,   required=True,
                        help="Saved ReLU AMEM checkpoint (.pth)")
    parser.add_argument("--data_path",   type=str,   required=True,
                        help="Observed-data NPZ containing x")
    parser.add_argument("--save_dir",    type=str,   required=True)
    parser.add_argument("--npmle_cache", type=str, default=None,
                        help="Saved MDV NPZ supplying NPMLE sigma-wise fits; AMEM is recomputed")
    parser.add_argument("--sigma_min",   type=float, default=0.1)
    parser.add_argument("--sigma_max",   type=float, default=2.5)
    parser.add_argument("--n_grid",      type=int,   default=50)
    parser.add_argument("--sigma_true",  type=float, default=None)
    parser.add_argument("--prior_type",  type=str,   default=None,
                        choices=["normal", "discrete", "uniform"],
                        help="Type of theta's true prior. Normal: true MDV = "
                             "sigma_true^2 + prior_var. Discrete or bounded "
                             "uniform: true MDV = sigma_true^2.")
    parser.add_argument("--prior_var",   type=float, default=None,
                        help="Variance sigma_prior^2 when the prior is "
                             "Normal(mean, prior_var). Required with --prior_type normal.")
    parser.add_argument("--hidden_dim",  type=int,   default=256)
    parser.add_argument("--num_layers",  type=int,   default=5)
    parser.add_argument("--n_support",   type=int,   default=100,
                        help="Must match the number of support points actually used in training (currently 100)")
    parser.add_argument("--num_group",   type=int,   default=None,
                        help="m (must equal n). Defaults to the data size if not given")
    parser.add_argument("--weight_threshold", type=float, default=0.001)
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Number of retries on a mosek fit failure (default: 5)")
    parser.add_argument("--no_npmle_subprocess", action="store_true",
                        help="Disable the per-sigma worker-restart safeguard "
                             "(maxtasksperchild=1) for the NPMLE(mosek) fit and "
                             "run sequentially in the same process instead. Not "
                             "recommended outside debugging, since mosek "
                             "sessions can accumulate and OOM over a long grid.")
    parser.add_argument("--npmle_workers", type=int, default=1,
                        help="Number of parallel NPMLE fits across sigma values (default: 1).")
    parser.add_argument("--save_npmle_by_sigma", action="store_true",
                        help="Save each raw NPMLE fit in save_dir/npmle_by_sigma as soon as it finishes; "
                             "reuse successful fits when resuming an interrupted run.")
    parser.add_argument("--tau_n_std_frac", type=float, default=0.02,
                        help="tau_n = this value x std(x) (e.g. 0.01 -> tau_n = 0.01 x std(x))")
    parser.add_argument("--dn_ymin", type=float, default=None,
                        help="Lower y-limit of the Dn plot (matplotlib auto if not given)")
    parser.add_argument("--dn_ymax", type=float, default=None,
                        help="Upper y-limit of the Dn plot (matplotlib auto if not given)")
    parser.add_argument("--dn_fig_height", type=float, default=6,
                        help="Height in inches of the MDV Dn plot (default: 6).")
    parser.add_argument("--dn_compact_layout", action="store_true",
                        help="Enlarge the MDV axes within the figure and increase legend text.")
    parser.add_argument("--xlim_left", type=float, default=1,
                        help="Left x-limit (sigma) of the var_pi-comparison and "
                             "Dn-curve plots. Defaults to --sigma_min if not given "
                             "(the previous code hardcoded 1.0). To see the "
                             "sigma<1 region too, set --xlim_left 0 explicitly.")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Recompute the AB-NPMLE/NPMLE fits from scratch even "
                             "if save_dir already has a no_bootstrap_result_{data_file_stem}.npz. "
                             "Default is to load and reuse the cache if present.")
    args = parser.parse_args()
    if args.npmle_workers < 1:
        parser.error("--npmle_workers must be at least 1")

    os.makedirs(args.save_dir, exist_ok=True)

    if not os.path.isfile(args.data_path):
        parser.error(f"data file not found: {args.data_path}")
    if not os.path.isfile(args.ckpt_path):
        parser.error(f"checkpoint not found: {args.ckpt_path}")
    if args.npmle_cache is not None and not os.path.isfile(args.npmle_cache):
        parser.error(f"NPMLE cache not found: {args.npmle_cache}")
    data_name = Path(args.data_path).stem

    result_path = os.path.join(
        args.save_dir, f"no_bootstrap_result_{data_name}.npz"
    )

    true_mdv = None
    if args.prior_type is not None and args.sigma_true is None:
        raise ValueError("--sigma_true is required when --prior_type is given")
    if args.prior_type == "normal":
        if args.prior_var is None:
            raise ValueError("--prior_type normal requires --prior_var (the prior variance sigma_prior^2) as well.")
        true_mdv = args.sigma_true ** 2 + args.prior_var
        print(f"\n  Prior: Normal, sigma_prior^2 = {args.prior_var:.6f}")
        print(f"  True MDV = sigma_0^2 + sigma_prior^2 "
              f"= {args.sigma_true**2:.6f} + {args.prior_var:.6f} = {true_mdv:.6f}"
              f"  (sqrt = {true_mdv**0.5:.6f})")
    elif args.prior_type in ("discrete", "uniform"):
        true_mdv = args.sigma_true ** 2
        print(f"\n  Prior: {args.prior_type}; no nonzero Gaussian factor")
        print(f"  True MDV = sigma_0^2 = {true_mdv:.6f}"
              f"  (sqrt = {true_mdv**0.5:.6f})")

    def to_obj(lst):
        arr = np.empty(len(lst), dtype=object)
        for i_, v in enumerate(lst):
            arr[i_] = v
        return arr

    use_cache = (os.path.exists(result_path) and not args.force_recompute
                 and args.npmle_cache is None)
    if use_cache:
        with np.load(result_path, allow_pickle=True) as previous:
            use_cache = (
                all(key in previous.files for key in
                    ("model_type", "source_data", "source_checkpoint"))
                and str(previous["model_type"]) == "relu"
                and str(previous["source_data"]) == str(Path(args.data_path).resolve())
                and str(previous["source_checkpoint"]) == str(Path(args.ckpt_path).resolve())
            )

    if use_cache:
        print(f"\n{'='*70}")
        print(f"  [using cache] Found an existing result file: {result_path}")
        print("  -> Loading from this file instead of refitting. "
              "(Add --force_recompute to force a recompute)")
        print(f"{'='*70}")
        cached = np.load(result_path, allow_pickle=True)
        sigma_grid = cached["sigma_grid"]
        var_pi_ab = cached["var_pi_ab"]
        var_pi_npmle = cached["var_pi_npmle"]
        Mhat_ab = cached["Mhat_ab"]
        Mhat_npmle = cached["Mhat_npmle"]
        Dn_ab = cached["Dn_ab"]
        Dn_npmle = cached["Dn_npmle"]
        S2_n = float(cached["S2_n"])
        atoms_ab = list(cached["atoms_ab"])
        weights_ab = list(cached["weights_ab"])
        atoms_npmle = list(cached["atoms_npmle"])
        weights_npmle = list(cached["weights_npmle"])
        old_tau_n_in_cache = float(cached["tau_n"]) if "tau_n" in cached.files else None

        print(f"  Loaded: n_grid={len(sigma_grid)}  S^2_n={S2_n:.6f}")
        print(f"  [note] Ignoring the old tau_n={old_tau_n_in_cache} stored in the cache. "
              f"Recomputing tau_n below from the current CLI args.")

        std_x = float(np.sqrt(S2_n))
        args.tau_n = resolve_tau_n(args, std_x)

        sigma_hat2_ab = estimate_mdv(sigma_grid, Dn_ab, args.tau_n)
        sigma_hat2_npmle = estimate_mdv(sigma_grid, Dn_npmle, args.tau_n)

        print(f"  -> Re-estimated MDV using the new tau_n={args.tau_n:.6f}")
        if true_mdv is not None:
            cache_values = {key: cached[key] for key in cached.files}
            cache_values.update(
                prior_type=args.prior_type,
                prior_var=np.float64(args.prior_var) if args.prior_var is not None else np.nan,
                true_mdv=np.float64(true_mdv),
            )
            cached.close()
            tmp_path = Path(result_path).with_suffix(".tmp.npz")
            np.savez(tmp_path, **cache_values)
            os.replace(tmp_path, result_path)
            print(f"  Updated true MDV metadata in {result_path}")

    else:
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data file not found: {args.data_path}")
        data = np.load(args.data_path)
        x_np = data["x"].astype(np.float32)
        n = len(x_np)
        print(f"\nLoaded data: {args.data_path}  (n={n})")

        m = args.num_group if args.num_group is not None else n
        if m != n:
            raise ValueError(f"MDV with w=ones requires num_group=n; got m={m}, n={n}")

        S2_n = float(np.var(x_np))
        std_x = float(np.std(x_np))
        print(f"  S^2_n = {S2_n:.6f}  (std(x) = {std_x:.6f})")

        args.tau_n = resolve_tau_n(args, std_x)

        sigma_grid = np.linspace(args.sigma_min, args.sigma_max, args.n_grid)
        print(f"  sigma grid: [{args.sigma_min}, {args.sigma_max}] x {args.n_grid} points")

        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
        x_mean = x_tensor.mean().item()
        x_std_val = x_tensor.std().item()
        x_std = (x_tensor - x_mean) / x_std_val

        N = args.n_support
        x_min_std = x_std.min().item()
        x_max_std = x_std.max().item()
        pad = (x_max_std - x_min_std) * 0.1
        support_init = torch.linspace(
            x_min_std - pad, x_max_std + pad, N, device=device
        ).reshape(1, 1, -1)

        if not os.path.exists(args.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

        model = AMEM_NET(
            hidden_dim=args.hidden_dim,
            output_dim=N,
            num_layers=args.num_layers,
            support=support_init.squeeze(0),
            n=n,
            m=m,
        ).to(device)
        ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=True)
        hp = ckpt.get("hyperparams") or {}
        expected_hp = {"hidden_dim": args.hidden_dim,
                       "num_layers": args.num_layers,
                       "n_support": N, "num_group": m}
        if any(hp.get(key) != value for key, value in expected_hp.items()):
            raise ValueError(f"checkpoint architecture differs from CLI arguments: {hp}")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        print(f"Loaded model: {args.ckpt_path}")

        print(f"\n{'='*60}")
        print("Computing AB-NPMLE (no bootstrap)...")
        print(f"{'='*60}")
        t0 = time.time()
        var_pi_ab, Mhat_ab, atoms_ab, weights_ab = compute_ab_no_bootstrap(
            model, sigma_grid, x_mean, x_std_val, m, device,
            weight_threshold=args.weight_threshold
        )
        print(f"  Done ({time.time()-t0:.1f}s)")

        print(f"\n{'='*60}")
        print("Computing NPMLE (no bootstrap)...")
        print(f"{'='*60}")
        t0 = time.time()
        if args.npmle_cache is not None:
            with np.load(args.npmle_cache, allow_pickle=True) as source:
                if (int(source["n"]) != n or
                    not np.isclose(float(source["S2_n"]), S2_n, rtol=0, atol=1e-7) or
                    not np.allclose(source["sigma_grid"], sigma_grid, rtol=0, atol=1e-12)):
                    raise ValueError("NPMLE cache does not match data or sigma grid")
                var_pi_npmle = np.asarray(source["var_pi_npmle"], dtype=float).copy()
                Mhat_npmle = np.asarray(source["Mhat_npmle"], dtype=float).copy()
                atoms_npmle = list(source["atoms_npmle"].copy())
                weights_npmle = list(source["weights_npmle"].copy())
            print(f"  Reused NPMLE sigma fits: {args.npmle_cache}")
        else:
            var_pi_npmle, Mhat_npmle, atoms_npmle, weights_npmle = compute_npmle_no_bootstrap(
                x_np, sigma_grid, weight_threshold=args.weight_threshold,
                max_retries=args.max_retries,
                use_subprocess=not args.no_npmle_subprocess,
                npmle_workers=args.npmle_workers,
                per_sigma_dir=(os.path.join(args.save_dir, "npmle_by_sigma")
                               if args.save_npmle_by_sigma else None),
            )
        print(f"  Done ({time.time()-t0:.1f}s)")

        diff = var_pi_ab - var_pi_npmle
        print(f"\n{'='*70}")
        print(f"  No-bootstrap var_pi comparison by sigma  ({data_name})")
        print(f"{'='*70}")
        print(f"  {'sigma':>8s} {'var_pi_AB':>12s} {'var_pi_NPMLE':>14s} {'diff':>10s} "
              f"{'K_AB':>6s} {'K_NPMLE':>8s}")
        for idx, sig in enumerate(sigma_grid):
            k_ab = len(atoms_ab[idx]) if atoms_ab[idx] is not None else None
            k_npmle = len(atoms_npmle[idx]) if atoms_npmle[idx] is not None else None
            k_ab_str = f"{k_ab}" if k_ab is not None else "fail"
            k_npmle_str = f"{k_npmle}" if k_npmle is not None else "fail"
            print(f"  {sig:>8.4f} {var_pi_ab[idx]:>12.6f} {var_pi_npmle[idx]:>14.6f} "
                  f"{diff[idx]:>10.6f} {k_ab_str:>6s} {k_npmle_str:>8s}")

        Dn_ab = np.abs(S2_n - Mhat_ab)
        Dn_npmle = np.abs(S2_n - Mhat_npmle)

        sigma_hat2_ab = estimate_mdv(sigma_grid, Dn_ab, args.tau_n)
        sigma_hat2_npmle = estimate_mdv(sigma_grid, Dn_npmle, args.tau_n)

        np.savez(
            result_path,
            sigma_grid=sigma_grid,
            var_pi_ab=var_pi_ab,
            var_pi_npmle=var_pi_npmle,
            Mhat_ab=Mhat_ab,
            Mhat_npmle=Mhat_npmle,
            Dn_ab=Dn_ab,
            Dn_npmle=Dn_npmle,
            S2_n=np.float64(S2_n),
            n=np.int64(n),
            tau_n=np.float64(args.tau_n),
            sigma_hat2_ab=np.float64(sigma_hat2_ab) if sigma_hat2_ab is not None else np.nan,
            sigma_hat2_npmle=np.float64(sigma_hat2_npmle) if sigma_hat2_npmle is not None else np.nan,
            prior_type=args.prior_type if args.prior_type is not None else "",
            prior_var=np.float64(args.prior_var) if args.prior_var is not None else np.nan,
            true_mdv=np.float64(true_mdv) if true_mdv is not None else np.nan,
            atoms_ab=to_obj(atoms_ab),
            weights_ab=to_obj(weights_ab),
            atoms_npmle=to_obj(atoms_npmle),
            weights_npmle=to_obj(weights_npmle),
            source_data=str(Path(args.data_path).resolve()),
            source_checkpoint=str(Path(args.ckpt_path).resolve()),
            source_npmle_cache=(str(Path(args.npmle_cache).resolve())
                                if args.npmle_cache is not None else ""),
            model_type="relu",
        )
        print(f"\nSaved result: {result_path}")

    print(f"\n{'='*70}")
    print(f"  No-bootstrap MDV estimate  ({data_name})")
    print(f"{'='*70}")
    if sigma_hat2_ab is not None:
        print(f"  AB-NPMLE   sigma_hat^2_n = {sigma_hat2_ab:.6f}  (sigma_hat_n = {sigma_hat2_ab**0.5:.4f})")
    else:
        print("  AB-NPMLE   MDV estimation failed (adjust tau_n or the sigma range)")
    if sigma_hat2_npmle is not None:
        print(f"  NPMLE      sigma_hat^2_n = {sigma_hat2_npmle:.6f}  (sigma_hat_n = {sigma_hat2_npmle**0.5:.4f})")
    else:
        print("  NPMLE      MDV estimation failed (adjust tau_n or the sigma range)")
    if args.sigma_true is not None:
        print(f"  Noise variance sigma^2_0 = {args.sigma_true**2:.6f} "
              "(shown as a reference, not the MDV target)")
    if sigma_hat2_ab is not None and sigma_hat2_npmle is not None:
        print(f"  AB - NPMLE difference (sigma_hat^2) = {sigma_hat2_ab - sigma_hat2_npmle:.6f}")
    if true_mdv is not None:
        print(f"\n  [prior={args.prior_type}]  True MDV = {true_mdv:.6f}  "
              f"(sqrt(MDV) = {true_mdv**0.5:.6f})")
        if sigma_hat2_ab is not None:
            print(f"  AB error    |sigma_hat^2_n - MDV| = {abs(sigma_hat2_ab - true_mdv):.6f}")
        if sigma_hat2_npmle is not None:
            print(f"  NPMLE error |sigma_hat^2_n - MDV| = {abs(sigma_hat2_npmle - true_mdv):.6f}")

    xlim_left = args.xlim_left if args.xlim_left is not None else args.sigma_min

    plot_path = os.path.join(
        args.save_dir, f"no_bootstrap_compare_{data_name}.png"
    )
    plot_no_bootstrap_comparison(
        sigma_grid, var_pi_ab, var_pi_npmle, args.sigma_true, plot_path, data_name,
        true_mdv=true_mdv, prior_var=args.prior_var,
        sigma_hat2_ab=sigma_hat2_ab, sigma_hat2_npmle=sigma_hat2_npmle,
        xlim_left=xlim_left,
    )

    dn_plot_path = os.path.join(
        args.save_dir, f"no_bootstrap_mdv_{data_name}.png"
    )
    plot_dn_curve_no_bootstrap(
        sigma_grid, Dn_ab, Dn_npmle, S2_n,
        sigma_hat2_ab, sigma_hat2_npmle, args.tau_n, args.sigma_true,
        dn_plot_path, data_name,
        dn_ymin=args.dn_ymin, dn_ymax=args.dn_ymax,
        true_mdv=true_mdv, prior_var=args.prior_var,
        xlim_left=xlim_left,
        fig_height=args.dn_fig_height,
        compact_layout=args.dn_compact_layout,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
