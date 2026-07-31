"""
Functional CI Coverage (Posterior Mean / Posterior Variance / Prior Mean / Prior Variance)
Model: x_i = theta_i + eps_i, eps_i ~ N(0, sigma^2)
Prior: pi (unknown, estimated by AB-NPMLE or NPMLE as pi_hat)
 
Supported functionals:
  posterior_mean     : E[theta | x, pi_hat]
  posterior_variance : Var(theta | x, pi_hat)
  prior_mean         : integral theta d(pi_hat)(theta)
  prior_variance     : integral theta^2 d(pi_hat)(theta) - (integral theta d(pi_hat)(theta))^2
 
True functional values under WCS1_Normal, SpikeAndSlab (JZ_A3_s500), and
Uniform priors are computed in closed form -- see true_functional() below.

Usage:
    # Fixed quantile conditioning example (same x_j for every dataset):
    #   Computes the x value at quantile_level from a large reference dataset
    #   (e.g. ./sim_data_100000/data1/WCS1_Normal.npz)
    #   and uses it as a fixed posterior conditioning value instead of obs_idx.
    python eval/functional_ci.py \
        --ab_base_dir       ./sim_data_gms_1000 \
        --npmle_base_dir    ./sim_data_npmle \
        --data_base_dir     ./sim_data \
        --data_name         WCS1_Normal \
        --prior_type        WCS1_Normal \
        --n_data            50 \
        --sigma              1.0 \
        --alpha              0.05 \
        --functional         all \
        --condition_mode     quantile \
        --quantile_level     0.75 \
        --quantile_ref_base_dir ./sim_data_100000 \
        --quantile_ref_idx   1 \
        --summary_plot \
        --save_dir           ./ci_results
"""

import numpy as np
import os
import csv
import argparse
import warnings
warnings.filterwarnings("ignore")


def posterior_stats_discrete(x: np.ndarray,
                              atoms: np.ndarray,
                              weights: np.ndarray,
                              sigma: float):
    """
    Posterior mean and posterior variance under discrete Ghat.

    P(theta = atoms[k] | x_j) proportional to N(x_j; atoms[k], sigma^2) * weights[k]

    Returns
    -------
    pm  : (n,)  E[theta | x, Ghat]
    pv  : (n,)  Var(theta | x, Ghat)
    """
    log_lik  = -0.5 * ((x[:, None] - atoms[None, :]) / sigma) ** 2
    log_lik -= log_lik.max(axis=1, keepdims=True)

    lik       = np.exp(log_lik) * weights[None, :]
    posterior = lik / (lik.sum(axis=1, keepdims=True) + 1e-300)

    pm  = (posterior * atoms[None, :]).sum(axis=1)
    pm2 = (posterior * atoms[None, :] ** 2).sum(axis=1)
    pv  = pm2 - pm ** 2

    return pm, pv


def prior_stats_discrete(atoms: np.ndarray,
                          weights: np.ndarray):
    """
    Prior mean and prior variance of Ghat.

    Returns
    -------
    prior_mean : float   integral theta dGhat(theta)
    prior_var  : float   integral theta^2 dGhat(theta) - (integral theta dGhat(theta))^2
    """
    pm = float((weights * atoms).sum())
    pv = float((weights * atoms ** 2).sum()) - pm ** 2
    return pm, pv


def true_functional(functional: str,
                    prior_type: str,
                    x_j: float,
                    sigma: float,
                    n_obs: int = None,
                    unif_low: float = -5.0,
                    unif_high: float = 10.0) -> float:
    """
    Exact true value of `functional` at observation x_j.
    For prior-level functionals x_j is ignored.

    Parameters
    ----------
    functional : str
    prior_type : str   "WCS1_Normal", "JZ_A3_s500", or "Uniform"
    x_j        : float  observed value (ignored for prior-level functionals)
    sigma      : float  noise std
    n_obs      : int    sample size of the dataset (required for JZ_A3_s500
                        so that pi1 = s/n can be computed)
    unif_low   : float  lower bound a of Unif(a, b)  [used when prior_type='Uniform']
    unif_high  : float  upper bound b of Unif(a, b)  [used when prior_type='Uniform']
    """
    if prior_type == "WCS1_Normal":
        tau2 = 4.0
        s2   = sigma ** 2

        if functional == "posterior_mean":
            return tau2 / (tau2 + s2) * x_j

        elif functional == "posterior_variance":
            return s2 * tau2 / (s2 + tau2)

        elif functional == "prior_mean":
            return 0.0

        elif functional == "prior_variance":
            return tau2

        else:
            raise ValueError(f"Unknown functional '{functional}'.")

    elif prior_type == "JZ_A3_s500":
        A   = 3.0
        s   = 500

        if n_obs is None:
            raise ValueError(
                "n_obs must be provided for prior_type='SpikeAndSlab' "
                "so that pi1 = s / n can be computed."
            )
        n    = n_obs
        pi1  = s / n
        pi0  = 1.0 - pi1
        s2   = sigma ** 2

        if functional == "prior_mean":
            return pi1 * A

        elif functional == "prior_variance":
            return pi1 * A ** 2 - (pi1 * A) ** 2

        elif functional in ("posterior_mean", "posterior_variance"):
            log_w0 = np.log(pi0 + 1e-300) - 0.5 * (x_j / sigma) ** 2
            log_wA = np.log(pi1 + 1e-300) - 0.5 * ((x_j - A) / sigma) ** 2

            log_max = max(log_w0, log_wA)
            w0_rel  = np.exp(log_w0 - log_max)
            wA_rel  = np.exp(log_wA - log_max)
            Z       = w0_rel + wA_rel

            post_0  = w0_rel / Z
            post_A  = wA_rel / Z

            e_theta  = post_0 * 0.0 + post_A * A
            e_theta2 = post_0 * 0.0 + post_A * A ** 2

            if functional == "posterior_mean":
                return e_theta
            else:
                return e_theta2 - e_theta ** 2

        else:
            raise ValueError(f"Unknown functional '{functional}'.")

    elif prior_type == "Uniform":
        from scipy.stats import norm as _norm
        a    = unif_low
        b    = unif_high
        span = b - a

        if functional == "prior_mean":
            return (a + b) / 2.0

        elif functional == "prior_variance":
            return span ** 2 / 12.0

        elif functional in ("posterior_mean", "posterior_variance"):
            s   = sigma
            za  = (x_j - a) / s
            zb  = (x_j - b) / s
            D   = _norm.cdf(za) - _norm.cdf(zb)

            if D < 1e-300:
                return float("nan")

            phi_a = _norm.pdf(za)
            phi_b = _norm.pdf(zb)

            e_theta = x_j + s * (phi_a - phi_b) / D

            if functional == "posterior_mean":
                return e_theta
            else:
                return s**2 * (1.0
                               - (phi_a - phi_b)**2 / D**2
                               + (zb * phi_b - za * phi_a) / D)

        else:
            raise ValueError(f"Unknown functional '{functional}'.")

    else:
        raise ValueError(
            f"Unknown prior_type '{prior_type}'. "
            "Supported: 'WCS1_Normal', 'JZ_A3_s500', 'Uniform'."
        )


def compute_functional_value(functional: str,
                              x_j: float,
                              atoms: np.ndarray,
                              weights: np.ndarray,
                              sigma: float) -> float:
    if functional in ("posterior_mean", "posterior_variance"):
        pm, pv = posterior_stats_discrete(np.array([x_j]), atoms, weights, sigma)
        return float(pm[0]) if functional == "posterior_mean" else float(pv[0])

    elif functional in ("prior_mean", "prior_variance"):
        pmean, pvar = prior_stats_discrete(atoms, weights)
        return pmean if functional == "prior_mean" else pvar

    else:
        raise ValueError(f"Unknown functional '{functional}'.")


def load_bootstrap(npz_path: str):
    data        = np.load(npz_path, allow_pickle=True)
    atoms_raw   = data["atoms"]
    weights_raw = data["weights"]

    atoms_list, weights_list = [], []
    for a, w in zip(atoms_raw, weights_raw):
        if a is None or w is None:
            continue
        a = np.asarray(a, dtype=np.float64).ravel()
        w = np.asarray(w, dtype=np.float64).ravel()
        w = w / (w.sum() + 1e-300)
        atoms_list.append(a)
        weights_list.append(w)

    return atoms_list, weights_list


def load_x(data_path: str) -> np.ndarray:
    raw   = np.load(data_path, allow_pickle=True)
    x_key = "x" if "x" in raw.files else ("z" if "z" in raw.files else raw.files[0])
    return np.asarray(raw[x_key], dtype=np.float64).ravel()


def compute_coverage(result_paths: list,
                     data_paths: list,
                     sigma: float,
                     alpha: float,
                     prior_type: str,
                     functional: str,
                     method_name: str,
                     obs_idx: int = 0,
                     fixed_x: float = None,
                     verbose: bool = True,
                     unif_low: float = -5.0,
                     unif_high: float = 10.0):
    """
    Parameters
    ----------
    obs_idx  : uses each dataset's x[obs_idx] as the conditioning value
               (only used when fixed_x is None).
    fixed_x  : if given, this value is used as the common conditioning value
               x_j for every dataset (e.g. a quantile computed on a large
               reference dataset). Has no effect on the prior-level functionals
               (prior_mean/prior_variance).

    Returns
    -------
    dict with keys:
      method, functional, covered_list, overall, obs_idx, fixed_x,
      alpha, target, n_data_ok,
      detail_rows   list of dicts, one per dataset (for CSV export)
    """
    condition_mode = "quantile" if fixed_x is not None else "obs_idx"
    lo, hi         = alpha / 2.0, 1.0 - alpha / 2.0
    is_prior_level = functional in ("prior_mean", "prior_variance")

    covered_list = []
    detail_rows  = []
    n_data_ok    = 0

    for i, (rpath, dpath) in enumerate(zip(result_paths, data_paths), start=1):

        base = {
            "data_idx":       i,
            "method":         method_name,
            "functional":     functional,
            "condition_mode": condition_mode if not is_prior_level else "",
            "obs_idx":        (obs_idx if condition_mode == "obs_idx" else "")
                               if not is_prior_level else "",
            "status":         "ok",
        }

        if rpath is None or not os.path.exists(rpath):
            if verbose:
                print(f"  [{method_name}][{functional}][data{i:>3}] SKIP: result not found")
            covered_list.append(np.nan)
            detail_rows.append({**base, "status": "skip_no_result"})
            continue

        if dpath is None or not os.path.exists(dpath):
            if verbose:
                print(f"  [{method_name}][{functional}][data{i:>3}] SKIP: data not found")
            covered_list.append(np.nan)
            detail_rows.append({**base, "status": "skip_no_data"})
            continue

        x = load_x(dpath)

        if is_prior_level:
            x_j = float("nan")
        elif fixed_x is not None:
            x_j = float(fixed_x)
        else:
            if obs_idx >= len(x):
                if verbose:
                    print(f"  [{method_name}][{functional}][data{i:>3}] "
                          f"SKIP: obs_idx={obs_idx} >= n={len(x)}")
                covered_list.append(np.nan)
                detail_rows.append({**base, "status": "skip_obs_idx_oob"})
                continue
            x_j = float(x[obs_idx])

        atoms_list, weights_list = load_bootstrap(rpath)
        B = len(atoms_list)

        if B == 0:
            if verbose:
                print(f"  [{method_name}][{functional}][data{i:>3}] "
                      f"SKIP: no valid bootstrap samples")
            covered_list.append(np.nan)
            detail_rows.append({**base, "x_j": x_j, "B": 0,
                                 "status": "skip_no_bootstrap"})
            continue

        boot_vals = np.array([
            compute_functional_value(functional, x_j, a, w, sigma)
            for a, w in zip(atoms_list, weights_list)
        ])

        ci_lo = float(np.quantile(boot_vals, lo))
        ci_hi = float(np.quantile(boot_vals, hi))

        true_val = true_functional(functional, prior_type, x_j, sigma,
                                   n_obs=len(x),
                                   unif_low=unif_low, unif_high=unif_high)

        c = int(ci_lo <= true_val <= ci_hi)
        covered_list.append(c)
        n_data_ok += 1

        if verbose:
            if is_prior_level:
                obs_str = ""
            elif fixed_x is not None:
                obs_str = f"x_fixed(q)={x_j:+.3f}  "
            else:
                obs_str = f"x[{obs_idx}]={x_j:+.3f}  "
            print(f"  [{method_name}][{functional}][data{i:>3}] B={B:4d}  "
                  f"{obs_str}"
                  f"true={true_val:+.4f}  "
                  f"CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  "
                  f"covered={c}")

        detail_rows.append({
            **base,
            "x_j":         x_j if not is_prior_level else "",
            "B":           B,
            "true_value":  true_val,
            "ci_lo":       ci_lo,
            "ci_hi":       ci_hi,
            "ci_width":    ci_hi - ci_lo,
            "boot_mean":   float(boot_vals.mean()),
            "boot_std":    float(boot_vals.std(ddof=1)),
            "boot_median": float(np.median(boot_vals)),
            "boot_q25":    float(np.quantile(boot_vals, 0.25)),
            "boot_q75":    float(np.quantile(boot_vals, 0.75)),
            "covered":     c,
            "status":      "ok",
        })

    arr          = np.array(covered_list, dtype=np.float64)
    valid        = arr[~np.isnan(arr)]
    overall_rate = float(valid.mean()) if len(valid) > 0 else float("nan")

    return {
        "method":         method_name,
        "functional":     functional,
        "covered_list":   arr,
        "overall":        overall_rate,
        "condition_mode": condition_mode,
        "obs_idx":        obs_idx,
        "fixed_x":        fixed_x,
        "alpha":          alpha,
        "target":         1.0 - alpha,
        "n_data_ok":      n_data_ok,
        "detail_rows":    detail_rows,
    }


_CSV_FIELDS = [
    "data_idx", "method", "functional",
    "condition_mode", "obs_idx", "x_j",
    "B",
    "true_value",
    "ci_lo", "ci_hi", "ci_width",
    "boot_mean", "boot_std", "boot_median", "boot_q25", "boot_q75",
    "covered",
    "status",
]


def save_detail_csv(results: list, csv_path: str):
    """
    Write one CSV with all per-dataset rows from every (functional x method) result.

    Columns
    -------
    data_idx    : int         folder index (1-based)
    method      : str         AB-NPMLE | NPMLE
    functional  : str         posterior_mean | posterior_variance | ...
    condition_mode : str | "" "obs_idx" or "quantile" (blank for prior-level)
    obs_idx     : int | ""    observation index; blank for prior-level or quantile mode
    x_j         : float | ""  conditioning value used (x[obs_idx], or the fixed
                              quantile value under quantile mode; blank for prior-level)
    B           : int         number of valid bootstrap samples
    true_value  : float       closed-form true value of the functional
    ci_lo       : float       lower quantile of bootstrap CI
    ci_hi       : float       upper quantile of bootstrap CI
    ci_width    : float       ci_hi - ci_lo
    boot_mean   : float       mean of bootstrap distribution
    boot_std    : float       std  of bootstrap distribution  (ddof=1)
    boot_median : float       median of bootstrap distribution
    boot_q25    : float       25th percentile of bootstrap distribution
    boot_q75    : float       75th percentile of bootstrap distribution
    covered     : int | ""    1 if true_value in [ci_lo, ci_hi], else 0; blank on skip
    status      : str         ok | skip_no_result | skip_no_data | ...
    """
    all_rows = []
    for res in results:
        all_rows.extend(res["detail_rows"])

    all_rows.sort(key=lambda r: (r.get("functional", ""),
                                 r.get("method", ""),
                                 r.get("data_idx", 0)))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            clean = {}
            for k in _CSV_FIELDS:
                v = row.get(k, "")
                if isinstance(v, float) and not (v != v):
                    clean[k] = f"{v:.6f}"
                elif isinstance(v, float):
                    clean[k] = ""
                else:
                    clean[k] = v
            writer.writerow(clean)

    print(f"Detail CSV saved: {csv_path}  ({len(all_rows)} rows)")


def build_paths_ab(ab_base_dir, data_base_dir, data_name, n_data):
    result_paths, data_paths = [], []
    for i in range(1, n_data + 1):
        result_paths.append(
            os.path.join(ab_base_dir, f"data{i}", data_name,
                         f"result_AB_npmle_{data_name}.npz")
        )
        data_paths.append(
            os.path.join(data_base_dir, f"data{i}", f"{data_name}.npz")
        )
    return result_paths, data_paths


def build_paths_npmle(npmle_base_dir, data_base_dir, data_name, n_data):
    result_paths, data_paths = [], []
    for i in range(1, n_data + 1):
        result_paths.append(
            os.path.join(npmle_base_dir, f"data{i}",
                         f"result_npmle_{data_name}.npz")
        )
        data_paths.append(
            os.path.join(data_base_dir, f"data{i}", f"{data_name}.npz")
        )
    return result_paths, data_paths


def make_summary_plot(results: list, save_path: str, cond_label: str, alpha: float):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        functionals = list(dict.fromkeys(r["functional"] for r in results))
        methods     = list(dict.fromkeys(r["method"]     for r in results))
        target      = 1.0 - alpha
        lookup      = {(r["functional"], r["method"]): r for r in results}
        n_func      = len(functionals)

        fig, axes = plt.subplots(1, n_func, figsize=(5 * n_func, 5), sharey=False)
        if n_func == 1:
            axes = [axes]

        colors = ["steelblue", "coral", "seagreen", "mediumpurple"]

        for ax, func in zip(axes, functionals):
            coverages, labels, bar_colors = [], [], []
            for j, meth in enumerate(methods):
                res = lookup.get((func, meth))
                if res is None:
                    continue
                cov = res["overall"]
                coverages.append(cov)
                labels.append(meth)
                bar_colors.append(colors[j % len(colors)]
                                  if not np.isnan(cov) else "lightgray")

            bars = ax.bar(labels, coverages, color=bar_colors,
                          alpha=0.85, edgecolor="white", width=0.4)
            for bar, val in zip(bars, coverages):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.005, f"{val:.4f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

            ax.axhline(target, color="black", lw=1.8, ls="--",
                       label=f"Target {target:.0%}")
            ax.set_title(func.replace("_", "\n"), fontsize=11)
            ax.set_ylabel("Coverage rate")
            ax.set_ylim(0.0, 1.05)
            ax.legend(fontsize=9)
            ax.grid(axis="y", linestyle=":", alpha=0.4)

            for k, meth in enumerate(methods):
                res = lookup.get((func, meth))
                if res:
                    ax.text(k, 0.02, f"{res['n_data_ok']} datasets",
                            ha="center", va="bottom", fontsize=8, color="gray")

        fig.suptitle(
            f"Functional CI Coverage  ({cond_label},  alpha={alpha},  target={target:.0%})",
            fontsize=13, y=1.01
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved:       {save_path}")

    except Exception as e:
        print(f"[Warning] Plot failed: {e}")


FUNCTIONAL_CHOICES = [
    "posterior_mean", "posterior_variance",
    "prior_mean", "prior_variance",
    "all",
]
ALL_FUNCTIONALS = [
    "posterior_mean", "posterior_variance",
    "prior_mean", "prior_variance",
]


def main():
    parser = argparse.ArgumentParser(
        description="Functional CI coverage: AB-NPMLE vs NPMLE"
    )
    parser.add_argument("--ab_base_dir",    type=str,
                        default="./sim_data_gms_1000")
    parser.add_argument("--npmle_base_dir", type=str,
                        default="./sim_data_npmle")
    parser.add_argument("--data_base_dir",  type=str,
                        default="./sim_data")
    parser.add_argument("--data_name",      type=str, default="WCS1_Normal",
                        help="Subfolder name inside ab/npmle/data base dirs "
                             "(e.g. WCS1_Normal, JZ_A3_s500). Also used to build "
                             "the NPMLE result filename (result_npmle_{data_name}.npz).")
    parser.add_argument("--prior_type",     type=str, default="WCS1_Normal",
                        choices=["WCS1_Normal", "JZ_A3_s500", "Uniform"],
                        help="True prior used to compute closed-form functional values. "
                             "Uniform: G = Unif(unif_low, unif_high).")
    parser.add_argument("--n_data",         type=int, default=50)
    parser.add_argument("--sigma",          type=float, default=1.0)
    parser.add_argument("--alpha",          type=float, default=0.05)
    parser.add_argument("--obs_idx",        type=int, default=0,
                        help="[condition_mode=obs_idx] uses each dataset's "
                             "x[obs_idx] as the posterior conditioning value")
    parser.add_argument("--condition_mode", type=str, default="quantile",
                        choices=["obs_idx", "quantile"],
                        help="How the conditioning value x_j is chosen for "
                             "posterior_mean/posterior_variance. "
                             "'obs_idx' (default): uses each dataset's own x[obs_idx] "
                             "(varies from dataset to dataset). "
                             "'quantile': fixes a single quantile value, computed on "
                             "a large reference dataset (e.g. n=100000), and uses it "
                             "for every dataset.")
    parser.add_argument("--quantile_level", type=float, default=0.75,
                        help="[condition_mode=quantile] quantile level to use (0-1). "
                             "e.g. 0.5=median, 0.9=90th percentile (default: 0.5)")
    parser.add_argument("--quantile_ref_base_dir", type=str, default=None,
                        help="[condition_mode=quantile] base_dir of the large "
                             "reference dataset used to compute the quantile. e.g. "
                             "./sim_data_100000 . If not given, "
                             "constructed automatically by appending '_100000' to --data_base_dir.")
    parser.add_argument("--quantile_ref_idx", type=int, default=1,
                        help="[condition_mode=quantile] reference dataset folder "
                             "number data{idx} (default: 1)")
    parser.add_argument("--functional",     type=str, default="all",
                        choices=FUNCTIONAL_CHOICES)
    parser.add_argument("--method",         type=str, default="both",
                        choices=["ab", "npmle", "both"])
    parser.add_argument("--summary_plot",   action="store_true")
    parser.add_argument("--save_dir",       type=str, default=".")
    parser.add_argument("--quiet",          action="store_true")
    parser.add_argument("--unif_low",  type=float, default=-5.0,
                        help="[Uniform prior] lower bound a of Unif(a, b)")
    parser.add_argument("--unif_high", type=float, default=10.0,
                        help="[Uniform prior] upper bound b of Unif(a, b)")
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, args.data_name)
    os.makedirs(save_dir, exist_ok=True)
    verbose     = not args.quiet
    functionals = ALL_FUNCTIONALS if args.functional == "all" else [args.functional]

    print("=" * 70)
    print("Functional CI Coverage  (AB-NPMLE vs NPMLE)")
    print("=" * 70)
    print(f"  functionals    : {functionals}")
    print(f"  method         : {args.method}")
    print(f"  prior_type     : {args.prior_type}")
    print(f"  n_data         : {args.n_data}")
    print(f"  sigma          : {args.sigma}")
    print(f"  alpha          : {args.alpha}  ->  target = {1-args.alpha:.2%}")
    print(f"  condition_mode : {args.condition_mode}  (posterior functionals only)")

    fixed_x = None
    if args.condition_mode == "quantile":
        if not (0.0 < args.quantile_level < 1.0):
            raise ValueError(
                f"--quantile_level must be in (0, 1). Got: {args.quantile_level}"
            )
        ref_base_dir = args.quantile_ref_base_dir or (args.data_base_dir.rstrip("/") + "_100000")
        ref_path = os.path.join(ref_base_dir, f"data{args.quantile_ref_idx}",
                                 f"{args.data_name}.npz")
        if not os.path.exists(ref_path):
            raise FileNotFoundError(
                f"Quantile reference dataset not found: {ref_path}\n"
                f"Check --quantile_ref_base_dir / --quantile_ref_idx ."
            )
        x_ref   = load_x(ref_path)
        fixed_x = float(np.quantile(x_ref, args.quantile_level))
        print(f"  quantile_level : {args.quantile_level}")
        print(f"  ref_dataset    : {ref_path}  (n_ref={len(x_ref)})")
        print(f"  fixed x_j      : {fixed_x:+.6f}  (applied to every dataset)")
    else:
        print(f"  obs_idx        : {args.obs_idx}")

    print(f"  save_dir       : {save_dir}")
    print("=" * 70 + "\n")

    ab_result_paths, ab_data_paths = build_paths_ab(
        args.ab_base_dir, args.data_base_dir, args.data_name, args.n_data
    )
    npmle_result_paths, npmle_data_paths = build_paths_npmle(
        args.npmle_base_dir, args.data_base_dir,
        args.data_name, args.n_data
    )

    results = []

    for func in functionals:
        print(f"\n{'-'*60}")
        print(f"  Functional: {func}")
        print(f"{'-'*60}")

        if args.method in ("ab", "both"):
            print("\n  [AB-NPMLE]")
            res = compute_coverage(
                result_paths = ab_result_paths,
                data_paths   = ab_data_paths,
                sigma        = args.sigma,
                alpha        = args.alpha,
                prior_type   = args.prior_type,
                functional   = func,
                method_name  = "AB-NPMLE",
                obs_idx      = args.obs_idx,
                fixed_x      = fixed_x,
                verbose      = verbose,
                unif_low     = args.unif_low,
                unif_high    = args.unif_high,
            )
            results.append(res)

        if args.method in ("npmle", "both"):
            print("\n  [NPMLE]")
            res = compute_coverage(
                result_paths = npmle_result_paths,
                data_paths   = npmle_data_paths,
                sigma        = args.sigma,
                alpha        = args.alpha,
                prior_type   = args.prior_type,
                functional   = func,
                method_name  = "NPMLE",
                obs_idx      = args.obs_idx,
                fixed_x      = fixed_x,
                verbose      = verbose,
                unif_low     = args.unif_low,
                unif_high    = args.unif_high,
            )
            results.append(res)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if args.condition_mode == "quantile":
        cond_str = f"x_fixed={fixed_x:+.4f} (q={args.quantile_level})"
    else:
        cond_str = f"obs_idx={args.obs_idx}"
    print(f"  {cond_str}  |  {args.n_data} datasets  |  alpha={args.alpha}")
    print(f"  {'Functional':<22} {'Method':<12} {'n_ok':>5}  {'Target':>8}  {'Coverage':>10}")
    print("  " + "-" * 63)
    for res in results:
        print(f"  {res['functional']:<22} {res['method']:<12} "
              f"{res['n_data_ok']:>5}  "
              f"{res['target']:>8.4f}  "
              f"{res['overall']:>10.4f}")
    print("=" * 70)

    save_dict = {}
    for res in results:
        tag = f"{res['method'].replace('-','_').lower()}_{res['functional']}"
        save_dict[f"{tag}_covered_list"] = res["covered_list"]
        save_dict[f"{tag}_overall"]      = np.float64(res["overall"])
        save_dict[f"{tag}_n_data_ok"]    = np.int64(res["n_data_ok"])
    save_dict["condition_mode"] = args.condition_mode
    save_dict["obs_idx"]        = np.int64(args.obs_idx)
    if fixed_x is not None:
        save_dict["fixed_x"]        = np.float64(fixed_x)
        save_dict["quantile_level"] = np.float64(args.quantile_level)
    save_dict["target"] = np.float64(1.0 - args.alpha)
    save_dict["alpha"]  = np.float64(args.alpha)

    npz_out = os.path.join(save_dir, "coverage_functionals.npz")
    np.savez(npz_out, **save_dict)
    print(f"\nResults saved:    {npz_out}")

    csv_out = os.path.join(save_dir, "coverage_detail.csv")
    save_detail_csv(results, csv_out)

    if args.summary_plot and results:
        plot_path = os.path.join(save_dir, "coverage_functionals.png")
        make_summary_plot(results, plot_path, cond_str, args.alpha)


if __name__ == "__main__":
    main()
