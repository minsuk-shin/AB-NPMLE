"""Functional bootstrap CI coverage from saved NPZ path templates.

Supply observed-data, NPEB, and AMEM templates containing {data_id}.
The script reads data1 through dataN, where N is --n-data. All input
triples must exist before calculation.

Example (replace the root directories with your result directories):

python eval/functional_ci.py \
    --observed-template '/path/to/sim_data/data{data_id}/JZ_A3_s500.npz' \
    --npeb-template '/path/to/npeb/data{data_id}/result_npmle_JZ_A3_s500.npz' \
    --amem-template '/path/to/amem/data{data_id}/JZ_A3_s500/bootstrap/result_AMEM_JZ_A3_s500.npz' \
    --n-data 50 --prior-type JZ_A3_s500 \
    --quantile-reference-data /path/to/reference/JZ_A3_s500.npz \
    --output-dir /path/to/functional_ci_results/JZ_A3_s500

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
        if not np.isfinite(boot_vals).all():
            raise ValueError(f"Non-finite bootstrap functional: {method_name} "
                             f"{functional} data{i} ({rpath})")

        ci_lo = float(np.quantile(boot_vals, lo))
        ci_hi = float(np.quantile(boot_vals, hi))

        true_val = true_functional(functional, prior_type, x_j, sigma,
                                   n_obs=len(x),
                                   unif_low=unif_low, unif_high=unif_high)
        if not np.isfinite(true_val):
            raise ValueError(f"Non-finite true functional: {functional} data{i}")

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



def save_summary_csv(results: list, csv_path: str):
    """Save data1 interval endpoints and coverage across all analysis datasets."""
    fields = ["functional", "method", "condition_mode", "fixed_x", "alpha",
              "true_value", "data1_ci_lo", "data1_ci_hi", "data1_B",
              "n_covered", "n_data_ok", "coverage"]
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            data1 = next((row for row in result["detail_rows"]
                          if row["data_idx"] == 1 and row["status"] == "ok"), None)
            if data1 is None:
                raise ValueError(f"Missing data1 interval for {result['method']} "
                                 f"{result['functional']}")
            writer.writerow({
                "functional": result["functional"],
                "method": result["method"],
                "condition_mode": result["condition_mode"],
                "fixed_x": result["fixed_x"] if result["fixed_x"] is not None else "",
                "alpha": result["alpha"],
                "true_value": data1["true_value"],
                "data1_ci_lo": data1["ci_lo"],
                "data1_ci_hi": data1["ci_hi"],
                "data1_B": data1["B"],
                "n_covered": int(np.nansum(result["covered_list"])),
                "n_data_ok": result["n_data_ok"],
                "coverage": result["overall"],
            })
    print(f"Summary CSV saved: {csv_path}")


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


def collect_dataset_paths(args, parser):
    templates = (args.observed_template, args.npeb_template, args.amem_template)
    if args.n_data < 1:
        parser.error("--n-data must be positive")
    if any(template is None or "{data_id}" not in template for template in templates):
        parser.error("all three path templates must contain {data_id}")
    specified = {
        data_id: tuple(template.replace("{data_id}", str(data_id))
                       for template in templates)
        for data_id in range(1, args.n_data + 1)
    }
    for data_id in range(1, args.n_data + 1):
        for path in specified[data_id]:
            if not os.path.isfile(path):
                parser.error(f"data{data_id} input not found: {path}")
    return specified


def main():
    parser = argparse.ArgumentParser(
        description="Functional bootstrap CI coverage from saved observed-data, NPEB and AMEM NPZ paths")
    parser.add_argument("--observed-template", required=True,
                        help="Observed-data NPZ path containing {data_id}")
    parser.add_argument("--npeb-template", required=True,
                        help="NPEB result NPZ path containing {data_id}")
    parser.add_argument("--amem-template", required=True,
                        help="AMEM result NPZ path containing {data_id}")
    parser.add_argument("--n-data", type=int, default=50,
                        help="Read data1 through dataN (default: 50)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for coverage NPZ, detail CSV and summary CSV")
    parser.add_argument("--prior-type", required=True,
                        choices=["WCS1_Normal", "JZ_A3_s500", "Uniform"],
                        help="True prior used to compute functional values")
    parser.add_argument("--functional", default="all", choices=FUNCTIONAL_CHOICES)
    parser.add_argument("--method", default="both", choices=["amem", "npeb", "both"])
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--condition-mode", default="quantile",
                        choices=["obs_idx", "quantile"],
                        help="Posterior conditioning: per-dataset observation or fixed reference quantile")
    parser.add_argument("--obs-idx", type=int, default=0)
    parser.add_argument("--quantile-reference-data",
                        help="Observed-data NPZ used to fix x for quantile conditioning")
    parser.add_argument("--quantile-level", type=float, default=0.75)
    parser.add_argument("--unif-low", type=float, default=-5.0)
    parser.add_argument("--unif-high", type=float, default=10.0)
    parser.add_argument("--summary-plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not np.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("--sigma must be positive and finite")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be in (0, 1)")
    if args.obs_idx < 0:
        parser.error("--obs-idx must be nonnegative")
    if not 0 < args.quantile_level < 1:
        parser.error("--quantile-level must be in (0, 1)")
    if args.unif_high <= args.unif_low:
        parser.error("--unif-high must exceed --unif-low")
    paths = collect_dataset_paths(args, parser)
    functionals = ALL_FUNCTIONALS if args.functional == "all" else [args.functional]
    needs_posterior = any(name.startswith("posterior_") for name in functionals)
    fixed_x = None
    if needs_posterior and args.condition_mode == "quantile":
        if not args.quantile_reference_data:
            parser.error("--quantile-reference-data is required for quantile conditioning")
        if not os.path.isfile(args.quantile_reference_data):
            parser.error(f"quantile reference not found: {args.quantile_reference_data}")
        x_ref = load_x(args.quantile_reference_data)
        fixed_x = float(np.quantile(x_ref, args.quantile_level))
    elif args.quantile_reference_data:
        parser.error("--quantile-reference-data requires posterior quantile conditioning")

    observed_paths = [paths[i][0] for i in range(1, args.n_data + 1)]
    npeb_paths = [paths[i][1] for i in range(1, args.n_data + 1)]
    amem_paths = [paths[i][2] for i in range(1, args.n_data + 1)]
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "input_paths.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["data_idx", "observed_data", "npeb_result", "amem_result"])
        for i in range(1, args.n_data + 1):
            writer.writerow([i, *(os.path.abspath(path) for path in paths[i])])

    print(f"Functional CI: {args.n_data} datasets, {args.prior_type}, "
          f"{args.method}, {functionals}, sigma={args.sigma}, alpha={args.alpha}")
    if fixed_x is not None:
        print(f"Fixed posterior x: {fixed_x:.6f} "
              f"(q={args.quantile_level}, {args.quantile_reference_data})")
    results = []
    for functional in functionals:
        for method_name, result_paths in (("AMEM", amem_paths), ("NPEB", npeb_paths)):
            if args.method != "both" and args.method != method_name.lower():
                continue
            result = compute_coverage(
                result_paths=result_paths, data_paths=observed_paths,
                sigma=args.sigma, alpha=args.alpha, prior_type=args.prior_type,
                functional=functional, method_name=method_name,
                obs_idx=args.obs_idx, fixed_x=fixed_x, verbose=not args.quiet,
                unif_low=args.unif_low, unif_high=args.unif_high)
            if result["n_data_ok"] != args.n_data:
                raise RuntimeError(f"{method_name} {functional}: only "
                                   f"{result['n_data_ok']}/{args.n_data} datasets succeeded")
            results.append(result)
            print(f"{functional} {method_name}: coverage={result['overall']:.4f} "
                  f"({result['n_data_ok']} datasets)")

    save_dict = {}
    for result in results:
        key = f"{result['method'].lower()}_{result['functional']}"
        save_dict[f"{key}_covered_list"] = result["covered_list"]
        save_dict[f"{key}_overall"] = np.float64(result["overall"])
        save_dict[f"{key}_n_data_ok"] = np.int64(result["n_data_ok"])
    save_dict.update(condition_mode=args.condition_mode, obs_idx=np.int64(args.obs_idx),
                     target=np.float64(1.0 - args.alpha), alpha=np.float64(args.alpha))
    if fixed_x is not None:
        save_dict["fixed_x"] = np.float64(fixed_x)
        save_dict["quantile_level"] = np.float64(args.quantile_level)
    np.savez(os.path.join(output_dir, "coverage_functionals.npz"), **save_dict)
    save_detail_csv(results, os.path.join(output_dir, "coverage_detail.csv"))
    save_summary_csv(results, os.path.join(output_dir, "coverage_summary.csv"))
    if args.summary_plot:
        condition = (f"x_fixed={fixed_x:+.4f}" if fixed_x is not None
                     else f"obs_idx={args.obs_idx}")
        make_summary_plot(results, os.path.join(output_dir, "coverage_functionals.png"),
                          condition, args.alpha)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
