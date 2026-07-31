"""
Simulation data generator for nonparametric empirical Bayes (normal means model)
    x_i = theta_i + eps_i,   eps_i ~ N(0, sigma^2)
    theta_i ~ pi0  (unknown mixing distribution)

Output .npz keys:
    x     : observed data x   (n,)
    theta : true latent mean  (n,)
    eps   : noise             (n,)
    sigma : noise std — shape (1,) (homoskedastic)

Settings (only the three used in the AAAI 2027 experiments):
  - Jiang & Zhang (2009), A=3, s=500                          → JZ_A3_s500
  - Willwerscheid, Carbonetto & Stephens (2025), WCS-1 Normal → WCS1_Normal
  - Willwerscheid, Carbonetto & Stephens (2025), WCS-3 Tophat → WCS3_Tophat
"""

import numpy as np
import os


# ===========================================================================
# Helpers
# ===========================================================================

def _add_noise(theta: np.ndarray,
               sigma,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """eps ~ N(0, sigma^2),  x = theta + eps"""
    eps = rng.normal(0.0, sigma, size=theta.shape)
    return theta + eps, eps


def _save(path: str, x, theta, eps, sigma, **extra):
    """Save a single dataset to .npz"""
    np.savez(path, x=x, theta=theta, eps=eps, sigma=np.atleast_1d(sigma), **extra)
    print(f"  saved → {path}")


# ===========================================================================
# 1. Jiang & Zhang (2009) — A=3, s=500
# ===========================================================================

def generate_JZ(A: float = 3, s: int = 500,
                n: int = 1000, sigma: float = 1.0,
                rng: np.random.Generator = None) -> dict:
    """
    G = (1 - s/n)*delta_0 + (s/n)*delta_A
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = np.zeros(n)
    theta[rng.choice(n, size=s, replace=False)] = A
    x, eps = _add_noise(theta, sigma, rng)

    return dict(x=x, theta=theta, eps=eps, sigma=np.array([sigma]),
                A=np.array(A), s=np.array(s))


# ===========================================================================
# 2. WCS Settings
# ===========================================================================

def generate_WCS1(n: int = 1000, sigma: float = 1.0,
                  rng: np.random.Generator = None) -> dict:
    """
    WCS-1 — Normal
        theta_i ~ N(0, 2^2)
    """
    if rng is None:
        rng = np.random.default_rng()
    theta = rng.normal(0.0, 2.0, size=n)
    x, eps = _add_noise(theta, sigma, rng)
    return dict(x=x, theta=theta, eps=eps, sigma=np.array([sigma]))


def generate_WCS3(n: int = 1000, sigma: float = 1.0,
                  rng: np.random.Generator = None) -> dict:
    """
    WCS-3 — Asymmetric Tophat
        theta_i ~ Uniform(-5, 10)
    """
    if rng is None:
        rng = np.random.default_rng()
    theta = rng.uniform(-5.0, 10.0, size=n)
    x, eps = _add_noise(theta, sigma, rng)
    return dict(x=x, theta=theta, eps=eps, sigma=np.array([sigma]))


# ===========================================================================
# Generate everything
# ===========================================================================

def generate_all(out_dir: str, seed: int = 42, n: int = 1000):
    """
    Generate the three experiment simulation datasets and save as .npz to out_dir/.

    Directory structure
    -------------------
    out_dir/
      JZ_A3_s500.npz
      WCS1_Normal.npz
      WCS3_Tophat.npz

    .npz keys
    ---------
    x     : (n,)      observed x_i
    theta : (n,)      true theta_i
    eps   : (n,)      noise (x = theta + eps)
    sigma : (1,)      noise std
    A, s              signal params [JZ only]
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("── Jiang & Zhang (2009), A=3, s=500 ──────────────────────")
    _save(os.path.join(out_dir, "JZ_A3_s500.npz"),
          **generate_JZ(A=3, s=500, n=n, rng=rng))

    print("── WCS-1 Normal ──────────────────────────────────────────")
    _save(os.path.join(out_dir, "WCS1_Normal.npz"),
          **generate_WCS1(n=n, rng=rng))

    print("── WCS-3 Tophat ──────────────────────────────────────────")
    _save(os.path.join(out_dir, "WCS3_Tophat.npz"),
          **generate_WCS3(n=n, rng=rng))

    # Summary
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".npz"))
    print(f"\n{len(files)} .npz file(s) saved → '{out_dir}/'")
    print(f"\n{'filename':<52} {'E[theta]':>8}  {'Var[theta]':>8}  {'E[x]':>8}")
    print("-" * 82)
    for f in files:
        d = np.load(os.path.join(out_dir, f))
        print(f"{f:<52} {d['theta'].mean():>8.3f}  {d['theta'].var():>8.3f}  {d['x'].mean():>8.3f}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate simulation .npz datasets")

    # Single-run mode (unchanged from before, e.g. generating one n=100000 dataset for the query)
    parser.add_argument("--out_dir", type=str, default="./sim_data_100000/data1",
                        help="Output directory for single-run mode (ignored when --iter is set)")
    parser.add_argument("--seed",    type=int, default=46)   # was originally 42 # always double-check seed before generating new data
    parser.add_argument("--n",       type=int, default=None,
                        help="Sample size n. If not set: 1000 in --iter mode, 100000 in single-run mode")

    # Iteration mode (replaces gen_data_iter.py: instead of re-launching this script 45 times
    # via subprocess, generate_all() is called repeatedly in-process)
    parser.add_argument("--iter", action="store_true",
                        help="Repeatedly generate data for i = iter_start..iter_end, varying seed and out_dir")
    parser.add_argument("--iter_start", type=int, default=6,
                        help="Starting iteration index i (matches gen_data_iter.py default: 6)")
    parser.add_argument("--iter_end", type=int, default=50,
                        help="Ending iteration index i, inclusive (matches gen_data_iter.py default: 50)")
    parser.add_argument("--iter_seed_base", type=int, default=41,
                        help="Seed for iteration i = iter_seed_base + i (matches gen_data_iter.py: 41)")
    parser.add_argument("--iter_out_base", type=str, default="./sim_data",
                        help="Save location for iterations: {iter_out_base}/data{i} (matches gen_data_iter.py: ./sim_data)")

    args = parser.parse_args()

    if args.iter:
        n = args.n if args.n is not None else 1000
        for i in range(args.iter_start, args.iter_end + 1):
            seed = args.iter_seed_base + i
            out_dir = os.path.join(args.iter_out_base, f"data{i}")
            print(f"\n===== iteration i={i}  (seed={seed}, out_dir={out_dir}) =====")
            generate_all(out_dir=out_dir, seed=seed, n=n)
    else:
        n = args.n if args.n is not None else 100000
        generate_all(out_dir=args.out_dir, seed=args.seed, n=n)
