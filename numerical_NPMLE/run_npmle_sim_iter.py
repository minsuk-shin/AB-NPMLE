r"""Fit numerical NPMLE to bootstrap replicates of a dataset.

Run from the repository root. Keep BLAS single-threaded when using workers:

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

Regular bootstrap (1,000 fits):

    python numerical_NPMLE/run_npmle_sim_iter.py \
        --data_idx 1 --data_name WCS1_Normal --data_mode sim \
        --n_bootstrap 1000 --n_workers 20

One fit with all observation weights equal to 1:

    python numerical_NPMLE/run_npmle_sim_iter.py \
        --data_idx 1 --data_name WCS1_Normal --data_mode sim \
        --n_bootstrap 1 --n_workers 1 --bootstrap_weights ones \
        --save_dir fixed_results/npeb/data1

Use --bootstrap_weights PATH to reuse weights from an NPZ file.
"""

# Requires the `npeb` package (MIT License): https://github.com/jake-soloff/NPEB
#   pip install npeb
# `m.fit(..., solver='mosek', ...)` below needs a MOSEK license (free for academic use):
# https://www.mosek.com/products/academic-licenses/

import numpy as np
import torch
import os
import time
import argparse
from multiprocessing import get_context
from npeb import GLMixture


def fit_one_bootstrap(args):
    i, x_arr, w_counts, prec_arr, threshold, row_condition = args
    try:
        x_boot = np.repeat(x_arr, w_counts)          # stored as a flat (n,) array
        m = GLMixture(prec_type='diagonal', homoscedastic=True)
        try:
            m.fit(x_boot.reshape(-1, 1), prec_arr, solver='mosek', numThreads=1,
                  row_condition=row_condition)
        except Exception:
            if not row_condition:
                raise
            # Some repeated group-weight samples make MOSEK's default
            # interior-point formulation report an unknown solution status.
            # Refit the same sample with the dual formulation.
            m = GLMixture(prec_type='diagonal', homoscedastic=True)
            m.fit(x_boot.reshape(-1, 1), prec_arr, solver='mosek', numThreads=1,
                  row_condition=True, intpntSolveForm='dual')
        atoms   = m.atoms.flatten().copy()
        weights = m.weights.flatten().copy()
        idx = weights > threshold
        return i, atoms[idx], weights[idx], x_boot.copy(), None
    except Exception as e:
        return i, None, None, None, str(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repeatedly fit NPMLE (via the npeb package) to bootstrap replicates of a "
                     "dataset, as a numerical benchmark to compare against AB-NPMLE."
    )
    parser.add_argument("--data_idx",    type=int, required=True,
                        help="Dataset index i")
    parser.add_argument("--data_name",   type=str, default="WCS1_Normal",
                        help="Dataset filename without extension (e.g. WCS1_Normal, JZ_A3_s500, WCS3_Tophat)")
    parser.add_argument("--data_mode",   type=str, default="sim", choices=["real", "sim"],
                        help="Which directory layout to use for the default data_dir/save_dir: "
                             "'sim' for ./sim_data/data{i}/... (matches gen_data/gen_sim_data.py), "
                             "'real' for ./real_data/... .")
    parser.add_argument("--data_dir",    type=str, default=None,
                        help="Overrides the data directory directly. If omitted, derived from --data_mode.")
    parser.add_argument("--save_dir",    type=str, default=None,
                        help="Overrides the save directory directly. If omitted, derived from --data_mode "
                             "and --sigma (results for different candidate sigma values are kept in "
                             "separate folders, e.g. for the noise-scale sensitivity study).")
    parser.add_argument("--sigma",       type=float, default=1.0,
                        help="Noise std used to build the precision passed to GLMixture.fit "
                             "(prec = 1 / sigma**2). This is a candidate scale for the NPMLE fit and "
                             "need not equal the data's true generating sigma (e.g. for the "
                             "noise-scale sensitivity / MDV study).")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--n_workers",   type=int, default=20)
    parser.add_argument("--checkpoint_every", type=int, default=0,
                        help="Save completed bootstrap fits every N results; 0 keeps the original end-only save.")
    parser.add_argument("--start_method", choices=["fork", "spawn"], default="fork",
                        help="Multiprocessing start method (default: fork).")
    parser.add_argument("--bootstrap_weights", type=str, default=None,
                        help="NPZ with saved 'bootstrap_weights' or 'counts' to reuse exactly, "
                             "or 'ones' to use one copy of every observation in each fit. "
                             "Failed fits retry the same weights.")
    parser.add_argument("--resample_failed_weights", action="store_true",
                        help="If a fit fails, draw a new bootstrap weight vector for that "
                             "index before retrying. The final weights are saved with the "
                             "result so paired AB inference can reuse them.")
    parser.add_argument("--row_condition", action="store_true",
                        help="Scale each likelihood row by its maximum before the MOSEK fit; "
                             "this preserves the mixing-weight optimizer and can improve stability.")
    parser.add_argument("--max_attempts", type=int, default=20,
                        help="Maximum attempts per bootstrap index; failed fits are resampled "
                             "until n_bootstrap successful fits are obtained.")
    parser.add_argument("--seed",        type=int, default=None,
                        help="Seed for sampling the bootstrap weights (W). If not given, it is "
                             "auto-generated from data_idx (1000 + data_idx). It doesn't need to "
                             "match the AB-NPMLE side for a paired comparison, since the "
                             "bootstrap_weights are saved to file and reused as-is anyway. The seed "
                             "value itself is saved in the result file for reproducibility record-keeping.")
    args = parser.parse_args()
    if args.n_bootstrap < 1 or args.n_workers < 1 or args.max_attempts < 1 or args.checkpoint_every < 0:
        parser.error("n_bootstrap, n_workers, and max_attempts must be positive; checkpoint_every must be nonnegative")
    if args.bootstrap_weights == "ones" and args.resample_failed_weights:
        parser.error("--resample_failed_weights cannot be used with --bootstrap_weights ones")

    i = args.data_idx
    seed = args.seed if args.seed is not None else 1000 + i

    if args.data_mode == "sim":
        default_data_dir = f"./sim_data/data{i}"
        default_save_dir = f"./sim_data_npmle/data{i}"
    else:  # 'real'
        default_data_dir = "./real_data"
        default_save_dir = f"./real_data_npmle"

    data_dir = args.data_dir or default_data_dir
    SAVE_DIR = args.save_dir or default_save_dir

    DATASET_PATH     = os.path.join(data_dir, args.data_name + ".npz")
    N_BOOTSTRAP      = args.n_bootstrap
    N_WORKERS        = args.n_workers
    WEIGHT_THRESHOLD = 0.001

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"result_npmle_{args.data_name}.npz")

    data = np.load(DATASET_PATH)
    x    = data['x']
    n    = len(x)
    sigma2 = args.sigma ** 2
    prec = np.array([1.0 / sigma2])

    print(f"[data{i}] n={n}, bootstrap={N_BOOTSTRAP}, workers={N_WORKERS}, seed={seed}, "
          f"sigma={args.sigma}, weight_source={args.bootstrap_weights or 'generated'}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    probs = torch.ones(n) / n
    prior_atoms = {}
    prior_weights = {}
    boot_samples = {}
    cum_times = []
    cum_compute_times = []
    attempts = np.zeros(N_BOOTSTRAP, dtype=np.int64)
    failed_attempts = 0
    elapsed_before = 0.0
    compute_before = 0.0
    weight_generation_seconds = 0.0

    if args.checkpoint_every and os.path.exists(save_path):
        with np.load(save_path, allow_pickle=True) as prior:
            if int(prior["n_bootstrap"]) != N_BOOTSTRAP or not np.isclose(float(prior["sigma"]), args.sigma):
                raise ValueError("Saved bootstrap count or sigma differs from this run; use another save_dir.")
            if "data_path" in prior.files and str(prior["data_path"]) != DATASET_PATH:
                raise ValueError("Saved data path differs from this run; use another save_dir.")
            w_all = prior["bootstrap_weights"].copy()
            if args.bootstrap_weights == "ones" and not np.all(w_all == 1):
                raise ValueError("Saved checkpoint does not use all-one weights; use another save_dir.")
            saved_atoms = prior["atoms"]
            saved_weights = prior["weights"]
            saved_samples = prior["bootstrap_samples"]
            status = prior["status"] if "status" in prior.files else np.array(
                [a is not None for a in saved_atoms], dtype=np.bool_)
            for b in np.flatnonzero(status):
                prior_atoms[int(b)] = saved_atoms[b]
                prior_weights[int(b)] = saved_weights[b]
                boot_samples[int(b)] = saved_samples[b]
            cum_times = list(prior["cum_times"])
            cum_compute_times = (list(prior["cum_compute_times"])
                                 if "cum_compute_times" in prior.files
                                 else [float("nan")] * len(cum_times))
            attempts = prior["attempts"].astype(np.int64).copy()
            failed_attempts = int(prior["failed_attempts"])
            elapsed_before = float(prior["total_time"])
            if "bootstrap_compute_seconds" in prior.files:
                compute_before = float(prior["bootstrap_compute_seconds"])
            elif len(prior_atoms):
                compute_before = float("nan")
                print("[resume] prior file has no compute-only time; full-run compute time is unavailable", flush=True)
        print(f"[resume] {len(prior_atoms)}/{N_BOOTSTRAP} bootstrap fits already saved", flush=True)
    elif args.bootstrap_weights == "ones":
        w_all = np.ones((N_BOOTSTRAP, n), dtype=np.int64)
    elif args.bootstrap_weights:
        with np.load(args.bootstrap_weights, allow_pickle=False) as source:
            key = "counts" if "counts" in source else "bootstrap_weights"
            if key not in source:
                parser.error("bootstrap NPZ must contain counts or bootstrap_weights")
            if "x" in source and not np.array_equal(source["x"].reshape(-1), x.reshape(-1)):
                parser.error("bootstrap weights were generated for different observations")
            w_all = np.asarray(source[key])
        if w_all.shape != (N_BOOTSTRAP, n) or not np.issubdtype(w_all.dtype, np.number):
            parser.error(f"bootstrap weights must have shape ({N_BOOTSTRAP}, {n})")
        if (not np.isfinite(w_all).all() or np.any(w_all < 0)
                or not np.all(w_all == np.rint(w_all))
                or not np.all(w_all.sum(axis=1) == n)):
            parser.error("bootstrap counts must be nonnegative integers summing to n")
        w_all = w_all.astype(np.int64, copy=True)
    else:
        weight_start = time.perf_counter()
        w_all = torch.distributions.Multinomial(
            total_count=n, probs=probs
        ).sample((N_BOOTSTRAP,)).long().numpy()
        weight_generation_seconds = time.perf_counter() - weight_start

    t_start = time.time()
    def save_progress(bootstrap_compute_seconds):
        tmp_path = save_path + ".tmp.npz"
        np.savez(
            tmp_path,
            atoms=np.array([prior_atoms.get(b) for b in range(N_BOOTSTRAP)], dtype=object),
            weights=np.array([prior_weights.get(b) for b in range(N_BOOTSTRAP)], dtype=object),
            bootstrap_samples=np.array([boot_samples.get(b) for b in range(N_BOOTSTRAP)], dtype=object),
            cum_times=np.array(cum_times, dtype=np.float64),
            cum_compute_times=np.array(cum_compute_times, dtype=np.float64),
            total_time=np.float64(elapsed_before + time.time() - t_start),
            bootstrap_compute_seconds=np.float64(bootstrap_compute_seconds),
            n_bootstrap=np.int64(N_BOOTSTRAP),
            bootstrap_weights=w_all,
            bootstrap_weights_source=np.str_(args.bootstrap_weights or "generated"),
            resample_failed_weights=np.bool_(args.resample_failed_weights),
            attempts=attempts,
            failed_attempts=np.int64(failed_attempts),
            seed=np.int64(seed),
            sigma=np.float64(args.sigma),
            data_path=np.str_(DATASET_PATH),
            status=np.array([b in prior_atoms for b in range(N_BOOTSTRAP)], dtype=np.bool_),
        )
        os.replace(tmp_path, save_path)

    ctx = get_context(args.start_method)
    with ctx.Pool(processes=N_WORKERS, maxtasksperchild=1) as pool:
        fit_loop_start = time.perf_counter()
        save_seconds = 0.0
        pending = [b for b in range(N_BOOTSTRAP) if b not in prior_atoms]
        completed_since_save = 0
        while pending:
            tasks = ((b, x, w_all[b], prec, WEIGHT_THRESHOLD, args.row_condition)
                     for b in pending)
            retry = []
            for b, atoms, weights, x_boot, err in pool.imap_unordered(
                    fit_one_bootstrap, tasks):
                attempts[b] += 1
                if err is not None:
                    failed_attempts += 1
                    if attempts[b] >= args.max_attempts:
                        raise RuntimeError(
                            f"Bootstrap {b} failed {attempts[b]} times; last error: {err}")
                    if args.resample_failed_weights or not args.bootstrap_weights:
                        w_all[b] = torch.distributions.Multinomial(
                            total_count=n, probs=probs).sample().long().numpy()
                    retry.append(b)
                    print(f"  [Bootstrap {b}] Error: {err}; retry {attempts[b] + 1}",
                          flush=True)
                    continue
                prior_atoms[b] = atoms
                prior_weights[b] = weights
                boot_samples[b] = x_boot
                cum_times.append(elapsed_before + time.time() - t_start)
                cum_compute_times.append(compute_before + weight_generation_seconds
                                         + time.perf_counter() - fit_loop_start - save_seconds)
                completed_since_save += 1
                if args.checkpoint_every and completed_since_save >= args.checkpoint_every:
                    save_start = time.perf_counter()
                    save_progress(compute_before + weight_generation_seconds
                                  + save_start - fit_loop_start - save_seconds)
                    save_seconds += time.perf_counter() - save_start
                    completed_since_save = 0
                count = len(prior_atoms)
                if count % 50 == 0:
                    elapsed = cum_times[-1]
                    elapsed_str = f"{elapsed/60:.1f}min" if elapsed >= 60 else f"{elapsed:.1f}s"
                    print(f"  Bootstrap {count}/{N_BOOTSTRAP} done  "
                          f"[elapsed: {elapsed_str}]", flush=True)
            pending = retry
            if args.checkpoint_every and completed_since_save:
                save_start = time.perf_counter()
                save_progress(compute_before + weight_generation_seconds
                              + save_start - fit_loop_start - save_seconds)
                save_seconds += time.perf_counter() - save_start
                completed_since_save = 0
        bootstrap_compute_seconds = (compute_before + weight_generation_seconds
                                     + time.perf_counter() - fit_loop_start - save_seconds)

    total_elapsed = elapsed_before + time.time() - t_start
    total_str = f"{total_elapsed/60:.1f}min" if total_elapsed >= 60 else f"{total_elapsed:.1f}s"
    print(f"[data{i}] done  [total elapsed: {total_str}]")
    print(f"[data{i}] bootstrap compute time: {bootstrap_compute_seconds:.2f}s", flush=True)

    save_progress(bootstrap_compute_seconds)
    print(f"Saved: {save_path}")
