"""
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

N_BOOTSTRAP=1000
N_WORKERS=40
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

nohup bash -c '
seq 1 1 | xargs -P 1 -I {} bash -c "
    echo \"[\$(date +%H:%M:%S)] data{} started\"
    python -u run_npmle_sim_iter.py \
        --data_idx    {} \
        --n_bootstrap '"$N_BOOTSTRAP"' \
        --n_workers   '"$N_WORKERS"' \
        > '"$LOG_DIR"'/data{}.log 2>&1
    echo \"[\$(date +%H:%M:%S)] data{} done\"
" >> '"$LOG_DIR"'/main.log 2>&1

echo "[$(date +%H:%M:%S)] All done!"  >> '"$LOG_DIR"'/main.log
' >> "$LOG_DIR/main.log" 2>&1 &

echo "PID: $!"
echo "Live tail: tail -f ${LOG_DIR}/main.log"
"""
### Increasing num_workers parallelizes across multiple bootstrap replicates;
### num_threads parallelizes within a single fit. Keep num_threads fixed at 1
### and increase num_workers instead. ###
"""
nohup bash -c '
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -u run_npmle_sim_iter.py --data_idx 1 --n_bootstrap 1000 --n_workers 40
' > log.txt 2>&1 &
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
    i, x_arr, w_counts, prec_arr, threshold = args
    try:
        x_boot = np.repeat(x_arr, w_counts)          # stored as a flat (n,) array
        m = GLMixture(prec_type='diagonal', homoscedastic=True)
        m.fit(x_boot.reshape(-1, 1), prec_arr, solver='mosek', numThreads=1)   
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
    parser.add_argument("--seed",        type=int, default=None,
                        help="Seed for sampling the bootstrap weights (W). If not given, it is "
                             "auto-generated from data_idx (1000 + data_idx). It doesn't need to "
                             "match the AB-NPMLE side for a paired comparison, since the "
                             "bootstrap_weights are saved to file and reused as-is anyway. The seed "
                             "value itself is saved in the result file for reproducibility record-keeping.")
    args = parser.parse_args()

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

    data = np.load(DATASET_PATH)
    x    = data['x']
    n    = len(x)
    sigma2 = args.sigma ** 2
    prec = np.array([1.0 / sigma2])

    print(f"[data{i}] n={n}, bootstrap={N_BOOTSTRAP}, workers={N_WORKERS}, seed={seed}, sigma={args.sigma}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    probs = torch.ones(n) / n
    w_all = torch.distributions.Multinomial(
        total_count=n, probs=probs
    ).sample((N_BOOTSTRAP,)).long().numpy()

    args_list = [
        (b, x, w_all[b], prec, WEIGHT_THRESHOLD)
        for b in range(N_BOOTSTRAP)
    ]

    prior_atoms   = {}
    prior_weights = {}
    boot_samples  = {}          # ← added
    cum_times     = []

    t_start = time.time()
    # ctx = get_context("spawn")
    ctx = get_context("fork")
    with ctx.Pool(processes=N_WORKERS, maxtasksperchild=1) as pool:
        for count, (b, atoms, weights, x_boot, err) in enumerate(   
            pool.imap_unordered(fit_one_bootstrap, args_list), start=1
        ):
            cum_times.append(time.time() - t_start)
            if err is not None:
                print(f"  [Bootstrap {b}] Error: {err}")
                prior_atoms[b]  = None
                prior_weights[b] = None
                boot_samples[b] = None                          
            else:
                prior_atoms[b]   = atoms
                prior_weights[b] = weights
                boot_samples[b]  = x_boot                         

            if count % 50 == 0:
                elapsed = cum_times[-1]
                elapsed_str = f"{elapsed/60:.1f}min" if elapsed >= 60 else f"{elapsed:.1f}s"
                print(f"  Bootstrap {count}/{N_BOOTSTRAP} done  [elapsed: {elapsed_str}]")

    total_elapsed = time.time() - t_start
    total_str = f"{total_elapsed/60:.1f}min" if total_elapsed >= 60 else f"{total_elapsed:.1f}s"
    print(f"[data{i}] done  [total elapsed: {total_str}]")

    atoms_list   = [prior_atoms[b]   for b in sorted(prior_atoms.keys())]
    weights_list = [prior_weights[b] for b in sorted(prior_weights.keys())]
    samples_list = [boot_samples[b]  for b in sorted(boot_samples.keys())]  

    save_path = os.path.join(SAVE_DIR, f"result_npmle_{args.data_name}.npz")
    np.savez(
        save_path,
        atoms             = np.array(atoms_list,   dtype=object),
        weights           = np.array(weights_list, dtype=object),
        bootstrap_samples = np.array(samples_list, dtype=object), 
        cum_times         = np.array(cum_times,    dtype=np.float64),
        total_time        = np.float64(total_elapsed),
        n_bootstrap       = np.int64(N_BOOTSTRAP),
        bootstrap_weights = w_all,
        seed              = np.int64(seed),
        sigma             = np.float64(args.sigma),
    )
    print(f"Saved: {save_path}")
