# AB-NPMLE: Amortized Bootstrap for Nonparametric Maximum Likelihood Estimation

AB-NPMLE learns a map from bootstrap weights and a candidate noise scale to a finite atomic mixing law
for the normal-means model, via the **Amortized Mixture–EM Network
(AMEM-Net)**, replacing repeated numerical NPMLE optimization.

> Provided for review/reproducibility purposes. Author identity has been
> removed in accordance with double-blind review policy.

## Requirements

```bash
pip install -r requirements.txt
```

**PyTorch (GPU) is installed separately**
PyTorch was installed separately with `--extra-index-url https://download.pytorch.org/whl/cu130` (CUDA 13.0).
If this doesn't match your driver, install the build that fits your setup instead.

`train.py`, `infer.py`, and `mdv.py` all pick the device automatically at
import time (`cuda` if available, else `mps`, else `cpu`) — no `--device`
flag is needed, and none of the example commands below change. On Apple
Silicon, just install the standard (non-CUDA) PyTorch build:

```bash
pip install torch
```

Do **not** use the `--extra-index-url` above on macOS/MPS; that index only
hosts CUDA builds.

> There is no `--device` flag — the scripts only auto-detect in the order
> `cuda` → `mps` → `cpu`. On a Mac this already means MPS is picked
> automatically (no CUDA to compete with). To force a specific device
> anyway, either edit the `device = ...` lines near the top of `train.py`,
> `infer.py`, and `mdv.py` directly, or hide CUDA from `torch` first, e.g.
> `CUDA_VISIBLE_DEVICES="" python train.py ...` to fall through to `mps`/`cpu`.

R (for `eval/wasstest.R`):

```r
install.packages("twosamples")
```

## Structure

```
.
├── run_npmle_sim_iter.py        # numerical Bootstrap-NPMLE baseline
├── train.py                     # trains AMEM-Net (AB-NPMLE)
├── infer.py                     # bootstrap inference with a trained checkpoint
├── mdv.py                       # maximum deconvolvable variance (MDV) comparison
├── sigma_sensitivity.py         # sigma sensitivity check (AB-NPMLE vs NPMLE)
├── gen_data/                    # data generation (simulated + real)
│   ├── gen_sim_data.py          # simulation data generation
│   ├── gen_real_data.R          # downloads + QCs the GIANT height GWAS data
│   └── giant_csv_to_npz.py      # converts gen_real_data.R's csv output to .npz
├── model/                       # AMEM-Net architecture + optimizer
│   ├── model.py
│   └── muon_optimizer.py
└── eval/                        # evaluation: comparisons, tests, figures
    ├── functional_ci.py
    ├── posterior_bootstrap.py
    ├── kstest.py
    ├── wasstest.R
    ├── marginal_density_comparison.py
    └── prior_density_comparison.py
```

## Pipeline

> The example commands below all use `n=1000` (dataset size) with no
> grouping, i.e. `--num_group` is set equal to `n` (observation-level
> AB-NPMLE, not the grouped AB-NPMLE (S) variant from the paper). If you
> want to change these settings (different `n`, grouped AB-NPMLE with
> `--num_group` < `n`, etc.), please read the arg descriptions in the code
> (`--help` on each script) rather than assuming these exact values apply.
> 
> This pipeline also mixes in R scripts (`gen_data/gen_real_data.R`,
> `eval/wasstest.R`) alongside the Python ones. Please carefully check the
> paths of the files each script reads and the paths of the files it creates
> after running.

**1. Generate simulation data**

```bash
python gen_data/gen_sim_data.py --iter --iter_start 1 --iter_end 50
# → ./sim_data/data{1..50}/{JZ_A3_s500, WCS1_Normal, WCS3_Tophat}.npz
```

**1b. (Optional) Generate real data — GIANT height GWAS**

```bash
Rscript gen_data/gen_real_data.R
# → giant_height_x.csv  (downloads + QCs the summary stats, computes Z = b/SE)

python gen_data/giant_csv_to_npz.py \
    --csv_path ./giant_height_x.csv --data_idx 1 --data_name GIANT_HEIGHT_n1000
# → ./real_data/GIANT_HEIGHT_n1000.npz  (key: "x"; loadable with --data_mode real)
```

**2. Fit the numerical Bootstrap-NPMLE baseline**

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u run_npmle_sim_iter.py --data_idx 1 --data_name WCS1_Normal \
    --n_bootstrap 1000 --n_workers 40
# → ./sim_data_npmle/data1/result_npmle_WCS1_Normal.npz
```

Single-threaded BLAS is required so that each worker process does not
oversubscribe the CPU; parallelism comes from `--n_workers`, not from
per-process threading.

**3. Train AMEM-Net (AB-NPMLE)**

```bash
python train.py --data_idx 1 --data_name WCS1_Normal --data_mode sim \
    --epochs 15000 --num_group 1000
# → ./sim_data_gms_{num_group}/data1/WCS1_Normal/checkpoint_AMEM_WCS1_Normal.pth
```

If this checkpoint will later be evaluated with `mdv.py` (step 5), add `--mdv`
here: it widens the training sigma range from Uniform(0, 1) to Uniform(0, 2),
matching the sigma grid (up to ~2.5) that `mdv.py` sweeps over. Without it,
`mdv.py` is extrapolating past what the model saw during training.

**4. Run bootstrap inference with the trained model**

```bash
python infer.py --data_idx 1 --data_name WCS1_Normal --npmle_path ./sim_data_npmle/data1/result_npmle_WCS1_Normal.npz \
    --data_mode sim --sigma_true 1.0 --n_bootstrap 1000
# → result_AB_npmle_WCS1_Normal.npz  (reuses the NPMLE baseline's bootstrap
#   weights automatically when num_group == n, for a paired comparison)
```

For real data, the true noise scale is unknown, so `--sigma_true` cannot simply
be given as ground truth: run `mdv.py` first to estimate it (the MDV), then
pass that estimate to `infer.py` as `--sigma_true`.

**5. Compare / evaluate**

```bash
# Marginal density overlay
# NB: --save_dir is set to "results_marginal_v2" (not the script's own default
# "results_marginal") because eval/kstest.py hardcodes
# RESULTS_DIR = "./results_marginal_v2" -- keep these two in sync.
python eval/marginal_density_comparison.py \
    --sim_data_dir ./sim_data --npmle_dir ./sim_data_npmle \
    --save_dir ./results_marginal --n_data 50 \
    --data_file WCS1_Normal.npz --npmle_file result_npmle_WCS1_Normal.npz \
    --ab_result "AB-NPMLE:./sim_data_gms_1000/data{i}/WCS1_Normal/result_AB_npmle_WCS1_Normal.npz" \
    --tag WCS1_Normal --sigma_eps 1.0

# KDE overlay of the estimated mixing distribution
python eval/prior_density_comparison.py \
    --npmle_path ./sim_data_npmle/data1/result_npmle_WCS1_Normal.npz \
    --ab_result "AB-NPMLE:./sim_data_gms_1000/data1/WCS1_Normal/result_AB_npmle_WCS1_Normal.npz" \
    --data_name WCS1_Normal --sigma_true 1.0

# MDV comparison across a noise-scale grid
# NB: mdv.py's own --ckpt_path default points at a different, older
# checkpoint naming scheme ("sim_data_gms_1000_supp10/.../checkpoint_film_*.pth")
# that does not match train.py's actual output -- always pass --ckpt_path
# and --data_path explicitly, as below.
python mdv.py --data_idx 1 --data_name WCS1_Normal \
    --ckpt_path ./sim_data_gms_1000/data1/WCS1_Normal/checkpoint_AMEM_WCS1_Normal.pth \
    --data_path ./sim_data/data1/WCS1_Normal.npz \
    --sigma_min 0.1 --sigma_max 2.5 --n_grid 50 \
    --save_dir ./mdv_out/data1/WCS1_Normal --sigma_true 1.0 \
    --prior_type normal --prior_var 4

# Bootstrap-of-the-posterior comparison
python eval/posterior_bootstrap.py \
      --npmle_result ./sim_data_npmle/data1/result_npmle_WCS1_Normal.npz \
      --ab_result    ./sim_data_gms_1000/data1/WCS1_Normal/result_AB_npmle_WCS1_Normal.npz \
      --data_path    ./sim_data/data1/WCS1_Normal.npz \
      --x_values     -0 \
      --sigma        1.0 \
      --summary_plot \
      --save_dir     ./posterior_bootstrap_out

# CI coverage for empirical-Bayes functionals
# NB: posterior_mean/posterior_variance with the default --condition_mode
# quantile requires a large (n=100000) reference dataset to fix the
# conditioning value x_j -- generate it first (single-run mode, not --iter):
python gen_data/gen_sim_data.py --out_dir ./sim_data_100000/data1 --n 100000
# functional_ci.py then computes the quantile itself from that file; you
# don't need to compute it yourself.
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

# Marginal-sample two-sample tests (across all 50 datasets)
python eval/kstest.py       # edit DATA_NAME at the top of the file
Rscript eval/wasstest.R           # edit DATA_NAME at the top of the file

# Sigma sensitivity: checks whether AB-NPMLE tracks NPMLE as sigma changes.
# NB: lives at repo root, not eval/. First fit NPMLE once per sigma
# candidate via run_npmle_sim_iter.py (matching sigma, separate output dir
# -- see --help), then pass those result_npmle_*.npz to --npmle_paths.
python sigma_sensitivity.py --data_name WCS1_Normal \
    --ckpt_path ./sim_data_gms_1000/data1/WCS1_Normal/checkpoint_AMEM_WCS1_Normal.pth \
    --sigma_candidates 0.5 0.8 1.0 \
    --npmle_paths ./sim_data_npmle_std0.5/data1/result_npmle_WCS1_Normal.npz \
                  ./sim_data_npmle_std0.8/data1/result_npmle_WCS1_Normal.npz \
                  ./sim_data_npmle/data1/result_npmle_WCS1_Normal.npz \
    --n_bootstrap 1000 \
    --result_dir ./plots_sigma/WCS1_Normal
```

All commands above assume they are run from the repository root.

## Reproducibility Notes

- Random seeds are set explicitly wherever randomness affects a reported
  result (see `--seed` in `gen_data/gen_sim_data.py`, `set.seed()` in
  `gen_data/gen_real_data.R`, the auto-derived seed in `run_npmle_sim_iter.py`,
  and `torch.manual_seed` calls in `infer.py`).
- `infer.py` reuses the NPMLE baseline's exact bootstrap weight draws
  (when `num_group == n`) so that both methods are evaluated on identical
  bootstrap replicates.
- Computing infrastructure used for the reported experiments:
  **[TODO: fill in — GPU/CPU model, memory, OS, CUDA/PyTorch versions]**.

## Third-Party Code & Licenses

- `model/muon_optimizer.py` is adapted from
  [KellerJordan/Muon](https://github.com/KellerJordan/Muon) (MIT License,
  © 2024 Keller Jordan) and is not a contribution of this paper. See the
  file header for the full license text and citation.
- `run_npmle_sim_iter.py` depends on the
  [`npeb`](https://github.com/jake-soloff/NPEB) package (MIT License) and,
  through it, on a MOSEK academic license for the underlying convex solver.

## License

Source code will be released under a license permitting free use for
research purposes upon publication of the paper.
