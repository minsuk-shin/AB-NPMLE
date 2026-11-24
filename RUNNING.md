# Running AMEM-Net

Run these commands from the repository root.

## Setup

```bash
pip install -r requirements.txt
```

Numerical NPMLE and MDV need a working MOSEK license. GB training and CUDA EM need a CUDA-compatible PyTorch build. The R scripts use `data.table`, `R.utils`, and `energy`; `gen_real_data.R` installs the first two.

## Main simulation workflow

These examples use `WCS1_Normal`, `data1`, `n=1000`, `sigma=1`, and 1,000 bootstrap samples. Repeat for other data indices/scenarios as needed. For AMEM, `--num_group 1000` equals `n`, allowing reuse of the numerical NPMLE bootstrap weights. Smaller group sizes generate their own inference weights.

### 1. Generate data

```bash
python gen_data/gen_sim_data.py --iter --iter_start 1 --iter_end 1 --n 1000
```

This writes `sim_data/data1/JZ_A3_s500.npz`, `WCS1_Normal.npz`, and `WCS3_Tophat.npz`. Each contains `x`, `theta`, `eps`, and `sigma`. For data1 through data50, use `--iter_end 50`. The JZ generator fixes `s=500`, so use `n >= 500` when generating all three scenarios.

### 2. Fit numerical bootstrap NPMLE

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python numerical_NPMLE/run_npmle_sim_iter.py \
  --data_idx 1 --data_name WCS1_Normal --data_mode sim \
  --n_bootstrap 1000 --n_workers 20
```

Output: `sim_data_npmle/data1/result_npmle_WCS1_Normal.npz`. It contains fitted `atoms`, `weights`, and the actual `bootstrap_weights`. Single-threaded BLAS prevents each worker from starting a large thread pool. This step needs `npeb` and MOSEK.

### 3. Train and infer with AMEM

```bash
python run/train.py \
  --data_idx 1 --data_name WCS1_Normal --data_mode sim \
  --num_group 1000 --epochs 30000 --w-ones-every 0

python run/infer.py \
  --data_idx 1 --data_name WCS1_Normal --data_mode sim \
  --num_group 1000 --sigma_true 1.0 --n_bootstrap 1000 \
  --npmle_path sim_data_npmle/data1/result_npmle_WCS1_Normal.npz
```

Outputs:

```text
sim_data_gms_1000/data1/WCS1_Normal/checkpoint_AMEM_WCS1_Normal.pth
sim_data_gms_1000/data1/WCS1_Normal/result_AMEM_WCS1_Normal.npz
```

`run/infer.py` reads the model dimensions from the checkpoint. When `num_group == n`, it reuses the NPMLE weights supplied through `--npmle_path`. For real observations, `--sigma_true` is a chosen or estimated noise scale, not known ground truth. Override default locations with `--data_dir`, `--run_dir`, or `--output_path`.

For MDV, add `--mdv` **when training**. It changes the sampled standardized sigma range from `Uniform(0, 1)` to `Uniform(0, 2)`. Train a separate checkpoint for MDV, as shown in the evaluation section below.

### 4. Train and sample GB-NPMLE

GB training requires CUDA and an input NPZ with `x` and one `sigma` value. The sampling CLI accepts a completed 30,000-iteration, sigma-1 checkpoint. The commands below do not insert all-one weights during training.

```bash
python gb_npmle/gbnpmle_function_gpu.py sim_data/data1/WCS1_Normal.npz \
  --output-dir gb_runs/data1/WCS1_Normal \
  --iterations 30000 --one-ones-every 0 --device-index 0

python gb_npmle/gbnpmle_sampling_with_weights.py \
  --pair 1 gb_runs/data1/WCS1_Normal/generator_last.pt \
  --output-dir gb_samples --n-draws 1000 --weight-mode npeb \
  --expected-one-ones-every 0 \
  --npeb-template 'sim_data_npmle/data{data_id}/result_npmle_WCS1_Normal.npz'
```

Training writes `generator_last.pt`, `generator_best.pt`, and `summary.json`. Sampling writes `gb_samples/bootstrap/data1/result_GB_WCS1_Normal.npz`, with `theta` and `bootstrap_weights`. For all-one weights, use `--weight-mode fixed` and omit `--npeb-template`; output then goes under `gb_samples/fixed_w_ones/data1/`. GB training requires a new, nonexistent `--output-dir`.

## Evaluation

The examples below use files from the main workflow. Replace paths and `--n-data` when evaluating more datasets. Scripts create output folders as needed.

### Prior and marginal density plots

```bash
python eval/prior_density_comparison.py \
  --npeb-result sim_data_npmle/data1/result_npmle_WCS1_Normal.npz \
  --amem-result sim_data_gms_1000/data1/WCS1_Normal/result_AMEM_WCS1_Normal.npz \
  --gb-result gb_samples/bootstrap/data1/result_GB_WCS1_Normal.npz \
  --output-stem eval_out/data1/prior

python eval/marginal_density_comparison.py \
  --observed-data sim_data/data1/WCS1_Normal.npz \
  --npeb-result sim_data_npmle/data1/result_npmle_WCS1_Normal.npz \
  --amem-result sim_data_gms_1000/data1/WCS1_Normal/result_AMEM_WCS1_Normal.npz \
  --gb-result gb_samples/bootstrap/data1/result_GB_WCS1_Normal.npz \
  --output-stem eval_out/data1/marginal
```

Both write PNG, PDF, and summary JSON files. The marginal script also writes an NPZ of computed densities. Omit `--gb-result` for an NPMLE/AMEM-only plot.

### Group-size comparisons

Train and infer separately for group sizes 5, 50, and 100:

```bash
for group in 5 50 100; do
  python run/train.py --data_idx 1 --data_name WCS1_Normal \
    --data_mode sim --num_group "$group" --epochs 30000 --w-ones-every 0
  python run/infer.py --data_idx 1 --data_name WCS1_Normal \
    --data_mode sim --num_group "$group" --sigma_true 1.0 --n_bootstrap 1000
done
```

Then run both plots:

```bash
python eval/group_prior_density_comparison.py \
  --npeb-result sim_data_npmle/data1/result_npmle_WCS1_Normal.npz \
  --group-5-result sim_data_gms_5/data1/WCS1_Normal/result_AMEM_WCS1_Normal.npz \
  --group-50-result sim_data_gms_50/data1/WCS1_Normal/result_AMEM_WCS1_Normal.npz \
  --group-100-result sim_data_gms_100/data1/WCS1_Normal/result_AMEM_WCS1_Normal.npz \
  --output-dir eval_out/group_prior

python eval/group_marginal_density_comparison.py \
  --observed-data sim_data/data1/WCS1_Normal.npz \
  --npeb-result sim_data_npmle/data1/result_npmle_WCS1_Normal.npz \
  --group-5-result sim_data_gms_5/data1/WCS1_Normal/result_AMEM_WCS1_Normal.npz \
  --group-50-result sim_data_gms_50/data1/WCS1_Normal/result_AMEM_WCS1_Normal.npz \
  --group-100-result sim_data_gms_100/data1/WCS1_Normal/result_AMEM_WCS1_Normal.npz \
  --output-dir eval_out/group_marginal
```

Each saves PNG/PDF plots; the marginal script also saves plotted values. For a combined three-panel figure, supply `--panel` three times as described in each script's `--help`.

### Noise-scale sensitivity and MDV

Fit NPMLE separately at each candidate sigma. Use different `--save_dir` values to avoid overwriting results:

```bash
for sigma in 0.5 0.8 1.0; do
  python numerical_NPMLE/run_npmle_sim_iter.py \
    --data_idx 1 --data_name WCS1_Normal --data_mode sim \
    --sigma "$sigma" --n_bootstrap 1000 --n_workers 20 \
    --save_dir "sim_data_npmle_sigma_${sigma}/data1"
done

python -m eval.sigma_sensitivity \
  --data_path sim_data/data1/WCS1_Normal.npz \
  --ckpt_path sim_data_gms_1000/data1/WCS1_Normal/checkpoint_AMEM_WCS1_Normal.pth \
  --sigma_candidates 0.5 0.8 1.0 \
  --npmle_paths sim_data_npmle_sigma_0.5/data1/result_npmle_WCS1_Normal.npz \
                sim_data_npmle_sigma_0.8/data1/result_npmle_WCS1_Normal.npz \
                sim_data_npmle_sigma_1.0/data1/result_npmle_WCS1_Normal.npz \
  --n_bootstrap 1000 --result_dir eval_out/sigma
```

`sigma_sensitivity.py` needs the `python -m` form from the repository root. `--npmle_paths` must match `--sigma_candidates` in order. Omit `--npmle_paths` for AMEM-only plots. It writes one result NPZ and KDE PDF per sigma.

Train the MDV checkpoint for 30,000 iterations with `--mdv` so the model samples standardized sigma over `0` to `2`. Supply its actual model dimensions if different from the defaults (`hidden_dim=256`, `num_layers=5`, `n_support=100`):

```bash
python run/train.py \
  --data_idx 1 --data_name WCS1_Normal --data_mode sim \
  --num_group 1000 --epochs 30000 --w-ones-every 0 --mdv \
  --run_dir sim_data_gms_1000_mdv/data1/WCS1_Normal

python eval/mdv.py \
  --data_path sim_data/data1/WCS1_Normal.npz \
  --ckpt_path sim_data_gms_1000_mdv/data1/WCS1_Normal/checkpoint_AMEM_WCS1_Normal.pth \
  --save_dir eval_out/mdv/data1 \
  --sigma_min 0.1 --sigma_max 2.5 --n_grid 50 \
  --sigma_true 1.0 --prior_type normal --prior_var 4
```

MDV writes a result NPZ and comparison plots. Its numerical fits require `npeb`/MOSEK.

### Functional confidence intervals and two-sample tests

`{data_id}` in each template is replaced with `1` through `--n-data`; all corresponding files must exist. `obs_idx` selects one observation per dataset for posterior functionals. A fixed reference quantile can instead be used with `--condition-mode quantile --quantile-reference-data PATH`.

```bash
python eval/functional_ci.py \
  --observed-template 'sim_data/data{data_id}/WCS1_Normal.npz' \
  --npeb-template 'sim_data_npmle/data{data_id}/result_npmle_WCS1_Normal.npz' \
  --amem-template 'sim_data_gms_1000/data{data_id}/WCS1_Normal/result_AMEM_WCS1_Normal.npz' \
  --n-data 50 --prior-type WCS1_Normal --functional all \
  --condition-mode obs_idx --obs-idx 0 \
  --output-dir eval_out/functional_ci

python eval/kstest.py \
  --npeb-template 'sim_data_npmle/data{data_id}/result_npmle_WCS1_Normal.npz' \
  --amem-template 'sim_data_gms_1000/data{data_id}/WCS1_Normal/result_AMEM_WCS1_Normal.npz' \
  --gb-template 'gb_samples/bootstrap/data{data_id}/result_GB_WCS1_Normal.npz' \
  --n-data 50 --output-dir eval_out/ks

Rscript eval/energy_test.R --sample-dir eval_out/ks/samples \
  --output-dir eval_out/energy --n-perm 1000

Rscript eval/wasstest.R --sample-dir eval_out/ks/samples \
  --output-dir eval_out/wasserstein --n-perm 1000
```

Functional CI writes coverage NPZ and detail/summary CSVs. KS writes `ks_each.csv`, `ks_summary.csv`, and `samples/data{data_id}.csv`. The R scripts read those 1,000-row sample CSVs and write their own CSVs. For paired comparisons, NPMLE and AMEM must have identical saved bootstrap weights; GB must use the same weights.

### Fixed all-one weight comparison

`eval/compare_fixed_w_ab_gb_sim.py` expects three prepared results with all-one weights. A standard bootstrap result does not meet this condition. Pass `--bootstrap_weights ones` to NPMLE to fit the original data once, with every observation's weight set to `1`. The saved NPMLE result contains these weights, which AMEM reuses during inference. GB samples in fixed mode:

```bash
python numerical_NPMLE/run_npmle_sim_iter.py \
  --data_idx 1 --data_name WCS1_Normal --data_mode sim \
  --n_bootstrap 1 --n_workers 1 --bootstrap_weights ones \
  --save_dir fixed_results/npeb/data1

python run/infer.py \
  --data_idx 1 --data_name WCS1_Normal --data_mode sim --num_group 1000 \
  --sigma_true 1.0 --n_bootstrap 1 \
  --npmle_path fixed_results/npeb/data1/result_npmle_WCS1_Normal.npz \
  --output_path fixed_results/amem/data1/result_AMEM_WCS1_Normal.npz

python gb_npmle/gbnpmle_sampling_with_weights.py \
  --pair 1 gb_runs/data1/WCS1_Normal/generator_last.pt \
  --output-dir gb_samples --n-draws 1000 --weight-mode fixed \
  --expected-one-ones-every 0

python eval/compare_fixed_w_ab_gb_sim.py \
  --npeb-result fixed_results/npeb/data1/result_npmle_WCS1_Normal.npz \
  --amem-result fixed_results/amem/data1/result_AMEM_WCS1_Normal.npz \
  --gb-result gb_samples/fixed_w_ones/data1/result_GB_WCS1_Normal.npz \
  --data-name WCS1_Normal --output-stem eval_out/fixed_w_prior
```

The comparison writes PNG, PDF, and summary JSON files.

## Other input and fitting options

For GIANT height observations, run `Rscript gen_data/gen_real_data.R` from the repository root. It downloads a large external file, writes `giant_height_x.csv`, and may install R packages. Then run:

```bash
python gen_data/giant_csv_to_npz.py \
  --csv_path giant_height_x.csv --data_idx 1 --data_name GIANT_HEIGHT_n10000
```

This writes `real_data/GIANT_HEIGHT_n10000.npz` with 10,000 observations in `x`. AMEM and numerical NPMLE read it with `--data_mode real`. GB training additionally needs a `sigma` field. Use an estimated or chosen noise scale for real-data inference.

The separate CUDA EM implementation is run with:

```bash
python numerical_NPMLE/run_cuda_em.py sim_data/data1/WCS1_Normal.npz \
  --output-dir cuda_em_out/data1 --n-bootstrap 1000
```

It writes `settings.json`, `bootstrap_counts.npz`, `summary.csv`, and individual `fit_0000.npz` files. These are not the aggregate `result_npmle_*.npz` expected by the evaluation commands above.

## Reproducibility

The simulation generator takes `--seed`; NPMLE, AMEM training, and GB training expose seed options. `run/infer.py` saves the weights it used. `paired_boots_weight/bootstrap_weights.py` is shared by AMEM and GB training when paired weight seeds are requested.
