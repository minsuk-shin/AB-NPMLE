"""KS tests for NPEB vs AMEM and NPEB vs GB, without permutations.

Use --sample-dir for existing data<ID>.csv files, or give three NPZ path
templates and --n-data to create 1,000 observations per method. GB NPZs may
contain direct theta draws or one generated theta in each bootstrap prior.

Examples (replace SETTING and /path/to/... with your paths):

    # Reuse existing sample CSVs (NPEB/AMEM, NPEB/GB, or all three columns).
    python eval/kstest.py \
      --sample-dir /path/to/samples --n-data 50 \
      --output-dir /path/to/ks_results

    # Create samples from three saved NPZ results, then run marginal KS tests.
    python eval/kstest.py \
      --npeb-template '/path/to/npeb/data{data_id}/result_npmle_SETTING.npz' \
      --amem-template '/path/to/amem/data{data_id}/SETTING/result_AMEM_SETTING.npz' \
      --gb-template '/path/to/gb/bootstrap/data{data_id}/result_GB_SETTING.npz' \
      --n-data 50 --output-dir /path/to/ks_results

    Use NPEB, AMEM, and GB results with matching bootstrap weights.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


N_SAMPLES = 1000


def draw_from_priors(atoms, weights, expected_priors, seed):
    if len(atoms) != expected_priors or len(weights) != expected_priors:
        raise ValueError(f"Expected {expected_priors} priors, found {len(atoms)} atoms and {len(weights)} weights")
    atom_parts, weight_parts = [], []
    for a, w in zip(atoms, weights):
        if a is None or w is None:
            raise ValueError("Missing fitted atoms or probabilities")
        a = np.asarray(a, dtype=float).reshape(-1)
        w = np.asarray(w, dtype=float).reshape(-1)
        if (a.size == 0 or a.shape != w.shape or
                not np.isfinite(a).all() or not np.isfinite(w).all() or
                (w < 0).any() or w.sum() <= 0):
            raise ValueError("Invalid fitted atoms or probabilities")
        atom_parts.append(a)
        weight_parts.append(w / w.sum())
    all_atoms = np.concatenate(atom_parts)
    all_weights = np.concatenate(weight_parts) / len(atom_parts)
    rng = np.random.default_rng(seed)
    theta = rng.choice(all_atoms, size=N_SAMPLES, p=all_weights)
    return theta + rng.normal(0, 1.0, size=N_SAMPLES)


def generate_samples(data_id, npeb_path, amem_path, gb_path):
    with (np.load(npeb_path, allow_pickle=True) as npeb,
          np.load(amem_path, allow_pickle=True) as amem,
          np.load(gb_path, allow_pickle=True) as gb):
        w = np.asarray(npeb["bootstrap_weights"])
        if (w.ndim != 2 or not all(w.shape) or
                not np.array_equal(w, amem["bootstrap_weights"])):
            raise ValueError(f"data{data_id}: NPEB/AMEM bootstrap weights differ")
        samples = {
            "NPEB": draw_from_priors(npeb["atoms"], npeb["weights"], w.shape[0],
                                     42000 + data_id),
            "AMEM": draw_from_priors(amem["atoms"], amem["weights"], w.shape[0],
                                     43000 + data_id),
        }
        gb_w = np.asarray(gb["bootstrap_weights"])
        if "theta" in gb.files:
            paired_bootstrap = gb_w.shape == w.shape and np.array_equal(gb_w, w)
            fixed_ones = (w.shape[0] == 1 and np.all(w == 1) and
                          gb_w.shape == (N_SAMPLES, w.shape[1]) and
                          np.all(gb_w == 1))
            if not (paired_bootstrap or fixed_ones):
                raise ValueError(f"data{data_id}: NPEB/GB weights do not match")
            theta_gb = np.asarray(gb["theta"], dtype=float).reshape(-1)
        elif "atoms" in gb.files and "weights" in gb.files:
            if not np.array_equal(w, gb_w):
                raise ValueError(f"data{data_id}: NPEB/GB bootstrap weights differ")
            atoms = gb["atoms"]
            probabilities = gb["weights"]
            if len(atoms) != N_SAMPLES or len(probabilities) != N_SAMPLES:
                raise ValueError(f"data{data_id}: expected {N_SAMPLES} GB theta draws")
            if any(np.asarray(a).size != 1 or np.asarray(p).size != 1
                   for a, p in zip(atoms, probabilities)):
                raise ValueError(f"data{data_id}: GB NPZ must store one theta per draw")
            theta_gb = np.asarray([np.asarray(a).item() for a in atoms], dtype=float)
        else:
            raise ValueError(f"data{data_id}: GB NPZ needs theta or atoms/weights")
    if theta_gb.shape != (N_SAMPLES,) or not np.isfinite(theta_gb).all():
        raise ValueError(f"data{data_id}: invalid GB theta draws")
    samples["GB"] = theta_gb + np.random.default_rng(44000 + data_id).normal(
        0, 1.0, size=N_SAMPLES)
    return samples


def run_saved_samples(data_id, path):
    samples = pd.read_csv(path)
    if (len(samples) != N_SAMPLES or "NPEB" not in samples or
            not set(samples.columns).issubset({"NPEB", "AMEM", "GB"}) or
            len(samples.columns) < 2 or
            not np.isfinite(samples.to_numpy(dtype=float)).all()):
        raise ValueError(f"Invalid saved sample CSV: {path}")
    records = []
    for method in ("AMEM", "GB"):
        if method in samples:
            result = ks_2samp(samples["NPEB"], samples[method], method="auto")
            records.append({"data_id": data_id, "comparison": f"NPEB vs {method}",
                            "ks_stat": float(result.statistic),
                            "p_value": float(result.pvalue),
                            "reject_0.05": int(result.pvalue < 0.05),
                            "p_method": "two_sample_ks"})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path,
                        help="Read existing data<ID>.csv samples; no sampling")
    parser.add_argument("--n-data", type=int, default=50,
                        help="Read data1 through dataN (default: 50)")
    parser.add_argument("--npeb-template", type=str,
                        help="NPEB NPZ path with {data_id} placeholder")
    parser.add_argument("--amem-template", type=str,
                        help="AMEM NPZ path with {data_id} placeholder")
    parser.add_argument("--gb-template", type=str,
                        help="GB NPZ path with {data_id} placeholder")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for sampled observations and KS results")
    args = parser.parse_args()

    if args.n_data < 1:
        parser.error("--n-data must be positive")
    templates = (args.npeb_template, args.amem_template, args.gb_template)
    if args.sample_dir:
        if any(template is not None for template in templates):
            parser.error("--sample-dir cannot be combined with NPZ templates")
    elif any(template is None or "{data_id}" not in template for template in templates):
        parser.error("provide all three NPZ templates containing {data_id}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    if args.sample_dir:
        for data_id in range(1, args.n_data + 1):
            path = args.sample_dir / f"data{data_id}.csv"
            if not path.is_file():
                parser.error(f"missing sample CSV: {path}")
            records.extend(run_saved_samples(data_id, path))
    else:
        samples_dir = args.output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        for data_id in range(1, args.n_data + 1):
            paths = [Path(template.replace("{data_id}", str(data_id)))
                     for template in templates]
            for path in paths:
                if not path.is_file():
                    parser.error(f"missing data{data_id} NPZ: {path}")
            samples = generate_samples(data_id, *paths)
            np.savez_compressed(samples_dir / f"data{data_id}.npz", **samples)
            csv_path = samples_dir / f"data{data_id}.csv"
            pd.DataFrame(samples).to_csv(csv_path, index=False)
            records.extend(run_saved_samples(data_id, csv_path))
    each = pd.DataFrame(records)
    each.to_csv(args.output_dir / "ks_each.csv", index=False)
    summary = each.groupby("comparison", sort=False).agg(
        n_tested=("data_id", "size"), mean_ks=("ks_stat", "mean"),
        median_ks=("ks_stat", "median"), max_ks=("ks_stat", "max"),
        median_p=("p_value", "median"),
        n_reject_0_05=("reject_0.05", "sum"),
    ).reset_index()
    summary.to_csv(args.output_dir / "ks_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
