"""Run standalone CUDA EM on original data or bootstrap samples.

Examples:
  CUDA_VISIBLE_DEVICES=0 python run_cuda_em.py sim_data/data1/WCS1_Normal.npz \
      --output-dir results/cuda_em_original
  CUDA_VISIBLE_DEVICES=0 python run_cuda_em.py sim_data/data1/WCS1_Normal.npz \
      --output-dir results/cuda_em_bootstrap --n-bootstrap 4 --seed 1001
"""

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from npeb_torch import mvn_pdf, normalize, score


@dataclass
class GPUFit:
    atoms: np.ndarray
    weights: np.ndarray
    loglik_history: np.ndarray
    atom_change_history: np.ndarray
    weight_change_history: np.ndarray
    atom_relative_change_history: np.ndarray
    weight_relative_change_history: np.ndarray
    converged: bool


def fit_without_mosek(x, sigma, tolerance, max_iterations,
                      stop_criterion="parameters", initial_atoms=None,
                      initial_weights=None, dtype=torch.float64):
    """Fit CUDA EM using a parameter-change or log-likelihood stopping rule."""
    if stop_criterion not in ("parameters", "loglik"):
        raise ValueError("stop_criterion must be 'parameters' or 'loglik'")
    X = torch.as_tensor(x, dtype=dtype, device="cuda").reshape(-1, 1)
    if X.numel() == 0 or not torch.isfinite(X).all():
        raise ValueError("x must contain finite observations")
    if (initial_atoms is None) != (initial_weights is None):
        raise ValueError("initial_atoms and initial_weights must be supplied together")
    if initial_atoms is None:
        atoms = X.clone()
        weights = torch.full((X.shape[0],), 1.0 / X.shape[0],
                             dtype=dtype, device="cuda")
    else:
        atoms = torch.as_tensor(initial_atoms, dtype=dtype,
                                device="cuda").reshape(-1, 1)
        weights = torch.as_tensor(initial_weights, dtype=dtype,
                                  device="cuda").reshape(-1)
        if (atoms.shape[0] != weights.shape[0] or atoms.shape[0] != X.shape[0]
                or not torch.isfinite(atoms).all() or not torch.isfinite(weights).all()
                or torch.any(weights < 0)):
            raise ValueError("invalid initial CUDA EM parameters")
    prec = torch.as_tensor([1.0 / sigma**2], dtype=dtype, device="cuda")

    loglik_history = [float(score(X, atoms, weights, prec).item())]
    atom_change_history = []
    weight_change_history = []
    atom_relative_change_history = []
    weight_relative_change_history = []
    converged = False
    for _ in range(max_iterations):
        B = mvn_pdf(X, atoms, prec, 'diagonal', True,
                    n_chunks=1, log_prob_thresh=-float('Inf'), row_condition=False)
        resp = B.multiply(weights)
        resp = normalize(resp, norm='l1', axis=1, copy=False)
        Nk = resp.sum(0).T
        new_weights = (Nk / Nk.sum()).reshape(-1)
        new_atoms = resp.transpose().dot(X) / Nk
        atom_change = float(torch.max(torch.abs(new_atoms - atoms)).item())
        weight_change = float(torch.max(torch.abs(new_weights - weights)).item())
        atom_relative_change = float(torch.max(
            torch.abs(new_atoms - atoms) / torch.maximum(
                torch.ones_like(atoms), torch.abs(atoms))).item())
        weight_relative_change = float(torch.max(
            torch.abs(new_weights - weights) / torch.maximum(
                weights, torch.full_like(weights, 1.0 / X.shape[0]))).item())
        if not all(np.isfinite(value) for value in (
                atom_change, weight_change,
                atom_relative_change, weight_relative_change)):
            raise RuntimeError("CUDA EM produced non-finite parameters")
        atoms, weights = new_atoms, new_weights
        current = float(score(X, atoms, weights, prec).item())
        if not np.isfinite(current):
            raise RuntimeError("CUDA EM produced a non-finite log-likelihood")
        loglik_history.append(current)
        atom_change_history.append(atom_change)
        weight_change_history.append(weight_change)
        atom_relative_change_history.append(atom_relative_change)
        weight_relative_change_history.append(weight_relative_change)
        if ((stop_criterion == "parameters"
             and atom_relative_change < tolerance
             and weight_relative_change < tolerance)
            or (stop_criterion == "loglik"
                and abs(loglik_history[-1] - loglik_history[-2]) < tolerance)):
            converged = True
            break

    # NumPy has no native bfloat16 dtype; store bfloat16 fit outputs as float32.
    output_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
    return GPUFit(atoms=atoms.to(output_dtype).cpu().numpy().reshape(-1),
                  weights=weights.to(output_dtype).cpu().numpy(),
                  loglik_history=np.asarray(loglik_history, dtype=np.float64),
                  atom_change_history=np.asarray(atom_change_history, dtype=np.float64),
                  weight_change_history=np.asarray(weight_change_history, dtype=np.float64),
                  atom_relative_change_history=np.asarray(
                      atom_relative_change_history, dtype=np.float64),
                  weight_relative_change_history=np.asarray(
                      weight_relative_change_history, dtype=np.float64),
                  converged=converged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path,
                        help="input .npz file containing x (and optionally sigma)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=0,
                        help="0 fits the original data once; positive values fit that many bootstrap samples")
    parser.add_argument("--bootstrap-weights", type=Path,
                        help="NPZ with counts or bootstrap_weights to reuse saved bootstrap samples")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--dtype", choices=("float64", "float32", "bfloat16"), default="float64",
                        help="CUDA floating-point dtype (default: float64)")
    parser.add_argument("--sigma", type=float,
                        help="override sigma in the input file (default: input sigma or 1)")
    parser.add_argument("--stop-criterion", choices=("loglik", "parameters"),
                        default="loglik")
    parser.add_argument("--tolerance", type=float,
                        help="default: 1e-8 for loglik, 1e-6 for parameters")
    parser.add_argument("--max-iterations", type=int,
                        help="default: 5000 for loglik, 120000 for parameters")
    args = parser.parse_args()
    if args.n_bootstrap < 0:
        parser.error("--n-bootstrap must be nonnegative")
    if args.bootstrap_weights and args.n_bootstrap < 1:
        parser.error("--bootstrap-weights requires --n-bootstrap greater than zero")
    if args.tolerance is None:
        args.tolerance = 1e-8 if args.stop_criterion == "loglik" else 1e-6
    if args.max_iterations is None:
        args.max_iterations = 5000 if args.stop_criterion == "loglik" else 120000
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be finite and positive")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    if not args.data_path.is_file():
        parser.error(f"input file does not exist: {args.data_path}")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    with np.load(args.data_path) as data:
        x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
        sigma = (args.sigma if args.sigma is not None else
                 float(np.asarray(data["sigma"]).reshape(-1)[0])
                 if "sigma" in data else 1.0)
    if len(x) == 0 or not np.isfinite(x).all():
        parser.error("x must contain finite observations")
    if not np.isfinite(sigma) or sigma <= 0:
        parser.error("sigma must be finite and positive")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch_dtype = {"float64": torch.float64, "float32": torch.float32,
                   "bfloat16": torch.bfloat16}[args.dtype]
    n = len(x)
    x_gpu = torch.as_tensor(x, dtype=torch_dtype, device="cuda")
    if args.bootstrap_weights:
        torch.manual_seed(args.seed)
        weight_start = time.perf_counter()
        timed_counts_gpu = torch.distributions.Multinomial(
            total_count=n, probs=torch.ones(n, device="cuda") / n
        ).sample((args.n_bootstrap,)).long()
        torch.cuda.synchronize()
        weight_generation_seconds = time.perf_counter() - weight_start
        del timed_counts_gpu
        if not args.bootstrap_weights.is_file():
            parser.error(f"bootstrap weights file does not exist: {args.bootstrap_weights}")
        with np.load(args.bootstrap_weights, allow_pickle=False) as source:
            key = "counts" if "counts" in source else "bootstrap_weights"
            if key not in source:
                parser.error("bootstrap weights NPZ must contain counts or bootstrap_weights")
            if "x" in source and not np.array_equal(source["x"].reshape(-1), x):
                parser.error("bootstrap weights belong to different observations")
            saved_counts = np.asarray(source[key])
        if saved_counts.ndim != 2 or saved_counts.shape[1] != n or saved_counts.shape[0] < args.n_bootstrap:
            parser.error(f"bootstrap weights must have at least {args.n_bootstrap} rows of length {n}")
        counts = saved_counts[:args.n_bootstrap]
        if (not np.issubdtype(counts.dtype, np.number) or not np.isfinite(counts).all()
                or np.any(counts < 0) or not np.all(counts == np.rint(counts))
                or not np.all(counts.sum(axis=1) == n)):
            parser.error("bootstrap counts must be nonnegative integers summing to n")
        counts = counts.astype(np.int64, copy=True)
        counts_gpu = torch.as_tensor(counts, dtype=torch.long, device="cuda")
        weight_source = f"saved weights from {args.bootstrap_weights.resolve()}; CUDA generation timed separately"
    elif args.n_bootstrap:
        torch.manual_seed(args.seed)
        weight_start = time.perf_counter()
        counts_gpu = torch.distributions.Multinomial(
            total_count=n, probs=torch.ones(n, device="cuda") / n
        ).sample((args.n_bootstrap,)).long()
        torch.cuda.synchronize()
        weight_generation_seconds = time.perf_counter() - weight_start
        counts = counts_gpu.cpu().numpy()
        weight_source = "CUDA multinomial"
    else:
        counts_gpu = torch.ones((1, n), dtype=torch.long, device="cuda")
        counts = counts_gpu.cpu().numpy()
        weight_generation_seconds = 0.0
        weight_source = "all ones"
    cumulative_compute_seconds = weight_generation_seconds

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "settings.json").write_text(json.dumps({
        "data_path": str(args.data_path.resolve()), "n": n, "sigma": sigma,
        "n_bootstrap": args.n_bootstrap, "seed": args.seed if args.n_bootstrap else None,
        "bootstrap_rule": weight_source,
        "stop_criterion": args.stop_criterion, "tolerance": args.tolerance,
        "max_iterations": args.max_iterations,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(), "gpu_count_used": 1,
        "initialization": "atoms=x_boot, weights=uniform",
        "weight_generation_seconds": weight_generation_seconds,
        "timing": "cumulative compute time includes CUDA bootstrap weight generation (generated weights discarded when saved weights are supplied), CUDA sample expansion, and synchronized CUDA EM; excludes result saves and loading saved weights",
    }, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "bootstrap_counts.npz",
                        counts=counts, x=x, sigma=np.float64(sigma))

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "fit", "seconds", "cum_compute_seconds", "iterations", "converged", "atom_count"))
        writer.writeheader()
        for index, weights in enumerate(counts):
            print(f"START fit={index}", flush=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            x_fit = torch.repeat_interleave(x_gpu, counts_gpu[index])
            result = fit_without_mosek(
                x_fit, sigma, args.tolerance, args.max_iterations,
                args.stop_criterion, dtype=torch_dtype)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            cumulative_compute_seconds += seconds
            np.savez_compressed(
                args.output_dir / f"fit_{index:04d}.npz",
                counts=weights, atoms=result.atoms, weights=result.weights,
                loglik_history=result.loglik_history,
                atom_change_history=result.atom_change_history,
                weight_change_history=result.weight_change_history,
                atom_relative_change_history=result.atom_relative_change_history,
                weight_relative_change_history=result.weight_relative_change_history,
                converged=np.bool_(result.converged), seconds=np.float64(seconds),
                cum_compute_seconds=np.float64(cumulative_compute_seconds))
            writer.writerow({
                "fit": index, "seconds": seconds,
                "cum_compute_seconds": cumulative_compute_seconds,
                "iterations": len(result.loglik_history) - 1,
                "converged": result.converged, "atom_count": len(result.atoms)})
            file.flush()
            print(f"DONE fit={index} seconds={seconds:.3f} "
                  f"iterations={len(result.loglik_history) - 1} "
                  f"converged={result.converged}", flush=True)
            del result
            torch.cuda.empty_cache()
    print(f"TOTAL compute time={cumulative_compute_seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()


