import os
import time
import argparse
import numpy as np
import torch

from model.model import AMEM_NET

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon GPU acceleration
else:
    device = torch.device("cpu")
print(f"Device: {device}")


def generate_bootstrap_weights(batch_size, m):
    probs = torch.ones(m) / m
    boots_weight = torch.distributions.Multinomial(total_count=m, probs=probs).sample((batch_size,))
    return boots_weight.unsqueeze(1)  # (B, 1, m)


def extract_npmle_bootstrap_w(npmle_data, n_bootstrap, m, n):
    """
    Reuses the bootstrap weights actually used by the NPMLE side
    (result_npmle_*.npz) for AMEM inference, so both methods are
    evaluated on the identical bootstrap replicates. Only meaningful
    when m (num_group) == n (number of data points); otherwise returns
    None and the caller generates independent weights instead.

    Returns
    -------
    torch.FloatTensor of shape (B, 1, m), or None if unavailable.
    """
    if m != n:
        print(f"  [w reuse skipped] num_group(m={m}) != n({n}) -> "
              f"generating independent bootstrap weights for AMEM.")
        return None

    candidate_keys = ["bootstrap_weights", "w_counts", "w", "boot_w", "weights_counts"]
    found_key = next((k for k in candidate_keys if k in npmle_data.files), None)
    if found_key is None:
        print(f"  [warning] m==n({n}) but no bootstrap-weight array found in --npmle_path "
              f"(checked keys: {candidate_keys}, actual keys: {list(npmle_data.files)}). "
              f"Generating independent bootstrap weights instead.")
        return None

    w_np = np.asarray(npmle_data[found_key])
    if w_np.ndim == 1:
        w_np = w_np[None, :]
    if w_np.shape[-1] != m:
        print(f"  [warning] last dim of '{found_key}' ({w_np.shape[-1]}) != m({m}); "
              f"generating independent bootstrap weights instead.")
        return None

    n_avail = w_np.shape[0]
    if n_avail < n_bootstrap:
        print(f"  [warning] only {n_avail} bootstrap weights available (< n_bootstrap={n_bootstrap}).")
    elif n_avail > n_bootstrap:
        print(f"  [info] using the first {n_bootstrap} of {n_avail} available bootstrap weights.")

    w_np = w_np[:n_bootstrap]
    w_t = torch.as_tensor(w_np, dtype=torch.float32).unsqueeze(1)  # (B,1,m)
    print(f"  Reusing NPMLE bootstrap weights ('{found_key}', shape={tuple(w_t.shape)}) for AMEM inference.")
    return w_t


def infer_bootstrap_npmle(model, sigma_val, n, n_bootstrap, m, x_mean, x_std, device, external_w=None):
    """
    external_w: torch.FloatTensor of shape (n_bootstrap, 1, m), or None.
        If given, reuse NPMLE's actual bootstrap weights (from
        extract_npmle_bootstrap_w) instead of generating new ones.
    """
    model.eval()

    if external_w is not None:
        all_w = external_w.to(device)
        if all_w.shape[0] != n_bootstrap:
            all_w = all_w[:n_bootstrap]
    else:
        torch.manual_seed(42)
        all_w = generate_bootstrap_weights(n_bootstrap, m).to(device)

    sigma = torch.tensor([[sigma_val]], dtype=torch.float32, device=device)
    sigma = sigma.expand(n_bootstrap, 1)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    t0 = time.time()

    with torch.no_grad():
        prob, supp = model(sigma, all_w)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    inference_time = time.time() - t0

    sup_np = (supp * x_std + x_mean).detach().cpu().numpy().squeeze(1)
    prob_np = prob[:, 0, :].cpu().numpy()

    all_atoms, all_weights = [], []
    for i in range(n_bootstrap):
        p = prob_np[i]
        mask = p > 0.001
        if mask.sum() > 0:
            all_atoms.append(sup_np[i][mask])
            all_weights.append(p[mask])
    return all_atoms, all_weights, inference_time


def resolve_hp(cli_val, saved_hp, key, default, name):
    """Checkpoint-saved hyperparams take priority (guarantees identical architecture);
    falls back to the CLI value, then to a hardcoded default, for older checkpoints
    saved without a 'hyperparams' field."""
    if key in saved_hp:
        if cli_val is not None and cli_val != saved_hp[key]:
            print(f"  [warning] --{name}={cli_val} given but checkpoint was trained with "
                  f"{key}={saved_hp[key]}; using the checkpoint's value.")
        return saved_hp[key]
    if cli_val is not None:
        return cli_val
    print(f"  [warning] checkpoint has no saved '{key}'; using default {name}={default}. "
          f"If this doesn't match training, pass --{name} explicitly.")
    return default


def main():
    parser = argparse.ArgumentParser(
        description="Run bootstrap inference for a trained AMEM checkpoint (does no training). "
                     "Model hyperparameters (hidden_dim, num_layers, n_support, num_group) are "
                     "read from the checkpoint automatically. Optionally reuses NPMLE's bootstrap "
                     "weights for a paired comparison, and saves result_AB_npmle_{data_name}.npz."
    )
    parser.add_argument('--data_idx', type=int, required=True)
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--sigma_true', type=float, required=True,
                         help='True noise std (sigma_eps) of the dataset. Only needed for inference '
                              '(standardizes sigma before feeding it to the model).')
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    parser.add_argument('--npmle_path', type=str, default=None,
                         help='Path to result_npmle_{data_name}.npz, used to reuse NPMLE\'s bootstrap '
                              'weights (when num_group == n) for a paired comparison. '
                              'Defaults to {data_dir}/result_npmle_{data_name}.npz.')
    parser.add_argument('--data_mode', type=str, default='real', choices=['real', 'sim'],
                         help="Must match the --data_mode used for train.py.")
    parser.add_argument('--data_dir', type=str, default=None,
                         help='Overrides the data directory directly. If omitted, derived from --data_mode.')
    parser.add_argument('--run_dir', type=str, default=None,
                         help='Overrides the checkpoint/output directory directly. '
                              'If omitted, derived from --data_mode. Must match train.py\'s run_dir.')
    parser.add_argument('--num_group', type=int, default=1000,
                         help='Only used to derive the default sim-mode run_dir path '
                              '(./sim_data_gms_{num_group}/...) -- the model itself uses the '
                              'num_group value saved in the checkpoint.')
    parser.add_argument('--hidden_dim', type=int, default=None,
                         help='Fallback if the checkpoint has no saved hyperparams (older checkpoint).')
    parser.add_argument('--num_layers', type=int, default=None,
                         help='Fallback if the checkpoint has no saved hyperparams (older checkpoint).')
    parser.add_argument('--n_support', type=int, default=None,
                         help='Fallback if the checkpoint has no saved hyperparams (older checkpoint).')
    args = parser.parse_args()

    i = args.data_idx
    data_name = args.data_name

    if args.data_mode == 'sim':
        default_data_dir = f"./sim_data/data{i}"
        default_run_dir = f"./sim_data_gms_{args.num_group}/data{i}/{data_name}"
    else:
        default_data_dir = "./real_data"
        default_run_dir = f"./real_data_gms/{data_name}"

    data_dir = args.data_dir or default_data_dir
    run_dir = args.run_dir or default_run_dir

    data_path = os.path.join(data_dir, data_name + ".npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    ckpt_path = os.path.join(run_dir, f"checkpoint_AMEM_{data_name}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path} (run train.py first)")

    if args.npmle_path is None:
        args.npmle_path = os.path.join(data_dir, f"result_npmle_{data_name}.npz")

    data = np.load(data_path)
    x_np = data["x"]
    n = len(x_np)
    sigma_true = args.sigma_true

    print(f"\n{'='*60}")
    print(f"Data index: {i}")
    print(f"Dataset   : {data_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"n={n}, sigma_eps={sigma_true:.4f}")
    print(f"{'='*60}\n")

    x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
    x_mean = x_tensor.mean()
    x_std_val = x_tensor.std()
    x_std = (x_tensor - x_mean) / x_std_val

    sigma_infer = sigma_true / x_std_val.item()
    print(f"sigma_infer (standardized): {sigma_infer:.4f}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    saved_hp = ckpt.get('hyperparams') or {}

    hidden_dim = resolve_hp(args.hidden_dim, saved_hp, 'hidden_dim', 256, 'hidden_dim')
    num_layers = resolve_hp(args.num_layers, saved_hp, 'num_layers', 5, 'num_layers')
    N = resolve_hp(args.n_support, saved_hp, 'n_support', 100, 'n_support')
    m = resolve_hp(None, saved_hp, 'num_group', args.num_group, 'num_group')

    x_min, x_max = x_std.min().item(), x_std.max().item()
    pad = (x_max - x_min) * 0.1
    support = torch.linspace(x_min - pad, x_max + pad, N, device=device).reshape(1, 1, -1)

    model = AMEM_NET(
        hidden_dim=hidden_dim, output_dim=N, num_layers=num_layers,
        support=support.squeeze(0), n=n, m=m
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    loss = ckpt.get('loss_for_graph', [])
    cum_times = ckpt.get('cum_times', [])
    train_time = float(ckpt.get('total_elapsed', 0.0))
    epochs = int(ckpt.get('epoch', 0))

    npmle_data = None
    if args.npmle_path and os.path.exists(args.npmle_path):
        print(f"Loading NPMLE results from: {args.npmle_path}")
        npmle_data = np.load(args.npmle_path, allow_pickle=True)
    else:
        print("[warning] --npmle_path not given or file missing -> "
              "bootstrap weights will be generated independently")

    npmle_w = None
    if npmle_data is not None:
        npmle_w = extract_npmle_bootstrap_w(npmle_data, args.n_bootstrap, m, n)

    print("\n" + "=" * 60)
    print(f"Inference: {args.n_bootstrap} bootstrap samples")
    print("=" * 60)

    atoms_ab, weights_ab, inference_time = infer_bootstrap_npmle(
        model, sigma_infer, n, args.n_bootstrap, m,
        x_mean.cpu().item(), x_std_val.cpu().item(), device,
        external_w=npmle_w
    )
    print(f"AMEM: collected {len(atoms_ab)} valid bootstrap distributions")
    print(f"  Train time (from checkpoint) : {train_time:.2f}s ({train_time/60:.2f} min)")
    print(f"  Infer time : {inference_time:.4f}s for {args.n_bootstrap} bootstraps")

    cum_times_arr = np.array(cum_times, dtype=np.float64)

    np.savez(
        os.path.join(run_dir, f"result_AB_npmle_{data_name}.npz"),
        atoms=np.array(atoms_ab, dtype=object),
        weights=np.array(weights_ab, dtype=object),
        cum_times=cum_times_arr,
        train_time=np.float64(train_time),
        inference_time=np.float64(inference_time),
        epochs=np.int64(epochs),
        n_bootstrap=np.int64(args.n_bootstrap),
        loss=loss,
    )
    print(f"\nSaved: {run_dir}/result_AB_npmle_{data_name}.npz")
    print(f"\nTo generate plots, run:")
    print(f"  python plot.py --data_idx {i} --data_name {data_name} --sigma_true {sigma_true}")
    print("\nDone!")


if __name__ == "__main__":
    main()
