import os
import time
import math
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.model import AMEM_NET, EM_based_loss
from paired_boots_weight.bootstrap_weights import draw_paired_weights

DEFAULT_W_ONES_INTERVAL = 0

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon GPU acceleration
else:
    device = torch.device("cpu")
print(f"Device: {device}")


def generate_bootstrap_weights(batch_size, m, device):
    probs = torch.full((m,), 1.0 / m, device=device)
    boots_weight = torch.distributions.Multinomial(total_count=m, probs=probs).sample((batch_size,))
    return boots_weight.unsqueeze(1)  # (B, 1, m)


def train_model(model, X, B, m, n, epochs, device, checkpoint_path, model_name="Model",
                 sigma_scale=1.0, hyperparams=None, schedule_epochs=None,
                 paired_weight_seed=None, w_ones_interval=DEFAULT_W_ONES_INTERVAL):
    from model.muon_optimizer import SingleDeviceMuon
    from torch.optim.lr_scheduler import LambdaLR

    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "support" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)

    opt_muon = SingleDeviceMuon(muon_params, lr=3e-3, momentum=0.95, weight_decay=0.00) if muon_params else None
    opt_adam = torch.optim.AdamW(adam_params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1) if adam_params else None

    schedule_epochs = schedule_epochs or epochs
    warmup_steps = max(1, schedule_epochs // 10)

    def cosine_with_warmup(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, schedule_epochs - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    sch_muon = LambdaLR(opt_muon, lr_lambda=cosine_with_warmup) if opt_muon else None
    sch_adam = LambdaLR(opt_adam, lr_lambda=cosine_with_warmup) if opt_adam else None

    print(f"  [{model_name}] Muon params: {sum(p.numel() for p in muon_params):,}, "
          f"Adam params: {sum(p.numel() for p in adam_params):,}")
    print(f"  [{model_name}] Warmup steps: {warmup_steps}, "
          f"Schedule total: {schedule_epochs}, Run steps: {epochs}")

    best_mean_loss = float("inf")
    loss_list, loss_for_graph = [], []
    start_epoch = 0
    cum_times = []
    total_elapsed_offset = 0.0
    iteration_compute_seconds = 0.0
    checkpoint_data = {}
    t_start = time.time()

    if os.path.exists(checkpoint_path):
        print(f"[{model_name}] Resuming from {checkpoint_path}")
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        if 'muon_optimizer_state_dict' in checkpoint_data and opt_muon:
            opt_muon.load_state_dict(checkpoint_data['muon_optimizer_state_dict'])
        if 'adam_optimizer_state_dict' in checkpoint_data and opt_adam:
            opt_adam.load_state_dict(checkpoint_data['adam_optimizer_state_dict'])
        start_epoch = checkpoint_data['epoch'] + 1
        best_mean_loss = checkpoint_data['best_mean_loss']
        loss_for_graph = checkpoint_data.get('loss_for_graph', [])
        cum_times = checkpoint_data.get('cum_times', [])
        total_elapsed_offset = checkpoint_data.get('total_elapsed', 0.0)
        iteration_compute_seconds = checkpoint_data.get('iteration_compute_seconds', 0.0)
        print(f"  Starting from epoch {start_epoch}, best_loss={best_mean_loss:.7f}")

    for t in range(start_epoch, epochs):
        iteration_start = time.perf_counter()
        model.train()
        sigma = torch.rand((B, 1, 1), device=device) * sigma_scale  # Uniform(0, sigma_scale)

        if paired_weight_seed is None:
            w = generate_bootstrap_weights(B, m, device)
        else:
            w = draw_paired_weights(B, m, device, paired_weight_seed, t,
                                    generate_bootstrap_weights)
        if w_ones_interval and (t + 1) % w_ones_interval == 0:
            w[0].fill_(1.0)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type != "cpu")):
            prob, support = model(sigma, w)
            loss = EM_based_loss(X, support, prob, sigma, w)

        if opt_muon: opt_muon.zero_grad()
        if opt_adam: opt_adam.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if opt_muon: opt_muon.step()
        if opt_adam: opt_adam.step()
        if sch_muon: sch_muon.step()
        if sch_adam: sch_adam.step()

        loss_value = float(loss.item())
        iteration_compute_seconds += time.perf_counter() - iteration_start
        loss_list.append(loss_value)
        if (t + 1) % 100 == 0:
            loss_mean = sum(loss_list) / len(loss_list)
            elapsed = time.time() - t_start + total_elapsed_offset
            elapsed_str = f"{elapsed/60:.1f}min" if elapsed >= 60 else f"{elapsed:.1f}s"
            cum_times.append((t + 1, elapsed))

            if loss_mean < best_mean_loss:
                best_mean_loss = loss_mean
                checkpoint_data = {
                    'epoch': t,
                    'best_epoch': t + 1,
                    'model_state_dict': model.state_dict(),
                    'muon_optimizer_state_dict': opt_muon.state_dict() if opt_muon else {},
                    'adam_optimizer_state_dict': opt_adam.state_dict() if opt_adam else {},
                    'best_mean_loss': best_mean_loss,
                    'loss_for_graph': loss_for_graph,
                    'cum_times': cum_times,
                    'total_elapsed': time.time() - t_start + total_elapsed_offset,
                    'iteration_compute_seconds': iteration_compute_seconds,
                    'hyperparams': hyperparams,
                }
                torch.save(checkpoint_data, checkpoint_path)
                ones_note = "  [w[0]=ones]" if w_ones_interval and (t + 1) % w_ones_interval == 0 else ""
                print(f"  [{model_name}] Epoch {t+1}: NEW BEST = {best_mean_loss:.7f}  [{elapsed_str}]{ones_note}", flush=True)
            else:
                ones_note = "  [w[0]=ones]" if w_ones_interval and (t + 1) % w_ones_interval == 0 else ""
                print(f"  [{model_name}] Epoch {t+1}: loss={loss_mean:.7f}, best={best_mean_loss:.7f}  [{elapsed_str}]{ones_note}", flush=True)

            loss_for_graph.append(best_mean_loss)
            loss_list = []

    checkpoint_data['epoch'] = epochs
    checkpoint_data['checkpoint_kind'] = 'last'
    checkpoint_data['model_state_dict'] = model.state_dict()
    checkpoint_data['muon_optimizer_state_dict'] = opt_muon.state_dict() if opt_muon else {}
    checkpoint_data['adam_optimizer_state_dict'] = opt_adam.state_dict() if opt_adam else {}
    checkpoint_data['best_mean_loss'] = best_mean_loss
    checkpoint_data['loss_for_graph'] = loss_for_graph
    checkpoint_data['cum_times'] = cum_times
    checkpoint_data['total_elapsed'] = time.time() - t_start + total_elapsed_offset
    checkpoint_data['iteration_compute_seconds'] = iteration_compute_seconds
    checkpoint_data['hyperparams'] = hyperparams  # ensure present even if resumed from an older checkpoint
    torch.save(checkpoint_data, checkpoint_path)
    print(f"[{model_name}] Last checkpoint saved at: {checkpoint_path}", flush=True)
    train_total_time = time.time() - t_start + total_elapsed_offset
    return model, loss_for_graph, cum_times, train_total_time, iteration_compute_seconds


def main():
    parser = argparse.ArgumentParser(
        description="Train the AMEM model only. Produces a checkpoint (.pth) that also stores "
                     "the architecture hyperparameters, so infer.py can rebuild the identical "
                     "model automatically. Run infer.py separately for bootstrap inference."
    )
    parser.add_argument('--data_idx', type=int, required=True,
                         help='Dataset index i')
    parser.add_argument('--data_name', type=str, required=True,
                         help='Dataset filename without extension (e.g. WCS1_Normal)')
    parser.add_argument('--epochs', type=int, default=30000)
    parser.add_argument('--schedule_epochs', type=int, default=None,
                        help='Learning-rate schedule horizon; defaults to --epochs.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_group', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--n_support', type=int, default=100)
    parser.add_argument('--data_mode', type=str, default='real', choices=['real', 'sim'],
                         help="Which directory layout to use for the default data_dir/run_dir: "
                              "'real' for ./real_data/... (default, matches earlier real-data runs), "
                              "'sim' for ./sim_data/data{i}/... (the gen_data/gen_sim_data.py pipeline).")
    parser.add_argument('--data_dir', type=str, default=None,
                         help='Overrides the data directory directly. If omitted, derived from --data_mode.')
    parser.add_argument('--run_dir', type=str, default=None,
                         help='Overrides the output/checkpoint directory directly. '
                              'If omitted, derived from --data_mode.')
    parser.add_argument('--mdv', action='store_true',
                         help='Sample training sigma from Uniform(0, 2) instead of Uniform(0, 1) '
                              '(i.e. sigma * 2). Used to prepare a checkpoint for MDV evaluation '
                              '(mdv.py), which sweeps sigma up to ~2.5.')
    parser.add_argument('--seed', type=int, default=None,
                         help='Seed for reproducible model initialization and training samples.')
    parser.add_argument('--paired-weight-seed', type=int, default=None,
                        help='Match GB multinomial weight draws at every training iteration')
    parser.add_argument('--w-ones-every', type=int, default=DEFAULT_W_ONES_INTERVAL,
                        help='Replace the first weight vector with all ones every N iterations; 0 disables this')
    args = parser.parse_args()
    if args.w_ones_every < 0:
        parser.error('--w-ones-every must be nonnegative')

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    i = args.data_idx
    data_name = args.data_name

    if args.data_mode == 'sim':
        default_data_dir = f"./sim_data/data{i}"
        default_run_dir = f"./sim_data_gms_{args.num_group}/data{i}/{data_name}"
    else:  # 'real'
        default_data_dir = "./real_data"
        default_run_dir = f"./real_data_gms/{data_name}"

    data_dir = args.data_dir or default_data_dir
    run_dir = args.run_dir or default_run_dir
    os.makedirs(run_dir, exist_ok=True)

    data_path = os.path.join(data_dir, args.data_name + ".npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = np.load(data_path)
    x_np = data["x"]
    n = len(x_np)

    print(f"\n{'='*60}")
    print(f"Data index: {i}")
    print(f"Dataset   : {args.data_name}")
    print(f"Data path : {data_path}")
    print(f"Result dir: {run_dir}")
    print(f"n={n}")
    print(f"{'='*60}\n")

    x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
    x_mean = x_tensor.mean()
    x_std_val = x_tensor.std()
    x_std = (x_tensor - x_mean) / x_std_val
    x_input = x_std.reshape(1, -1, 1)

    B = args.batch_size
    m = args.num_group
    N = args.n_support

    x_min, x_max = x_std.min().item(), x_std.max().item()
    pad = (x_max - x_min) * 0.1
    support = torch.linspace(x_min - pad, x_max + pad, N, device=device).reshape(1, 1, -1)
    support_for_model = support.clone()

    print("=" * 60)
    print("Training AMEM model")
    print("=" * 60)

    model = AMEM_NET(
        hidden_dim=args.hidden_dim, output_dim=N,
        num_layers=args.num_layers, support=support_for_model.squeeze(0), n=n, m=m
    ).to(device)
    print(f"AMEM model params: {sum(p.numel() for p in model.parameters()):,}")

    with torch.no_grad():
        nn.init.zeros_(model.layer_supp_delta.weight)
        nn.init.zeros_(model.layer_supp_delta.bias)

    hyperparams = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'n_support': N,
        'num_group': m,
        'paired_weight_seed': args.paired_weight_seed,
        'w_ones_interval': args.w_ones_every,
    }

    ckpt = os.path.join(run_dir, f"checkpoint_AMEM_{args.data_name}.pth")
    sigma_scale = 2.0 if args.mdv else 1.0
    model, loss, cum_times, train_time, iteration_compute_seconds = train_model(
        model, x_input, B, m, n, args.epochs, device,
        checkpoint_path=ckpt, model_name="AMEM", sigma_scale=sigma_scale,
        hyperparams=hyperparams,
        schedule_epochs=args.schedule_epochs,
        paired_weight_seed=args.paired_weight_seed,
        w_ones_interval=args.w_ones_every,
    )

    print(f"\nTraining done. Train time: {train_time:.2f}s ({train_time/60:.2f} min)")
    print(f"Iteration compute time (through loss.item): {iteration_compute_seconds:.2f}s "
          f"({iteration_compute_seconds/60:.2f} min)")
    print(f"Checkpoint saved at: {ckpt} (includes hyperparams for infer.py)")
    print(f"\nNext, run inference:")
    mode_flag = " --data_mode sim" if args.data_mode == 'sim' else ""
    print(f"  python infer.py --data_idx {i} --data_name {args.data_name} "
          f"--sigma_true <true_noise_std>{mode_flag}")


if __name__ == "__main__":
    main()
