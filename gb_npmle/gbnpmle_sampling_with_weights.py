# Adapted from https://github.com/shijiew97/GBnpmle/blob/master/inst/GB_NPMLE_Sampling.py
# Upstream package: GBnpmle 0.1.0, GPL-3. Changes: optional supplied W,
# explicit device, and optional diagnostics for the NPZ runner.
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import argparse
import json
import os
from pathlib import Path

def GB_NPMLE_Sampling(fit, Boot_size, W=None, Device=None, Return_details=False,
                     Component_mode='multinomial'):
  
    device = torch.device(Device) if Device is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    
    tau = fit[1]
    p = fit[2]
    k = fit[3] #k=l
    dist = fit[4]
    s = fit[5]
    n = fit[6]
    zm = fit[7] #zm=q
    NN = fit[0].to(device)
    NN.eval()
    
    size = int(Boot_size) 
    if size < 1:
      raise ValueError('Boot_size must be positive')
    if W is not None:
      W = torch.as_tensor(W, device=device, dtype=torch.float32)
      if (W.shape != (size, s) or not bool(torch.isfinite(W).all()) or
          bool((W < 0).any())):
        raise ValueError(f'W must be finite nonnegative weights of shape {(size, s)}')
    
    if device.type == 'cuda':
      torch.cuda.synchronize(device)
    Generation_start = time.perf_counter()
    with torch.no_grad(): 
      if Component_mode == 'multinomial':
        tau_prob = torch.as_tensor(tau, device=device, dtype=torch.float32).reshape(-1)
        if (tau_prob.numel() != k or not bool(torch.isfinite(tau_prob).all()) or
            bool((tau_prob < 0).any()) or float(tau_prob.sum()) <= 0):
          raise ValueError('tau must contain k finite nonnegative weights')
        tau_prob = tau_prob / tau_prob.sum()
        sample = torch.distributions.Multinomial(
            total_count=1, probs=tau_prob).sample((size,)).argmax(dim=1)
      elif Component_mode == 'fixed_counts':
        index_num = torch.round(
            size * torch.as_tensor(tau, device=device, dtype=torch.float32)).long()
        index_num[index_num.argmax()] += size - index_num.sum()
        sample = torch.repeat_interleave(torch.arange(k, device=device), index_num)
      else:
        raise ValueError("Component_mode must be 'multinomial' or 'fixed_counts'")
      if W is None:
        w = torch.empty((size, s), device=device).exponential_()
        w = w / w.mean(dim=1, keepdim=True)
      else:
        w = W
      z = torch.randn((size, zm), device=device)
      output = NN(torch.cat((w, z), dim=1)).reshape(size, k, p)
      output = output.transpose(1, 2)
      theta = output.gather(2, sample[:, None, None].expand(-1, p, 1)).squeeze(2)

      if dist in ("Gaussian scale", "Poisson", "Gamma rate", "Gamma shape", "Weibull scale"):
        theta = theta.exp()
      elif dist == "Binomial":
        theta = theta.sigmoid()
      elif dist != "Gaussian location":
        raise ValueError(f'Unsupported distribution: {dist}')

    if device.type == 'cuda':
      torch.cuda.synchronize(device)
    
    Generation_time = time.perf_counter() - Generation_start
    Theta_dist = pd.DataFrame(theta.cpu().numpy())
    if Return_details:
      return (Theta_dist, Generation_time, p,
              sample.cpu().numpy(),
              output.reshape(size, p*k).cpu().numpy(), z.cpu().numpy())
    return Theta_dist, Generation_time, p


class _CheckpointGenerator(nn.Module):
    """Network architecture saved by gbnpmle_function_gpu.py."""

    def __init__(self, input_dim, hidden_dim, n_components):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, n_components)

    def forward(self, values):
        values = torch.relu(self.fc1(values))
        values = torch.relu(self.fc2(values))
        values = torch.relu(self.fc3(values))
        return self.fc_out(values)


def _save_checkpoint_draws(data_id, checkpoint_path, output_dir,
                           n_draws=1000, device=None,
                           expected_one_ones_every=0,
                           weight_mode='fixed', npeb_path=None, seed=None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    summary = json.loads(checkpoint_path.with_name('summary.json').read_text())
    n = int(summary['n'])
    if (summary.get('one_ones_every') != expected_one_ones_every or
            summary.get('iterations') != 30000 or
            not np.isclose(float(summary['sigma']), 1.0)):
        raise ValueError(f'Unexpected GB training settings: {checkpoint_path}')
    data_name = Path(summary['data_path']).stem
    mode_dir = 'fixed_w_ones' if weight_mode == 'fixed' else 'bootstrap'
    if weight_mode == 'fixed':
        weight_matrix = np.ones((n_draws, n), dtype=np.float32)
        npmle_source = ''
    elif weight_mode == 'npeb':
        if npeb_path is None or not npeb_path.is_file():
            raise FileNotFoundError(f'data{data_id}: NPEB result not found: {npeb_path}')
        with np.load(npeb_path, allow_pickle=False) as npeb:
            weight_matrix = np.asarray(npeb['bootstrap_weights'], dtype=np.float32)
        if weight_matrix.ndim != 2 or weight_matrix.shape[1] != n or weight_matrix.shape[0] < n_draws:
            raise ValueError(f'data{data_id}: NPEB bootstrap_weights must have at least {(n_draws, n)} entries')
        weight_matrix = weight_matrix[:n_draws]
        if (not np.isfinite(weight_matrix).all() or np.any(weight_matrix < 0) or
                not np.all(weight_matrix.sum(axis=1) == n)):
            raise ValueError(f'data{data_id}: invalid NPEB bootstrap counts')
        npmle_source = str(npeb_path.resolve())
    else:
        raise ValueError(f'Unknown weight mode: {weight_mode}')
    path = output_dir / mode_dir / f'data{data_id}' / f'result_GB_{data_name}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = int(seed) if seed is not None else (44000 + data_id if weight_mode == 'fixed' else 1002)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as previous:
                if (previous['theta'].shape == (n_draws,) and
                        np.isfinite(previous['theta']).all() and
                        'atoms' in previous.files and
                        'weights' in previous.files and
                        previous['atoms'].shape == (n_draws, 1) and
                        previous['weights'].shape == (n_draws, 1) and
                        np.array_equal(previous['atoms'][:, 0], previous['theta']) and
                        np.all(previous['weights'] == 1) and
                        np.array_equal(previous['bootstrap_weights'], weight_matrix) and
                        str(previous['weight_mode']) == mode_dir and
                        str(previous['npmle_source']) == npmle_source and
                        int(previous['seed']) == seed and
                        'device' in previous.files and
                        str(previous['device']) == str(device) and
                        'component_mode' in previous.files and
                        str(previous['component_mode']) == 'multinomial' and
                        str(previous['checkpoint']) == str(checkpoint_path.resolve())):
                    print(f'data{data_id}: reuse saved GB theta NPZ', flush=True)
                    return path
        except (OSError, ValueError, KeyError):
            pass
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if saved.get('checkpoint_kind') != 'last' or saved.get('epoch') != 30000:
        raise ValueError(f'Incomplete GB checkpoint: {checkpoint_path}')
    state = saved['state_dict']
    input_dim = state['fc1.weight'].shape[1]
    hidden_dim = state['fc1.weight'].shape[0]
    n_components = state['fc_out.weight'].shape[0]
    q = input_dim - n
    if q <= 0:
        raise ValueError(f'GB input width differs from n={n}: {checkpoint_path}')
    tau = np.asarray(saved['tau'], dtype=float).reshape(-1)
    if (tau.size != n_components or not np.isfinite(tau).all() or
            (tau < 0).any() or tau.sum() <= 0):
        raise ValueError(f'Invalid GB component probabilities: {checkpoint_path}')
    tau /= tau.sum()
    model = _CheckpointGenerator(input_dim, hidden_dim, n_components)
    model.load_state_dict(state)
    model.eval()
    torch.manual_seed(seed)
    fit = (model, tau, 1, n_components, 'Gaussian location', n, n, q)
    theta_frame, seconds, _, components, _, _ = GB_NPMLE_Sampling(
        fit, n_draws, W=weight_matrix, Device=device, Return_details=True,
        Component_mode='multinomial')
    theta = theta_frame.to_numpy(dtype=float).reshape(-1)
    if theta.shape != (n_draws,) or not np.isfinite(theta).all():
        raise ValueError(f'Invalid GB theta draws: {checkpoint_path}')
    temporary = path.with_name(path.name + '.tmp.npz')
    atoms = theta.reshape(-1, 1)
    np.savez_compressed(temporary, theta=theta, atoms=atoms,
                        weights=np.ones_like(atoms), component=components,
                        bootstrap_weights=weight_matrix, tau=tau,
                        fixed_w=np.bool_(weight_mode == 'fixed'),
                        weight_mode=np.str_(mode_dir), npmle_source=np.str_(npmle_source),
                        seed=np.int64(seed), inference_seconds=np.float64(seconds),
                        device=np.str_(str(device)),
                        component_mode=np.str_('multinomial'),
                        checkpoint=np.str_(str(checkpoint_path.resolve())))
    os.replace(temporary, path)
    print(f'data{data_id}: saved GB theta NPZ', flush=True)
    return path


def _main():
    parser = argparse.ArgumentParser(
        description='Save GB theta draws using fixed ones or paired NPEB bootstrap weights')
    parser.add_argument('--pair', nargs=2, action='append', required=True,
                        metavar=('DATA_ID', 'GB_CHECKPOINT'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--n-draws', type=int, default=1000)
    parser.add_argument('--seed', type=int,
                        help='Sampling seed (default: 44000+DATA_ID for fixed, 1002 for NPEB weights)')
    parser.add_argument('--weight-mode', choices=('fixed', 'npeb'), default='fixed',
                        help='Use all-one weights or saved NPEB bootstrap weights')
    parser.add_argument('--npeb-template', type=str,
                        help='NPEB result NPZ path containing {data_id}; required for --weight-mode npeb')
    parser.add_argument('--expected-one-ones-every', type=int, default=0,
                        help='Required w=1 insertion interval in the GB training checkpoint (0 disables it)')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto',
                        help='Device for sampling and generator inference')
    args = parser.parse_args()
    if args.n_draws < 1:
        parser.error('--n-draws must be positive')
    if args.expected_one_ones_every < 0:
        parser.error('--expected-one-ones-every must be nonnegative')
    if args.weight_mode == 'npeb':
        if not args.npeb_template or '{data_id}' not in args.npeb_template:
            parser.error('--weight-mode npeb requires --npeb-template containing {data_id}')
    elif args.npeb_template:
        parser.error('--npeb-template requires --weight-mode npeb')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for data_id_text, checkpoint_text in args.pair:
        data_id = int(data_id_text)
        if data_id < 1:
            parser.error('DATA_ID must be positive')
        checkpoint = Path(checkpoint_text)
        npeb_path = (Path(args.npeb_template.replace('{data_id}', str(data_id)))
                      if args.npeb_template else None)
        result_path = _save_checkpoint_draws(
            data_id, checkpoint, args.output_dir, args.n_draws, device,
            args.expected_one_ones_every, args.weight_mode, npeb_path, args.seed)
        sampling_seed = (args.seed if args.seed is not None else
                         (44000 + data_id if args.weight_mode == 'fixed' else 1002))
        manifest.append({'data_id': data_id,
                         'gb_checkpoint': str(checkpoint.resolve()),
                         'result_path': str(result_path.resolve()),
                         'weight_mode': ('fixed_w_ones' if args.weight_mode == 'fixed'
                                         else 'bootstrap'),
                         'npmle_source': str(npeb_path.resolve()) if npeb_path else None,
                         'n_theta': args.n_draws,
                         'component_mode': 'multinomial',
                         'device': str(device),
                         'seed': sampling_seed})
    mode_dir = 'fixed_w_ones' if args.weight_mode == 'fixed' else 'bootstrap'
    (args.output_dir / mode_dir / 'sample_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    _main()
