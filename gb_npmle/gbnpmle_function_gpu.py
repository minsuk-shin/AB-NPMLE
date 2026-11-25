# Adapted from GBnpmle: https://github.com/shijiew97/GBnpmle
# Upstream package: GBnpmle 0.1.0, GPL-3 (see its DESCRIPTION file).
# This version adds GPU training checkpoints and an NPZ command-line runner.
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import sys
import argparse
import json
import math
import os
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paired_boots_weight.bootstrap_weights import draw_paired_weights
from run.train import generate_bootstrap_weights


def _cpu_state(network):
    return {name: tensor.detach().cpu().clone()
            for name, tensor in network.state_dict().items()}


@torch.no_grad()
def _gaussian_tau(network, X, Y, s, n, zm, p, k, param, tol, device):
    """Package Stage II update for a saved Gaussian-location generator."""
    tau = torch.ones(k, device=device) / k
    sub_size = 20
    y = Y.view(1, n).expand(sub_size, n)
    for iteration in range(1, 5001):
        pz_sum = torch.empty((n, k), device=device)
        for j in range(k):
            W, _ = dirichlet(s, n, m=sub_size, M=1, device=device)
            inputs = input_G(n=zm, m=sub_size, M=1, w=W)
            output = network(inputs).reshape(sub_size, p * k)
            index = torch.arange(k * p).reshape(k, p).transpose(1, 0)
            output = torch.index_select(output, 1, index.reshape(-1).to(device))
            theta = output.reshape(sub_size, p, k)[:, :, j]
            location = torch.matmul(theta, X.transpose(1, 0))
            probability = torch.exp(
                torch.distributions.normal.Normal(location, scale=param)
                .log_prob(y))
            pz_sum[:, j] = probability.mean(dim=0)
        next_tau = (pz_sum / pz_sum.sum(dim=1, keepdim=True)).mean(dim=0)
        change = torch.max(torch.abs(next_tau - tau)).item()
        tau = next_tau
        if change < tol:
            break
    else:
        raise RuntimeError("Stage II tau did not converge")
    return tau.detach().cpu(), iteration

def dirichlet(s, n, m, M, device=None):
    if device is not None:
        weights = torch.empty((m, s), device=device).exponential_(1.0)
        w_s = weights / weights.mean(dim=1, keepdim=True)
        w = w_s.repeat_interleave(M, dim=0)
        if s == n:
            w_m = w_s
        elif n % s == 0:
            w_m = w_s.repeat_interleave(n // s, dim=1)
        else:
            w_m = w_s.reshape(-1).repeat_interleave(n // s + 1)[:m*n].reshape(m,n)
        return w, w_m
    w = np.random.exponential(scale=1, size=s*m).reshape(m,s)
    w_mean = np.repeat(np.mean(w, axis=1),s).reshape(m,s)
    w_s = w / w_mean
    w = np.repeat(w_s, M, axis=0)
    if s == n:
        w_m = w_s
    elif n % s == 0:
        w_m = np.repeat(w_s, n/s).reshape(m,n)
    else:
        w_m = np.repeat(w_s, int(n/s)+1)[0:m*n]
        w_m = w_m.reshape(m,n)
    return w, w_m


def input_G(n, m, M, w):
    if torch.is_tensor(w):
        z = torch.randn((m*M, n), device=w.device)
        return torch.cat((w, z), dim=1)
    z = np.random.normal(loc=0, scale=1, size = n*m*M).reshape(m*M,n)
    input_g = np.concatenate((w,z),axis=1)
    return input_g

def trans_cuda(obj):
    if torch.is_tensor(obj):
       return obj.to(dtype=torch.float)
    if torch.cuda.is_available():
       device = torch.device('cuda')
    else:
       device = torch.device('cpu')
    obj = np.asarray(obj)
    obj = torch.from_numpy(obj)
    obj = obj.to(device, dtype=torch.float)
    return obj

def GB_NPMLE_train(X, Y, S, Hidden_size, Num_it, L, M, Q, Gpu_ind, N1, n1, P, Dist, 
Param, Lr0, Verb, Tol, LrDecay, Lrpower, Save, Save_path,
Best_path=None, Last_path=None, Checkpoint_every=100, One_ones_every=0,
Last_only=False, Paired_weight_seed=None):
  #GPU Computing
  gpu_ind = int(Gpu_ind)
  if gpu_ind == -1:
    print("WARNING: CPU computing would be very slow!")
  else:
    if torch.cuda.is_available():
      device = torch.device('cuda', gpu_ind)
    else:
      device = torch.device('cpu')
      print("WARNING: CPU computing would be very slow!")
    
  #Parameter initization
  s = int(S)     
  n = int(n1)   
  m = int(M)   
  zm = int(Q) 
  N = int(N1)   
  p = int(P)   
  k = int(L)   
  save = int(Save)
  param = float(Param)
  dist = str(Dist)
  iteration = int(Num_it) 
  hidden_size = int(Hidden_size)
  lr = float(Lr0)
  lr0 = float(Lr0)
  verb = int(Verb)
  tol = float(Tol)
  lrDecay = int(LrDecay)
  lrpower = float(Lrpower)
  
  #Dataset initization
  if torch.is_tensor(X) == False:
    X = torch.from_numpy(X).to(device, dtype=torch.float)
  else:
    X = X.to(device, dtype=torch.float)
    
  if torch.is_tensor(Y) == False:
    Y = torch.from_numpy(Y).to(device, dtype=torch.float).view(n,1)
  else:
    Y = Y.to(device, dtype=torch.float).view(n,1)
  #Determine which algorithm is using
  if k == 1:
    em0 = 0
    method = "GMS Algorithm 1"
    print("Training G via GMS Algorithm 1!")
  else:
    em0 = 1
    method = "Two-stage Algorithm"
    print("Training G via Two-stage Algorithm")
    
  #NN structure: Default as three-layer-MLP
  class Net(nn.Module):
    def __init__(self, hidden_size): 
      super(Net, self).__init__()
      self.relu = nn.ReLU()
      self.fc1 = nn.Linear(s+zm, hidden_size) 
      self.fc2 = nn.Linear(hidden_size, hidden_size)
      self.fc3 = nn.Linear(hidden_size, hidden_size)
      self.fc_out = nn.Linear(hidden_size, p*k) 
  
    def forward(self, X1):
      out = self.relu( self.fc1(X1) )
      out = self.relu( self.fc2(out) )
      out = self.relu( self.fc3(out) )
      out = self.fc_out(out)
      return out
  
  #Generator initilization
  NN = Net(hidden_size).to(device)
  optimizer = torch.optim.Adam(NN.parameters(), lr=lr)
  LOSS = torch.zeros(iteration)
  best_mean_loss = math.inf
  best_epoch = None
  best_state = None
  loss_for_graph = []
  cum_times = []
  stage1_compute_seconds = 0.0
  if Best_path is not None and (dist != "Gaussian location" or iteration < Checkpoint_every):
    raise ValueError("Best/last checkpoints require Gaussian location and one full window")
  
  Training_start = time.perf_counter()
  #Generator training
  for it in range(iteration):
    iteration_start = time.perf_counter()
    
    if lrDecay == 1:
      lr = lr0/(float(it+1.0)**Lrpower) 
      for param_group in optimizer.param_groups:
        param_group["lr"] = lr
      
    if Paired_weight_seed is None:
      w, w_m = dirichlet(s=s, n=n, m=N, M=m, device=device)
    else:
      # Unlike the original exponential bootstrap weights, use AMEM's bootstrap
      # weight sampler with a fixed per-iteration seed so both methods use the
      # same sampled w at each training iteration.
      w_s = draw_paired_weights(N, s, device, Paired_weight_seed, it,
                                generate_bootstrap_weights).squeeze(1)
      w = w_s.repeat_interleave(m, dim=0)
      if s == n:
        w_m = w_s
      elif n % s == 0:
        w_m = w_s.repeat_interleave(n // s, dim=1)
      else:
        w_m = w_s.reshape(-1).repeat_interleave(n // s + 1)[:N*n].reshape(N, n)
    if One_ones_every and (it + 1) % One_ones_every == 0:
      # The first of N weight vectors is repeated m times for Monte Carlo z draws.
      w[:m].fill_(1.0)
      w_m[0].fill_(1.0)
    
    input_g = input_G(n=zm, m=N, M=m, w=w)
    if dist == "Gaussian location": output = NN(trans_cuda(input_g)) 
    if dist == "Gaussian scale": output = torch.exp(NN(trans_cuda(input_g)))
    if dist == "Poisson": output = torch.exp(NN(trans_cuda(input_g)))
    if dist == "Gamma rate": output = torch.exp(NN(trans_cuda(input_g)))
    if dist == "Gamma shape": output = torch.exp(NN(trans_cuda(input_g)))
    if dist == "Binomial": output = torch.exp(NN(trans_cuda(input_g)))/(1+torch.exp(NN(trans_cuda(input_g))))
    if dist == "Uniform": output = torch.exp(NN(trans_cuda(input_g)))
    if dist == "Weibull scale": output = torch.exp(NN(trans_cuda(input_g)))
    
    output1 = torch.index_select(output,
                                1,
                                torch.arange(0,k*p).reshape(k,p).transpose(1,0).reshape(-1).to(device)).reshape(N*m*p,k)

    Theta = torch.gather(output1, 
                         1, 
                         torch.multinomial(torch.ones(k, device=device)/k, 
                                           N*m*p,
                                           replacement=True).reshape(N*m*p,1)).reshape(N,m,p)

    result = torch.index_select(torch.matmul(Theta,X.transpose(1,0)).reshape(N,n*m),
                                 1,
                                 torch.arange(0,n*m).reshape(m,n).transpose(1,0).reshape(-1).to(device)).reshape(N,n,m) 

    if dist == "Gaussian location":
      normal = torch.distributions.normal.Normal(result, param)
      pdf = torch.exp(normal.log_prob(Y))
      out = trans_cuda(w_m).reshape(N, n) * torch.log(torch.mean(pdf,dim=2))

    if dist == "Gaussian scale":
      normal = torch.distributions.normal.Normal(param, result)
      pdf = torch.exp(normal.log_prob(Y))
      out = trans_cuda(w_m).reshape(N, n) * torch.log(torch.mean(pdf,dim=2))
      
    if dist == "Poisson": 
      poi = torch.distributions.poisson.Poisson(result)
      pmf = torch.exp(poi.log_prob(Y)) + 1E-20
      out = trans_cuda(w_m).reshape(N,n) * torch.log(torch.mean(pmf,dim=2))
      
    if dist == "Gamma rate":
      gamm = torch.distributions.gamma.Gamma(param,result)
      pdf = torch.exp(gamm.log_prob(Y))
      out = trans_cuda(w_m).reshape(N,n) * torch.log(torch.mean(pdf,dim=2))

    if dist == "Gamma shape":
      gamm = torch.distributions.gamma.Gamma(result,param)
      pdf = torch.exp(gamm.log_prob(Y))
      out = trans_cuda(w_m).reshape(N,n) * torch.log(torch.mean(pdf,dim=2))
     
    if dist == "Binomial":
      bino = torch.distributions.binomial.Binomial(param,probs=result)
      pdf_bin0 = torch.exp(bino.log_prob(Y))
      out = trans_cuda(w_m).reshape(N,n) * torch.log(torch.mean(pdf_bin0,dim=2))  
    
    if dist == "Uniform":
      unif = torch.distributions.uniform.Uniform(low=param,high=result)
      pdf = torch.exp(unif.log_prob(Y))
      out = trans_cuda(w_m).reshape(N,n) * torch.log(torch.mean(pdf,dim=2))  
    
    if dist == "Weibull scale":
      weib = torch.distributions.weibull.Weibull(scale=result, concentration=param)
      pdf = torch.exp(weib.log_prob(Y))
      out = trans_cuda(w_m).reshape(N,n) * torch.log(torch.mean(pdf,dim=2))
    
    loss = - torch.mean(torch.sum(out, dim=1))
    
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()
      
    LOSS[it] = loss.item()
    stage1_compute_seconds += time.perf_counter() - iteration_start
    if Best_path is not None and (it + 1) % Checkpoint_every == 0:
      mean_loss = float(LOSS[it + 1 - Checkpoint_every:it + 1].mean().item())
      if not math.isfinite(mean_loss):
        raise RuntimeError(f"Nonfinite mean loss at iteration {it + 1}")
      cum_times.append((it + 1, time.perf_counter() - Training_start))
      if mean_loss < best_mean_loss:
        best_mean_loss = mean_loss
        best_epoch = it + 1
        best_state = _cpu_state(NN)
        torch.save({"state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": it, "checkpoint_kind": "best",
                    "best_epoch": best_epoch, "best_mean_loss": best_mean_loss,
                    "loss_for_graph": loss_for_graph,
                    "cum_times": cum_times,
                    "stage1_compute_seconds": stage1_compute_seconds,
                    "total_elapsed": time.perf_counter() - Training_start},
                   Best_path)
        print(f"BEST iteration={best_epoch} mean_loss={best_mean_loss:.8f}", flush=True)
      elif (it + 1) % 500 == 0:
        print(f"PROGRESS iteration={it + 1}/{iteration} mean_loss={mean_loss:.8f} "
              f"best_iteration={best_epoch}", flush=True)
      loss_for_graph.append(best_mean_loss)
    if (it+1)%10==0 and verb == 1:
      percent = float((it+1)*100) /iteration
      arrow   = '-' * int(percent/100 *20 -1) + '>'
      spaces  = ' ' * (20-len(arrow))
      train_time = time.perf_counter() - Training_start
      print('\r[%s/%s]'% (it+1, iteration), 'Progress: [%s%s] %d %%' % (arrow, spaces, percent),
      " Current/Initial Loss: {:.2f}/{:.0f}, Method: {}, Learning rate: {:.6f}, Training time: {:.2f}".format(LOSS[it], LOSS[0], str(method), lr, train_time,), end='')
      sys.stdout.flush()
  #EM algorithm starts
  stage2_start = time.perf_counter()
  max_iter = 5000
  tau = torch.ones(k, device=device)/k
  if em0 == 1:
    pz_sum = torch.zeros((n,k), device=device)
    it = 1;sub_size = 20;M = 1
    y = torch.cat(sub_size*[Y.view(1,n)],0)
    while it <= max_iter :
        for j in range(k):
            W, w_m = dirichlet(s, n, m=sub_size, M=M, device=device)
            with torch.no_grad():
              input_g = trans_cuda(input_G(n=zm, m=sub_size, M=1, w=W))
              output = NN(input_g).reshape(sub_size, p*k)
              output = torch.index_select(output,
                                          1,
                                          torch.arange(0,k*p).reshape(k,p).transpose(1,0).reshape(-1).to(device))
              Theta = output.reshape(sub_size,p,k)
              output = torch.matmul(Theta[:,:,j], X.transpose(1,0))
              
            if dist == "Gaussian location":
              normal = torch.distributions.normal.Normal(output,scale=param)
              prob = torch.exp((normal.log_prob(y)))
            if dist == "Gaussian scale":
              normal = torch.distributions.normal.Normal(param,scale=output)
              prob = torch.exp((normal.log_prob(y)))
            if dist == "Poisson": 
              poi = torch.distributions.poisson.Poisson(torch.exp(output))
              prob = torch.exp(poi.log_prob(y)) + 1E-20
            if dist == "Gamma rate":
              gamm = torch.distributions.gamma.Gamma(param,torch.exp(output))
              prob = torch.exp(gamm.log_prob(y))
            if dist == "Gamma shape":
              gamm = torch.distributions.gamma.Gamma(torch.exp(output),param)
              prob = torch.exp(gamm.log_prob(y))
            if dist == "Binomial":
              bino = torch.distributions.binomial.Binomial(param,probs=torch.exp(output)/(1+torch.exp(output)))
              prob = torch.exp(bino.log_prob(y))
            if dist == "Uniform":
              unif = torch.distributions.uniform.Uniform(low=param, high=torch.exp(output))
              prob = torch.exp(unif.log_prob(y))
            if dist == "Weibull scale":
              weib = torch.distributions.weibull.Weibull(scale=torch.exp(output), concentration=param)
              prob = torch.exp(weib.log_prob(y))
            pz_sum[:, j] = prob.mean(dim=0)
        pz = pz_sum/pz_sum.sum(dim=1, keepdim=True)
        next_tau = pz.mean(dim=0)
        change = torch.max(torch.abs(next_tau - tau)).item()
        tau = next_tau
        
        if change < tol : break
        else: it = it+1
  if device.type == "cuda":
    torch.cuda.synchronize(device)
  stage2_compute_seconds = time.perf_counter() - stage2_start
  best_stage2_compute_seconds = 0.0
  if Best_path is not None:
    last_state = _cpu_state(NN)
    last_tau = tau.detach().cpu().clone()
    if Last_path is not None:
      torch.save({"state_dict": last_state, "tau": last_tau,
                  "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": iteration, "checkpoint_kind": "last",
                  "stage1_epoch": iteration,
                  "best_epoch": best_epoch, "best_mean_loss": best_mean_loss,
                  "loss_for_graph": loss_for_graph,
                  "cum_times": cum_times,
                  "stage1_compute_seconds": stage1_compute_seconds,
                  "stage2_compute_seconds": stage2_compute_seconds,
                  "total_elapsed": time.perf_counter() - Training_start},
                 Last_path)
    if Last_only:
      best_em_iterations = 0
    else:
      NN.load_state_dict(best_state)
      best_stage2_start = time.perf_counter()
      best_tau, best_em_iterations = _gaussian_tau(
          NN, X, Y, s, n, zm, p, k, param, tol, device)
      best_stage2_compute_seconds = time.perf_counter() - best_stage2_start
      best_checkpoint = torch.load(Best_path, map_location="cpu", weights_only=True)
      best_checkpoint.update({"tau": best_tau, "stage1_epoch": best_epoch,
                              "stage2_compute_seconds": best_stage2_compute_seconds})
      torch.save(best_checkpoint, Best_path)
      NN.load_state_dict(last_state)
    NN.best_epoch = best_epoch
    NN.best_mean_loss = best_mean_loss
    NN.best_em_iterations = best_em_iterations
    print(f"STAGE II best iterations={best_em_iterations}", flush=True)
  
  #Saving NN
  if save == 1:
    torch.save(NN.state_dict(), Save_path)

  NN.training_seconds = time.perf_counter() - Training_start
  NN.stage1_compute_seconds = stage1_compute_seconds
  NN.stage2_compute_seconds = stage2_compute_seconds
  NN.best_stage2_compute_seconds = best_stage2_compute_seconds
  print(f"TOTAL training through final save={NN.training_seconds:.2f}s", flush=True)
  print(f"Stage I compute time (through loss.item)={stage1_compute_seconds:.2f}s", flush=True)
  print(f"Stage II compute time={stage2_compute_seconds:.2f}s", flush=True)
  return NN, tau.cpu().detach().numpy(), p, k, dist, s, n, zm, NN.training_seconds, hidden_size, param, lr, N, M, tol, lrDecay


def main():
    parser = argparse.ArgumentParser(
        description="GPU GB-NPMLE Gaussian-location training with best and last checkpoints")
    parser.add_argument("data_path", type=Path, help="NPZ with x and sigma")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--weight-batch-size", type=int, default=128, help="N")
    parser.add_argument("--num-group", type=int, default=None,
                        help="weight group count; defaults to n")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--paired-weight-seed", type=int, default=1001,
                        help="Use the same multinomial weights as train.py")
    parser.add_argument("--one-ones-every", type=int, default=0,
                        help="Set the first weight vector to all ones every K iterations; 0 disables it")
    parser.add_argument("--last-only", action="store_true",
                        help="Keep periodic best checks, but overwrite their checkpoint with last and skip best Stage II EM")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if not args.data_path.is_file():
        parser.error(f"missing input: {args.data_path}")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    if args.iterations < 100 or args.iterations % 100 or args.weight_batch_size < 1:
        parser.error("iterations must be a positive multiple of 100 and N must be positive")
    if args.one_ones_every < 0:
        parser.error("--one-ones-every must be nonnegative")
    with np.load(args.data_path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float32).reshape(-1)
        sigma_values = np.asarray(data["sigma"]).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all() or sigma_values.size != 1:
        parser.error("input must contain finite x and one sigma")
    sigma = float(sigma_values[0])
    if not math.isfinite(sigma) or sigma <= 0:
        parser.error("sigma must be positive and finite")
    n = x.size
    group_size = args.num_group if args.num_group is not None else n
    if group_size < 1 or group_size > n or (args.paired_weight_seed is not None and n % group_size):
        parser.error("num-group must be in [1, n] and divide n for paired weights")
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True)
    last_path = args.output_dir / "generator_last.pt"
    best_path = last_path if args.last_only else args.output_dir / "generator_best.pt"
    print(f"START n={n} S={group_size} N={args.weight_batch_size} M=100 q=100 l=100 "
          f"iterations={args.iterations} sigma={sigma} GPU={args.device_index} "
          f"one_ones_every={args.one_ones_every}", flush=True)
    fit = GB_NPMLE_train(
        np.ones((n, 1), dtype=np.float32), x,
        group_size, 500, args.iterations, 100, 100, 100, args.device_index,
        args.weight_batch_size, n, 1, "Gaussian location", sigma,
        0.0005, 0, 0.005, 0, 0.2, 0, "",
        str(best_path), str(last_path), 100, args.one_ones_every,
        args.last_only, args.paired_weight_seed)
    summary = {
        "data_path": str(args.data_path.resolve()), "n": n, "sigma": sigma,
        "iterations": args.iterations, "group_size": group_size,
        "weight_batch_size": args.weight_batch_size,
        "mc_samples": 100, "noise_dim": 100, "components": 100,
        "hidden_size": 500, "learning_rate": 0.0005,
        "checkpoint_every": 100, "seed": args.seed,
        "paired_weight_seed": args.paired_weight_seed,
        "one_ones_every": args.one_ones_every,
        "last_only": args.last_only,
        "best_epoch": fit[0].best_epoch,
        "best_mean_loss": fit[0].best_mean_loss,
        "training_seconds": fit[0].training_seconds,
        "stage1_compute_seconds": fit[0].stage1_compute_seconds,
        "stage2_compute_seconds": fit[0].stage2_compute_seconds,
        "best_stage2_compute_seconds": fit[0].best_stage2_compute_seconds,
        "best_checkpoint": None if args.last_only else str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"DONE best={None if args.last_only else best_path} last={last_path}", flush=True)


def batch_main():
    """Run one half of the 150 simulation datasets on a dedicated GPU."""
    parser = argparse.ArgumentParser(description="GB-NPMLE 150-dataset GPU worker")
    parser.add_argument("--data-root", type=Path, default=Path("sim_data"))
    parser.add_argument("--output-root", type=Path, default=Path("sim_data_gbnpmle_1000"))
    parser.add_argument("--worker-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--weight-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args(sys.argv[2:])
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / f"worker_gpu{args.device_index}.log"
    status_path = output_root / f"worker_gpu{args.device_index}.status.tsv"

    def report(message):
        print(message, flush=True)
        with log_path.open("a") as log:
            log.write(message + "\n")

    tasks = [(index, name) for index in range(1, 51)
             for name in ("WCS1_Normal", "WCS3_Tophat", "JZ_A3_s500")]
    assigned = [(index, name) for position, (index, name) in enumerate(tasks)
                if position % 2 == args.worker_index]
    report(f"START GPU={args.device_index} worker={args.worker_index} "
           f"assigned={len(assigned)} iterations={args.iterations} "
           f"N={args.weight_batch_size} seed={args.seed}")
    with status_path.open("a") as status:
        if status.tell() == 0:
            status.write("data\tname\tstatus\ttraining_seconds\tbest_epoch\n")
        for index, name in assigned:
            data_path = data_root / f"data{index}" / f"{name}.npz"
            output_dir = output_root / f"data{index}" / f"{name}_gpu_nogroup_15k"
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            summary_path = output_dir / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                if (summary.get("data_path") != str(data_path)
                        or summary.get("iterations") != args.iterations
                        or summary.get("n") != 1000
                        or summary.get("group_size") != 1000
                        or summary.get("weight_batch_size") != args.weight_batch_size
                        or summary.get("seed") != args.seed
                        or not (output_dir / "generator_best.pt").is_file()
                        or not (output_dir / "generator_last.pt").is_file()):
                    raise RuntimeError(f"Existing result has different settings: {output_dir}")
                report(f"SKIP data{index}/{name}: completed result exists")
                seconds = summary.get("training_seconds", summary.get("stage1_seconds"))
                status.write(f"{index}\t{name}\tskipped\t"
                             f"{seconds:.6f}\t{summary['best_epoch']}\n")
                status.flush()
                continue
            if output_dir.exists():
                raise RuntimeError(f"Incomplete output directory needs review: {output_dir}")
            run_log = output_dir.parent / f"{output_dir.name}.run.log"
            if run_log.exists():
                raise RuntimeError(f"Existing run log needs review: {run_log}")
            command = [sys.executable, str(Path(__file__).resolve()),
                       str(data_path), "--output-dir", str(output_dir),
                       "--iterations", str(args.iterations),
                       "--weight-batch-size", str(args.weight_batch_size),
                       "--device-index", str(args.device_index),
                       "--seed", str(args.seed)]
            env = os.environ.copy()
            for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[variable] = "1"
            report(f"RUN data{index}/{name} GPU={args.device_index}")
            with run_log.open("w") as log:
                completed = subprocess.run(command, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, check=False)
            if completed.returncode != 0:
                status.write(f"{index}\t{name}\tfailed\t\t\n")
                status.flush()
                raise RuntimeError(f"Training failed for data{index}/{name}; see {run_log}")
            run_log.rename(output_dir / "train.log")
            summary = json.loads(summary_path.read_text())
            status.write(f"{index}\t{name}\tcompleted\t"
                         f"{summary['training_seconds']:.6f}\t{summary['best_epoch']}\n")
            status.flush()
            report(f"DONE data{index}/{name} GPU={args.device_index} "
                   f"seconds={summary['training_seconds']:.2f} "
                   f"best_epoch={summary['best_epoch']}")
    report(f"FINISHED GPU={args.device_index} all {len(assigned)} assigned tasks")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        batch_main()
    else:
        main()
  
  
  
  
  
  
  
