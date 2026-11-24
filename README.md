# Amortized Mixture–EM Networks for Bootstrap Inference in Nonparametric Maximum Likelihood Estimation

AMEM-Net learns a finite mixing distribution from a candidate noise standard deviation and bootstrap weights. This repository also includes numerical NPMLE and GB-NPMLE baselines, data generators, and evaluation scripts.

See [RUNNING.md](RUNNING.md) for installation, data generation, training, inference, and evaluation commands. Python dependencies are listed in [requirements.txt](requirements.txt).

## Platform

AMEM training and inference support CUDA, Apple Silicon Macs through PyTorch MPS, and CPU. The reported AMEM-Net and GB-NPMLE training experiments used CUDA GPUs; results on CPU or MPS may differ because numerical precision and device operations can differ.

## Repository structure

```text
AMEM_Net/
├── gen_data/              Simulation and GIANT data preparation
├── model/                 AMEM-Net and Muon optimizer
├── run/                   AMEM training and inference
├── numerical_NPMLE/       Numerical NPMLE and CUDA EM baselines
├── gb_npmle/              GB-NPMLE training and sampling
├── paired_boots_weight/   Shared bootstrap weight sampler
├── eval/                  Plots, MDV, confidence intervals, and tests
├── requirements.txt       Python dependencies
└── RUNNING.md             Execution commands
```

## Third-Party Code & Licenses

- `model/muon_optimizer.py` is adapted from [KellerJordan/Muon](https://github.com/KellerJordan/Muon) (MIT License, © 2024 Keller Jordan) and is not a contribution of this paper. Its file header contains the full license text and citation.
- `model/model.py` implements FiLM conditioning from [Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*](https://arxiv.org/abs/1709.07871). This is an architectural reference; FiLM is implemented locally.
- `gb_npmle/gbnpmle_function_gpu.py` and `gb_npmle/gbnpmle_sampling_with_weights.py` are adapted from [GBnpmle](https://github.com/shijiew97/GBnpmle) (GPL-3). Their file headers identify the upstream source and license.
- `numerical_NPMLE/run_npmle_sim_iter.py` and `eval/mdv.py` depend on [`npeb`](https://github.com/jake-soloff/NPEB) (MIT License, © 2023 Jake A. Soloff). Their MOSEK solver runs require a [MOSEK license](https://www.mosek.com/products/academic-licenses/), such as an academic license.
