# AMEM-Net

AMEM-Net learns a finite mixing distribution from a candidate noise standard deviation and bootstrap weights. This repository also includes numerical NPMLE and GB-NPMLE baselines, data generators, and evaluation scripts.

See [RUNNING.md](RUNNING.md) for installation, data generation, training, inference, and evaluation commands. Python dependencies are listed in [requirements.txt](requirements.txt).

## References

- The model uses [FiLM conditioning](https://arxiv.org/abs/1709.07871).
- The optimizer is adapted from [Muon](https://github.com/KellerJordan/Muon).
- The GB-NPMLE baseline is based on [GBnpmle](https://github.com/shijiew97/GBnpmle).
- Numerical fits use [npeb](https://github.com/jake-soloff/NPEB) and MOSEK.
