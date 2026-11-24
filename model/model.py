"""
Bootstrap NPMLE with FiLM conditioning on bootstrap weights.

FiLM reference: Perez et al., "FiLM: Visual Reasoning with a General
Conditioning Layer" (AAAI 2018), https://arxiv.org/abs/1709.07871.

Model: sigma is embedded as the trunk input; the bootstrap weight vector w
conditions every block via FiLM (Feature-wise Linear Modulation). The model
outputs a discrete distribution (prob, support) over a fixed support grid,
i.e. it directly parameterizes an estimated mixing distribution G-hat.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean-centering, no bias)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: gamma * x + beta, conditioned on cond."""
    def __init__(self, cond_dim, feature_dim, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        # Init to identity: gamma=1, beta=0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[:feature_dim] = 1.0

    def forward(self, x, cond):
        params = self.net(cond)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma * x + beta


class MLP_BLOCK_FiLM(nn.Module):
    """One residual MLP block, FiLM-conditioned on the bootstrap weight embedding."""
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.layer = nn.Linear(hidden_dim, hidden_dim)
        self.ln = RMSNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.film = FiLM(cond_dim, hidden_dim)

    def forward(self, x, cond):
        out = self.layer(x)
        out = self.relu(out)
        out = self.ln(out + x)  # residual + layernorm
        out = self.film(out, cond)
        return out


class AMEM_NET(nn.Module):
    """Bootstrap NPMLE model with FiLM conditioning on the bootstrap weight w.

    forward(sigma, w) -> (prob, support): a discrete distribution over a
    learned support grid, representing the estimated mixing distribution
    G-hat for that (sigma, w) bootstrap replicate.
    """
    def __init__(self, hidden_dim, output_dim, num_layers, support, n, m):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n = n

        self.layer_sig1 = nn.Linear(1, hidden_dim)
        self.layer_sig2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_w = nn.Linear(m, hidden_dim)
        self.layer_w2 = nn.Linear(hidden_dim, m)
        self.relu = nn.ReLU()
        self.relu_f = nn.ReLU()
        self.relu_w = nn.ReLU()
        self.relu_fw = nn.ReLU()
        self.ln1 = RMSNorm(hidden_dim)

        cond_dim = m
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(MLP_BLOCK_FiLM(hidden_dim, cond_dim))

        self.layer_out = nn.Linear(hidden_dim, output_dim)
        self.layer_l1 = nn.Linear(hidden_dim, output_dim)
        self.layer_l1w = nn.Linear(hidden_dim, output_dim)
        self.layer_l2 = nn.Linear(m, output_dim)
        self.layer_l2w = nn.Linear(m, output_dim)
        self.layer_f = nn.Linear(hidden_dim, m)
        self.layer_fw = nn.Linear(hidden_dim, m)
        self.softmax = nn.Softmax(dim=-1)
        self.support = nn.Parameter(support)

        self.layer_supp_delta = nn.Linear(output_dim, output_dim)

        self.min_spacing = (support.max() - support.min()).item() / (output_dim * 5.0)
        self.s_min = support.min().item()
        self.s_max = support.max().item()

    def forward(self, sigma, w):
        w_cond = w.squeeze(1)                     # (B, n)
        sig_in = sigma.view(sigma.shape[0], -1)    # (B, 1)

        out = self.relu(self.layer_sig1(sig_in))
        out = self.layer_sig2(out)                 # (B, hidden_dim)
        w_emb = self.relu_w(self.layer_w(w_cond))
        w_emb = self.layer_w2(w_emb)

        for layer in self.layers:
            out = layer(out, w_emb)

        out_supp = out

        out = self.layer_l1(out) + self.layer_l2(self.relu_f(self.layer_f(out) * w_emb))
        prob = self.softmax(out)

        out_supp = self.layer_l1w(out_supp) + self.layer_l2w(self.relu_fw(self.layer_fw(out_supp) * w_emb))

        raw = self.layer_supp_delta(out_supp)               # (B, N)
        delta = F.softplus(raw) + self.min_spacing           # positive, minimum atom spacing
        cumulative = torch.cumsum(delta, dim=-1)              # (B, N), automatically monotonic

        norm = (cumulative - cumulative[:, :1]) / (cumulative[:, -1:] - cumulative[:, :1] + 1e-8)
        support = self.s_min + norm * (self.s_max - self.s_min)
        support = support.unsqueeze(1)              # (B, 1, N)

        prob = prob.unsqueeze(1)                     # (B, 1, N)
        return prob, support


def EM_based_loss(X_, support, prob, sigma, w):
    """Weighted NPMLE-style Gaussian mixture negative log-likelihood."""
    sigma = sigma.clamp(min=1e-2)  # clamped to a minimum of 1e-2 for numerical stability of the loss
    w_t = w.transpose(1, 2)
    w_t = w_t.expand(w_t.shape[0], w_t.shape[1], X_.shape[1] // w_t.shape[1]).reshape(w_t.shape[0], -1)
    LOG_2PI = 1.8378770664093453  # precomputed constant, avoids recomputing log(2*pi) every call for faster computation
    log_f_y = (0.5 * LOG_2PI + torch.log(sigma)
               + (X_ - support) ** 2 / (2 * sigma ** 2)
               - torch.log(prob + 1e-30))

    weight = torch.softmax(-log_f_y, dim=-1).detach()

    loss = torch.mean(weight * log_f_y, dim=-1)
    loss = torch.mean(w_t * loss, dim=-1)
    return loss.mean()
