"""
Checks whether the bootstrap distribution changes consistently with NPMLE as sigma varies.

Usage:
    python eval/sigma_sensitivity.py --data_name WCS1_Normal \
        --ckpt_path ./sim_data_gms_1000/data1/WCS1_Normal/checkpoint_AMEM_WCS1_Normal.pth \
        --sigma_candidates 0.5 0.8 1.0 \
        --npmle_paths ./sim_data_npmle_std0.5/data1/result_npmle_WCS1_Normal.npz ./sim_data_npmle_std0.8/data1/result_npmle_WCS1_Normal.npz ./sim_data_npmle/data1/result_npmle_WCS1_Normal.npz \
        --n_bootstrap 1000 \
        --result_dir ./plots_sigma/WCS1_Normal
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#F7F7F7",
    "axes.edgecolor"    : "#CCCCCC",
    "axes.linewidth"    : 0.8,
    "axes.grid"         : False,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "xtick.direction"   : "out",
    "ytick.direction"   : "out",
    "xtick.major.size"  : 13,
    "ytick.major.size"  : 13,
    "xtick.labelsize"   : 38,
    "ytick.labelsize"   : 38,
    "xtick.color"       : "#555555",
    "ytick.color"       : "#555555",
    "font.family"       : "DejaVu Sans",
    "font.size"         : 38,
    "axes.labelsize"    : 38,
    "axes.titlesize"    : 13,
    "axes.titleweight"  : "bold",
    "legend.frameon"    : True,
    "legend.framealpha" : 0.92,
    "legend.edgecolor"  : "#CCCCCC",
    "legend.fontsize"   : 38,
    "axes.unicode_minus": False,
})
import matplotlib.pyplot as plt
import seaborn as sns

from model.model import AMEM_NET
from infer import (
    infer_bootstrap_npmle,
    extract_npmle_bootstrap_w,
    resolve_hp,
    device,
)


def load_x(data_path):
    """Loads the observed-data array, preferring the 'x' key (falls back to
    'z', then the first key) -- same convention as posterior_bootstrap.py."""
    raw   = np.load(data_path, allow_pickle=True)
    x_key = "x" if "x" in raw.files else ("z" if "z" in raw.files else raw.files[0])
    return np.asarray(raw[x_key], dtype=np.float64).ravel()


def aggregate(atoms_list, weights_list):
    combined_a, combined_w = [], []
    valid = 0
    for a, w in zip(atoms_list, weights_list):
        if a is not None and len(a) > 0:
            combined_a.extend(a)
            combined_w.extend(w)
            valid += 1
    if valid == 0:
        return None, None
    combined_a = np.array(combined_a)
    combined_w = np.array(combined_w) / valid
    return combined_a, combined_w


def plot_one_sigma(sigma_val, ab_atoms, ab_weights,
                  npmle_atoms=None, npmle_weights=None,
                  data_name="", save_path="sigma_kde.pdf",
                  show_legend=True):
    """
    AB-NPMLE vs NPMLE KDE plot for a single sigma.

    show_legend: if False, no legend is drawn. Useful when looping over
    several sigmas and only wanting a legend on the first panel.
    """
    PALETTE = {
        "bg":    "white",
        "grid":  "#E0E0E0",
        "text":  "#1A1A1A",
        "ab":    "#56B4E9",
        "npmle": "#000000",
    }

    fig = plt.figure(figsize=(8, 9), facecolor=PALETTE["bg"])
    ax  = fig.add_subplot(111, facecolor=PALETTE["bg"])

    all_pts = []
    if ab_atoms is not None:
        all_pts.append(ab_atoms)
    if npmle_atoms is not None:
        all_pts.append(npmle_atoms)

    if not all_pts:
        plt.close()
        print(f"[skip] sigma={sigma_val:.2f}: no valid atoms")
        return

    q_lo, q_hi = np.percentile(np.concatenate(all_pts), [0.5, 99.5])

    if ab_atoms is not None and len(ab_atoms) > 0:
        sns.kdeplot(
            x=ab_atoms, weights=ab_weights,
            bw_adjust=0.05,
            color=PALETTE["ab"], lw=2.5,
            linestyle="solid",
            label="AB-NPMLE",
            ax=ax, zorder=5,
        )

    if npmle_atoms is not None and npmle_weights is not None:
        sns.kdeplot(
            x=npmle_atoms, weights=npmle_weights,
            bw_adjust=0.05,
            color=PALETTE["npmle"], lw=2.0,
            linestyle="solid",
            label="Bootstrap-NPMLE",
            ax=ax, zorder=6,
        )

    ax.set_xlim(q_lo, q_hi)

    for spine in ax.spines.values():
        spine.set_edgecolor("#BBBBBB")
        spine.set_linewidth(0.8)

    ax.tick_params(colors=PALETTE["text"])
    ax.set_xlabel(r"$\theta$",  labelpad=10, color=PALETTE["text"])
    ax.set_ylabel("Density",    labelpad=10, color=PALETTE["text"])

    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, linestyle=":")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.4, linestyle=":")

    if show_legend:
        ax.legend(
            loc="upper right",
            handlelength=1.0,
            handletextpad=0.4,
            borderpad=0.3,
            labelspacing=0.3,
        )

    plt.tight_layout()
    plt.savefig(save_path, facecolor=PALETTE["bg"], format="pdf")
    plt.close()
    print(f"Saved -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name",        type=str,   required=True)
    parser.add_argument("--ckpt_path",        type=str,   required=True,
                        help="Path to checkpoint_AMEM_{data_name}.pth")
    parser.add_argument("--data_idx",         type=int,   default=1,
                        help="Dataset index i, only used to build the default data_dir "
                             "(./sim_data/data{i} or ./real_data).")
    parser.add_argument("--data_mode",        type=str,   default="sim",
                        choices=["real", "sim"],
                        help="Directory layout for the default data_dir: 'sim' (default) "
                             "for ./sim_data/data{i}/..., 'real' for ./real_data/... .")
    parser.add_argument("--data_dir",         type=str,   default=None,
                        help="Overrides the data directory directly. If omitted, derived "
                             "from --data_mode/--data_idx.")
    parser.add_argument("--sigma_candidates", type=float, nargs="+",
                        default=[0.5, 0.8, 1.0],
                        help="sigma (noise std) candidates to compare (default: 0.5 0.8 1.0).")
    parser.add_argument("--npmle_paths",      type=str,   nargs="+", default=None,
                        help="NPMLE .npz paths, one per --sigma_candidates entry, in the "
                             "same order. When given, AB-NPMLE reuses that npz's actual "
                             "bootstrap weights (if num_group == n) for a paired comparison. "
                             "If omitted, only AB-NPMLE is plotted (no NPMLE reference), and "
                             "bootstrap weights are generated independently.")
    parser.add_argument("--n_bootstrap",      type=int,   default=1000)
    parser.add_argument("--hidden_dim",       type=int,   default=None,
                        help="Fallback if the checkpoint has no saved hyperparams (older checkpoint).")
    parser.add_argument("--num_layers",       type=int,   default=None,
                        help="Fallback if the checkpoint has no saved hyperparams (older checkpoint).")
    parser.add_argument("--n_support",        type=int,   default=None,
                        help="Fallback if the checkpoint has no saved hyperparams (older checkpoint).")
    parser.add_argument("--num_group",        type=int,   default=1000,
                        help="Fallback if the checkpoint has no saved hyperparams (older checkpoint).")
    parser.add_argument("--result_dir",       type=str,   default=None,
                        help="Output directory (defaults to ckpt_path's folder).")
    args = parser.parse_args()

    base_dir = args.result_dir or os.path.dirname(args.ckpt_path)
    os.makedirs(base_dir, exist_ok=True)

    i = args.data_idx
    default_data_dir = f"./sim_data/data{i}" if args.data_mode == "sim" else "./real_data"
    data_dir  = args.data_dir or default_data_dir
    data_path = os.path.join(data_dir, args.data_name + ".npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    x_np = load_x(data_path)
    n    = len(x_np)

    x_tensor  = torch.tensor(x_np, dtype=torch.float32).to(device)
    x_mean    = x_tensor.mean().item()
    x_std_val = x_tensor.std().item()
    x_std     = (x_tensor - x_mean) / x_std_val

    print(f"n={n}  x_mean={x_mean:.4f}  x_std={x_std_val:.4f}")

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    print(f"Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    saved_hp = ckpt.get("hyperparams") or {}

    hidden_dim = resolve_hp(args.hidden_dim, saved_hp, "hidden_dim", 256, "hidden_dim")
    num_layers = resolve_hp(args.num_layers, saved_hp, "num_layers", 5, "num_layers")
    N          = resolve_hp(args.n_support,  saved_hp, "n_support", 100, "n_support")
    m          = resolve_hp(None, saved_hp, "num_group", args.num_group, "num_group")

    x_min, x_max = x_std.min().item(), x_std.max().item()
    pad     = (x_max - x_min) * 0.1
    support = torch.linspace(x_min - pad, x_max + pad, N, device=device).reshape(1, 1, -1)

    model = AMEM_NET(
        hidden_dim=hidden_dim, output_dim=N,
        num_layers=num_layers,
        support=support.squeeze(0),
        n=n, m=m,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt_path}")

    if args.npmle_paths and len(args.npmle_paths) != len(args.sigma_candidates):
        raise ValueError(
            f"--npmle_paths count ({len(args.npmle_paths)}) does not match "
            f"--sigma_candidates count ({len(args.sigma_candidates)})."
        )
    npmle_path_map = dict(zip(args.sigma_candidates, args.npmle_paths or []))

    for idx, s_true in enumerate(args.sigma_candidates):
        sigma_infer = s_true / x_std_val
        print(f"\n  sigma={s_true:.4f}  ->  sigma_infer={sigma_infer:.4f}")

        npmle_data, npmle_atoms, npmle_weights = None, None, None
        npmle_path = npmle_path_map.get(s_true)
        if npmle_path:
            if os.path.exists(npmle_path):
                npmle_data     = np.load(npmle_path, allow_pickle=True)
                atoms_list_n   = [a for a in npmle_data["atoms"]   if a is not None]
                weights_list_n = [w for w in npmle_data["weights"] if w is not None]
                npmle_atoms, npmle_weights = aggregate(atoms_list_n, weights_list_n)
                print(f"  NPMLE loaded ({len(atoms_list_n)} bootstraps): {npmle_path}")
            else:
                print(f"  NPMLE not found (skipped): {npmle_path}")

        npmle_w = (extract_npmle_bootstrap_w(npmle_data, args.n_bootstrap, m, n)
                   if npmle_data is not None else None)

        atoms_list, weights_list, _ = infer_bootstrap_npmle(
            model, sigma_infer, n, args.n_bootstrap, m,
            x_mean, x_std_val, device, external_w=npmle_w,
        )
        ab_atoms, ab_weights = aggregate(atoms_list, weights_list)
        print(f"    valid bootstraps: {len(atoms_list)}")

        s_tag     = f"{s_true:.4f}".replace(".", "_")
        save_path = os.path.join(
            base_dir,
            f"sigma_kde_{args.data_name}_s{s_tag}.pdf"
        )

        plot_one_sigma(
            sigma_val=s_true,
            ab_atoms=ab_atoms,
            ab_weights=ab_weights,
            npmle_atoms=npmle_atoms,
            npmle_weights=npmle_weights,
            data_name=args.data_name,
            save_path=save_path,
            show_legend=(idx == 0),
        )


if __name__ == "__main__":
    main()
