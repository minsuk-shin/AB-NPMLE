r"""Compare AMEM bootstrap prior estimates across noise standard deviations.

Run from the repository root:

    cd /path/to/AMEM_Net

AMEM inference and paired comparison with saved NPMLE results:

    python -m eval.sigma_sensitivity \
      --data_path /path/to/observed_data.npz \
      --ckpt_path /path/to/checkpoint_AMEM_DATA_NAME.pth \
      --sigma_candidates 0.5 0.8 1.0 \
      --npmle_paths /path/to/npmle_sigma_0.5.npz /path/to/npmle_sigma_0.8.npz /path/to/npmle_sigma_1.0.npz \
      --n_bootstrap 1000 --result_dir /path/to/output

For AMEM-only plots, omit --npmle_paths. Its paths must match the sigma
candidates in number and order, and their bootstrap weights must be available
for a paired comparison. One result NPZ and one KDE PDF are saved per sigma.

Plot three panels from results already saved above (no inference):

    python -m eval.sigma_sensitivity \
      --panel "sigma=0.5" /path/to/npmle_sigma_0.5.npz /path/to/amem_sigma_0.5.npz \
      --panel "sigma=0.8" /path/to/npmle_sigma_0.8.npz /path/to/amem_sigma_0.8.npz \
      --panel "sigma=1.0" /path/to/npmle_sigma_1.0.npz /path/to/amem_sigma_1.0.npz \
      --output_stem /path/to/three_sigma_prior
"""

import argparse
import os
from pathlib import Path

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
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import seaborn as sns

AMEM_COLOR = "#D55E00"
NPMLE_COLOR = "#000000"
AMEM_LINESTYLE = (0, (7, 3))


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
                  show_legend=False, axis_fontsize=36, tick_fontsize=34,
                  y_tick_decimals=None, paper_style=False,
                  bw_adjust=0.1):
    """
    AB-NPMLE vs NPMLE KDE plot for a single sigma.

    show_legend: if False, no legend is drawn. Useful when looping over
    several sigmas and only wanting a legend on the first panel.
    """
    PALETTE = {
        "bg":    "white",
        "grid":  "#E0E0E0",
        "text":  "#1A1A1A",
        "ab":    AMEM_COLOR,
        "npmle": NPMLE_COLOR,
    }

    if paper_style:
        axis_fontsize, tick_fontsize = 13, 11
    fig, ax = plt.subplots(figsize=(3.2, 2.55) if paper_style else (7.2, 6.4),
                           facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

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
            bw_adjust=bw_adjust,
            color=PALETTE["ab"], lw=2.1 if paper_style else 5.0,
            linestyle=AMEM_LINESTYLE,
            label="AMEM-Net",
            ax=ax, zorder=5,
        )

    if npmle_atoms is not None and npmle_weights is not None:
        sns.kdeplot(
            x=npmle_atoms, weights=npmle_weights,
            bw_adjust=bw_adjust,
            color=PALETTE["npmle"], lw=1.8 if paper_style else 2.0,
            linestyle="solid",
            label="Bootstrap-NPMLE",
            ax=ax, zorder=6,
        )

    ax.set_xlim(q_lo, q_hi)

    for spine in ax.spines.values():
        if paper_style:
            spine.set_visible(True)
        spine.set_edgecolor("#BBBBBB")
        spine.set_linewidth(0.8)

    ax.tick_params(colors=PALETTE["text"], labelsize=tick_fontsize,
                   length=3 if paper_style else 7)
    ax.tick_params(axis="y", labelrotation=90)
    if y_tick_decimals is not None:
        ax.yaxis.set_major_formatter(
            FormatStrFormatter(f"%.{y_tick_decimals}f"))
        if y_tick_decimals == 1:
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel(r"$\theta$", labelpad=4, fontsize=axis_fontsize,
                  color=PALETTE["text"])
    ax.set_ylabel("Density", labelpad=4, fontsize=axis_fontsize,
                  color=PALETTE["text"])

    if not paper_style:
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, linestyle=":")
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.4, linestyle=":")
    else:
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylim(bottom=0)

    if show_legend:
        ax.legend(
            loc="upper right",
            fontsize=22,
            handlelength=1.0,
            handletextpad=0.4,
            borderpad=0.25,
            labelspacing=0.3,
        )

    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.15, top=0.985)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02,
                facecolor=PALETTE["bg"], format="pdf")
    if paper_style:
        fig.savefig(str(Path(save_path).with_suffix(".png")), dpi=300,
                    bbox_inches="tight", pad_inches=0.02,
                    facecolor=PALETTE["bg"])
    plt.close()
    print(f"Saved -> {save_path}")


def plot_three_panel(panel_specs, output_stem, bw_adjust=0.1,
                     figure_height=3.9189):
    """Plot three saved NPEB/AMEM bootstrap pairs with a shared top legend."""
    fig, axes = plt.subplots(1, 3, figsize=(8.1189, figure_height), facecolor="white")
    for index, (ax, (title, npmle_path, amem_path)) in enumerate(zip(axes, panel_specs)):
        ax.set_facecolor("white")
        all_atoms = []
        for path, color, linestyle, linewidth in (
            (npmle_path, NPMLE_COLOR, "-", 1.8),
            (amem_path, AMEM_COLOR, AMEM_LINESTYLE, 2.1),
        ):
            with np.load(path, allow_pickle=True) as result:
                atoms, weights = aggregate(result["atoms"], result["weights"])
            if atoms is None:
                raise ValueError(f"No valid bootstrap atoms in {path}")
            all_atoms.append(atoms)
            sns.kdeplot(x=atoms, weights=weights, bw_adjust=bw_adjust,
                        color=color, linestyle=linestyle, linewidth=linewidth,
                        fill=False, ax=ax)
        lo, hi = np.percentile(np.concatenate(all_atoms), [0.5, 99.5])
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.06)
        ax.set_title(title, fontsize=13, fontweight="normal", pad=3)
        ax.set_xlabel(r"$\theta$", fontsize=13, labelpad=2)
        ax.set_ylabel("Density" if index == 0 else "", fontsize=13, labelpad=2)
        ax.tick_params(axis="both", labelsize=11, length=3)
        ax.tick_params(axis="y", labelrotation=90)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ymax = ax.get_ylim()[1]
        step = 0.5 if ymax > 1.0 else 0.2 if ymax > 0.35 else 0.1
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#BBBBBB")
            spine.set_linewidth(0.8)
    handles = [
        Line2D([0], [0], color=NPMLE_COLOR, lw=1.8, label="Bootstrap-NPMLE"),
        Line2D([0], [0], color=AMEM_COLOR, lw=2.1,
               linestyle=AMEM_LINESTYLE, label="AMEM-Net"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=2, fontsize=13, frameon=False, handlelength=1.7,
               columnspacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.81,
                        bottom=0.15, wspace=0.32)
    Path(output_stem).parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(f"{output_stem}.{extension}", dpi=300,
                    bbox_inches="tight", pad_inches=0.0368)
    plt.close(fig)
    print(f"Saved -> {output_stem}.pdf and .png")


def save_legend_pdf(save_path):
    """Save one shared legend for a row of sigma KDE panels."""
    handles = [
        Line2D([0], [0], color=AMEM_COLOR, lw=5.0,
               linestyle=AMEM_LINESTYLE, label="AMEM-Net"),
        Line2D([0], [0], color=NPMLE_COLOR, lw=2.0,
               linestyle="solid", label="Bootstrap-NPMLE"),
    ]
    fig = plt.figure(figsize=(7.2, 0.7), facecolor="white")
    fig.legend(handles=handles, loc="center", ncol=2, frameon=False,
               fontsize=22, handlelength=2.2, columnspacing=1.5)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02,
                facecolor="white", format="pdf")
    plt.close(fig)
    print(f"Saved -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        help="Observed-data NPZ; its filename supplies the data name")
    parser.add_argument("--ckpt_path",        type=str,
                        help="Path to a ReLU AMEM checkpoint (.pth or .pt)")
    parser.add_argument("--panel", nargs=3, action="append",
                        metavar=("TITLE", "NPEB_NPZ", "AMEM_NPZ"),
                        help="Repeat three times to plot saved results as one figure")
    parser.add_argument("--output_stem", type=str,
                        help="Output path without extension for --panel")
    parser.add_argument("--paper_style", action="store_true",
                        help="Compact paper style for individual KDE plots")
    parser.add_argument("--bw_adjust", type=float, default=0.1,
                        help="KDE bandwidth multiplier (default: 0.1)")
    parser.add_argument("--panel_height", type=float, default=3.9189,
                        help="Height in inches for a three-panel figure (default: 3.9189)")
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
    parser.add_argument("--y_tick_decimals", type=int, default=None,
                        help="Optional number of decimal places for KDE y-axis tick labels.")
    args = parser.parse_args()

    if not np.isfinite(args.bw_adjust) or args.bw_adjust <= 0:
        parser.error("--bw_adjust must be positive and finite")
    if not np.isfinite(args.panel_height) or args.panel_height <= 0:
        parser.error("--panel_height must be positive and finite")
    if args.panel:
        if len(args.panel) != 3 or not args.output_stem or args.data_path or args.ckpt_path:
            parser.error("--panel requires three groups and --output_stem, without inference inputs")
        plot_three_panel(args.panel, args.output_stem,
                         bw_adjust=args.bw_adjust,
                         figure_height=args.panel_height)
        return
    if not args.data_path or not args.ckpt_path:
        parser.error("--data_path and --ckpt_path are required for inference")

    from model.model import AMEM_NET
    from run.infer import (infer_bootstrap_npmle,
                            extract_npmle_bootstrap_w, resolve_hp, device)

    base_dir = args.result_dir or os.path.dirname(args.ckpt_path)
    os.makedirs(base_dir, exist_ok=True)

    data_path = args.data_path
    data_name = Path(data_path).stem
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

    for s_true in args.sigma_candidates:
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
        if args.npmle_paths and npmle_w is None:
            raise ValueError(
                f"Paired bootstrap weights unavailable for sigma={s_true}: {npmle_path}. "
                f"Expected {args.n_bootstrap} weight vectors of length {n} "
                f"with num_group={m}."
            )

        atoms_list, weights_list, _, infer_w = infer_bootstrap_npmle(
            model, sigma_infer, n, args.n_bootstrap, m,
            x_mean, x_std_val, device, external_w=npmle_w,
        )
        ab_atoms, ab_weights = aggregate(atoms_list, weights_list)
        print(f"    valid bootstraps: {len(atoms_list)}")

        s_tag     = f"{s_true:.4f}".replace(".", "_")
        result_path = os.path.join(
            base_dir, f"result_AB_npmle_{data_name}_s{s_tag}.npz"
        )
        np.savez(
            result_path,
            atoms=np.array(atoms_list, dtype=object),
            weights=np.array(weights_list, dtype=object),
            sigma=np.float64(s_true),
            n_bootstrap=np.int64(args.n_bootstrap),
            bootstrap_weights=infer_w,
            checkpoint=np.str_(args.ckpt_path),
            model_variant=np.str_("relu"),
            npmle_path=np.str_(npmle_path or ""),
            data_path=np.str_(data_path),
        )
        print(f"Saved -> {result_path}")

        save_path = os.path.join(
            base_dir,
            f"sigma_kde_{data_name}_s{s_tag}.pdf"
        )

        plot_one_sigma(
            sigma_val=s_true,
            ab_atoms=ab_atoms,
            ab_weights=ab_weights,
            npmle_atoms=npmle_atoms,
            npmle_weights=npmle_weights,
            data_name=data_name,
            save_path=save_path,
            show_legend=False,
            y_tick_decimals=args.y_tick_decimals,
            paper_style=args.paper_style,
            bw_adjust=args.bw_adjust,
        )

    if not args.paper_style:
        save_legend_pdf(os.path.join(base_dir, "sigma_kde_legend.pdf"))


if __name__ == "__main__":
    main()
