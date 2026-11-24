r"""Prior KDEs for NPEB and AMEM groups 5, 50, and 100 from saved NPZs.

Run from the repository root:

    cd /path/to/AMEM_Net
    python eval/group_prior_density_comparison.py \
      --npeb-result /path/to/result_npmle_DATA_NAME.npz \
      --group-5-result /path/to/group_5/result_AMEM_DATA_NAME.npz \
      --group-50-result /path/to/group_50/result_AMEM_DATA_NAME.npz \
      --group-100-result /path/to/group_100/result_AMEM_DATA_NAME.npz \
      --output-dir /path/to/output

For a combined figure, provide --panel TITLE NPEB_NPZ GROUP5_NPZ
GROUP50_NPZ GROUP100_NPZ three times instead of the single-case result options.
The script saves each panel and the combined three-panel figure. It only reads
fitted distributions; it does not infer or resample.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import seaborn as sns

from prior_density_comparison import get_aggregated, get_weighted_range, load_result


STYLES = (
    ("NPEB", "#2563EB", "-", 4.2),
    ("AMEM (S=5)", "#DC2626", "--", 3.8),
    ("AMEM (S=50)", "#059669", "-.", 2.8),
    ("AMEM (S=100)", "#000000", ":", 2.8),
)


def read_case(spec):
    title, npeb, group5, group50, group100 = spec
    paths = tuple(Path(p) for p in (npeb, group5, group50, group100))
    distributions = []
    for (label, *_), path in zip(STYLES, paths):
        atoms_list, weights_list = load_result(path, label)
        atoms, weights, _ = get_aggregated(atoms_list, weights_list)
        if atoms is None:
            raise ValueError(f"No valid prior atoms: {path}")
        distributions.append((atoms, weights))
    return title, paths, distributions


def draw_panel(ax, case, bw_adjust, ylabel):
    title, _, distributions = case
    ranges = []
    for index, ((_, color, linestyle, linewidth), (atoms, weights)) in enumerate(
            zip(STYLES, distributions)):
        sns.kdeplot(x=atoms, weights=weights, bw_adjust=bw_adjust, gridsize=1000,
                    fill=False, color=color, linestyle=linestyle,
                    linewidth=linewidth, ax=ax)
        ax.lines[-1].set_zorder((2, 1, 3, 4)[index])
        ranges.append(get_weighted_range(atoms, weights))
    lo, hi = min(r[0] for r in ranges), max(r[1] for r in ranges)
    pad = max((hi - lo) * 0.05, 0.1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.06)
    ax.set_title(title, fontsize=27, pad=5)
    ax.set_xlabel(r"$\theta$", fontsize=28)
    ax.set_ylabel(ylabel, fontsize=28)
    ax.tick_params(axis="both", labelsize=24, length=3)
    ax.tick_params(axis="y", labelrotation=90)
    ax.tick_params(axis="y", pad=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def draw_figure(cases, output_stem, bw_adjust):
    combined = len(cases) == 3
    fig, axes = plt.subplots(1, len(cases), squeeze=False,
                             figsize=(16, 7.7) if combined else (14, 8.5),
                             facecolor="white")
    for index, (ax, case) in enumerate(zip(axes[0], cases)):
        draw_panel(ax, case, bw_adjust, "Density" if index == 0 else "")
    handles = [Line2D([0], [0], color=color, linestyle=linestyle,
                      linewidth=width, label=label)
               for label, color, linestyle, width in STYLES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=4, fontsize=26, frameon=False, handlelength=1.8,
               columnspacing=1.2)
    fig.subplots_adjust(left=0.07 if combined else 0.10, right=0.99,
                        top=0.80, bottom=0.18,
                        wspace=0.28)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(f"{output_stem}.{extension}", dpi=300,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {output_stem}.pdf and .png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npeb-result", type=Path)
    parser.add_argument("--group-5-result", type=Path)
    parser.add_argument("--group-50-result", type=Path)
    parser.add_argument("--group-100-result", type=Path)
    parser.add_argument("--title", type=str)
    parser.add_argument("--panel", nargs=5, action="append",
                        metavar=("TITLE", "NPEB_NPZ", "GROUP5_NPZ", "GROUP50_NPZ",
                                 "GROUP100_NPZ"),
                        help="Repeat three times for individual and combined figures")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bw-adjust", type=float, default=0.1)
    args = parser.parse_args()
    if not np.isfinite(args.bw_adjust) or args.bw_adjust <= 0:
        parser.error("--bw-adjust must be positive and finite")
    if args.panel:
        if len(args.panel) != 3:
            parser.error("--panel must be given exactly three times")
        cases = [read_case(spec) for spec in args.panel]
    else:
        paths = (args.npeb_result, args.group_5_result,
                 args.group_50_result, args.group_100_result)
        if any(path is None for path in paths):
            parser.error("provide NPEB and all three group result paths")
        cases = [read_case((args.title or args.npeb_result.stem, *paths))]

    tag = str(args.bw_adjust).replace(".", "p")
    for case in cases:
        title, paths, _ = case
        name = paths[0].stem.removeprefix("result_npmle_")
        stem = args.output_dir / name / f"{name}_group_prior_bw{tag}"
        draw_figure([case], stem, args.bw_adjust)
        Path(f"{stem}_sources.json").write_text(json.dumps({
            "title": title, "bw_adjust": args.bw_adjust,
            "sources": {style[0]: str(path.resolve())
                        for style, path in zip(STYLES, paths)},
        }, indent=2) + "\n")
    if len(cases) == 3:
        draw_figure(cases, args.output_dir / f"three_panel_group_prior_bw{tag}",
                    args.bw_adjust)


if __name__ == "__main__":
    main()
