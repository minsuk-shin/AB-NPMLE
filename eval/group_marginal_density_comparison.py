r"""Marginal densities for NPEB and AMEM groups 5, 50, and 100.

Run from the repository root:

    cd /path/to/AMEM_Net
    python eval/group_marginal_density_comparison.py \
      --observed-data /path/to/observed_DATA_NAME.npz \
      --npeb-result /path/to/result_npmle_DATA_NAME.npz \
      --group-5-result /path/to/group_5/result_AMEM_DATA_NAME.npz \
      --group-50-result /path/to/group_50/result_AMEM_DATA_NAME.npz \
      --group-100-result /path/to/group_100/result_AMEM_DATA_NAME.npz \
      --output-dir /path/to/output

For a combined figure, provide --panel TITLE OBSERVED_NPZ NPEB_NPZ
GROUP5_NPZ GROUP50_NPZ GROUP100_NPZ three times instead of the single-case
result options. The script saves each panel and the combined three-panel
figure. It only reads saved files; it does not infer or resample.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
import numpy as np

from marginal_density_comparison import (
    get_aggregated, load_bootstrap_results, marginal_density_discrete,
)


STYLES = (
    ("NPEB", "#2563EB", "-", 6.5),
    ("AMEM (S=5)", "#DC2626", (0, (5, 3)), 6.0),
    ("AMEM (S=50)", "#059669", "-.", 4.5),
    ("AMEM (S=100)", "#000000", ":", 4.5),
)


def read_case(spec, sigma_override=None):
    title, observed, npeb, group5, group50, group100 = spec
    observed = Path(observed)
    paths = tuple(Path(p) for p in (npeb, group5, group50, group100))
    with np.load(observed, allow_pickle=False) as data:
        x = np.sort(np.asarray(data["x"], dtype=float).reshape(-1))
        sigma = (sigma_override if sigma_override is not None
                 else float(np.asarray(data["sigma"], dtype=float).reshape(-1)[0]))
    if x.size == 0 or not np.isfinite(x).all() or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"Invalid observed data or sigma: {observed}")
    densities = []
    for path in paths:
        atoms_list, weights_list = load_bootstrap_results(path)
        atoms, weights, _, valid = get_aggregated(atoms_list, weights_list)
        if valid == 0:
            raise ValueError(f"No valid bootstrap distributions in {path}")
        densities.append(marginal_density_discrete(x, atoms, weights,
                                                   sigma_eps=sigma))
    return title, observed, paths, x, sigma, densities


def draw_panel(ax, case, ylabel):
    title, _, _, x, _, densities = case
    for index, ((_, color, linestyle, linewidth), density) in enumerate(
            zip(STYLES, densities)):
        ax.plot(x, density, color=color, linestyle=linestyle,
                linewidth=linewidth, zorder=(2, 5, 3, 4)[index])
    pad = max((x[-1] - x[0]) * 0.04, 0.1)
    ax.set_xlim(x[0] - pad, x[-1] + pad)
    ax.set_ylim(0, max(0.11, max(float(y.max()) for y in densities) * 1.12))
    ax.set_title(title, fontsize=27, pad=5)
    ax.set_xlabel("$x$", fontsize=28)
    ax.set_ylabel(ylabel, fontsize=28)
    ax.tick_params(axis="both", labelsize=24, length=3)
    ax.tick_params(axis="y", labelrotation=90)
    ax.tick_params(axis="y", pad=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def draw_figure(cases, output_stem, show_legend=False):
    combined = len(cases) == 3
    fig, axes = plt.subplots(1, len(cases), squeeze=False,
                             figsize=(16, 6.7) if combined else (14, 7.5),
                             facecolor="white")
    for index, (ax, case) in enumerate(zip(axes[0], cases)):
        draw_panel(ax, case, "Density" if index == 0 else "")
    if show_legend:
        handles = [Line2D([0], [0], color=color, linestyle=linestyle,
                          linewidth=width, label=label)
                   for label, color, linestyle, width in STYLES]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
                   ncol=4, fontsize=26, frameon=False, handlelength=1.8,
                   columnspacing=1.2)
    fig.subplots_adjust(left=0.07 if combined else 0.10, right=0.99,
                        top=0.79 if show_legend else 0.89,
                        bottom=0.18, wspace=0.28)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(f"{output_stem}.{extension}", dpi=300,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {output_stem}.pdf and .png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-data", type=Path)
    parser.add_argument("--npeb-result", type=Path)
    parser.add_argument("--group-5-result", type=Path)
    parser.add_argument("--group-50-result", type=Path)
    parser.add_argument("--group-100-result", type=Path)
    parser.add_argument("--title", type=str)
    parser.add_argument("--panel", nargs=6, action="append",
                        metavar=("TITLE", "OBSERVED_NPZ", "NPEB_NPZ", "GROUP5_NPZ",
                                 "GROUP50_NPZ", "GROUP100_NPZ"),
                        help="Repeat three times for individual and combined figures")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sigma-eps", type=float,
                        help="Override sigma in observed-data NPZ")
    parser.add_argument("--legend", action="store_true",
                        help="Show a shared top legend (default: no marginal legend)")
    args = parser.parse_args()
    if args.sigma_eps is not None and (not np.isfinite(args.sigma_eps)
                                       or args.sigma_eps <= 0):
        parser.error("--sigma-eps must be positive and finite")
    if args.panel:
        if len(args.panel) != 3:
            parser.error("--panel must be given exactly three times")
        cases = [read_case(spec, args.sigma_eps) for spec in args.panel]
    else:
        values = (args.observed_data, args.npeb_result, args.group_5_result,
                  args.group_50_result, args.group_100_result)
        if any(value is None for value in values):
            parser.error("provide observed data, NPEB, and all three group result paths")
        cases = [read_case((args.title or args.observed_data.stem, *values),
                           args.sigma_eps)]

    for case in cases:
        title, observed, paths, x, sigma, densities = case
        name = observed.stem
        folder = args.output_dir / name
        stem = folder / f"{name}_group_marginal"
        draw_figure([case], stem, args.legend)
        np.savez_compressed(f"{stem}_values.npz", x=x, npeb=densities[0],
                            group_5=densities[1], group_50=densities[2],
                            group_100=densities[3])
        Path(f"{stem}_sources.json").write_text(json.dumps({
            "title": title, "observed_data": str(observed.resolve()),
            "sigma_eps": sigma,
            "sources": {style[0]: str(path.resolve())
                        for style, path in zip(STYLES, paths)},
        }, indent=2) + "\n")
    if len(cases) == 3:
        draw_figure(cases, args.output_dir / "three_panel_group_marginal",
                    args.legend)


if __name__ == "__main__":
    main()
