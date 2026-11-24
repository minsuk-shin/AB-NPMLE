r"""Plot fixed-W=1 priors from saved AMEM, NPEB, and GB NPZ results.

Run from the repository root:

    python eval/compare_fixed_w_ab_gb_sim.py \
      --amem-result /path/to/amem_fixed_w_ones.npz \
      --npeb-result /path/to/npeb_fixed_w_ones.npz \
      --gb-result /path/to/gb_fixed_w_ones.npz \
      --data-name SCENARIO_NAME \
      --output-stem /path/to/output/prior_overlay

For SCENARIO_NAME, use for example JZ_A3_s500, WCS1_Normal, or WCS3_Tophat.
All three input files must use the same all-ones bootstrap weight vector.
The output stem is written as PNG, PDF, and summary JSON files.

This file only reads fitted NPZ files; it performs no inference or sampling.
The original checkpoint inference and diagnostic plots live in
compare_fixed_w_ab_gb_sim_legacy.py.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
import numpy as np


def merge_nearby_atoms(atoms, probs, min_mass=0.001, max_span=0.25):
    """Group displayed atoms whose cluster's total theta span is <= max_span."""
    visible = probs > min_mass
    order = np.argsort(atoms[visible])
    locations = atoms[visible][order]
    masses = probs[visible][order]
    groups = []
    start = 0
    for stop in range(1, len(locations) + 1):
        if stop < len(locations) and locations[stop] - locations[start] <= max_span:
            continue
        xs, ps = locations[start:stop], masses[start:stop]
        groups.append((float(np.average(xs, weights=ps)), float(ps.sum()),
                       len(xs), float(xs.min()), float(xs.max())))
        start = stop
    return groups


def displayed_atom_groups(atoms, probs, max_span, min_mass=0.001):
    """Return individual displayed atoms when span is zero, otherwise merge."""
    if max_span > 0:
        return merge_nearby_atoms(atoms, probs, min_mass=min_mass,
                                  max_span=max_span)
    visible = probs > min_mass
    order = np.argsort(atoms[visible])
    return [(float(x), float(p), 1, float(x), float(x))
            for x, p in zip(atoms[visible][order], probs[visible][order])]


def draw_overlay(mass_ax, gb_groups, ab_groups, npmle_groups, case,
                 left_ylabel=True, right_ylabel=True, panel=False):
    """Draw grouped fitted values on an existing axis without sampling."""
    density_ax = mass_ax.twinx()
    blue, orange, purple = "#2563EB", "#D55E00", "#7952A2"
    bin_width = 0.2
    gb_centers = np.asarray([group[0] for group in gb_groups])
    gb_counts = np.asarray([group[2] for group in gb_groups])
    bin_indices, inverse = np.unique(np.rint(gb_centers / bin_width).astype(int),
                                     return_inverse=True)
    gb_density = np.bincount(inverse, weights=gb_counts) / (gb_counts.sum() * bin_width)
    density_ax.bar(bin_indices * bin_width, gb_density, width=bin_width * 0.88,
                   color=purple, edgecolor=purple, alpha=0.28,
                   linewidth=0.7, zorder=1)
    density_ax.set_ylim(0, gb_density.max() * 1.12)
    if right_ylabel:
        density_ax.set_ylabel("Probability density", color=purple,
                              fontsize=11, labelpad=2)
    density_ax.tick_params(axis="y", colors=purple, labelsize=10, pad=2)
    if panel:
        density_ax.tick_params(axis="y", labelrotation=90)
    density_ax.spines["right"].set_color(purple)
    density_ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    density_ax.set_zorder(0)
    mass_ax.set_zorder(1)
    mass_ax.patch.set_visible(False)
    for groups, color, marker, style, shift, size, hollow in (
        (npmle_groups, blue, "o", "solid", -0.025, 31, True),
        (ab_groups, orange, "^", "dashed", 0.025, 34, False),
    ):
        centers = np.asarray([group[0] for group in groups]) + shift
        mass = np.asarray([group[1] for group in groups])
        mass_ax.vlines(centers, 0, mass, colors=color, linestyles=style,
                       linewidth=1.5, zorder=3)
        mass_ax.scatter(centers, mass, s=size, marker=marker,
                        facecolors="white" if hollow else color,
                        edgecolors=color, linewidths=1.1, zorder=4)
    max_mass = max(max(group[1] for group in npmle_groups),
                   max(group[1] for group in ab_groups))
    mass_ax.set_ylim(0, max_mass * 1.15)
    all_centers = np.concatenate((gb_centers,
                                  [group[0] for group in ab_groups],
                                  [group[0] for group in npmle_groups]))
    pad = max(0.28, np.ptp(all_centers) * 0.035)
    mass_ax.set_xlim(all_centers.min() - pad, all_centers.max() + pad)
    mass_ax.set_title({"JZ_A3_s500": "JZ", "WCS1_Normal": "WCS1",
                       "WCS3_Tophat": "WCS3"}.get(case, case), fontsize=12, pad=2)
    mass_ax.set_xlabel(r"$\theta$", fontsize=12, labelpad=1)
    if left_ylabel:
        mass_ax.set_ylabel("Probability mass", fontsize=11, labelpad=2)
    mass_ax.tick_params(axis="both", labelsize=10, pad=2)
    if panel:
        mass_ax.tick_params(axis="y", labelrotation=90)
    mass_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    mass_ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    mass_ax.grid(axis="y", color="#D9DDE3", linewidth=0.5, alpha=0.8)
    mass_ax.set_axisbelow(True)
    if panel:
        density_step = (0.1 if gb_density.max() <= 0.7 else
                        0.2 if gb_density.max() <= 1.4 else 0.5)
        density_ax.set_ylim(0, np.ceil(gb_density.max() * 1.12 / density_step)
                            * density_step)
        density_ax.yaxis.set_major_locator(MultipleLocator(density_step))
        density_ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        mass_ax.set_ylim(0, np.ceil(max_mass * 1.15 / 0.1) * 0.1)
        mass_ax.yaxis.set_major_locator(MultipleLocator(0.1))
        mass_ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        mass_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        mass_ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        mass_ax.set_title(mass_ax.get_title(), fontsize=13.65, pad=3)
        mass_ax.set_xlabel(r"$\theta$", fontsize=13, labelpad=2)
        if left_ylabel:
            mass_ax.set_ylabel("Probability mass", fontsize=13, labelpad=2)
        if right_ylabel:
            density_ax.set_ylabel("Probability density", color=purple,
                                  fontsize=13, labelpad=2)
        mass_ax.tick_params(axis="both", labelsize=11.05)
        density_ax.tick_params(axis="y", labelsize=11.05)


def plot_compact_overlay(path, gb_groups, ab_groups, npmle_groups, case):
    widths = {"JZ_A3_s500": 3.2, "WCS1_Normal": 4.0, "WCS3_Tophat": 4.45}
    fig, mass_ax = plt.subplots(figsize=(widths.get(case, 4.0), 3.35))
    draw_overlay(mass_ax, gb_groups, ab_groups, npmle_groups, case)
    fig.subplots_adjust(left=0.18, right=0.81, top=0.92, bottom=0.15)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def load_grouped_priors(paths, merge_span):
    laws = {}
    reference_w = None
    for name, path in paths.items():
        with np.load(path, allow_pickle=True) as result:
            w = np.asarray(result["bootstrap_weights"])
            if (w.ndim != 2 or w.shape[0] < 1 or not np.all(w == 1) or
                    (reference_w is not None and not np.array_equal(w[0], reference_w))):
                raise ValueError(f"Expected the same all-ones W in {path}")
            reference_w = w[0]
            if name == "gb" and "theta" in result.files:
                atoms = np.asarray(result["theta"], dtype=float).reshape(-1)
                if w.shape[0] != len(atoms):
                    raise ValueError(f"Expected one fixed W per GB theta in {path}")
                weights = np.ones(len(atoms), dtype=float)
            else:
                if w.shape[0] != 1:
                    raise ValueError(f"Expected one fitted prior in {path}")
                atoms = np.asarray(result["atoms"][0], dtype=float).reshape(-1)
                weights = np.asarray(result["weights"][0], dtype=float).reshape(-1)
            if (not len(atoms) or len(atoms) != len(weights) or
                    not np.isfinite(atoms).all() or not np.isfinite(weights).all() or
                    (weights < 0).any() or weights.sum() <= 0):
                raise ValueError(f"Invalid fitted prior in {path}")
            laws[name] = atoms, weights
    ab_groups = displayed_atom_groups(*laws["ab"], max_span=merge_span)
    npmle_groups = displayed_atom_groups(*laws["npmle"], max_span=merge_span)
    gb_atoms = laws["gb"][0]
    gb_groups = displayed_atom_groups(gb_atoms, np.ones(len(gb_atoms)),
                                      max_span=merge_span, min_mass=0)
    if not ab_groups or not npmle_groups or not gb_groups:
        raise ValueError("No fitted atoms remain after grouping")
    return gb_groups, ab_groups, npmle_groups


def source_summary(paths, groups, merge_span):
    gb_groups, ab_groups, npmle_groups = groups
    return {"sources": {name: str(path) for name, path in paths.items()},
            "w": "all ones", "merge_span": merge_span,
            "ab_groups": len(ab_groups), "npmle_groups": len(npmle_groups),
            "gb_groups": len(gb_groups)}


def plot_saved_priors(paths, stem, case, merge_span):
    groups = load_grouped_priors(paths, merge_span)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        plot_compact_overlay(Path(f"{stem}.{extension}"), *groups, case)
    summary = source_summary(paths, groups, merge_span)
    stem.with_name(stem.name + "_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"Saved {stem}.png and .pdf")


def plot_three_panel(panel_specs, stem, merge_span):
    if len(panel_specs) != 3:
        raise ValueError("Exactly three panels are required")
    fig, axes = plt.subplots(1, 3, figsize=(8.1189, 3.9189))
    summaries = {}
    for index, (ax, (case, paths)) in enumerate(zip(axes, panel_specs)):
        groups = load_grouped_priors(paths, merge_span)
        draw_overlay(ax, *groups, case, left_ylabel=(index == 0),
                     right_ylabel=(index == 2), panel=True)
        summaries[case] = source_summary(paths, groups, merge_span)
    fig.legend(handles=[
        Line2D([0], [0], color="#2563EB", marker="o", markerfacecolor="white",
               linewidth=1.15, markersize=4.5, label="NPMLE"),
        Line2D([0], [0], color="#D55E00", linestyle="--", marker="^",
               linewidth=1.15, markersize=4.5, label="AMEM-Net"),
        Patch(facecolor="#7952A2", edgecolor="#7952A2", alpha=0.4,
              label="GB-NPMLE"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3,
       fontsize=13, frameon=False, handlelength=1.55, columnspacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.81,
                        bottom=0.13, wspace=0.32)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(Path(f"{stem}.{extension}"), dpi=300,
                    bbox_inches="tight", pad_inches=0.0368)
    plt.close(fig)
    stem.with_name(stem.name + "_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n")
    print(f"Saved {stem}.png and .pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amem-result", type=Path,
                        help="Saved AMEM NPZ file")
    parser.add_argument("--npeb-result", type=Path,
                        help="Saved NPEB NPZ file")
    parser.add_argument("--gb-result", type=Path,
                        help="Saved GB NPZ file")
    parser.add_argument("--panel", nargs=4, action="append",
                        metavar=("NAME", "AMEM_NPZ", "NPEB_NPZ", "GB_NPZ"),
                        help="Add one panel; provide exactly three --panel groups")
    parser.add_argument("--data-name", default=None,
                        help="Plot title/scenario; inferred from AMEM path if omitted")
    parser.add_argument("--output-stem", type=Path, default=None,
                        help="Output path without extension; defaults to current directory")
    parser.add_argument("--merge-span", type=float, default=0.2,
                        help="Maximum theta span within a displayed group")
    args = parser.parse_args()
    if not np.isfinite(args.merge_span) or args.merge_span < 0:
        parser.error("--merge-span must be nonnegative")
    if args.panel:
        if len(args.panel) != 3:
            parser.error("provide exactly three --panel groups")
        if any((args.amem_result, args.npeb_result, args.gb_result)):
            parser.error("use --panel or the three single-plot result options")
        specs = [(name, {"ab": Path(amem), "npmle": Path(npeb), "gb": Path(gb)})
                 for name, amem, npeb, gb in args.panel]
        stem = args.output_stem or Path.cwd() / "three_panel_prior_overlay"
        plot_three_panel(specs, stem, args.merge_span)
        return
    if not all((args.amem_result, args.npeb_result, args.gb_result)):
        parser.error("provide --amem-result, --npeb-result, and --gb-result")
    case = args.data_name or args.amem_result.parent.parent.name
    stem = args.output_stem or Path.cwd() / f"{case}_prior_overlay"
    paths = {"ab": args.amem_result, "npmle": args.npeb_result,
             "gb": args.gb_result}
    plot_saved_priors(paths, stem, case, args.merge_span)


if __name__ == "__main__":
    main()
