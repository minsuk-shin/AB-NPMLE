"""Plot prior KDEs from saved result NPZ paths; no inference or sampling."""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# =====================================================================
# Color / style palette
# =====================================================================

NPMLE_COLOR = "#2563EB"
NPMLE_FILL  = "#93C5FD"

AB_PALETTE = [
    ("#DC2626", "--"),
    ("#E69F00", ":"),
    ("#059669", "-."),
    ("#EA580C", ":"),
    ("#0891B2", "-"),
    ("#DB2777", "--"),
    ("#65A30D", "-."),
    ("#9333EA", ":"),
]


# =====================================================================
# Aggregation (split out of ab_npmle_iter.py so this script doesn't
# depend on importing it)
# =====================================================================

def get_aggregated(atoms_list, weights_list):
    combined_a, combined_w = [], []
    valid_count = 0
    for a, w in zip(atoms_list, weights_list):
        if a is not None and w is not None:
            combined_a.extend(a)
            combined_w.extend(w)
            valid_count += 1
    if valid_count == 0:
        return None, None, None
    combined_a = np.array(combined_a)
    combined_w = np.array(combined_w) / valid_count
    sort_idx = np.argsort(combined_a)
    sorted_a = combined_a[sort_idx]
    sorted_w = combined_w[sort_idx]
    cum_w = np.cumsum(sorted_w)
    return sorted_a, sorted_w, cum_w


def get_weighted_range(atoms, weights, lo=0.005, hi=0.995):
    sort_idx = np.argsort(atoms)
    sorted_atoms = atoms[sort_idx]
    sorted_w = weights[sort_idx] / weights.sum()
    cum_w = np.cumsum(sorted_w)
    x_min = float(sorted_atoms[np.searchsorted(cum_w, lo)])
    x_max = float(sorted_atoms[np.searchsorted(cum_w, hi)])
    return x_min, x_max


# =====================================================================
# Load result files
# =====================================================================

def load_result(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{label}] result file not found: {path}")
    data = np.load(path, allow_pickle=True)
    atoms = [a for a in data["atoms"] if a is not None]
    weights = [w for w in data["weights"] if w is not None]
    print(f"[{label}] loaded {len(atoms)} valid bootstrap distributions from {path}")
    return atoms, weights


def parse_ab_result_arg(raw):
    color = None
    if "@" in raw:
        raw, color = raw.rsplit("@", 1)
        color = color.strip() or None

    if ":" in raw:
        label, path = raw.split(":", 1)
        label = label.strip()
        path = path.strip()
        if label == "":
            label = os.path.splitext(os.path.basename(path))[0]
    else:
        path = raw.strip()
        label = os.path.splitext(os.path.basename(path))[0]
    return label, path, color


# =====================================================================
# Plotting: KDE overlay of N methods
# =====================================================================

def plot_multi_comparison(methods, sigma_val, n, save_path, bw_adjust=0.40,
                          presentation=False, paper=False, title=None,
                          landscape=False, highlight_on_top=False,
                          npmle_linewidth=None, amem_linewidth=None,
                          group_linewidth=None, compact_three_panel=False,
                          figure_size=None, png_dpi=300, fixed_canvas=False,
                          line_width_scale=1.0, kde_gridsize=200):
    n_methods = len(methods)
    if n_methods < 2:
        raise ValueError("Need at least 2 methods to compare (NPMLE + at least 1 AB-NPMLE variant).")

    agg = []
    x_lo_list, x_hi_list = [], []
    for m in methods:
        a, w, cum = get_aggregated(m['atoms'], m['weights'])
        valid = sum(1 for aa, ww in zip(m['atoms'], m['weights'])
                    if aa is not None and ww is not None)
        if a is None:
            print(f"[warning] '{m['label']}' has no valid bootstrap results -> excluding.")
            agg.append(None)
            continue
        xmin, xmax = get_weighted_range(a, w)
        x_lo_list.append(xmin)
        x_hi_list.append(xmax)
        agg.append({'a': a, 'w': w, 'cum': cum, 'valid': valid})
        print(f"[plot] {m['label']}: valid={valid}")

    if all(g is None for g in agg):
        raise RuntimeError("No method has valid results.")

    x_min = min(x_lo_list)
    x_max = max(x_hi_list)
    x_pad = (x_max - x_min) * 0.05
    x_lo, x_hi = x_min - x_pad, x_max + x_pad

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F7F7F7",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 13,
        "ytick.major.size": 13,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.labelsize": 21,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "legend.fontsize": 18,
    })
    if paper:
        plt.rcParams.update({
            "axes.facecolor": "white",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.labelsize": 9,
        })

    base, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    shared_legend = paper and landscape and (title or '').startswith('JZ')
    large_landscape_axes = paper and landscape
    compact_panel = shared_legend or compact_three_panel
    fig_size = (2.18, 2.24) if compact_panel else (3.2, 2.55) if large_landscape_axes else (
        (2.4, 2.2) if paper else
        (9.5, 7) if presentation else (8, 5.5))
    if figure_size is not None:
        fig_size = figure_size
    fig, ax = plt.subplots(figsize=fig_size)
    for m, g in zip(methods, agg):
        if g is None:
            continue
        is_hl = m.get('highlight', False)
        is_amem = (presentation or paper) and m['label'] == 'AMEM-Net'
        line_width = (1.8 if is_amem else (1.0 if is_hl else 1.25)) if paper else (
            3.6 if is_amem else (1.8 if is_hl else 2.3))
        if npmle_linewidth is not None and m.get('is_npmle', False):
            line_width = npmle_linewidth
        elif amem_linewidth is not None and is_amem:
            line_width = amem_linewidth
        elif group_linewidth is not None and is_hl:
            line_width = group_linewidth
        if m.get('linewidth') is not None:
            line_width = m['linewidth']
        line_width *= line_width_scale
        line = sns.kdeplot(x=g['a'], weights=g['w'], bw_adjust=bw_adjust,
                            gridsize=kde_gridsize,
                            fill=m.get('fill', True), color=m['color'],
                            alpha=0.65 if is_hl else (0.30 if m.get('fill', True) else 1.0),
                            linewidth=line_width,
                            linestyle=m['linestyle'],
                            label=m['label'], ax=ax)
        artist = line.get_lines()[-1]
        artist.set_zorder(6 if is_hl and highlight_on_top else
                          5 if is_amem else (2 if is_hl else 3))
    ax.set_xlim(x_lo, x_hi)
    if fixed_canvas:
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.06)
    ax.set_xlabel(r'$\theta$', labelpad=1 if large_landscape_axes else 4)
    ax.set_ylabel('Density', labelpad=1 if large_landscape_axes else 4)
    if large_landscape_axes:
        axis_label_size = 12 if compact_panel else 14
        tick_label_size = 10 if compact_panel else 13
        ax.xaxis.label.set_size(axis_label_size)
        ax.tick_params(axis='x', labelsize=tick_label_size)
        ax.yaxis.label.set_size(axis_label_size)
        ax.tick_params(axis='y', pad=1, labelsize=tick_label_size)
    if presentation or paper:
        ax.tick_params(axis='y', labelrotation=90)
    if paper:
        ax.locator_params(axis='x', nbins=5)
        ymax = ax.get_ylim()[1]
        tick_step = 0.5 if ymax > 1.0 else (0.2 if ymax > 0.35 else 0.1)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_step))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        handles, labels = ax.get_legend_handles_labels()
        labels = ['AMEM-Net\n(g=100)' if label == 'AMEM-Net (group=100)'
                  else label for label in labels]
        if shared_legend:
            labels = ['AMEM (S=100)' if label.startswith('AMEM-Net\n')
                      else label for label in labels]
            fig.legend(handles, labels, loc='upper center',
                       bbox_to_anchor=(0.52, 0.985), ncol=2, fontsize=8.8,
                       frameon=False, handlelength=0.85, handletextpad=0.25,
                       columnspacing=0.45, labelspacing=0.2)
        elif not landscape:
            legend_loc = ('upper right' if (title or '').startswith('WCS3')
                          else 'center' if (title or '').startswith('JZ')
                          else 'upper left')
            ax.legend(handles, labels, loc=legend_loc, fontsize=6.5, frameon=True,
                      framealpha=0.95, facecolor='white', edgecolor='#DDDDDD',
                      handlelength=1.7, labelspacing=0.22, borderpad=0.3)
        fig.subplots_adjust(left=0.14 if compact_panel else 0.11 if large_landscape_axes else 0.17,
                            right=0.995 if landscape else 0.99,
                            top=0.77 if shared_legend else (0.995 if large_landscape_axes else 0.99),
                            bottom=0.15 if compact_panel else 0.13 if large_landscape_axes else 0.15)
    else:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                  ncol=4 if presentation else 3, fontsize=14, frameon=False)
    if presentation and not paper:
        fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.14)
    elif not paper:
        fig.subplots_adjust(left=0.16, right=0.97, top=0.84, bottom=0.17)
    if fixed_canvas:
        fig.subplots_adjust(left=0.18, right=0.94,
                            top=0.82 if shared_legend else 0.96, bottom=0.13)

    panel_png = f"{base}_kde.png"
    panel_pdf = f"{base}_kde.pdf"
    save_kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.015} if paper and not fixed_canvas else {}
    fig.savefig(panel_png, dpi=png_dpi, **save_kwargs)
    print(f"Saved PNG : {panel_png}")

    with PdfPages(panel_pdf) as pdf:
        pdf.savefig(fig, **save_kwargs)
        d = pdf.infodict()
        d['Title'] = 'NPMLE vs AB-NPMLE vs Group AB-NPMLE Comparison (KDE)'
        n_str = f"{n}" if n is not None else "unknown"
        d['Subject'] = f'{n_methods} methods, n={n_str}, sigma={sigma_val:.4f}'
    print(f"Saved PDF : {panel_pdf}")

    plt.close(fig)
    return [('kde', panel_png, panel_pdf)]


# =====================================================================
# Main
# =====================================================================

def plot_three_panel(panel_specs, output_stem, bw_adjust,
                     line_width_scale=1.0, kde_gridsize=200):
    """Draw three saved prior comparisons with one legend above the panels."""
    styles = (
        ("NPEB", "#2563EB", "-", 1.8),
        ("AMEM-Net", "#DC2626", "--", 2.1),
        ("GB-NPMLE", "#374151", ":", 1.9),
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.1189, 3.9189))
    summary = {}
    for index, (ax, (title, paths)) in enumerate(zip(axes, panel_specs)):
        ranges = []
        for label, color, linestyle, linewidth in styles:
            atoms, weights = load_result(paths[label], label)
            locations, masses, _ = get_aggregated(atoms, weights)
            if locations is None:
                raise ValueError(f"No prior atoms for {title}: {label}")
            ranges.append(get_weighted_range(locations, masses))
            sns.kdeplot(x=locations, weights=masses, bw_adjust=bw_adjust,
                        gridsize=kde_gridsize,
                        color=color, linestyle=linestyle,
                        linewidth=linewidth * line_width_scale,
                        fill=False, ax=ax)
        lo = min(value[0] for value in ranges)
        hi = max(value[1] for value in ranges)
        pad = max(0.05 * (hi - lo), 0.1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.06)
        ymax = ax.get_ylim()[1]
        step = 0.5 if ymax > 1.0 else 0.2 if ymax > 0.35 else 0.1
        ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.set_title(title, fontsize=13.65, pad=3)
        ax.set_xlabel(r"$\theta$", fontsize=13, labelpad=2)
        ax.set_ylabel("Density" if index == 0 else "", fontsize=13, labelpad=2)
        ax.tick_params(axis="both", labelsize=11, length=3)
        ax.tick_params(axis="y", labelrotation=90)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        summary[title] = {"sources": {key: str(value.resolve())
                                      for key, value in paths.items()},
                          "bw_adjust": bw_adjust}
    handles = [Line2D([0], [0], color=color, linestyle=linestyle,
                      linewidth=linewidth * line_width_scale, label=label)
               for label, color, linestyle, linewidth in styles]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=3, fontsize=13, frameon=False, handlelength=1.7,
               columnspacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.81, bottom=0.15,
                        wspace=0.32)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(Path(f"{output_stem}.{extension}"), dpi=300,
                    bbox_inches="tight", pad_inches=0.0368)
    plt.close(fig)
    Path(f"{output_stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"Saved {output_stem}.pdf and .png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npeb-result", type=Path,
                        help="Saved NPEB result NPZ")
    parser.add_argument("--amem-result", type=Path,
                        help="Saved AMEM result NPZ")
    parser.add_argument("--gb-result", type=Path,
                        help="Optional saved GB result NPZ")
    parser.add_argument("--panel", nargs=4, action="append",
                        metavar=("TITLE", "NPEB_NPZ", "AMEM_NPZ", "GB_NPZ"),
                        help="Repeat three times for a combined three-panel figure")
    parser.add_argument("--output-stem", type=Path, required=True,
                        help="Output path without extension; writes PNG and PDF")
    parser.add_argument("--data-name", type=str,
                        help="Plot title; defaults to the AMEM scenario folder")
    parser.add_argument("--bw-adjust", type=float, default=0.1,
                        help="KDE bandwidth multiplier")
    parser.add_argument("--sigma-eps", type=float, default=1.0,
                        help="Noise standard deviation for PDF metadata")
    parser.add_argument("--paper", action="store_true",
                        help="Use the existing compact paper style")
    parser.add_argument("--landscape", action="store_true",
                        help="Use the existing landscape paper style")
    parser.add_argument("--line-width-scale", type=float, default=1.0,
                        help="Multiply KDE curve widths (default: 1.0)")
    parser.add_argument("--kde-gridsize", type=int, default=200,
                        help="Number of KDE evaluation points (default: 200)")
    args = parser.parse_args()
    if args.bw_adjust <= 0 or args.sigma_eps <= 0:
        parser.error("--bw-adjust and --sigma-eps must be positive")
    if not np.isfinite(args.line_width_scale) or args.line_width_scale <= 0:
        parser.error("--line-width-scale must be a positive finite number")
    if args.kde_gridsize < 100:
        parser.error("--kde-gridsize must be at least 100")
    if args.panel:
        if len(args.panel) != 3 or any((args.npeb_result, args.amem_result,
                                       args.gb_result)):
            parser.error("provide exactly three --panel groups, without single-plot results")
        specs = [(title, {"NPEB": Path(npeb), "AMEM-Net": Path(amem),
                          "GB-NPMLE": Path(gb)})
                 for title, npeb, amem, gb in args.panel]
        plot_three_panel(specs, args.output_stem, args.bw_adjust,
                         line_width_scale=args.line_width_scale,
                         kde_gridsize=args.kde_gridsize)
        return
    if args.npeb_result is None or args.amem_result is None:
        parser.error("provide --npeb-result and --amem-result for a single plot")

    title = args.data_name or args.amem_result.parent.parent.name
    sources = [
        ("NPEB", args.npeb_result, "#2563EB", "-"),
        ("AMEM-Net", args.amem_result, "#DC2626", "--"),
    ]
    if args.gb_result is not None:
        sources.append(("GB-NPMLE", args.gb_result, "#374151", ":"))

    methods = []
    for label, path, color, linestyle in sources:
        atoms, weights = load_result(path, label)
        methods.append({"label": label, "atoms": atoms, "weights": weights,
                        "color": color, "linestyle": linestyle, "fill": False,
                        "is_npmle": label == "NPEB"})

    stem = args.output_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    plot_multi_comparison(
        methods, sigma_val=args.sigma_eps, n=None,
        save_path=str(stem) + ".png", bw_adjust=args.bw_adjust,
        paper=args.paper, landscape=args.landscape,
        title=title.replace("_", " "),
        line_width_scale=args.line_width_scale,
        kde_gridsize=args.kde_gridsize)
    for extension in ("png", "pdf"):
        source = Path(f"{stem}_kde.{extension}")
        source.replace(Path(f"{stem}.{extension}"))
    Path(f"{stem}_summary.json").write_text(json.dumps({
        "sources": {label: str(path.resolve()) for label, path, _, _ in sources},
        "data_name": title, "bw_adjust": args.bw_adjust,
        "kde_gridsize": args.kde_gridsize,
        "sigma_eps": args.sigma_eps,
    }, indent=2) + "\n")
    print(f"Saved {stem}.png and {stem}.pdf")


if __name__ == "__main__":
    main()
