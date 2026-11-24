"""Plot marginal densities from saved result NPZ paths; no inference or sampling."""


import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.stats import norm
import os
import argparse
import json
from pathlib import Path


OI_BLUE   = "#0072B2"
OI_VERMIL = "#D55E00"

COLOR_NPMLE = OI_BLUE
LINESTYLE_NPMLE = "-"

MARKER_NPMLE = "o"
MARKEVERY    = 40

AB_PALETTE = [
    (OI_VERMIL, "--"),
    ("#009E73", ":"),
    ("#E69F00", "-."),
    ("#CC79A7", ":"),
    ("#56B4E9", "-"),
    ("#F0E442", "--"),
]


# =============================================================================
# 1. Aggregation
# =============================================================================

def get_aggregated(atoms_list, weights_list):
    combined_a, combined_w = [], []
    valid_count = 0
    for a, w in zip(atoms_list, weights_list):
        if a is not None and w is not None:
            combined_a.extend(a)
            combined_w.extend(w)
            valid_count += 1
    if valid_count == 0:
        return None, None, None, 0
    combined_a = np.array(combined_a)
    combined_w = np.array(combined_w) / valid_count
    sort_idx   = np.argsort(combined_a)
    sorted_a   = combined_a[sort_idx]
    sorted_w   = combined_w[sort_idx]
    cum_w      = np.cumsum(sorted_w)
    return sorted_a, sorted_w, cum_w, valid_count


# =============================================================================
# 2. Estimated marginal density from discrete mixing distribution
# =============================================================================

def marginal_density_discrete(x, atoms, weights, sigma_eps=1.0):
    w = weights / weights.sum()
    # GB-NPMLE can contribute many candidate atoms; avoid a dense
    # (number of observations) x (number of atoms) allocation.
    m = np.zeros(len(x), dtype=np.float64)
    for start in range(0, len(atoms), 4096):
        stop = min(start + 4096, len(atoms))
        m += norm.pdf(x[:, None], loc=atoms[None, start:stop], scale=sigma_eps) @ w[start:stop]
    return m


# =============================================================================
# 3. Load
# =============================================================================

def load_bootstrap_results(path):
    data         = np.load(path, allow_pickle=True)
    atoms_list   = list(data["atoms"])
    weights_list = list(data["weights"])
    valid        = sum(1 for a in atoms_list if a is not None)
    print(f"    Loaded {valid} valid bootstraps from {os.path.basename(path)}")
    return atoms_list, weights_list


# =============================================================================
# 3b. Parsing multiple AB-NPMLE results (same rule as plot_group_ab_npmle_comparison.py)
# =============================================================================

def parse_ab_result_arg(raw):
    color = None
    if "@" in raw:
        raw, color = raw.rsplit("@", 1)
        color = color.strip() or None

    if ":" in raw:
        label, path_template = raw.split(":", 1)
        label = label.strip()
        path_template = path_template.strip()
        if label == "":
            label = os.path.splitext(os.path.basename(path_template))[0]
    else:
        path_template = raw.strip()
        label = os.path.splitext(os.path.basename(path_template))[0]
    return label, path_template, color


def resolve_ab_methods(ab_result_args, data_idx):
    methods = []
    for idx, raw in enumerate(ab_result_args):
        label, path_template, forced_color = parse_ab_result_arg(raw)
        path = path_template.replace("{i}", str(data_idx))
        if forced_color is not None:
            color, linestyle, highlight = forced_color, "-", True
        else:
            color, linestyle = AB_PALETTE[idx % len(AB_PALETTE)]
            highlight = False
        methods.append({
            "label":     label,
            "path":      path,
            "color":     color,
            "linestyle": linestyle,
            "highlight": highlight,
        })
    return methods


# =============================================================================
# 4. Style
# =============================================================================

FONTSIZE = 18


def setup_style():
    plt.rcParams.update({
        "figure.facecolor"  : "white",
        "axes.facecolor"    : "#F7F7F7",
        "axes.edgecolor"    : "#CCCCCC",
        "axes.linewidth"    : 0.8,
        "axes.grid"         : True,
        "grid.color"        : "white",
        "grid.linewidth"    : 1.0,
        "grid.linestyle"    : "-",
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "xtick.direction"   : "out",
        "ytick.direction"   : "out",
        "xtick.major.size"  : 13,
        "ytick.major.size"  : 13,
        "xtick.labelsize"   : FONTSIZE,
        "ytick.labelsize"   : FONTSIZE,
        "xtick.color"       : "#555555",
        "ytick.color"       : "#555555",
        "font.family"       : "DejaVu Sans",
        "font.size"         : FONTSIZE,
        "axes.labelsize"    : 21,
        "axes.titlesize"    : FONTSIZE,
        "axes.titleweight"  : "bold",
        "legend.frameon"    : True,
        "legend.framealpha" : 0.92,
        "legend.edgecolor"  : "#CCCCCC",
        "legend.fontsize"   : FONTSIZE,
    })


LEGEND_KW = dict(handlelength=0.6, handletextpad=0.2, borderpad=0.15,
                  labelspacing=0.15, borderaxespad=0.2)


def _wrap_label(label):
    return label.replace(" (", "\n(", 1) if " (" in label else label


def _save_png_pdf(fig, save_path, tight=False, png_dpi=300):
    base, _ = os.path.splitext(save_path)
    png_path, pdf_path = base + ".png", base + ".pdf"
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.015} if tight else {}
    fig.savefig(png_path, dpi=png_dpi, **save_kwargs)
    fig.savefig(pdf_path, dpi=300, **save_kwargs)
    print(f"    Saved: {png_path}  /  {pdf_path}")


# =============================================================================
# 5. Density plot — main (density only; the only plot this script produces)
# =============================================================================

def plot_density_main(x_sorted, m_npmle, ab_methods, data_idx,
                      save_path=None, npmle_label="Bootstrap-NPMLE",
                      paper=False, title=None, landscape=False,
                      three_panel_no_gb=False, npmle_linewidth=None,
                      figure_size=None, png_dpi=300, fixed_canvas=False,
                      show_legend=True, line_width_scale=1.0):
    setup_style()
    if paper:
        shared_legend = show_legend and landscape and (title or "").startswith("JZ")
        large_landscape_axes = landscape
        plt.rcParams.update({
            "axes.facecolor": "white",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.labelsize": 9,
        })
        fig_size = (2.18, 2.24) if three_panel_no_gb else (
            (3.2, 2.55) if large_landscape_axes else (2.4, 2.2))
        if figure_size is not None:
            fig_size = figure_size
        fig, ax_main = plt.subplots(figsize=fig_size)
        ax_main.plot(x_sorted, m_npmle, color="#2563EB",
                     lw=(npmle_linewidth if npmle_linewidth is not None else
                         (2.8 if three_panel_no_gb else 1.25)) * line_width_scale,
                     label=npmle_label, zorder=3)
        for m in ab_methods:
            is_amem = m["label"] == "AMEM-Net"
            is_group = "group=100" in m["label"]
            ax_main.plot(x_sorted, m["m_s"],
                         color=m.get("color", "#666666" if is_group else (
                             "#DC2626" if is_amem else "#E69F00")),
                         lw=m.get("linewidth", (2.0 if is_group else (2.4 if is_amem else 1.25))
                                  if three_panel_no_gb else
                                  (1.0 if is_group else (1.3 if is_amem else 1.25))) * line_width_scale,
                         linestyle=(":" if three_panel_no_gb else "-.") if is_group else (
                             (0, (6, 4)) if is_amem else ":"),
                         alpha=1.0 if three_panel_no_gb else
                         (0.65 if is_group else (0.85 if is_amem else 1.0)),
                         label=m["label"],
                         zorder=(6 if is_group else (5 if is_amem else 4))
                         if three_panel_no_gb else
                         (4.5 if is_group else (5 if is_amem else 4)))
            if is_group and not three_panel_no_gb:
                ax_main.plot(x_sorted[::120], m["m_s"][::120],
                             linestyle="None", marker="o", markersize=2.8,
                             markerfacecolor="white", markeredgecolor="#666666",
                             markeredgewidth=0.7, zorder=6)
            elif is_group and three_panel_no_gb:
                mark_x = np.linspace(x_sorted[0], x_sorted[-1], 18)
                mark_y = np.interp(mark_x, x_sorted, m["m_s"])
                ax_main.plot(mark_x, mark_y, linestyle="None", marker="o",
                             markersize=3.2, markerfacecolor="#4B5563",
                             markeredgecolor="white", markeredgewidth=0.4,
                             zorder=7)
        ax_main.set_xlabel("$x$", labelpad=1)
        ax_main.set_ylabel("Density", labelpad=1)
        ax_main.tick_params(axis="y", labelrotation=90)
        if large_landscape_axes:
            axis_label_size = 12 if three_panel_no_gb else 14
            tick_label_size = 10 if three_panel_no_gb else 13
            if fixed_canvas:
                axis_label_size = 12
                tick_label_size = 10
            ax_main.xaxis.label.set_size(axis_label_size)
            ax_main.yaxis.label.set_size(axis_label_size)
            ax_main.tick_params(axis="x", labelsize=tick_label_size)
            ax_main.tick_params(axis="y", pad=1, labelsize=tick_label_size)
        x_pad = 0.04 * (x_sorted[-1] - x_sorted[0])
        ax_main.set_xlim(x_sorted[0] - x_pad, x_sorted[-1] + x_pad)
        max_density = max(float(m_npmle.max()),
                          *(float(m["m_s"].max()) for m in ab_methods))
        ax_main.set_ylim(0, max(0.1, 1.12 * max_density) if landscape
                         else 1.12 * max_density)
        ax_main.locator_params(axis="x", nbins=5)
        ax_main.yaxis.set_major_locator(
            ticker.MultipleLocator(0.1) if landscape else ticker.MaxNLocator(nbins=5))
        ax_main.yaxis.set_major_formatter(
            ticker.FormatStrFormatter("%.1f" if landscape else "%.2f"))
        if shared_legend:
            handles, labels = ax_main.get_legend_handles_labels()
            handles = [Line2D([0], [0], color="#666666",
                              linestyle=":" if three_panel_no_gb else "-.",
                              linewidth=2.0 if three_panel_no_gb else 1.0,
                              marker=None if three_panel_no_gb else "o", markersize=3,
                              markerfacecolor="white", markeredgecolor="#666666")
                       if label == "AMEM-Net (group=100)" else handle
                       for handle, label in zip(handles, labels)]
            labels = ["AMEM (S=100)" if label == "AMEM-Net (group=100)"
                      else label for label in labels]
            fig.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(0.52, 0.985) if three_panel_no_gb else (0.52, 0.965),
                       ncol=2 if three_panel_no_gb or fixed_canvas else len(labels),
                       fontsize=8.8 if three_panel_no_gb else 8.0,
                       frameon=False, handlelength=0.85 if three_panel_no_gb else 1.0,
                       handletextpad=0.25, columnspacing=0.45 if three_panel_no_gb else 0.35,
                       labelspacing=0.2 if three_panel_no_gb else 0.5)
        elif show_legend and not landscape:
            legend_loc = "lower left" if (title or "").startswith(("WCS3", "JZ")) else "upper left"
            ax_main.legend(loc=legend_loc, fontsize=6.5, frameon=True,
                           framealpha=0.95, facecolor="white", edgecolor="#DDDDDD",
                           handlelength=1.7, labelspacing=0.22, borderpad=0.3)
        fig.subplots_adjust(left=0.14 if three_panel_no_gb else 0.12 if large_landscape_axes else 0.18,
                            right=0.995 if landscape else 0.99,
                            top=0.77 if three_panel_no_gb and shared_legend else
                            0.875 if shared_legend else (0.995 if large_landscape_axes else 0.99),
                            bottom=0.15 if three_panel_no_gb else 0.13 if large_landscape_axes else 0.15)
        if fixed_canvas:
            fig.subplots_adjust(left=0.18, right=0.94,
                                top=0.82 if shared_legend else 0.96, bottom=0.16)
        if save_path:
            _save_png_pdf(fig, save_path, tight=not fixed_canvas, png_dpi=png_dpi)
        plt.close(fig)
        return
    fig, ax_main = plt.subplots(figsize=(8, 5.5))

    ax_main.plot(x_sorted, m_npmle,
                 color=COLOR_NPMLE, lw=2.0 * line_width_scale, linestyle=LINESTYLE_NPMLE,
                 marker=MARKER_NPMLE, markersize=5, markevery=MARKEVERY,
                 label=npmle_label, alpha=0.90, zorder=4)
    for m in ab_methods:
        is_hl = m.get("highlight", False)
        ax_main.plot(x_sorted, m["m_s"],
                     color=m["color"], lw=(3.0 if is_hl else 2.0) * line_width_scale,
                     linestyle=m["linestyle"], label=_wrap_label(m["label"]),
                     alpha=1.0 if is_hl else 0.90, zorder=6 if is_hl else 4)
    ax_main.set_xlabel("$x$", labelpad=4)
    ax_main.set_ylabel("Density", labelpad=4)
    if show_legend:
        ax_main.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
                       fontsize=14, frameon=False)
    x_pad = 0.04 * (x_sorted[-1] - x_sorted[0])
    ax_main.set_xlim(x_sorted[0] - x_pad, x_sorted[-1] + x_pad)
    max_density = max(float(m_npmle.max()), *(float(m["m_s"].max()) for m in ab_methods))
    ax_main.set_ylim(0, 1.15 * max_density)
    ax_main.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.84, bottom=0.17)
    if save_path:
        _save_png_pdf(fig, save_path)
    plt.close()


# =============================================================================
# 6. Process a single data folder
# =============================================================================

def process_one(data_idx, sim_data_dir, npmle_dir, save_dir,
                data_file, npmle_file, ab_result_args, tag,
                sigma_eps, npmle_label="Bootstrap-NPMLE"):

    folder = f"data{data_idx}"

    data_path  = os.path.join(sim_data_dir, folder, data_file)
    npmle_path = os.path.join(npmle_dir,    folder, npmle_file)

    missing = [p for p in [data_path, npmle_path] if not os.path.exists(p)]
    if missing:
        print(f"  [skip] {folder}: missing files:")
        for p in missing:
            print(f"         {p}")
        return False

    ab_methods = resolve_ab_methods(ab_result_args, data_idx)
    kept = []
    for m in ab_methods:
        if os.path.exists(m["path"]):
            kept.append(m)
        else:
            print(f"  [warning] {folder}: '{m['label']}' result file not found -> skipping: {m['path']}")
    ab_methods = kept
    if not ab_methods:
        print(f"  [skip] {folder}: no valid AB-NPMLE result files.")
        return False

    out_dir = os.path.join(save_dir, folder, tag)
    os.makedirs(out_dir, exist_ok=True)

    data     = np.load(data_path)
    x_obs    = data['x']
    x_sorted = np.sort(x_obs)
    print(f"  n={len(x_obs)},  x range=[{x_obs.min():.3f}, {x_obs.max():.3f}]")

    npmle_atoms, npmle_weights   = load_bootstrap_results(npmle_path)
    npmle_a, npmle_w, _, npmle_B = get_aggregated(npmle_atoms, npmle_weights)

    for m in ab_methods:
        atoms, weights = load_bootstrap_results(m["path"])
        a, w, _, B = get_aggregated(atoms, weights)
        if a is None:
            print(f"  [warning] {folder}: '{m['label']}' has no valid bootstrap results -> excluding.")
            m["valid"] = False
        else:
            m["a"], m["w"], m["B"], m["valid"] = a, w, B, True
    ab_methods = [m for m in ab_methods if m.get("valid")]
    if not ab_methods:
        print(f"  [skip] {folder}: no AB-NPMLE method has valid bootstrap results.")
        return False

    m_npmle_s = marginal_density_discrete(x_sorted, npmle_a, npmle_w, sigma_eps)
    for m in ab_methods:
        m["m_s"] = marginal_density_discrete(x_sorted, m["a"], m["w"], sigma_eps)

    np.savez(
        os.path.join(out_dir, f"{tag}_marginal_all_agg.npz"),
        x_sorted   = x_sorted,
        m_npmle_s  = m_npmle_s,
        npmle_a    = npmle_a,        npmle_w   = npmle_w,
        ab_labels  = np.array([m["label"] for m in ab_methods], dtype=object),
        ab_colors  = np.array([m["color"] for m in ab_methods], dtype=object),
        ab_m_s     = np.array([m["m_s"] for m in ab_methods]),
        ab_atoms   = np.array([m["a"] for m in ab_methods], dtype=object),
        ab_weights = np.array([m["w"] for m in ab_methods], dtype=object),
    )
    print(f"  Saved npz: {tag}_marginal_all_agg.npz  "
          f"({len(ab_methods)} AB-NPMLE method(s): " + ", ".join(m["label"] for m in ab_methods) + ")")

    plot_density_main(
        x_sorted, m_npmle_s, ab_methods, data_idx,
        save_path=os.path.join(out_dir, f"{tag}_density_comparison_agg.pdf"),
        npmle_label=npmle_label,
    )

    return True


# =============================================================================
# 7. Main
# =============================================================================

def plot_three_panel(panel_specs, output_stem, sigma_override=None,
                     show_legend=True, line_width_scale=1.0):
    """Draw three saved marginal comparisons with a shared top legend."""
    styles = (
        ("NPEB", "#2563EB", "-", 1.8),
        ("AMEM-Net", "#DC2626", "--", 2.1),
        ("GB-NPMLE", "#374151", ":", 1.9),
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.1189, 3.9189))
    summary = {}
    for index, (ax, (title, observed_path, paths)) in enumerate(zip(axes, panel_specs)):
        with np.load(observed_path, allow_pickle=False) as data:
            x = np.sort(np.asarray(data["x"], dtype=float).reshape(-1))
            sigma = (float(sigma_override) if sigma_override is not None
                     else float(np.asarray(data["sigma"]).reshape(-1)[0]))
        if (x.size == 0 or not np.isfinite(x).all() or
                not np.isfinite(sigma) or sigma <= 0):
            raise ValueError(f"Invalid observed data or sigma: {observed_path}")
        ymax = 0.0
        for label, color, linestyle, linewidth in styles:
            atoms, weights = load_bootstrap_results(paths[label])
            locations, masses, _, valid = get_aggregated(atoms, weights)
            if valid == 0:
                raise ValueError(f"No fitted priors for {title}: {label}")
            density = marginal_density_discrete(x, locations, masses,
                                                sigma_eps=sigma)
            ymax = max(ymax, float(density.max()))
            ax.plot(x, density, color=color, linestyle=linestyle,
                    linewidth=linewidth * line_width_scale)
        pad = max(0.04 * (x[-1] - x[0]), 0.1)
        ax.set_xlim(x[0] - pad, x[-1] + pad)
        ax.set_ylim(0, max(0.11, ymax * 1.12))
        upper = ax.get_ylim()[1]
        step = 0.5 if upper > 1.0 else 0.2 if upper > 0.35 else 0.1
        ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.set_title(title, fontsize=13.65, pad=3)
        ax.set_xlabel("$x$", fontsize=13, labelpad=2)
        ax.set_ylabel("Density" if index == 0 else "", fontsize=13, labelpad=2)
        ax.tick_params(axis="both", labelsize=11, length=3)
        ax.tick_params(axis="y", labelrotation=90)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        summary[title] = {"sources": {key: str(value.resolve())
                                      for key, value in paths.items()},
                          "observed_data": str(observed_path.resolve()),
                          "sigma_eps": sigma}
    if show_legend:
        handles = [Line2D([0], [0], color=color, linestyle=linestyle,
                          linewidth=linewidth * line_width_scale, label=label)
                   for label, color, linestyle, linewidth in styles]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01),
                   ncol=3, fontsize=13, frameon=False, handlelength=1.7,
                   columnspacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.98,
                        top=0.81, bottom=0.15,
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
    parser.add_argument("--observed-data", type=Path,
                        help="Observed-data NPZ containing x and sigma")
    parser.add_argument("--npeb-result", type=Path,
                        help="Saved NPEB result NPZ")
    parser.add_argument("--amem-result", type=Path,
                        help="Saved AMEM result NPZ")
    parser.add_argument("--gb-result", type=Path,
                        help="Optional saved GB result NPZ")
    parser.add_argument("--panel", nargs=5, action="append",
                        metavar=("TITLE", "OBSERVED_NPZ", "NPEB_NPZ",
                                 "AMEM_NPZ", "GB_NPZ"),
                        help="Repeat three times for a combined three-panel figure")
    parser.add_argument("--output-stem", type=Path, required=True,
                        help="Output path without extension; writes PNG and PDF")
    parser.add_argument("--data-name", type=str,
                        help="Plot title; defaults to the observed-data filename")
    parser.add_argument("--sigma-eps", type=float,
                        help="Noise standard deviation; defaults to sigma in observed-data NPZ")
    parser.add_argument("--paper", action="store_true",
                        help="Use the existing compact paper style")
    parser.add_argument("--landscape", action="store_true",
                        help="Use the existing landscape paper style")
    parser.add_argument("--no-legend", action="store_true",
                        help="Omit the legend from individual and three-panel plots")
    parser.add_argument("--line-width-scale", type=float, default=1.0,
                        help="Multiply curve widths (default: 1.0)")
    args = parser.parse_args()
    if not np.isfinite(args.line_width_scale) or args.line_width_scale <= 0:
        parser.error("--line-width-scale must be a positive finite number")

    if args.panel:
        if len(args.panel) != 3 or any((args.observed_data, args.npeb_result,
                                       args.amem_result, args.gb_result)):
            parser.error("provide exactly three --panel groups, without single-plot results")
        specs = [(title, Path(observed),
                  {"NPEB": Path(npeb), "AMEM-Net": Path(amem),
                   "GB-NPMLE": Path(gb)})
                 for title, observed, npeb, amem, gb in args.panel]
        plot_three_panel(specs, args.output_stem, args.sigma_eps,
                         show_legend=not args.no_legend,
                         line_width_scale=args.line_width_scale)
        return
    if any(value is None for value in (args.observed_data, args.npeb_result,
                                       args.amem_result)):
        parser.error("provide --observed-data, --npeb-result and --amem-result for a single plot")

    with np.load(args.observed_data, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=float).reshape(-1)
        if args.sigma_eps is not None:
            sigma_eps = args.sigma_eps
        elif "sigma" in data.files:
            sigma_values = np.asarray(data["sigma"], dtype=float).reshape(-1)
            if sigma_values.size != 1:
                parser.error("observed-data sigma must have one value")
            sigma_eps = float(sigma_values[0])
        else:
            parser.error("provide --sigma-eps when observed-data NPZ has no sigma")
    if x.size == 0 or not np.isfinite(x).all() or not np.isfinite(sigma_eps) or sigma_eps <= 0:
        parser.error("observed x and sigma must be finite, and sigma must be positive")
    x_sorted = np.sort(x)

    sources = [
        ("NPEB", args.npeb_result, "#2563EB"),
        ("AMEM-Net", args.amem_result, "#DC2626"),
    ]
    if args.gb_result is not None:
        sources.append(("GB-NPMLE", args.gb_result, "#374151"))
    densities = {}
    for label, path, _ in sources:
        atoms, weights = load_bootstrap_results(path)
        locations, masses, _, valid = get_aggregated(atoms, weights)
        if valid == 0:
            raise ValueError(f"No fitted priors in {path}")
        densities[label] = marginal_density_discrete(
            x_sorted, locations, masses, sigma_eps=sigma_eps)

    stem = args.output_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    title = args.data_name or args.observed_data.stem
    plot_density_main(
        x_sorted, densities["NPEB"],
        [{"label": label, "m_s": densities[label], "color": color,
          "linestyle": "--" if label == "AMEM-Net" else ":"}
         for label, _, color in sources[1:]],
        data_idx=1, save_path=str(stem) + ".png",
        npmle_label="NPEB", paper=args.paper,
        landscape=args.landscape, title=title.replace("_", " "),
        show_legend=not args.no_legend,
        line_width_scale=args.line_width_scale)
    values = {"x": x_sorted, "npeb": densities["NPEB"],
              "amem": densities["AMEM-Net"]}
    if "GB-NPMLE" in densities:
        values["gb"] = densities["GB-NPMLE"]
    np.savez_compressed(f"{stem}_values.npz", **values)
    Path(f"{stem}_summary.json").write_text(json.dumps({
        "sources": {label: str(path.resolve()) for label, path, _ in sources},
        "observed_data": str(args.observed_data.resolve()),
        "data_name": title, "sigma_eps": sigma_eps,
    }, indent=2) + "\n")
    print(f"Saved {stem}.png and {stem}.pdf")


if __name__ == "__main__":
    main()
