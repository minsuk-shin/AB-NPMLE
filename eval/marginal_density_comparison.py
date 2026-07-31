"""
Marginal Density Comparison — generic setting
x_i = theta_i + eps_i,  eps_i ~ N(0, sigma_eps^2)

Overlays NPMLE (baseline) vs. several AB-NPMLE variants on one figure.

Example:
    python marginal_density_comparison.py \
        --sim_data_dir   ./sim_data \
        --npmle_dir      ./sim_data_npmle \
        --save_dir       ./results_marginal \
        --n_data         1 \
        --data_file      WCS1_Normal.npz \
        --npmle_file     result_npmle_WCS1.npz \
        --ab_result      "AB-NPMLE:./sim_data_gms_1000/data{i}/WCS1_Normal/result_AB_npmle_WCS1_Normal.npz" \
        --ab_result      "AB-NPMLE (group=100):./sim_data_gms_1000/data{i}/WCS1_Normal_group100/result_AB_npmle_WCS1_Normal.npz" \
        --tag            WCS1_Normal \
        --sigma_eps      1.0
"""


import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm
import os
import argparse


OI_BLUE   = "#0072B2"
OI_VERMIL = "#D55E00"

COLOR_NPMLE = OI_BLUE
LINESTYLE_NPMLE = "--"

MARKER_NPMLE = "o"
MARKEVERY    = 40

AB_PALETTE = [
    (OI_VERMIL, (0, (3, 1, 1, 1))),
    ("#009E73", "--"),
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
    w       = weights / weights.sum()
    log_phi = norm.logpdf(x[:, None], loc=atoms[None, :], scale=sigma_eps)
    log_max = log_phi.max(axis=1, keepdims=True)
    m       = np.exp(log_max.squeeze(1)) * (np.exp(log_phi - log_max) @ w)
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

FONTSIZE = 25


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
        "axes.labelsize"    : FONTSIZE,
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


def _save_png_pdf(fig, save_path):
    base, _ = os.path.splitext(save_path)
    png_path, pdf_path = base + ".png", base + ".pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300)
    print(f"    Saved: {png_path}  /  {pdf_path}")


# =============================================================================
# 5. Density plot — main (density only; the only plot this script produces)
# =============================================================================

def plot_density_main(x_sorted, m_npmle, ab_methods, data_idx, save_path=None):
    setup_style()
    fig, ax_main = plt.subplots(figsize=(4, 4))

    ax_main.plot(x_sorted, m_npmle,
                 color=COLOR_NPMLE, lw=2.0, linestyle=LINESTYLE_NPMLE,
                 marker=MARKER_NPMLE, markersize=5, markevery=MARKEVERY,
                 label="Bootstrap-NPMLE", alpha=0.90, zorder=4)
    for m in ab_methods:
        is_hl = m.get("highlight", False)
        ax_main.plot(x_sorted, m["m_s"],
                     color=m["color"], lw=3.0 if is_hl else 2.0,
                     linestyle=m["linestyle"], label=_wrap_label(m["label"]),
                     alpha=1.0 if is_hl else 0.90, zorder=6 if is_hl else 4)
    ax_main.set_xlabel("$x$", labelpad=4)
    ax_main.set_ylabel("Density", labelpad=4)
    legend_fontsize = FONTSIZE - 4 if len(ab_methods) >= 4 else FONTSIZE
    # ax_main.legend(loc="upper right", fontsize=legend_fontsize, **LEGEND_KW)
    ax_main.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    fig.subplots_adjust(left=0.32, right=0.97, top=0.95, bottom=0.24)
    if save_path:
        _save_png_pdf(fig, save_path)
    plt.close()


# =============================================================================
# 6. Process a single data folder
# =============================================================================

def process_one(data_idx, sim_data_dir, npmle_dir, save_dir,
                data_file, npmle_file, ab_result_args, tag,
                sigma_eps):

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
    )

    return True


# =============================================================================
# 7. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--sim_data_dir', type=str, default='./sim_data')
    parser.add_argument('--npmle_dir',    type=str, default='./sim_data_npmle')
    parser.add_argument('--save_dir',     type=str, default='./results_marginal')
    parser.add_argument('--n_data',       type=int, default=50)

    parser.add_argument('--data_file',  type=str, default='JZ_A3_s500.npz',
                        help='Observed-data filename inside the sim_data folder')
    parser.add_argument('--npmle_file', type=str, default='result_npmle_JZ_A3_s500.npz',
                        help='NPMLE result filename inside the npmle_dir folder')
    parser.add_argument('--ab_result', type=str, action='append', required=True,
                        help='e.g. "AB-NPMLE (group=100):./sim_data_gms_1000/data{i}/WCS1_Normal/result_AB_npmle_WCS1_Normal.npz@black"')
    parser.add_argument('--tag',        type=str, default='JZ_A3_s500',
                        help='Output filename prefix (npz, pdf)')

    parser.add_argument('--sigma_eps', type=float, default=1.0,
                        help='noise std  (eps_i ~ N(0, sigma_eps^2))')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    ab_labels_preview = [parse_ab_result_arg(r)[0] for r in args.ab_result]
    print(f"[info] Comparing {len(args.ab_result)} AB-NPMLE method(s): "
          + ", ".join(ab_labels_preview))

    success, skipped = 0, 0
    for i in range(1, args.n_data + 1):
        print(f"\n{'='*50}")
        print(f"[data{i} / data{args.n_data}]")
        ok = process_one(
            data_idx       = i,
            sim_data_dir   = args.sim_data_dir,
            npmle_dir      = args.npmle_dir,
            save_dir       = args.save_dir,
            data_file      = args.data_file,
            npmle_file     = args.npmle_file,
            ab_result_args = args.ab_result,
            tag            = args.tag,
            sigma_eps      = args.sigma_eps,
        )
        if ok:
            success += 1
        else:
            skipped += 1

    print(f"\n{'='*50}")
    print(f"Done.  processed={success}  skipped={skipped}")


if __name__ == "__main__":
    main()
