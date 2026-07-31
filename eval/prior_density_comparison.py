"""
prior_density_comparison.py
----------------------------------
Overlays NPMLE vs AB-NPMLE vs N group-AB-NPMLE variants as KDE plots of the
aggregated bootstrap mixing distribution. Does not train anything —
each method's atoms/weights are assumed to already be saved as result_*.npz
by ab_npmle_iter.py; this script just loads and overlays them.

Example:
    python prior_density_comparison.py \
        --npmle_path ./sim_data_npmle_v2/data1/result_npmle_WCS1.npz \
        --ab_result "AB-NPMLE:./sim_data_gms_1000/data1/WCS1_Normal/result_AB_npmle_WCS1_Normal.npz" \
        --ab_result "AB-NPMLE (group=100):./sim_data_gms_1000/data1/WCS1_Normal_group100/result_AB_npmle_WCS1_Normal.npz@black" \
        --data_name WCS1_Normal \
        --sigma_true 1.0 \
        --save_path ./compare_out/AB_group_vs_npmle_WCS1_Normal.png
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


# =====================================================================
# Color / style palette
# =====================================================================

NPMLE_COLOR = "#2563EB"
NPMLE_FILL  = "#93C5FD"

AB_PALETTE = [
    ("#DC2626", "-"),
    ("#059669", "--"),
    ("#7C3AED", "-."),
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

def plot_multi_comparison(methods, sigma_val, n, save_path, bw_adjust=0.05):
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
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "font.family": "DejaVu Sans",
        "font.size": 25,
        "axes.labelsize": 25,
        "axes.titlesize": 25,
        "axes.titleweight": "bold",
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "legend.fontsize": 23,
    })

    base, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    for m, g in zip(methods, agg):
        if g is None:
            continue
        is_hl = m.get('highlight', False)
        line = sns.kdeplot(x=g['a'], weights=g['w'], bw_adjust=bw_adjust,
                            fill=m.get('fill', True), color=m['color'],
                            alpha=0.30 if m.get('fill', True) else 1.0,
                            linewidth=3.4 if is_hl else 2.3,
                            linestyle=m['linestyle'],
                            label=m['label'], ax=ax)
        if is_hl:
            line.get_lines()[-1].set_zorder(6)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(r'$\theta$', labelpad=4)
    ax.set_ylabel('Density', labelpad=4)

    plt.tight_layout(pad=0.15)

    panel_png = f"{base}_kde.png"
    panel_pdf = f"{base}_kde.pdf"
    plt.savefig(panel_png, dpi=300, bbox_inches='tight')
    print(f"Saved PNG : {panel_png}")

    with PdfPages(panel_pdf) as pdf:
        pdf.savefig(fig)
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

def main():
    parser = argparse.ArgumentParser(
        description="Overlay NPMLE vs AB-NPMLE vs N group-AB-NPMLE variants (KDE)"
    )
    parser.add_argument('--npmle_path', type=str, required=True,
                         help='Path to NPMLE result npz (e.g. output of run_npmle_real.py)')
    parser.add_argument('--npmle_label', type=str, default='NPMLE',
                         help='NPMLE legend label (default: NPMLE)')
    parser.add_argument('--ab_result', type=str, action='append', required=True,
                         help='e.g. "AB-NPMLE (group=100):./result_group100.npz@black"')
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--n', type=int, default=None,
                         help='Original data size n (for PDF metadata only, optional)')
    parser.add_argument('--sigma_true', type=float, default=1.0)
    parser.add_argument('--bw_adjust', type=float, default=0.05,
                         help='KDE bandwidth multiplier (default 0.05, matches ab_npmle_iter.py)')
    parser.add_argument('--save_path', type=str, default=None,
                         help='Default: ./compare_out/AB_group_vs_npmle_{data_name}.png')
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = os.path.join(
            "./compare_out", f"AB_group_vs_npmle_{args.data_name}.png"
        )

    npmle_atoms, npmle_weights = load_result(args.npmle_path, args.npmle_label)
    methods = [{
        'label': args.npmle_label,
        'atoms': npmle_atoms,
        'weights': npmle_weights,
        'color': NPMLE_COLOR,
        'linestyle': '--',
        'fill': False,
    }]

    for i, raw in enumerate(args.ab_result):
        label, path, forced_color = parse_ab_result_arg(raw)
        atoms, weights = load_result(path, label)
        if forced_color is not None:
            color, linestyle = forced_color, '-'
            highlight = True
        else:
            color, linestyle = AB_PALETTE[i % len(AB_PALETTE)]
            highlight = False
        methods.append({
            'label': label,
            'atoms': atoms,
            'weights': weights,
            'color': color,
            'linestyle': linestyle,
            'fill': not highlight,
            'highlight': highlight,
        })

    print(f"\nComparing {len(methods)} method(s): " + ", ".join(m['label'] for m in methods))

    plot_multi_comparison(
        methods,
        sigma_val=args.sigma_true,
        n=args.n,
        save_path=args.save_path,
        bw_adjust=args.bw_adjust,
    )


if __name__ == "__main__":
    main()
