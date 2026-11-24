"""
giant_csv_to_npz.py
--------------------
Converts the Z = b/SE vector (csv, generated in R by gen_real_data.R) from
the GIANT height GWAS into the .npz format (key="x") read by the rest of
this pipeline (train.py / infer.py / mdv.py / run_npmle_sim_iter.py).

Usage:
    python gen_data/giant_csv_to_npz.py \
        --csv_path ./giant_height_x.csv \
        --data_idx 1 \
        --data_name GIANT_HEIGHT_n10000

Output:
    ./real_data/{data_name}.npz   (key: "x", shape (n,))
    Loadable directly by the rest of the pipeline via
        --data_name {data_name} --data_mode real
"""
import argparse
import os
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to the x-vector csv saved from R (column name: x)")
    parser.add_argument("--data_idx", type=int, required=True,
                        help="data_idx for logging/consistency with the rest of the "
                             "pipeline's CLI (not used to build the output path in "
                             "real-data mode -- see --data_mode real in train.py/infer.py)")
    parser.add_argument("--data_name", type=str, required=True,
                        help="Dataset name to save under, without extension "
                             "(e.g. GIANT_HEIGHT_n10000)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. Defaults to ./real_data, matching "
                             "train.py/infer.py's --data_mode real convention. Pass "
                             "./sim_data/data{data_idx} instead if you intend to load "
                             "this with --data_mode sim.")
    args = parser.parse_args()

    # -- Load csv --------------------------------------------------------
    df = pd.read_csv(args.csv_path)
    if "x" not in df.columns:
        raise ValueError(
            f"csv has no 'x' column. Columns found: {list(df.columns)}"
        )
    x = df["x"].to_numpy(dtype=np.float64)
    x = x[np.isfinite(x)]  # defensively drop any stray NaN/Inf

    n = len(x)
    print(f"Loaded: {args.csv_path}")
    print(f"  n = {n}")
    print(f"  mean(x) = {x.mean():.4f}, var(x) = {x.var(ddof=1):.4f}")

    # -- Output path -------------------------------------------------------
    out_dir = args.out_dir or "./real_data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.data_name}.npz")

    # theta/eps are unknown for real GWAS data (only the observed x is
    # available), so only "x" is saved here -- matching the key convention
    # used by gen_data/gen_sim_data.py and read by every downstream script.
    np.savez(out_path, x=x.astype(np.float32))

    print(f"\nSaved: {out_path}")
    print("Example usage:")
    print(f"  python train.py --data_idx {args.data_idx} --data_name {args.data_name} --data_mode real ...")
    print(f"  python infer.py --data_idx {args.data_idx} --data_name {args.data_name} --data_mode real ...")


if __name__ == "__main__":
    main()
