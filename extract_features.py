"""
CereSpMM Feature Extractor
==========================

Instruction:
    python extract_features.py --dir ./matrices --out features.csv

    # If in the same dir:
    python extract_features.py

dependencies:
    pip install numpy scipy pandas
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix


def extract_features(mtx_path: str) -> dict:



    try:
        raw = mmread(mtx_path)
        A = csr_matrix(raw)
    except Exception as e:
        print(f"  [SKIP] {mtx_path}: {e}")
        return None

    n_rows, n_cols = A.shape
    nnz = A.nnz

    if nnz == 0:
        print(f"  [SKIP] {mtx_path}: empty matrix")
        return None


    nnz_ratio = nnz / (n_rows * n_cols)


    row_nnz = np.diff(A.indptr).astype(np.float64)   # shape (n_rows,)

    max_nnz = row_nnz.max()
    min_nnz = row_nnz.min()
    avg_nnz = row_nnz.mean()
    var_nnz = row_nnz.var()
    cv_nnz  = row_nnz.std() / avg_nnz if avg_nnz > 0 else 0.0


    rows_idx, cols_idx = A.nonzero()
    diag_offsets = cols_idx.astype(np.int64) - rows_idx.astype(np.int64)
    unique_diags = np.unique(diag_offsets)

    dia_num   = len(unique_diags)                        
    all_diags = n_rows + n_cols - 1                      
    dia_ratio = dia_num / all_diags

    dia_storage = dia_num * max(n_rows, n_cols)
    dia_pad = 1.0 - (nnz / dia_storage) if dia_storage > 0 else 1.0

    return {
        "file":      os.path.basename(mtx_path),
        "row":       n_rows,
        "col":       n_cols,
        "nnz":       nnz,
        "nnz_ratio": round(nnz_ratio, 8),
        "max":       int(max_nnz),
        "min":       int(min_nnz),
        "average":   round(avg_nnz, 4),
        "VAR":       round(var_nnz, 4),
        "CV":        round(cv_nnz,  4),
        "dia_num":   dia_num,
        "dia_ratio": round(dia_ratio, 6),
        "dia_pad":   round(dia_pad,   6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".",
                        help="include .mtx file path ")
    parser.add_argument("--out", default="features.csv",
                        help="output CSV filename (default: features.csv)")
    args = parser.parse_args()


    mtx_files = sorted([
        os.path.join(args.dir, f)
        for f in os.listdir(args.dir)
        if f.lower().endswith(".mtx")
    ])

    if not mtx_files:
        print(f"[ERROR] on '{args.dir}' .mtx file not found")
        return

    print(f"find {len(mtx_files)}  .mtx file, extracting features...\n")

    records = []
    for i, path in enumerate(mtx_files, 1):
        print(f"  [{i:2d}/{len(mtx_files)}] {os.path.basename(path)}")
        feat = extract_features(path)
        if feat is not None:
            records.append(feat)

    if not records:
        print("\n[ERROR] No features extracted successfully")
        return

    df = pd.DataFrame(records)


    df["label"] = ""   # fill: CSR / BCOO / BDIA

    out_path = os.path.join(args.dir, args.out)
    df.to_csv(out_path, index=False)

    print(f"\n✓ Features saved to: {out_path}")
    print(f"  Totoal of  {len(df)} records, {len(df.columns)} columns\n")
    print(df.to_string(index=False))
    print()
    print("─" * 55)
    print("Next step: Fill in the 'label' column of the CSV file for each matrix on the CS-3.")
    print("        Empirically Fastest Algorithms (CSR / BCOO / BDIA)")
    print("        pass it to `cerespMM_selector.py` to train the model.")


if __name__ == "__main__":
    main()
