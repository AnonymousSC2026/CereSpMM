"""
CereSpMM Feature Extractor
==========================
从目录下所有 .mtx 文件提取稀疏矩阵特征，保存为 CSV。

用法:
    python extract_features.py --dir ./matrices --out features.csv

    # 如果矩阵和脚本在同一目录:
    python extract_features.py

依赖:
    pip install numpy scipy pandas
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix


def extract_features(mtx_path: str) -> dict:
    """从单个 .mtx 文件提取全部 12 个特征"""

    # ── 读取并转换为 CSR ──────────────────────────────────────
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

    # ── Group 1: 基本特征 ─────────────────────────────────────
    nnz_ratio = nnz / (n_rows * n_cols)

    # ── Group 2: 每行 nnz 分布 ────────────────────────────────
    row_nnz = np.diff(A.indptr).astype(np.float64)   # shape (n_rows,)

    max_nnz = row_nnz.max()
    min_nnz = row_nnz.min()
    avg_nnz = row_nnz.mean()
    var_nnz = row_nnz.var()
    cv_nnz  = row_nnz.std() / avg_nnz if avg_nnz > 0 else 0.0

    # ── Group 3: 对角线结构特征 ───────────────────────────────
    # 找出矩阵中实际出现的对角线编号 (col - row)
    rows_idx, cols_idx = A.nonzero()
    diag_offsets = cols_idx.astype(np.int64) - rows_idx.astype(np.int64)
    unique_diags = np.unique(diag_offsets)

    dia_num   = len(unique_diags)                        # 实际对角线数
    all_diags = n_rows + n_cols - 1                      # 所有可能对角线数
    dia_ratio = dia_num / all_diags

    # DIA 格式的 padding 率:
    #   DIA 用 dia_num 条长度为 max(n_rows,n_cols) 的数组存储
    #   实际有效元素 = nnz，其余为 padding
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
                        help="包含 .mtx 文件的目录 (默认: 当前目录)")
    parser.add_argument("--out", default="features.csv",
                        help="输出 CSV 文件名 (默认: features.csv)")
    args = parser.parse_args()

    # 找到所有 .mtx 文件
    mtx_files = sorted([
        os.path.join(args.dir, f)
        for f in os.listdir(args.dir)
        if f.lower().endswith(".mtx")
    ])

    if not mtx_files:
        print(f"[错误] 在 '{args.dir}' 下没有找到 .mtx 文件")
        return

    print(f"找到 {len(mtx_files)} 个 .mtx 文件，开始提取特征...\n")

    records = []
    for i, path in enumerate(mtx_files, 1):
        print(f"  [{i:2d}/{len(mtx_files)}] {os.path.basename(path)}")
        feat = extract_features(path)
        if feat is not None:
            records.append(feat)

    if not records:
        print("\n[错误] 没有成功提取任何特征")
        return

    df = pd.DataFrame(records)

    # 添加一列空的 label 列，供你手动填写最优算法
    df["label"] = ""   # 填写: CSR / BCOO / BDIA

    out_path = os.path.join(args.dir, args.out)
    df.to_csv(out_path, index=False)

    print(f"\n✓ 特征已保存到: {out_path}")
    print(f"  共 {len(df)} 条记录，{len(df.columns)} 列\n")
    print(df.to_string(index=False))
    print()
    print("─" * 55)
    print("下一步: 在 CSV 的 label 列填写每个矩阵在 CS-3 上")
    print("        实测最快的算法 (CSR / BCOO / BDIA)，")
    print("        然后传给 cerespMM_selector.py 训练模型。")


if __name__ == "__main__":
    main()
