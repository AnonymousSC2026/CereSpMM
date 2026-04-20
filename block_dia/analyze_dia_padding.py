import numpy as np
from scipy.io import mmread
from collections import defaultdict
import math
import sys

# ------------------------------------------------------------
# Fixed PE grid (hardware)
# ------------------------------------------------------------
h = 1170
w = 755

if len(sys.argv) != 2:
    print("Usage: python mm_to_dia_analyze.py input.mtx")
    sys.exit(1)

mm_path = sys.argv[1]

# ------------------------------------------------------------
# Load MM
# ------------------------------------------------------------
A = mmread(mm_path).tocoo()
M, K = A.shape
rows, cols, data = A.row, A.col, A.data

print(f"[MM] shape=({M},{K}), nnz={len(data)}")

# ------------------------------------------------------------
# Auto padding
# ------------------------------------------------------------
Mt = math.ceil(M / h)
Kt = math.ceil(K / w)

M_pad = Mt * h
K_pad = Kt * w

print("---------------------------------------------------")
print("Auto tiling (with padding)")
print(f"  Mt = {Mt}, Kt = {Kt}")
print(f"  M_pad = {M_pad}, K_pad = {K_pad}")
print("---------------------------------------------------")

# ------------------------------------------------------------
# Build DIA logically (no padding arrays)
# ------------------------------------------------------------
tiles = [[defaultdict(dict) for _ in range(w)] for _ in range(h)]

for i, k, v in zip(rows, cols, data):
    py = i // Mt
    px = k // Kt
    if py >= h or px >= w:
        continue
    d = k - i
    local_row = i - py * Mt
    tiles[py][px][d][local_row] = float(v)

# ------------------------------------------------------------
# Analyze per-PE DIA
# ------------------------------------------------------------
max_num_diags = 0
max_vals_len  = 0
empty_tiles   = 0

for py in range(h):
    for px in range(w):
        diag_map = tiles[py][px]
        if not diag_map:
            empty_tiles += 1
            continue

        num_diags = 0
        vals_len  = 0

        for d in diag_map:
            rows_dict = diag_map[d]
            sorted_rows = sorted(rows_dict.keys())

            if not sorted_rows:
                continue

            # segmented DIA
            seg_len = 1
            for r0, r1 in zip(sorted_rows[:-1], sorted_rows[1:]):
                if r1 == r0 + 1:
                    seg_len += 1
                else:
                    num_diags += 1
                    vals_len  += seg_len
                    seg_len = 1
            num_diags += 1
            vals_len  += seg_len

        max_num_diags = max(max_num_diags, num_diags)
        max_vals_len  = max(max_vals_len,  vals_len)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------
print("DIA analysis result")
print("---------------------------------------------------")
print(f"PE grid            : h={h}, w={w} (tiles={h*w})")
print(f"Empty tiles        : {empty_tiles}")
print(f"MAX_NUM_DIAGS      : {max_num_diags}")
print(f"MAX_VALS_PER_PX    : {max_vals_len}")
print("---------------------------------------------------")
print("Suggested compile params:")
print(f"  Mt={Mt}, Kt={Kt}")
print(f"  MAX_NUM_DIAGS={max_num_diags}")
print(f"  MAX_VALS_PER_PX={max_vals_len}")
print("---------------------------------------------------")
