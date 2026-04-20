#!/usr/bin/env cs_python
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime,
    MemcpyDataType,
    MemcpyOrder,
)

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()

# ============================================================
# 1) Read compile parameters
# ============================================================
with open(f"{args.name}/out.json", encoding="utf-8") as jf:
    compile_data = json.load(jf)

w  = int(compile_data["params"]["w"])
h  = int(compile_data["params"]["h"])
Mt = int(compile_data["params"]["Mt"])
Kt = int(compile_data["params"]["Kt"])
Nt = int(compile_data["params"]["Nt"])

M = Mt * h
K = Kt * w
N = Nt * w

# ============================================================
# 2) Init runtime
# ============================================================
runner = SdkRuntime(args.name, cmaddr=args.cmaddr, suppress_simfab_trace=True)

memcpy_dtype_fp16 = MemcpyDataType.MEMCPY_16BIT
memcpy_dtype_f32  = MemcpyDataType.MEMCPY_32BIT
memcpy_dtype_u32  = MemcpyDataType.MEMCPY_32BIT
memcpy_order      = MemcpyOrder.ROW_MAJOR

# ============================================================
# 3) Prepare inputs
# ============================================================
np.random.seed(7)

B = np.random.rand(K, N).astype(np.float16)

val_pad     = np.loadtxt("tmp_val_pad.csv",     delimiter=",", dtype=np.float32)
col_idx_pad = np.loadtxt("tmp_col_idx_pad.csv", delimiter=",", dtype=np.int32)
row_ptr_pad = np.loadtxt("tmp_row_ptr_pad.csv", delimiter=",", dtype=np.int32)

val_pad     = val_pad.reshape(h, w, -1).astype(np.float16)
col_idx_pad = col_idx_pad.reshape(h, w, -1).astype(np.uint16)
row_ptr_pad = row_ptr_pad.reshape(h, w, -1).astype(np.uint16)

MAX_META_LEN = val_pad.shape[2]
MAX_PTR_LEN  = row_ptr_pad.shape[2]

if MAX_META_LEN % 2 == 1:
    val_pad     = np.pad(val_pad,     ((0,0),(0,0),(0,1)), constant_values=0)
    col_idx_pad = np.pad(col_idx_pad, ((0,0),(0,0),(0,1)), constant_values=0)
    MAX_META_LEN += 1

if MAX_PTR_LEN % 2 == 1:
    row_ptr_pad = np.pad(row_ptr_pad, ((0,0),(0,0),(0,1)), constant_values=0)
    MAX_PTR_LEN += 1

# ============================================================
# 4) Device symbols
# ============================================================
sym_val     = runner.get_id("A_val")
sym_col     = runner.get_id("A_col_idx")
sym_row     = runner.get_id("A_row_ptr")
sym_B       = runner.get_id("B")
sym_C_band  = runner.get_id("C_band")
sym_tsc     = runner.get_id("tsc_totals")

# ============================================================
# 5) Start device + H2D CSR
# ============================================================
runner.load()
runner.run()

runner.memcpy_h2d(
    sym_val,
    val_pad.view(np.uint16).view(np.uint32).ravel(),
    0, 0, w, h, MAX_META_LEN // 2,
    streaming=False, data_type=memcpy_dtype_f32,
    order=memcpy_order, nonblock=True,
)

runner.memcpy_h2d(
    sym_col,
    col_idx_pad.view(np.uint16).view(np.uint32).ravel(),
    0, 0, w, h, MAX_META_LEN // 2,
    streaming=False, data_type=memcpy_dtype_f32,
    order=memcpy_order, nonblock=True,
)

runner.memcpy_h2d(
    sym_row,
    row_ptr_pad.view(np.uint16).view(np.uint32).ravel(),
    0, 0, w, h, MAX_PTR_LEN // 2,
    streaming=False, data_type=memcpy_dtype_f32,
    order=memcpy_order, nonblock=True,
)

runner.launch("f_enable_tsc", nonblock=False)

# ============================================================
# 6) Band loop
# ============================================================
C = np.zeros((M, N), dtype=np.float32)

for band in range(w):
    col0 = band * Nt
    col1 = col0 + Nt

    B_band = B[:, col0:col1]
    B3 = np.zeros((h, w, Kt * Nt), dtype=np.float16)

    for px in range(w):
        tile = B_band[px * Kt : (px + 1) * Kt, :]
        B3[0, px] = tile.reshape(-1)

    B16 = B3.reshape(-1).view(np.uint16)
    B32 = (
        (B16[1::2].astype(np.uint32) << 16)
        | B16[0::2].astype(np.uint32)
    )

    runner.memcpy_h2d(
        sym_B,
        B32,
        0, 0, w, h, (Kt * Nt) // 2,
        streaming=False, data_type=memcpy_dtype_u32,
        order=memcpy_order, nonblock=True,
    )

    runner.launch("main", nonblock=False)

    C_band_u32 = np.zeros(h * Mt * Nt, np.uint32)
    runner.memcpy_d2h(
        C_band_u32, sym_C_band,
        w - 1, 0, 1, h, (Mt * Nt),
        streaming=False, data_type=memcpy_dtype_f32,
        order=memcpy_order, nonblock=False,
    )

    C_band = C_band_u32.view(np.float32).reshape(h, Mt, Nt)
    C[:, col0:col1] = C_band.reshape(M, Nt)

# ============================================================
# 7) TSC  
# ============================================================
runner.launch("f_get_timestamps", nonblock=False)
tsc_array = np.zeros(8, dtype=np.uint32)
runner.memcpy_d2h(
    tsc_array, sym_tsc,
    730, 4, 1, 1, 8,
    streaming=False, data_type=memcpy_dtype_f32,
    order=MemcpyOrder.ROW_MAJOR, nonblock=False,
)

runner.stop()

# ============================================================
# Print TSC
# ============================================================
def u64(lo, hi): return int(lo) + (int(hi) << 32)

freq_mhz = 850
freq_hz  = freq_mhz * 1e6

bcast_cycles  = u64(tsc_array[0], tsc_array[1])
comp_cycles   = u64(tsc_array[2], tsc_array[3])
reduce_cycles = u64(tsc_array[4], tsc_array[5])   # [CHANGE] was main_start/end

nnz  = np.count_nonzero(val_pad)
FLOPs = nnz * 2 * N
compute_time_sec = comp_cycles / freq_hz
gflops = FLOPs / compute_time_sec / 1e9

print("\n===== TSC Accumulated Timing =====")
print(f"Broadcast cycles : {bcast_cycles}  ({bcast_cycles/freq_mhz:.3f} us)")
print(f"Compute  cycles  : {comp_cycles}  ({comp_cycles/freq_mhz:.3f} us)")
print(f"Reduce   cycles  : {reduce_cycles}  ({reduce_cycles/freq_mhz:.3f} us)")
print(f"GFLOPS : {gflops:.6f}")

print("SUCCESS")
