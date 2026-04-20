#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime,
    MemcpyDataType,
    MemcpyOrder,
)

# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--cmaddr")
args = parser.parse_args()

# ------------------------------------------------------------
# Load compile metadata
# ------------------------------------------------------------
with open(f"{args.name}/out.json", encoding="utf-8") as jf:
    compile_data = json.load(jf)

h  = int(compile_data["params"]["h"])
w  = int(compile_data["params"]["w"])
Mt = int(compile_data["params"]["Mt"])
Kt = int(compile_data["params"]["Kt"])
Nt = int(compile_data["params"]["Nt"])

MAX_NUM_DIAGS   = int(compile_data["params"]["MAX_NUM_DIAGS"])
MAX_VALS_PER_PX = int(compile_data["params"]["MAX_VALS_PER_PX"])

assert MAX_VALS_PER_PX % 2 == 0, "MAX_VALS_PER_PX must be even (fp16 packing)"

M = Mt * h
K = Kt * w
N = Nt * w

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

np.random.seed(7)

# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------
runner = SdkRuntime(
    args.name,
    cmaddr=args.cmaddr,
    suppress_simfab_trace=True,
)

# ------------------------------------------------------------
# Symbols
# ------------------------------------------------------------
sym_A_diag_offset = runner.get_id("A_diag_offset")
sym_A_row_start   = runner.get_id("A_row_start")
sym_A_len         = runner.get_id("A_len")
sym_A_diag_ptr    = runner.get_id("A_diag_ptr")
sym_A_vals        = runner.get_id("A_vals")
sym_num_diags     = runner.get_id("num_diags")
sym_vals_len      = runner.get_id("vals_len")

sym_B      = runner.get_id("B")
sym_C_band = runner.get_id("C_band")
sym_tsc    = runner.get_id("tsc_totals")

# ------------------------------------------------------------
# Load device
# ------------------------------------------------------------
runner.load()
runner.run()

# ------------------------------------------------------------
# Load DIA CSV
# ------------------------------------------------------------
offs_pad = np.loadtxt("A_diag_offset.csv", delimiter=",", dtype=np.int32)
rsta_pad = np.loadtxt("A_row_start.csv",   delimiter=",", dtype=np.uint32)
len_pad  = np.loadtxt("A_len.csv",         delimiter=",", dtype=np.uint32)
ptr_pad  = np.loadtxt("A_diag_ptr.csv",    delimiter=",", dtype=np.uint32)
vals_pad = np.loadtxt("A_vals.csv",        delimiter=",", dtype=np.float32)

numd_arr = np.loadtxt("num_diags.csv", delimiter=",", dtype=np.uint32)
vlen_arr = np.loadtxt("vals_len.csv",  delimiter=",", dtype=np.uint32)

offs_pad = offs_pad.reshape(h, w, MAX_NUM_DIAGS)
rsta_pad = rsta_pad.reshape(h, w, MAX_NUM_DIAGS)
len_pad  = len_pad.reshape (h, w, MAX_NUM_DIAGS)
ptr_pad  = ptr_pad.reshape (h, w, MAX_NUM_DIAGS + 1)
vals_pad = vals_pad.reshape(h, w, MAX_VALS_PER_PX)
numd_arr = numd_arr.reshape(h, w, 1)
vlen_arr = vlen_arr.reshape(h, w, 1)

# ------------------------------------------------------------
# H2D: DIA metadata
# ------------------------------------------------------------
runner.memcpy_h2d(sym_A_diag_offset, offs_pad.view(np.uint32).ravel(),
                  0, 0, w, h, MAX_NUM_DIAGS,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=True)

runner.memcpy_h2d(sym_A_row_start, rsta_pad.view(np.uint32).ravel(),
                  0, 0, w, h, MAX_NUM_DIAGS,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=True)

runner.memcpy_h2d(sym_A_len, len_pad.view(np.uint32).ravel(),
                  0, 0, w, h, MAX_NUM_DIAGS,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=True)

runner.memcpy_h2d(sym_A_diag_ptr, ptr_pad.view(np.uint32).ravel(),
                  0, 0, w, h, MAX_NUM_DIAGS + 1,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=True)

# ------------------------------------------------------------
# H2D: A_vals (fp16 → u32 packing)
# ------------------------------------------------------------
vals16 = vals_pad.astype(np.float16)
flat16 = vals16.reshape(-1).view(np.uint16)

v32 = (flat16[1::2].astype(np.uint32) << 16) | flat16[0::2].astype(np.uint32)

runner.memcpy_h2d(sym_A_vals, v32,
                  0, 0, w, h, MAX_VALS_PER_PX // 2,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=True)

runner.memcpy_h2d(sym_num_diags, numd_arr.view(np.uint32).ravel(),
                  0, 0, w, h, 1,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=True)

runner.memcpy_h2d(sym_vals_len, vlen_arr.view(np.uint32).ravel(),
                  0, 0, w, h, 1,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=True)

# ------------------------------------------------------------
# Enable TSC
# ------------------------------------------------------------
runner.launch("f_enable_tsc", nonblock=False)

# ------------------------------------------------------------
# Generate B
# ------------------------------------------------------------
B = np.random.rand(K, N).astype(np.float32)
C = np.zeros((M, N), dtype=np.float32)

# ------------------------------------------------------------
# Band loop
# ------------------------------------------------------------
for band in range(w):

    col0 = band * Nt
    col1 = col0 + Nt

    B_band = B[:, col0:col1]  # (K, Nt)

    B3 = np.zeros((h, w, Kt * Nt), dtype=np.float16)

    for px in range(w):
        tile = B_band[px * Kt:(px + 1) * Kt, :]
        for py in range(h):
            B3[py, px] = tile.reshape(-1).astype(np.float16)

    runner.memcpy_h2d(
        sym_B,
        B3.view(np.uint32).reshape(-1),
        0, 0, w, h,
        (Kt * Nt) // 2,
        streaming=False,
        data_type=memcpy_dtype,
        order=memcpy_order,
        nonblock=True,
    )

    runner.launch("main", nonblock=False)

    C_band_u32 = np.zeros(h * Mt * Nt, np.uint32)

    runner.memcpy_d2h(
        C_band_u32, sym_C_band,
        w - 1, 0, 1, h, (Mt * Nt),
        streaming=False,
        data_type=memcpy_dtype,
        order=memcpy_order,
        nonblock=False,
    )

    C_band = C_band_u32.view(np.float32).reshape(h, Mt, Nt)
    C[:, col0:col1] += C_band.reshape(M, Nt)

# ------------------------------------------------------------
# Read TSC
# ------------------------------------------------------------
runner.launch("f_get_timestamps", nonblock=False)

tsc_array = np.zeros(8, dtype=np.uint32)
runner.memcpy_d2h(tsc_array, sym_tsc,
                  730, 4, 1, 1, 8,
                  streaming=False,
                  data_type=memcpy_dtype,
                  order=MemcpyOrder.ROW_MAJOR,
                  nonblock=False)

runner.stop()

# ------------------------------------------------------------
# Timing
# ------------------------------------------------------------
def u64(lo, hi): return int(lo) + (int(hi) << 32)
def make_u48(lo32, hi16): return (lo32 & 0xFFFFFFFF) + ((hi16 & 0xFFFF) << 32)

bcast_cycles = u64(tsc_array[0], tsc_array[1])
comp_cycles  = u64(tsc_array[2], tsc_array[3])

nnz = int(np.sum(vlen_arr))
freq_mhz = 850
freq_hz = freq_mhz * 1e6

FLOPs = nnz * 2 * N
compute_time_sec = comp_cycles / freq_hz
gflops = FLOPs / compute_time_sec / 1e9


print("\n===== TSC Accumulated Timing =====")
print(f"Broadcast cycles : {bcast_cycles} ({bcast_cycles/freq_mhz:.3f} us)")
print(f"Compute cycles   : {comp_cycles} ({comp_cycles/freq_mhz:.3f} us)")
print(f"GFLOPS         : {gflops:.6f}")
print("SUCCESS")
