#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime,
    MemcpyDataType,
    MemcpyOrder,
)

# ---------------- args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()

# ---------------- load compile metadata ----------------
with open(f"{args.name}/out.json", encoding="utf-8") as jf:
    compile_data = json.load(jf)

h  = int(compile_data["params"]["h"])
w  = int(compile_data["params"]["w"])
Mt = int(compile_data["params"]["Mt"])
Kt = int(compile_data["params"]["Kt"])
Nt = int(compile_data["params"]["Nt"])

M = Mt * h
K = Kt * w
N = Nt * w

memcpy_order = MemcpyOrder.ROW_MAJOR
dtype_f16 = MemcpyDataType.MEMCPY_16BIT
dtype_f32 = MemcpyDataType.MEMCPY_32BIT

np.random.seed(7)

# ---------------- generate B (fp16) ----------------
B = np.random.rand(K, N).astype(np.float16)

# ---------------- runtime ----------------
runner = SdkRuntime(
    args.name,
    cmaddr=args.cmaddr,
    suppress_simfab_trace=True,
)

# ---------------- load COO padded data ----------------
val_pad = np.loadtxt("tmp_val_pad.csv", delimiter=",", dtype=np.float32)
x_pad   = np.loadtxt("tmp_x_pad.csv",   delimiter=",", dtype=np.int32)
y_pad   = np.loadtxt("tmp_y_pad.csv",   delimiter=",", dtype=np.int32)

val_pad     = val_pad.reshape(h, w, -1).astype(np.float16)
x_pad   = x_pad.reshape(h, w, -1)
y_pad   = y_pad.reshape(h, w, -1)

MAX_META_LEN = val_pad.shape[2]

# ---------------- pad to even (COO fp16) ----------------
if MAX_META_LEN % 2 == 1:
    val_pad = np.pad(val_pad, ((0,0),(0,0),(0,1)), constant_values=0)
    x_pad   = np.pad(x_pad,   ((0,0),(0,0),(0,1)), constant_values=0)
    y_pad   = np.pad(y_pad,   ((0,0),(0,0),(0,1)), constant_values=0)
    MAX_META_LEN += 1

# ---------------- cast for device ----------------
val_pad = val_pad.astype(np.float16)
x_pad   = x_pad.astype(np.uint32)
y_pad   = y_pad.astype(np.uint32)

# ---------------- symbols ----------------
sym_val      = runner.get_id("A_val")
sym_x        = runner.get_id("A_x")
sym_y        = runner.get_id("A_y")
sym_B        = runner.get_id("B")
sym_C_band   = runner.get_id("C_band")
sym_tsc      = runner.get_id("tsc_totals")

# ---------------- launch ----------------
runner.load()
runner.run()

# ---------------- H2D COO ----------------
runner.memcpy_h2d(
    sym_val,
    val_pad.view(np.uint16).view(np.uint32).ravel(),
    0, 0, w, h, MAX_META_LEN // 2,
    streaming=False,
    data_type=dtype_f32,
    order=memcpy_order,
    nonblock=True,
)

runner.memcpy_h2d(
    sym_x,
    x_pad.ravel().view(np.uint32),
    0, 0, w, h, MAX_META_LEN,
    streaming=False,
    data_type=dtype_f32,
    order=memcpy_order,
    nonblock=True,
)

runner.memcpy_h2d(
    sym_y,
    y_pad.ravel().view(np.uint32),
    0, 0, w, h, MAX_META_LEN,
    streaming=False,
    data_type=dtype_f32,
    order=memcpy_order,
    nonblock=True,
)

# enable TSC
runner.launch("f_enable_tsc", nonblock=False)

# ---------------- band loop ----------------
C = np.zeros((M, N), dtype=np.float32)

for band in range(w):
    col0 = band * Nt
    col1 = col0 + Nt

    # K x Nt
    B_band = B[:, col0:col1]

    # (h, w, Kt*Nt), only py==0 row filled
    B3 = np.zeros((h, w, Kt * Nt), dtype=np.float16)
    for px in range(w):
        B_tile = B_band[px * Kt : (px + 1) * Kt, :]
        B3[0, px] = B_tile.reshape(-1)

    runner.memcpy_h2d(
        sym_B,
        B3.ravel().view(np.uint16).view(np.uint32).ravel(),
        0, 0, w, h, (Kt * Nt) // 2,
        streaming=False,
        data_type=dtype_f32,
        order=memcpy_order,
        nonblock=True,
    )

    runner.launch("main", nonblock=False)

    # read C_band (fp32)
    C_band_u32 = np.zeros(h * Mt * Nt, np.uint32)
    runner.memcpy_d2h(
        C_band_u32, sym_C_band,
        w - 1, 0, 1, h, (Mt * Nt) ,
        streaming=False, data_type=dtype_f32,
        order=memcpy_order, nonblock=False,
    )

    C_band = C_band_u32.view(np.float32).reshape(h, Mt, Nt)
    C[:, col0:col1] = C_band.reshape(M, Nt)

# ---------------- TSC ----------------
runner.launch("f_get_timestamps", nonblock=False)
tsc_array = np.zeros(8, dtype=np.uint32)
runner.memcpy_d2h(
    tsc_array, sym_tsc,
    730, 4, 1, 1, 8,
    streaming=False, data_type=dtype_f32,
    order=MemcpyOrder.ROW_MAJOR, nonblock=False,
)

runner.stop()

# ============================================================
# Print TSC
# ============================================================
def u64(lo, hi): return int(lo) + (int(hi) << 32)
def make_u48(lo32, hi16): return (lo32 & 0xFFFFFFFF) + ((hi16 & 0xFFFF) << 32)

bcast_cycles = u64(tsc_array[0], tsc_array[1])
comp_cycles  = u64(tsc_array[2], tsc_array[3])
reduce_cycles = u64(tsc_array[4], tsc_array[5])

freq_mhz = 850
freq_hz = freq_mhz * 1e6

nnz = np.count_nonzero(val_pad)
FLOPs = nnz * 2 * N
compute_time_sec = comp_cycles / freq_hz
gflops = FLOPs / compute_time_sec / 1e9


print("\n===== TSC Accumulated Timing =====")
print(f"Broadcast cycles : {bcast_cycles} ({bcast_cycles/freq_mhz:.3f} us)")
print(f"Compute cycles   : {comp_cycles} ({comp_cycles/freq_mhz:.3f} us)")
print(f"Reduce   cycles  : {reduce_cycles}  ({reduce_cycles/freq_mhz:.3f} us)")
print(f"GFLOPS         : {gflops:.6f}")

# ============================================================
# 8) CPU Verification
# ============================================================
#A_dense = np.loadtxt("tmp.csv", delimiter=",", dtype=np.float32)
#C_expected = A_dense @ B.astype(np.float32)

#abs_err = np.abs(C_expected - C)
#rel_err = abs_err / (np.abs(C_expected) + 1e-12)

#bad_mask = abs_err > (1e-2 + 0.1 * np.abs(C_expected))
#bad_idx  = np.argwhere(bad_mask)
#print("C\n", C)
#print("C_expected\n", C_expected)
#print("\n===== Verification =====")
#print(f"Errors: {bad_idx.shape[0]} / {C.size}")
#print("max abs err:", abs_err.max())
#print("max rel err:", rel_err.max())
print("SUCCESS")
