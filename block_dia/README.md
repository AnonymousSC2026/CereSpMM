# BDIA-based SpMM Kernel

This directory contains the Block Diagonal (BDIA) format-specific SpMM kernel for Cerebras CS-3. BDIA organizes non-zero elements along diagonal segments, enabling streaming memory access without indirect indexing. It is best suited for matrices with banded or diagonal-dominant sparsity patterns.

---

## Prerequisites

- Cerebras SDK (`cs_appliance_sdk`) installed and activated
- `mm_to_dia` compiled from `../preprocessing/mm_to_dia.cpp`
- Input matrix in Matrix Market (`.mtx`) format

Compile the converter if not already done:

```bash
cd ../preprocessing
g++ -O2 -o mm_to_dia mm_to_dia.cpp
```

---

## Step 1: Analyze Tiling Parameters

Before converting the matrix, run the analyzer to determine the required tiling parameters:

```bash
python analyze_dia_padding.py <matrix.mtx>
```

Example output:

```
[MM] shape=(5620,5620), nnz=79650
---------------------------------------------------
Auto tiling (with padding)
  Mt = 5, Kt = 8
  M_pad = 5850, K_pad = 6040
---------------------------------------------------
DIA analysis result
---------------------------------------------------
PE grid            : h=1170, w=755 (tiles=883350)
Empty tiles        : 876244
MAX_NUM_DIAGS      : 12
MAX_VALS_PER_PX    : 40
---------------------------------------------------
Suggested compile params:
  Mt=5, Kt=8
  MAX_NUM_DIAGS=12
  MAX_VALS_PER_PX=40
---------------------------------------------------
```

Note down the values of `Mt`, `Kt`, `MAX_NUM_DIAGS`, and `MAX_VALS_PER_PX`.

---

## Step 2: Convert Matrix to BDIA Format

Run `mm_to_dia` with the parameters obtained in Step 1:

```bash
./mm_to_dia <matrix.mtx> <Mt> <Kt> <MAX_NUM_DIAGS> <MAX_VALS_PER_PX>
```

Example:

```bash
./mm_to_dia optdigits_10NN.mtx 5 8 12 40
```

This produces the following CSV files used as kernel inputs:

| File | Description |
|---|---|
| `A_diag_offset.csv` | Diagonal offset for each segment |
| `A_row_start.csv` | Starting row index of each segment |
| `A_len.csv` | Length of each diagonal segment |
| `A_diag_ptr.csv` | Pointer into value array for each segment |
| `A_vals.csv` | Non-zero values along diagonals |
| `num_diags.csv` | Number of active diagonal segments per PE |
| `vals_len.csv` | Number of values per PE |
| `has_work.csv` | Whether each PE has any non-zero work |

---

## Step 3: Compile the Kernel

```bash
python appliance_compile.py
```

This compiles the CSL kernel (`csl/pe.csl` and `csl/layout.csl`) using the Cerebras SDK compiler with the parameters obtained from Step 1. Compilation typically takes 3--5 minutes.

---

## Step 4: Run on Cerebras CS-3

```bash
python launcher_run.py --name <artifact_dir> --cmaddr <CS3_IP:port>
```

Example:

```bash
python launcher_run.py --name out --cmaddr 192.168.1.1:9000
```

The output reports timing from hardware timestamp counters (TSC):

```
===== TSC Accumulated Timing =====
Broadcast cycles : XXXXXX  (XXX.XXX us)
Compute  cycles  : XXXXXX  (XXX.XXX us)
Reduce   cycles  : XXXXXX  (XXX.XXX us)
GFLOPS (per band): XXXXXX
```

---

## Notes

- BDIA performance degrades for matrices with highly irregular sparsity or very short diagonal segments. For such matrices, consider using BCSR or BCOO instead.
- The `has_work.csv` file can be used to identify non-empty PEs for targeted performance analysis. See `../preprocessing/check_pe_dist.py` for PE workload distribution utilities.
- If `MAX_NUM_DIAGS` or `MAX_VALS_PER_PX` is set too small, some non-zero segments may be silently dropped. Always use the values reported by `analyze_dia_padding.py`.