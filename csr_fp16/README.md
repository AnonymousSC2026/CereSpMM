# BCSR-based SpMM Kernel

This directory contains the Block Compressed Sparse Row (BCSR) format-specific SpMM kernel for Cerebras CS-3. BCSR organizes non-zero elements into block rows, improving data locality and reducing indexing overhead. It is best suited for matrices with block-structured sparsity or locally dense submatrices.

---

## Prerequisites

- Cerebras SDK (`cs_appliance_sdk`) installed and activated
- `format` compiled from `../preprocessing/format.c`
- Input matrix in Matrix Market (`.mtx`) format

Compile the converter if not already done:

```bash
cd ../preprocessing
gcc -O2 -o format format.c
```

---

## Step 1: Convert Matrix to BCSR Format

```bash
./format <matrix.mtx> 1170 755 1
```

Example:

```bash
./format ca-HepTh.mtx 1170 755 1
```

Example output:

```
[INFO] Matrix 9877 x 9877 padded to 10530 x 10570
[INFO] PE grid 1170 x 755, block size 9 x 14
[INFO] NNZ in file    : 51971
[INFO] Expanded NNZ (actual stored nnz): 51971
[DONE] BCSR generated (supports all common mtx kinds)
```

This produces the following CSV files:

| File | Description |
|---|---|
| `tmp_row_ptr.csv` | Row pointers for each PE tile |
| `tmp_col_idx.csv` | Column indices of non-zero elements |
| `tmp_val.csv` | Non-zero values |

---

## Step 2: Add Padding

```bash
python add_padding.py 1
```

This pads the CSR arrays to uniform length across all PE tiles, which is required for host-to-device transfer on Cerebras CS-3. The output files are:

| File | Description |
|---|---|
| `tmp_row_ptr_pad.csv` | Padded row pointer arrays |
| `tmp_col_idx_pad.csv` | Padded column index arrays |
| `tmp_val_pad.csv` | Padded value arrays |

---

## Step 3: Compile the Kernel

```bash
python appliance_compile.py
```

This compiles the CSL kernel (`csl/pe.csl` and `csl/layout.csl`) using the Cerebras SDK compiler. Compilation typically takes 3--5 minutes.

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

- BCSR performs best on matrices with block-structured sparsity or high diagonal padding ratio. For matrices with highly irregular sparsity, consider using BCOO instead.
- Use `../preprocessing/check_pe_dist.py` to inspect the NNZ distribution across PE tiles before running:

```bash
python ../preprocessing/check_pe_dist.py 1
```

- Use `../preprocessing/pe_query.py` to query the workload of a specific PE:

```bash
python ../preprocessing/pe_query.py <px> <py> 1
```
