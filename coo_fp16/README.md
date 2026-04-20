# BCOO-based SpMM Kernel

This directory contains the Block Coordinate (BCOO) format-specific SpMM kernel for Cerebras CS-3. BCOO stores each non-zero entry explicitly as a coordinate-value tuple, providing maximum flexibility for irregular sparsity patterns. It is best suited for matrices with highly irregular or random non-zero distributions, such as GNN adjacency matrices and unstructured attention patterns.

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

## Step 1: Convert Matrix to BCOO Format

```bash
./format <matrix.mtx> 1170 755 2
```

Example:

```bash
./format amazon_computer.mtx 1170 755 2
```

Example output:

```
[INFO] Matrix 13752 x 13752 padded to 14040 x 13590
[INFO] PE grid 1170 x 755, block size 12 x 18
[INFO] NNZ in file    : 491722
[INFO] Expanded NNZ (actual stored nnz): 491722
[DONE] BCOO generated (supports all common mtx kinds)
```

This produces the following CSV files:

| File | Description |
|---|---|
| `tmp_x.csv` | Row coordinates of non-zero elements per PE tile |
| `tmp_y.csv` | Column coordinates of non-zero elements per PE tile |
| `tmp_val.csv` | Non-zero values |

---

## Step 2: Add Padding

```bash
python add_padding.py 2
```

This pads the COO arrays to uniform length across all PE tiles, which is required for host-to-device transfer on Cerebras CS-3. The output files are:

| File | Description |
|---|---|
| `tmp_x_pad.csv` | Padded row coordinate arrays |
| `tmp_y_pad.csv` | Padded column coordinate arrays |
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

- BCOO introduces higher indexing overhead compared to BCSR and BDIA. For matrices with structured sparsity (e.g., diagonal or block patterns), consider using BDIA or BCSR instead.
- Use `../preprocessing/check_pe_dist.py` to inspect the NNZ distribution across PE tiles before running:

```bash
python ../preprocessing/check_pe_dist.py 2
```

- Use `../preprocessing/pe_query.py` to query the workload of a specific PE:

```bash
python ../preprocessing/pe_query.py <px> <py> 2
```
