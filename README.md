![CereSpMM Logo](imgs/CereSpMM_logo.png)

# CereSpMM
CereSpMM: Accelerating General Sparse Matrix-Matrix Multiplication on Cerebras CS-3

## Overview

CereSpMM is a unified SpMM framework designed for the Cerebras CS-3 wafer-scale processor. It introduces a novel Stationary-A Broadcast-B (SA-BB) computation method and three format-specific SpMM algorithms (BCSR, BCOO, BDIA), combined with a lightweight decision tree-based algorithm selection module that automatically selects the most suitable algorithm for a given sparse matrix.

## Project Structure

```
CereSpMM/
├── imgs/                        
│   └── CereSpMM_logo.png
│
├── datasets/                    
│
├── csr_fp16/                   
│   ├── pe.csl              
│   ├── layout.csl           
│   ├── run.py                   
│   └── appliance_compile.py
│   └── add_padding.py     
│   ├── commands_wse2.sh                  
│   └── launcher_run.py
│   └── format.c    
│   └── README.md 
│
├── coo_fp16/                    
│   ├── pe.csl              
│   ├── layout.csl           
│   ├── run.py                   
│   └── appliance_compile.py
│   └── add_padding.py     
│   ├── commands_wse2.sh                  
│   └── launcher_run.py
│   └── format.c 
│   └── README.md 
│   
├── block_dia/                  
│   ├── pe.csl              
│   ├── layout.csl           
│   ├── run.py                   
│   └── appliance_compile.py
│   └── launcher_run.py     
│   ├── mm_to_dia.cpp                  
│   └── analyze_dia_padding.py
│   └── README.md 
│
├── cerespmm_selector.py                   
│
├── extract_features.py
│
├── requirements.txt
│
└── README.md
```

## Requirements

### Hardware
- Cerebras CS-3 system with Cerebras appliance network access

### Software
- Cerebras SDK (`cs_appliance_sdk`) v2.9.0
- Python 3.11
- NumPy == 1.25.0
- GCC (for compiling `format.c`)
- G++ (for compiling `mm_to_dia.cpp`)

## Installation

```bash
git clone https://github.com/AnonymousSC2026/CereSpMM.git
cd CereSpMM
```

Compile the preprocessing tools:

```bash
cd preprocessing
gcc -O2 -o format format.c
g++ -O2 -o mm_to_dia mm_to_dia.cpp
```

## Usage

### Step 1: Prepare Input Matrix

CereSpMM accepts sparse matrices in [Matrix Market (.mtx)](https://math.nist.gov/MatrixMarket/formats.html) format. Real-world matrices can be downloaded from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/).

### Step 2: Format Conversion

**BCSR / BCOO:**
```bash
./format <matrix.mtx> 1170 755 1   # BCSR
./format <matrix.mtx> 1170 755 2   # BCOO
```

**BDIA** (first analyze tiling parameters, then convert):
```bash
python analyze_dia_padding.py <matrix.mtx>
./mm_to_dia <matrix.mtx> <Mt> <Kt> <MAX_NUM_DIAGS> <MAX_VALS_PER_PX>
```

### Step 3: Algorithm Selection

Extract sparse matrix features and predict the best algorithm:

```bash
python algorithm_selection/predict.py --matrix <matrix.mtx>
```

### Step 4: Compile and Run

```bash
# Compile the selected kernel (e.g., BCSR)
cd csr_fp16
python appliance_compile.py

# Run on CS-3
python launcher_run.py
```

The output reports broadcast cycles, compute cycles, reduce cycles, and GFLOPS throughput:

```
===== TSC Accumulated Timing =====
Broadcast cycles : XXXXXX  (XXX.XXX us)
Compute  cycles  : XXXXXX  (XXX.XXX us)
Reduce   cycles  : XXXXXX  (XXX.XXX us)
GFLOPS (per band): XXXXXX
```

## Datasets

### HPC Matrices
Sparse matrices are sourced from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/).

### Sparse Attention Matrices (LRA)
Sparse attention masks are generated following the task specifications of the [Long Range Arena (LRA) benchmark](https://github.com/google-research/long-range-arena):

```bash
python datasets/lra_to_mtx_all.py --masks local strided bigbird block
```

### GNN Matrices
- **Cora**: downloaded from [LINQS](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz)
- **Amazon Photo / Computer**: available via [DGL](https://www.dgl.ai/) dataset loader

```python
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset
```

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.
