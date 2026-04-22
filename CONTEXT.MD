# ADES-v2: Project Context Document
> Last updated: April 2026
> Purpose: Continuity document — upload this to any new chat to restore full project context.
> To resume: read this file, then ask to continue from the last completed step.

---

## 1. Project Overview

**Goal:** Predict the full reliability curve of embedded systems from their architecture graph
using a Graph Neural Network (GNN), enabling real-time system adaptation without
running expensive CTMC simulations.

**Institution:** University / CSC Finland — compute runs on **Puhti** (CSC supercomputer),
managed via **MyCSC** under project `project_2016976`.

**Key value proposition:** CTMC simulation takes hours per configuration. The GNN predicts
in ~0.35ms. The model enables evaluation of unseen configurations at runtime.

**Collaborators:** Manuel Padilla (mpadilla). Previous intake also included Tariq Aziz.

---

## 2. Infrastructure & File Paths

### Puhti Account
```
User         : mpadilla
Project      : project_2016976
Login node   : puhti-login14 or r18c01
```

### Directory Layout
```
/projappl/project_2016976/
├── ADES-v2/                              ← ACTIVE repo (this project)
│   ├── src/
│   │   ├── scripts/                      ← pipeline (00 → 01 → 02 → 03 → 04)
│   │   ├── GAT_LN_HEAD.py                ← best model architecture (carried over)
│   │   ├── GAT_LN.py
│   │   └── GAT.py
│   ├── data/
│   │   └── new_raw -> /scratch/project_2016976/data_2026_manuel/new_raw  ← SYMLINK
│   ├── results/
│   │   └── 00_verify/                    ← EDA outputs already generated
│   ├── logs/
│   ├── .gitignore
│   ├── README.md
│   └── CONTEXT.md                        ← this file
│
└── ADES-reliability-estimation/          ← OLD repo, keep for reference, do not modify

/scratch/project_2016976/
└── data_2026_manuel/
    ├── new_raw/                          ← ONLY dataset that matters
    │   ├── config_all_0_22000_100.csv   ← reliability curves (targets)
    │   └── matrices.zip                 ← graph adjacency files (inputs)
    ├── 3-switches-5-slaves/             ← IGNORE — intake 4 only, different problem
    ├── 5-switches-7-slaves/             ← IGNORE — incomplete, no reliability data
    ├── interim_graph_features.pt        ← OLD, do not use
    ├── processed_compact/               ← OLD, do not use
    └── processed_intake5.pt             ← OLD, do not use
```

### .gitignore (critical — never commit data or model weights)
```
data/
*.pt
*.h5
*.zip
*.csv
results/
logs/
*.png
*.out
__pycache__/
*.pyc
.venv/
*.egg-info/
.idea/
.vscode/
```

---

## 3. Compute Budget (as of April 2026)

| Resource | Used | Remaining | Total |
|---|---|---|---|
| GPU | 28K BU | 122K BU | 150K BU |
| CPU | 1.5K BU | 118K BU | 120K BU |
| Storage | 0 BU | 60K BU | 60K BU |

- V100 GPU on Puhti ≈ 60 BU/hour → ~2,000 GPU-hours remaining
- Only 1 concurrent job allowed on interactive/small partition
- Never run `du -sh $SCRATCH/*` — hangs on Lustre. Use `--max-depth=1`

---

## 4. Raw Data Structure

### CSV — Reliability Curves
```
File    : data/new_raw/config_all_0_22000_100.csv
Rows    : 144,255
Columns : CONFIG, 0, 100, 200, ..., 22000  (221 time columns + 1 ID column)
Index   : CONFIG = "{alloc_id}_{config_num}"  e.g. "0000_0736"
Values  : Reliability ∈ (0, 1], always exactly 1.0 at t=0
Step    : every 100 hours, 0h to 22,000h
NaN     : none
```

### ZIP — Graph Files
```
File    : data/new_raw/matrices.zip
Total   : 144,286 txt files
Matched : 144,255 (have CSV target)
Skipped : 31 files (*_0001 for alloc 0000–0030, suspected data generation bug)

Internal path: {alloc_id}/matrix/config_{config_num}.txt
Example:       0000/matrix/config_0736.txt
```

Each txt file format:
```
# ['N1', 'N1S1', 'N1S2', 'N2', ..., 'S1', 'S1S2', 'T1_1', 'T2_4']
[[0. 0. 1. ...]
 [1. 0. 0. ...]
 ...]
Line 1   : Python list of node names (prefixed with #)
Remaining: Adjacency matrix (floats, bracket-wrapped)
```

---

## 5. Node Naming Convention & Classifier

```
N1, N2, N3 ...          → compute   (slave/processing unit)
S1, S2, S3 ...          → switch
N1S1, N1S2, S1S2 ...   → link      (two device names fused = link between them)
T1_1, T1_2, T2_4 ...   → task      (underscore present; T1 or T2 = task type)
```

**Canonical classifier — single source of truth, use this everywhere:**
```python
def classify_node(name: str) -> str:
    if '_' in name:
        return 'task'
    stripped = ''.join(c for c in name if not c.isdigit())
    if len(stripped) >= 2:
        return 'link'
    if name.startswith('N'):
        return 'compute'
    if name.startswith('S'):
        return 'switch'
    return 'unknown'
```

**Graph structure (fixed across all 144,255 samples):**
```
Total nodes : 30 (always exactly 30, no variation)
  compute   : 6
  switch    : 3
  link      : 15
  task      : 6  (T1 × 3 instances, T2 × 3 instances)
```

Tasks are actual nodes in the graph connected to compute nodes.
Different allocations = different task-to-compute connections = different topology.

---

## 6. EDA Findings — Complete

All findings confirmed by running `src/scripts/00_verify_data.py` on April 2026.

### Reliability Curve Distribution
```
t=0h     : all exactly 1.0
t=1,000h : min=0.9430  mean=0.9770  max=0.9997  (<0.9:  0.0%)
t=2,000h : min=0.8893  mean=0.9543  max=0.9987  (<0.9:  0.2%)
t=4,000h : min=0.7908  mean=0.9103  max=0.9950  (<0.9: 38.3%)
t=8,000h : min=0.6254  mean=0.8270  max=0.9812  (<0.9: 88.3%)
t=12,000h: min=0.4946  mean=0.7503  max=0.9599  (<0.9: 99.1%)
t=22,000h: min=0.2751  mean=0.5854  max=0.8834  (<0.9:100.0%)
```

The "nines" binning scheme (0.9, 0.99, 0.999...) from intake 4/5 is completely
invalid for this data — all samples fall below 0.9 at t=22,000h and 88% fall
below 0.9 by t=8,000h. Classification by nines was only meaningful in the narrow
0–8,500h window used previously, and was measuring a degenerate distribution.

### Curve Uniqueness — Critical Finding
```
Total curves    : 144,255
Unique curves   : 3,336   (2.3% of dataset)
Duplicate groups: 3,137
Redundant copies: 140,919 (97.7% of dataset are exact copies of another curve)
Most duplicated : 3,716 copies of one curve
```

**Confirmed: different graph topologies produce identical reliability curves.**
200 structurally unique adjacency matrices verified to share the same curve in
the top duplicate group. This is physics, not a bug — many topologies achieve
equivalent redundancy/reliability properties through different physical layouts.

**Implication for splitting:** Never split a curve hash group across train/test.
All samples sharing a curve hash must go entirely to one side.

### Curve Crossings — Critical Finding
```
Method: pairwise comparison of 3,336 unique curves
        t=0 excluded (shared anchor), min diff threshold > 0.01
Total pairs checked : 2,835,500
Crossing pairs      : 562,620  (19.84%)

Crossing depth distribution:
  shallow  (0.01–0.05): 56.8%
  moderate (0.05–0.10): 29.7%  ← meaningful for design decisions
  deep     (>0.10)    : 13.4%  ← major behavioral differences

Example deep crossings (real crossover times):
  Curve A vs B: crossover at t=7,200h  (A ends 0.108 higher at t=22,000h)
  Curve A vs C: crossover at t=5,900h
  Curve A vs D: crossover at t=2,400h
```

**Conclusion 1 — static ranking is impossible.** 43.1% of crossing pairs are
moderate or deep. The best configuration depends on operating time horizon.
ML is necessary — a lookup table cannot answer "which is best for my mission?"

**Conclusion 2 — full curve prediction is necessary.** Single timepoint prediction
(e.g. R at t=8,000h) gives wrong rankings for crossing pairs whose crossover
happens before or after that point. The operating lifetime determines the optimal
choice and must be accounted for.

### Allocation Balance & Symmetry
```
Number of allocations : 31
Min samples           : 1,157  (alloc 0000)
Max samples           : 7,372  (alloc 0019)
Imbalance ratio       : 6.4×
```

Symmetry groups — many allocations are statistically identical (permutations):
```
Group A (mean R8k≈0.898): 0000, 0001, 0007, 0008, 0009
Group B (mean R8k≈0.849): 0003, 0021
Group C (mean R8k≈0.848): 0011, 0022
Group D (mean R8k≈0.848): 0013, 0023
Group E (mean R8k≈0.804): 0006, 0028
Group F (mean R8k≈0.804): 0020, 0029
Group G (mean R8k≈0.799): 0002, 0004, 0010, 0014
Group H (mean R8k≈0.799): 0005, 0012, 0015, 0016, 0018
Group I (mean R8k≈0.799): 0017, 0026
Unique  (no match)       : 0024, 0025, 0027, 0030
```
The 31 allocations collapse into ~10 genuinely distinct reliability regimes.

---

## 7. Design Decisions — All Locked In

| Decision | Choice | Evidence |
|---|---|---|
| Prediction task | Regression — full curve (221 points) | Crossings confirmed, nines bins invalid |
| Storage format | HDF5 (.h5) | Lustre-safe, memory-mappable, random access |
| Time range stored | Full 0h–22,000h, never truncate | Crossings occur anywhere in range |
| Splitting strategy | Dual test sets (see below) | Both generalization questions matter |
| Model family | Single GNN — no ensemble | Ensemble approach discarded entirely |
| Old topology folders | Ignored entirely | intake 4 only, different problem |
| Old processed files | Ignored entirely | Wrong time range, wrong features, leakage |

### Splitting Strategy — Dual Test Sets
Two scientific questions are tested independently with one training run:

```
Full dataset (144,255 samples, 3,336 unique curves)
         │
         ├── Allocation Test Set (~18.6%, ~26,791 samples)
         │     Held-out allocations: 0009, 0013, 0025, 0019
         │     One per reliability regime (high/medium/wide/low)
         │     Tests: "Does model generalize to unseen task strategies?"
         │
         └── Remaining (~81.4%)
                  │
                  ├── Config Test Set (~17% of remaining, ~14% overall)
                  │     Curve-hash groups held out, stratified by mean reliability
                  │     Tests: "Does model predict unseen reliability values?"
                  │
                  └── Training Set (~70% overall)
```

**Interpreting the two scores:**
- High config-test, low alloc-test → model memorizes allocation patterns
- High config-test, high alloc-test → model learns true topology→reliability physics
- The gap between the two scores is itself a publishable finding

---

## 8. Pipeline — Current Status

```
src/scripts/
├── 00_verify_data.py   ✅ COMPLETE — run April 2026, outputs in results/00_verify/
├── 01_preprocess.py    ⏳ NEXT — raw zip + csv → dataset.h5 (full fidelity)
├── 02_split.py         ⏳ TODO  — dual split → splits.json (just indices)
├── 03_train.py         ⏳ TODO  — GNN training, lazy HDF5 loading
└── 04_evaluate.py      ⏳ TODO  — evaluate on BOTH test sets, report separately
```

### HDF5 Schema (to be created by 01_preprocess.py)
Each of 144,255 samples stored as a group under its config_id:
```
/0000_0736/
    config_id     : str     "0000_0736"
    allocation    : str     "0000"
    config_num    : str     "0736"
    node_names    : str[]   ["N1", "N1S1", ...]  (preserved for re-featurization)
    node_features : f32[30, F]
    edge_index    : i32[2, E]
    y_curve       : f32[221]  full 0h–22,000h curve
    curve_hash    : str     md5 of y_curve
    adj_hash      : str     md5 of adjacency matrix
```

---

## 9. Model Architecture (carried over from old repo)

**GAT_LN_HEAD** — best previous model, use as baseline:
```
conv1 : GATConv(input_dim, 64, heads=8)  → 512-dim
ln1   : LayerNorm(512)
relu
conv2 : GATConv(512, 32, heads=8)        → 256-dim
ln2   : LayerNorm(256)
drop  : Dropout(0.3)
pool  : global_mean_pool
fc    : Linear(256, output_dim)
```
For full curve regression: output_dim=221, loss=MSE.
For single-point baseline: output_dim=1, target=R(8000h), use first.

Previous results used classification + random split → not a valid comparison.

---

## 10. Known Issues from Old Repo — Never Repeat

1. Nines binning applied where all values < 0.9 — degenerate classification
2. Time range truncated to 8,500h — misses crossings in the 8,500–22,000h range
3. Random splitting — 97.7% curve redundancy means massive leakage
4. Two incompatible preprocessing outputs existed simultaneously (5-feat vs 12-feat)
5. Feature index 3 hardcoded to 0.0 as timestamp placeholder, never filled
6. Node classifier used fragile else-catch-all silently misclassifying nodes
7. betweenness_centrality O(VE) caused slow preprocessing
8. du -sh $SCRATCH/* hangs on Puhti Lustre — always use --max-depth=1
9. 97.7% curve redundancy was unknown and unhandled throughout