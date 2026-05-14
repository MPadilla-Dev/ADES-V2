# ADES-v2: Project Context Document
> Last updated: May 2026
> Purpose: Continuity document — upload this to any new chat to restore full project context.
> To resume: read this file, then ask to continue from the last completed step.

---

## 1. Project Overview

**Goal:** Predict the full reliability curve of embedded systems from their architecture graph
using a Graph Neural Network (GNN), enabling real-time system adaptation without
running expensive CTMC simulations.

**Institution:** University / CSC Finland — Puhti supercomputer, project `project_2016976`.
**User:** mpadilla. Previous intake also included Tariq Aziz.

---

## 2. Infrastructure & File Paths

```
/projappl/project_2016976/
├── ADES-v2/                          ← ACTIVE repo
│   ├── src/scripts/                  ← pipeline scripts 00–05
│   ├── data/
│   │   ├── new_raw -> /scratch/.../data_2026_manuel/new_raw  (symlink)
│   │   ├── dataset.h5                ← preprocessed HDF5 (on scratch via symlink)
│   │   └── splits.json               ← train/val/test indices for all 4 splits
│   ├── models/                       ← 7 trained model checkpoints (.pth)
│   ├── results/
│   │   ├── 02_eda_deep/              ← EDA plots
│   │   └── 05_evaluate/              ← evaluation results, plots, summary
│   ├── logs/                         ← job outputs
│   └── .venv/                        ← Python venv (pytorch/2.4 base)
│
└── ADES-reliability-estimation/      ← OLD repo, do not modify

/scratch/project_2016976/data_2026_manuel/new_raw/
├── config_all_0_22000_100.csv        ← reliability curves (targets)
└── matrices.zip                      ← graph adjacency files (inputs)
```

**Environment activation (always required before running scripts):**
```bash
module load pytorch/2.4
source /projappl/project_2016976/ADES-v2/.venv/bin/activate
```

---

## 3. Raw Data Structure

```
144,255 samples, 30 nodes each (fixed), 221 time points (0h–22,000h, step 100h)

Node types:
  6 compute (N1–N6)    — each connects to 0, 1, or 2 switches
  3 switch  (S1–S3)
  15 link   (N1S1...)  — encode endpoints in name
  6 task    (T1_1, T2_4...) — connect to compute nodes; vary by allocation

31 allocations (task-to-compute assignments), 8 functional symmetry groups
11,903 unique hardware wirings (MD5, task edges stripped)
392 unique hardware topology classes (WL hash, VF2-proven exact)
3,336 unique reliability curves
```

**Node classifier (canonical, use everywhere):**
```python
def classify_node(name):
    if '_' in name: return 'task'
    stripped = ''.join(c for c in name if not c.isdigit())
    if len(stripped) >= 2: return 'link'
    if name.startswith('N'): return 'compute'
    if name.startswith('S'): return 'switch'
    return 'unknown'
```

---

## 4. Key EDA Findings

| Finding | Value |
|---|---|
| Curve redundancy | 97.7% — only 3,336 unique curves from 144,255 samples |
| Task allocation variance | Explains 73% of overall reliability variance |
| Hardware variance | Explains only 27% |
| Curve crossing rate | 13.03% of unique curve pairs cross |
| Moderate/deep crossings | 38% — static ranking impossible |
| WL isomorphism | 392 groups, VF2-proven on 356,589 pairs, 0 collisions |
| WL vs MD5 curve diff | Within one WL group same allocation can produce different curves (e.g. N6 connecting to 2 vs 1 switch when N6 hosts all tasks) |

**Open domain expert question:** Do all compute nodes have identical failure rates in the CTMC? If yes, WL is the correct hardware identity. Empirical evidence (5% gap between WL and MD5 splits) suggests approximately yes.

**Nines classification is invalid** — all samples fall below 0.9 by t=22,000h.

---

## 5. HDF5 Schema

```
meta/config_ids      [N] str   — "0000_0736"
meta/allocations     [N] str   — "0000"
meta/config_nums     [N] str   — "0736"
meta/curve_hashes    [N] str   — md5 of y_curve (for curve-hash split)
meta/adj_hashes      [N] str   — md5 of full adjacency matrix
meta/hw_md5_hashes   [N] str   — md5 of hardware-only edges (11,903 unique)
meta/hw_wl_hashes    [N] str   — WL hash of hardware subgraph (392 unique)
meta/node_names      [N,30] str
features/node_features [N,30,5] f32  — one-hot [compute,switch,link,T1,T2]
edges/edge_index     [2,E_tot] i32   — CSR format
edges/edge_ptr       [N+1] i64
targets/y_curve      [N,221] f32    — full 0h–22,000h reliability curve
```

---

## 6. Splits (splits.json)

| Split | Groups | Train | Val | Test | Tests |
|---|---|---|---|---|---|
| curve_hash | 3,336 | 69.9% | 14.9% | 15.2% | Unseen reliability values |
| allocation | 31 | 76.6% | 4.8% | 18.6% | Unseen task strategies |
| hw_md5 | 11,903 | 68.2% | 14.4% | 17.4% | Unseen physical wirings |
| hw_wl | 392 | 69.2% | 13.5% | 17.3% | Unseen topology classes |

Allocation test: 0009, 0013, 0025, 0019 (one per reliability regime)
Allocation val:  0001, 0008

---

## 7. Model Architecture

**GAT_LN_HEAD** — 138,241 parameters
```
GATConv(input_dim, 64, heads=8) → LayerNorm(512) → ReLU
GATConv(512, 32, heads=8)       → LayerNorm(256) → Dropout(0.3)
global_mean_pool → Linear(256, output_dim)

input_dim  = 6 for time_conditioned (5 one-hot + t_norm)
           = 5 for full_curve / single_timestep
output_dim = 1   for time_conditioned / single_timestep
           = 221 for full_curve
```

Training: Adam lr=0.0005, ReduceLROnPlateau (factor=0.5, patience=10),
early stopping patience=20, batch_size=64, max_epochs=300.

---

## 8. Trained Models (all in models/)

| File | Split | Formulation | Best Val MSE |
|---|---|---|---|
| curve_hash_time_conditioned_best.pth | curve_hash | time_conditioned | 0.001389 (ep 84) |
| curve_hash_full_curve_best.pth | curve_hash | full_curve | 0.001394 |
| curve_hash_single_timestep_best.pth | curve_hash | single_timestep | 0.001003 |
| allocation_time_conditioned_best.pth | allocation | time_conditioned | 0.002122 |
| allocation_full_curve_best.pth | allocation | full_curve | 0.002134 |
| hw_md5_time_conditioned_best.pth | hw_md5 | time_conditioned | 0.001598 |
| hw_wl_time_conditioned_best.pth | hw_wl | time_conditioned | 0.001674 |

---

## 9. Evaluation Results

### Standard Metrics at t=8,000h

| Experiment | MSE | MAE | R² |
|---|---|---|---|
| curve_hash_time_conditioned | 0.001238 | 0.02681 | 0.6504 |
| curve_hash_full_curve | 0.001202 | 0.02631 | 0.6606 |
| curve_hash_single_timestep | 0.001212 | 0.02667 | 0.6578 |
| allocation_time_conditioned | 0.001914 | 0.03209 | 0.4495 |
| allocation_full_curve | 0.001889 | 0.03270 | 0.4568 |
| hw_md5_time_conditioned | 0.001149 | 0.02589 | 0.6661 |
| hw_wl_time_conditioned | 0.001207 | 0.02629 | 0.6564 |

### Crossing Accuracy (% of crossing pairs correctly ranked)

| Experiment | 2k | 4k | 8k | 12k | 16k | 22k |
|---|---|---|---|---|---|---|
| curve_hash_time_conditioned | 26.3% | 34.9% | 43.7% | 50.8% | 67.9% | 89.1% |
| curve_hash_full_curve | 50.9% | 45.5% | 46.9% | 51.5% | 68.3% | 88.5% |
| curve_hash_single_timestep | 0% | 0% | 41.9% | 0% | 0% | 0% |
| allocation_time_conditioned | 72.0% | 73.7% | 79.1% | 77.7% | 69.6% | 59.5% |
| allocation_full_curve | 72.4% | 70.5% | 75.1% | 71.8% | 61.1% | 67.4% |
| hw_md5_time_conditioned | 43.9% | 39.6% | 46.8% | 54.8% | 67.7% | 90.8% |
| hw_wl_time_conditioned | 40.9% | 40.2% | 51.6% | 58.9% | 67.9% | 79.5% |

### Key Findings from Results

1. **Formulation MSE is identical, crossing accuracy differs.** Time-conditioned and full curve have MSE difference of 0.000036. But full curve is better calibrated at early times (50.9% vs 26.3% at t=2k).

2. **Model learned late-curve behavior better.** curve_hash crossing accuracy: 26.3% at t=2k → 89.1% at t=22k. Early reliability values cluster near 1.0 (hard to rank); late values spread widely (easy to rank).

3. **Allocation split shows opposite crossing pattern.** Better at early times (72–79%), worse at late (59.5%). Task placement is most informative at short operating lifetimes.

4. **Hardware generalisation is achievable.** hw_md5 R²=0.666, comparable to curve_hash. WL only 5% harder than MD5 — WL isomorphism approximately physically meaningful.

5. **Allocation split is 54.6% harder** (R² drops 0.66 → 0.45). Primary area for future improvement.

6. **Single timestep is harmful for crossing accuracy** — scores 0% at all times except its fixed t=8,000h, and even there scores 41.9% (below random). Unusable for system adaptation.

---

## 10. Pipeline Status — COMPLETE

```
00_verify_data.py    ✅ Done
01_preprocess.py     ✅ Done — dataset.h5 with all hashes
02_eda_deep.py       ✅ Done — 14 plots in results/02_eda_deep/
03_split.py          ✅ Done — splits.json
04_train.py          ✅ Done — 7 models trained
05_evaluate.py       ✅ Done — results in results/05_evaluate/
```

---

## 11. Suggested Next Steps

1. Add explicit structural features (task-host node degree, switch connectivity of task-hosting node) to improve early-time crossing accuracy
2. Improve allocation generalisation — R²=0.45 on unseen task strategies
3. Clarify domain expert question on compute node failure rates
4. Experiment with larger model (increase hidden_dim from 64)
5. Try allocation-aware training strategy

---

## 12. Known Issues — Never Repeat

1. `du -sh $SCRATCH/*` hangs on Puhti — use `--max-depth=1`
2. Nines binning is invalid for this dataset at full time range
3. Random splitting causes massive leakage (97.7% curve redundancy)
4. WL hash on hardware-only subgraph groups non-equivalent configs (N6 task-host degree varies within WL group)
5. Target shape mismatch bug (fixed): `batch.y` shape `[B]` vs pred `[B,1]` — always reshape with `target.view(pred.shape)`
6. Never commit: `wandb/`, `models/`, `data/`, `*.h5`, `*.pth`, `*.log`, `*.out`