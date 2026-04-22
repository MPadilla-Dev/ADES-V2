"""
02_eda_deep.py
==============
STEP 2 of the ADES-v2 pipeline — Deep Exploratory Data Analysis.

Runs AFTER 01_preprocess.py has created data/dataset.h5.
Analysis only — does NOT produce splits. Splitting is in 03_split.py.

Sections:
    1.  Dataset inventory and allocation balance
    2.  Two-axis dataset structure
        — allocation-first view (per-allocation HW and curve counts)
        — hardware-first view (allocations and curves per HW layout)
        — task sensitivity scatter (HW layouts × allocations × curves)
    3.  Graph node composition
        — node type counts globally and per allocation
        — within-allocation hardware uniqueness (WL-aware)
    4.  Cross-allocation hardware identity
        — how many HW layouts are shared across allocations?
        — task sensitivity: does allocation change the curve for same HW?
        — allocation symmetry groups confirmed via WL grouping
    5.  Reliability curve uniqueness
        — full duplicate group analysis
        — curve families visualization
    6.  Curve crossings — full pairwise on unique curves
        — all 5,562,780 pairs examined
        — diverse examples: early / mid / late crossover times
        — one example per unique source curve (no curve dominates)
        — zoomed insets around crossover point
    7.  Crossing hub analysis (unique curves only)
        — crossing count distribution
        — hub curves vs low-crossing curves
    8.  Behavioral slope space
        — early vs late degradation rate (continuous, no forced families)
        — slope ratio distribution
    9.  Architectural correlation (continuous)
        — task concentration, switch degree vs reliability scatter plots

Module : python-data/3.10-24.04
Input  : data/dataset.h5
Output : results/02_eda_deep/
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict, Counter
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
H5_PATH  = "data/dataset.h5"
OUT_DIR  = "results/02_eda_deep"
LOG_PATH = "logs/02_eda_deep.log"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

ALLOC_TEST_SET = ['0009', '0013', '0025', '0019']

# ── LOGGING ───────────────────────────────────────────────────────────────────
log_lines = []
def log(msg=""):
    print(msg, flush=True)
    log_lines.append(msg)

def save_log():
    with open(LOG_PATH, 'w') as f:
        f.write('\n'.join(log_lines))

def save_fig(fig, name, dpi=180):
    path = f"{OUT_DIR}/{name}"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log(f"  -> Saved {name}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log("=" * 70)
    log("  ADES-v2 DEEP EDA (revised)")
    log("  Full dataset — no sampling, no split execution.")
    log("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────────────
    log(f"\nLoading dataset from {H5_PATH}...")
    with h5py.File(H5_PATH, 'r') as h5:
        config_ids   = h5['meta/config_ids'][:].astype(str)
        allocations  = h5['meta/allocations'][:].astype(str)
        config_nums  = h5['meta/config_nums'][:].astype(str)
        curve_hashes = h5['meta/curve_hashes'][:].astype(str)
        adj_hashes   = h5['meta/adj_hashes'][:].astype(str)
        hw_hashes    = h5['meta/hw_hashes'][:].astype(str)
        node_names   = h5['meta/node_names'][:].astype(str)
        y_curve      = h5['targets/y_curve'][:]
        node_feats   = h5['features/node_features'][:]
        edge_index   = h5['edges/edge_index'][:]
        edge_ptr     = h5['edges/edge_ptr'][:]
        time_vals    = h5.attrs['time_values']

    N          = len(config_ids)
    N_TIME     = len(time_vals)
    N_NODES    = node_feats.shape[1]
    feat_names = ['compute', 'switch', 'link', 'task_T1', 'task_T2']
    alloc_unique = sorted(set(allocations))
    t8k_idx    = np.argmin(np.abs(time_vals - 8000))

    log(f"  Samples      : {N:,}")
    log(f"  Time steps   : {N_TIME}  ({time_vals[0]:.0f}h - {time_vals[-1]:.0f}h)")
    log(f"  Allocations  : {len(alloc_unique)}")
    log(f"  Unique WL hw : {len(set(hw_hashes)):,}  (expected 392)")
    log(f"  Unique curves: {len(set(curve_hashes)):,}  (expected 3,336)")

    # Pre-build index structures used across sections
    alloc_counts   = Counter(allocations)
    hash_to_indices = defaultdict(list)
    for i, ch in enumerate(curve_hashes):
        hash_to_indices[ch].append(i)
    unique_hashes = list(hash_to_indices.keys())
    Y_unique      = np.array([y_curve[hash_to_indices[h][0]]
                               for h in unique_hashes])  # [3336, 221]
    r_at_8000     = Y_unique[:, t8k_idx]

    hw_to_indices = defaultdict(list)
    for i, hw in enumerate(hw_hashes):
        hw_to_indices[hw].append(i)
    unique_hw = list(hw_to_indices.keys())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Dataset Inventory
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 1: Dataset Inventory")
    log("=" * 70)

    counts_arr = [alloc_counts[a] for a in alloc_unique]
    log(f"  Total samples           : {N:,}")
    log(f"  Allocations             : {len(alloc_unique)}")
    log(f"  Min samples/alloc       : {min(counts_arr):,}")
    log(f"  Max samples/alloc       : {max(counts_arr):,}")
    log(f"  Imbalance ratio         : {max(counts_arr)/min(counts_arr):.1f}x")

    fig, ax = plt.subplots(figsize=(16, 5))
    colors  = ['tomato' if a in ALLOC_TEST_SET else 'steelblue'
               for a in alloc_unique]
    bars    = ax.bar(alloc_unique, counts_arr, color=colors, edgecolor='white')
    for bar, cnt in zip(bars, counts_arr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(cnt), ha='center', va='bottom', fontsize=7, rotation=90)
    ax.set_title("Samples per Allocation\n"
                 "(red = allocation test set candidates)", fontsize=13)
    ax.set_xlabel("Allocation ID")
    ax.set_ylabel("Number of configurations")
    ax.grid(axis='y', alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='steelblue', label='Training pool'),
        Patch(color='tomato',    label='Allocation test set (held out)')
    ])
    save_fig(fig, "01_allocation_balance.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Two-Axis Dataset Structure
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 2: Two-Axis Dataset Structure")
    log("=" * 70)

    # 2a. Allocation-first table
    log("\n  ALLOCATION-FIRST VIEW:")
    log(f"  {'Alloc':>6}  {'Samples':>8}  {'HW layouts':>11}  "
        f"{'Unique curves':>13}  {'Curves/HW':>9}")
    log(f"  {'-'*6}  {'-'*8}  {'-'*11}  {'-'*13}  {'-'*9}")

    alloc_hw_counts    = []
    alloc_curve_counts = []
    alloc_hw_unique    = []

    for alloc in alloc_unique:
        mask      = allocations == alloc
        n_samp    = mask.sum()
        n_hw      = len(set(hw_hashes[mask]))
        n_curves  = len(set(curve_hashes[mask]))
        ratio     = n_curves / n_hw if n_hw > 0 else 0
        alloc_hw_counts.append(n_samp)
        alloc_curve_counts.append(n_curves)
        alloc_hw_unique.append(n_hw)
        log(f"  {alloc:>6}  {n_samp:>8,}  {n_hw:>11,}  "
            f"{n_curves:>13,}  {ratio:>9.2f}")

    log(f"\n  Note: within every allocation, samples = HW layouts (WL-unique)")
    log(f"  Unique curves per allocation range: "
        f"{min(alloc_curve_counts)}–{max(alloc_curve_counts)}")

    # Allocation-first multi-bar chart
    x      = np.arange(len(alloc_unique))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(18, 6))
    bars1 = ax.bar(x - width/2, alloc_hw_counts,    width, label='Samples (= HW layouts)',
                   color='steelblue',    edgecolor='white', alpha=0.85)
    bars2 = ax.bar(x + width/2, alloc_curve_counts, width, label='Unique reliability curves',
                   color='darkorange', edgecolor='white', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(alloc_unique, rotation=90, fontsize=7)
    ax.set_title("Allocation-First View: Samples and Unique Curves per Allocation",
                 fontsize=13)
    ax.set_xlabel("Allocation ID")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "02_allocation_first_view.png")

    # 2b. Hardware-first distributions
    log("\n  HARDWARE-FIRST VIEW:")
    hw_to_allocs_set  = defaultdict(set)
    hw_to_curves_set  = defaultdict(set)
    for hw, alloc, ch in zip(hw_hashes, allocations, curve_hashes):
        hw_to_allocs_set[hw].add(alloc)
        hw_to_curves_set[hw].add(ch)

    n_allocs_per_hw = [len(v) for v in hw_to_allocs_set.values()]
    n_curves_per_hw = [len(v) for v in hw_to_curves_set.values()]

    log(f"  Unique HW layouts (WL)  : {len(unique_hw):,}")
    log(f"  Allocations per HW      : min={min(n_allocs_per_hw)}  "
        f"max={max(n_allocs_per_hw)}  mean={np.mean(n_allocs_per_hw):.1f}")
    log(f"  Unique curves per HW    : min={min(n_curves_per_hw)}  "
        f"max={max(n_curves_per_hw)}  mean={np.mean(n_curves_per_hw):.1f}")

    n_fully_explored   = sum(1 for v in hw_to_allocs_set.values() if len(v) == 31)
    n_task_insensitive = sum(1 for v in hw_to_curves_set.values() if len(v) == 1)
    n_task_sensitive   = sum(1 for v in hw_to_curves_set.values()
                             if len(v) == max(n_curves_per_hw))
    log(f"  HW in all 31 allocs     : {n_fully_explored:,}")
    log(f"  Task-insensitive HW     : {n_task_insensitive:,}  (1 curve regardless of alloc)")
    log(f"  Max task-sensitive HW   : {n_task_sensitive:,}  ({max(n_curves_per_hw)} curves)")

    alloc_dist = Counter(n_allocs_per_hw)
    log(f"\n  Allocations per HW layout distribution:")
    for k in sorted(alloc_dist.keys()):
        log(f"    {k:2d} alloc(s): {alloc_dist[k]:,} HW layouts")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].hist(n_allocs_per_hw, bins=31, color='steelblue',
                 edgecolor='white', range=(0.5, 31.5))
    axes[0].axvline(np.mean(n_allocs_per_hw), color='red', linestyle='--',
                    label=f'Mean: {np.mean(n_allocs_per_hw):.1f}')
    axes[0].set_title(f"Allocations per HW Layout\n"
                      f"({len(unique_hw):,} unique HW layouts)", fontsize=12)
    axes[0].set_xlabel("Number of allocations this HW appears in")
    axes[0].set_ylabel("Number of HW layouts")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    curve_dist = Counter(n_curves_per_hw)
    axes[1].bar(sorted(curve_dist.keys()),
                [curve_dist[k] for k in sorted(curve_dist.keys())],
                color='darkorange', edgecolor='white')
    axes[1].set_title(f"Unique Curves per HW Layout\n"
                      f"(task sensitivity — higher = more sensitive)", fontsize=12)
    axes[1].set_xlabel("Number of unique reliability curves produced")
    axes[1].set_ylabel("Number of HW layouts")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "03_hardware_first_view.png")

    # 2c. Task sensitivity scatter
    # One point per unique WL hw layout
    # x = number of allocations, y = number of unique curves
    # color = mean reliability at t=8000h
    log("\n  Building task sensitivity scatter...")
    hw_mean_r8k = {}
    for hw, indices in hw_to_indices.items():
        hw_mean_r8k[hw] = y_curve[indices, t8k_idx].mean()

    sc_x     = np.array(n_allocs_per_hw)
    sc_y     = np.array(n_curves_per_hw)
    sc_color = np.array([hw_mean_r8k[hw] for hw in unique_hw])

    fig, ax = plt.subplots(figsize=(12, 7))
    sc = ax.scatter(sc_x, sc_y, c=sc_color, cmap='RdYlGn',
                    alpha=0.5, s=20, linewidths=0)
    plt.colorbar(sc, ax=ax, label='Mean R at t=8,000h')
    ax.set_title("Task Sensitivity Landscape\n"
                 "Each point = one unique hardware layout (392 total)\n"
                 "x = allocations tested, y = unique curves produced, "
                 "color = mean reliability",
                 fontsize=12)
    ax.set_xlabel("Number of allocations this hardware appears in")
    ax.set_ylabel("Number of unique reliability curves produced")
    ax.grid(alpha=0.3)

    # Annotate extremes
    max_curves_idx = np.argmax(sc_y)
    ax.annotate(f"Most task-sensitive\n({sc_y[max_curves_idx]} curves)",
                xy=(sc_x[max_curves_idx], sc_y[max_curves_idx]),
                xytext=(sc_x[max_curves_idx] + 1, sc_y[max_curves_idx] - 5),
                fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))

    save_fig(fig, "04_task_sensitivity_scatter.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Graph Node Composition
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 3: Graph Node Composition")
    log("=" * 70)

    type_totals = node_feats.sum(axis=0).sum(axis=0)
    log("\n  Global node type totals (all samples):")
    for fname, total in zip(feat_names, type_totals):
        log(f"    {fname:10s}: {int(total):,}  ({int(total)/N:.2f} per graph)")

    # Within-allocation WL uniqueness confirmation
    log("\n  Within-allocation hardware uniqueness (WL-aware):")
    all_unique = True
    for alloc in alloc_unique:
        mask     = allocations == alloc
        hw_alloc = hw_hashes[mask]
        n_total  = len(hw_alloc)
        n_unique = len(set(hw_alloc))
        if n_unique < n_total:
            all_unique = False
            log(f"    alloc {alloc}: {n_total} samples, {n_unique} unique HW — "
                f"{n_total - n_unique} duplicates!")
    if all_unique:
        log("  RESULT: All hardware layouts unique within every allocation (WL-confirmed).")

    fig, axes = plt.subplots(1, len(feat_names), figsize=(18, 5), sharey=True)
    for fi, (fname, ax) in enumerate(zip(feat_names, axes)):
        per_alloc = [
            node_feats[allocations == a, :, fi].sum(axis=1).mean()
            for a in alloc_unique
        ]
        ax.bar(alloc_unique, per_alloc, color='mediumseagreen', edgecolor='white')
        ax.set_title(fname, fontsize=10)
        ax.set_xlabel("Allocation")
        ax.tick_params(axis='x', rotation=90, labelsize=6)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel("Mean node count per graph")
    fig.suptitle("Mean Node Count per Type per Allocation", fontsize=13)
    plt.tight_layout()
    save_fig(fig, "05_node_types_per_allocation.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Cross-Allocation Hardware Identity
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 4: Cross-Allocation Hardware Identity (WL-aware)")
    log("=" * 70)
    log("  Using WL hashes — isomorphic graphs treated as the same hardware.")

    # How many HW layouts are shared across allocations?
    n_shared_hw = sum(1 for v in hw_to_allocs_set.values() if len(v) > 1)
    n_unique_hw_alloc = sum(1 for v in hw_to_allocs_set.values() if len(v) == 1)
    log(f"  HW layouts in exactly 1 allocation : {n_unique_hw_alloc:,}")
    log(f"  HW layouts shared across allocations: {n_shared_hw:,}")

    # Allocation symmetry groups — allocations sharing the same unique curve set
    log("\n  Allocation symmetry groups (same unique curve count = same behavior space):")
    curve_count_to_allocs = defaultdict(list)
    for alloc, n_curves in zip(alloc_unique, alloc_curve_counts):
        curve_count_to_allocs[n_curves].append(alloc)

    for n_curves, allocs in sorted(curve_count_to_allocs.items()):
        if len(allocs) > 1:
            log(f"    {n_curves:4d} unique curves: allocs {allocs}")
        else:
            log(f"    {n_curves:4d} unique curves: alloc  {allocs[0]}  (unique)")

    # Task sensitivity: for HW layouts in multiple allocations,
    # does task assignment reliably change the curve?
    log("\n  Task sensitivity analysis:")
    log("  For HW layouts appearing in 2+ allocations:")
    sensitivity_counts = Counter()
    for hw, curves in hw_to_curves_set.items():
        if len(hw_to_allocs_set[hw]) > 1:
            sensitivity_counts[len(curves)] += 1

    total_multi = sum(sensitivity_counts.values())
    log(f"  HW layouts in 2+ allocations: {total_multi:,}")
    for n_curves in sorted(sensitivity_counts.keys()):
        pct = sensitivity_counts[n_curves] / total_multi * 100
        label = "(insensitive)" if n_curves == 1 else ""
        log(f"    {n_curves:3d} unique curve(s): {sensitivity_counts[n_curves]:,}  "
            f"({pct:.1f}%)  {label}")

    # Cross-alloc identity bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].bar(['In 1 allocation\nonly', 'Shared across\nallocations'],
                [n_unique_hw_alloc, n_shared_hw],
                color=['#3498db', '#e74c3c'], edgecolor='white', width=0.5)
    for bar, val in zip(axes[0].patches,
                        [n_unique_hw_alloc, n_shared_hw]):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 2,
                     f'{val:,}\n({val/len(unique_hw)*100:.1f}%)',
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
    axes[0].set_title("Hardware Layout Sharing Across Allocations\n"
                      "(WL-isomorphism-aware)", fontsize=11)
    axes[0].set_ylabel("Number of unique HW layouts")
    axes[0].grid(axis='y', alpha=0.3)

    # Task sensitivity for multi-allocation HW
    sorted_keys = sorted(sensitivity_counts.keys())
    sorted_vals = [sensitivity_counts[k] for k in sorted_keys]
    bar_colors  = ['#2ecc71' if k == 1 else '#e67e22' if k <= 5
                   else '#e74c3c' for k in sorted_keys]
    axes[1].bar([str(k) for k in sorted_keys], sorted_vals,
                color=bar_colors, edgecolor='white')
    axes[1].set_title("Task Sensitivity: Unique Curves per HW Layout\n"
                      "(for HW layouts tested in 2+ allocations)", fontsize=11)
    axes[1].set_xlabel("Number of unique reliability curves produced")
    axes[1].set_ylabel("Number of HW layouts")
    axes[1].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "06_cross_alloc_identity.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Reliability Curve Uniqueness
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 5: Reliability Curve Uniqueness")
    log("=" * 70)

    n_unique_curves = len(hash_to_indices)
    dup_groups      = {h: idx for h, idx in hash_to_indices.items()
                       if len(idx) > 1}
    n_redundant     = sum(len(v) - 1 for v in dup_groups.values())
    group_sizes     = sorted([len(v) for v in hash_to_indices.values()],
                              reverse=True)

    log(f"  Total samples         : {N:,}")
    log(f"  Unique curves         : {n_unique_curves:,}")
    log(f"  Duplicate groups      : {len(dup_groups):,}")
    log(f"  Redundant copies      : {n_redundant:,} ({n_redundant/N*100:.1f}%)")
    log(f"  Largest group         : {group_sizes[0]:,} copies")
    log(f"  Median group size     : {np.median(group_sizes):.1f}")

    # Curve families — all 3336 unique curves colored by R(8k)
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap    = plt.cm.RdYlGn
    norm    = plt.Normalize(r_at_8000.min(), r_at_8000.max())
    sm      = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    for idx in np.argsort(r_at_8000):
        ax.plot(time_vals, Y_unique[idx],
                color=cmap(norm(r_at_8000[idx])),
                alpha=0.15, linewidth=0.5)
    plt.colorbar(sm, ax=ax, label='R at t=8,000h (coloring only)')
    ax.set_title(f"All {n_unique_curves:,} Unique Reliability Curves\n"
                 f"({N:,} total — {n_redundant:,} exact duplicates, "
                 f"{n_redundant/N*100:.1f}%)", fontsize=12)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Reliability")
    ax.grid(alpha=0.2)
    save_fig(fig, "07_curve_families.png")

    # Group size distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(group_sizes, bins=80, color='steelblue', edgecolor='none', log=True)
    ax.axvline(np.median(group_sizes), color='orange', linestyle='--',
               label=f'Median: {np.median(group_sizes):.0f}')
    ax.axvline(group_sizes[0], color='red', linestyle='--',
               label=f'Max: {group_sizes[0]:,}')
    ax.set_title("Duplicate Group Size Distribution", fontsize=13)
    ax.set_xlabel("Copies sharing the same reliability curve")
    ax.set_ylabel("Number of groups (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "08_curve_duplicate_distribution.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Full Pairwise Crossing Analysis
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 6: Curve Crossing Analysis (Full Pairwise, Unique Curves Only)")
    log("=" * 70)

    n_pairs = n_unique_curves * (n_unique_curves - 1) // 2
    log(f"  Unique curves : {n_unique_curves:,}")
    log(f"  Total pairs   : {n_pairs:,}")
    log("  t=0 excluded. Min diff threshold: 0.01.")

    t0_idx = np.where(time_vals == 0)[0]
    Y_work = np.delete(Y_unique, t0_idx, axis=1)
    t_work = np.delete(time_vals, t0_idx)
    n_c    = Y_work.shape[0]

    depth_buckets = {
        'shallow (0.01–0.05)' : 0,
        'moderate (0.05–0.10)': 0,
        'deep (>0.10)'        : 0,
    }
    crossing_count         = 0
    total_pairs            = 0
    cross_count_per_curve  = np.zeros(n_c, dtype=np.int32)

    # Diverse example collection
    # Goal: one deep example per unique source curve i,
    # distributed across early/mid/late crossover times
    # t_work ranges from 100h to 22000h
    t_early_max = 5000
    t_mid_max   = 12000
    # late = > 12000h

    examples_early = []  # crossover < 5000h
    examples_mid   = []  # 5000 <= crossover < 12000h
    examples_late  = []  # crossover >= 12000h
    seen_source    = set()  # ensure one example per source curve

    for i in tqdm(range(n_c), desc="  Pairwise crossing", unit="curve"):
        diffs      = Y_work[i] - Y_work[i+1:]
        max_abs    = np.abs(diffs).max(axis=1)
        meaningful = max_abs > 0.01
        has_pos    = (diffs >  1e-6).any(axis=1)
        has_neg    = (diffs < -1e-6).any(axis=1)
        crossing   = has_pos & has_neg & meaningful

        n_cross        = int(crossing.sum())
        crossing_count += n_cross
        total_pairs    += int(meaningful.sum())

        cross_count_per_curve[i] += n_cross
        if n_cross > 0:
            for jo in np.where(crossing)[0]:
                cross_count_per_curve[i + 1 + jo] += 1

        if n_cross > 0:
            cross_diffs  = diffs[crossing]
            cross_maxabs = max_abs[crossing]

            for k, d in enumerate(cross_maxabs):
                if   d < 0.05: depth_buckets['shallow (0.01–0.05)']  += 1
                elif d < 0.10: depth_buckets['moderate (0.05–0.10)'] += 1
                else:
                    depth_buckets['deep (>0.10)'] += 1

                    # Collect diverse deep examples
                    if i not in seen_source and d > 0.10:
                        j        = i + 1 + np.where(crossing)[0][k]
                        sign_chg = np.where(
                            np.diff(np.sign(cross_diffs[k]))
                        )[0]
                        if len(sign_chg) > 0:
                            t_cross = t_work[sign_chg[0]]
                            ex = {
                                'i'        : i,
                                'j'        : j,
                                't_cross'  : t_cross,
                                'max_diff' : d,
                                'curve_i'  : Y_work[i].copy(),
                                'curve_j'  : Y_work[j].copy(),
                                'diff'     : cross_diffs[k].copy(),
                            }
                            if t_cross < t_early_max and len(examples_early) < 2:
                                examples_early.append(ex)
                                seen_source.add(i)
                            elif t_early_max <= t_cross < t_mid_max \
                                    and len(examples_mid) < 2:
                                examples_mid.append(ex)
                                seen_source.add(i)
                            elif t_cross >= t_mid_max and len(examples_late) < 2:
                                examples_late.append(ex)
                                seen_source.add(i)

    crossing_rate = crossing_count / total_pairs * 100 if total_pairs > 0 else 0
    total_depth   = sum(depth_buckets.values())

    log(f"\n  Meaningful pairs  : {total_pairs:,}")
    log(f"  Crossing pairs    : {crossing_count:,}")
    log(f"  Crossing rate     : {crossing_rate:.2f}%")
    log(f"\n  Depth distribution:")
    for label, count in depth_buckets.items():
        pct = count / total_depth * 100 if total_depth > 0 else 0
        log(f"    {label:25s}: {count:8,}  ({pct:.1f}%)")

    log(f"\n  Early examples (<{t_early_max}h crossover)  : {len(examples_early)}")
    log(f"  Mid examples ({t_early_max}–{t_mid_max}h crossover): {len(examples_mid)}")
    log(f"  Late examples (>{t_mid_max}h crossover) : {len(examples_late)}")

    moderate_deep = (depth_buckets['moderate (0.05–0.10)'] +
                     depth_buckets['deep (>0.10)'])
    log(f"\n  Moderate+deep crossings: {moderate_deep/total_depth*100:.1f}%")
    log("  → Static ranking impossible. Full-curve regression required.")

    # Depth distribution bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    labels   = list(depth_buckets.keys())
    vals     = [depth_buckets[l] for l in labels]
    colors_d = ['#3498db', '#e67e22', '#e74c3c']
    bars     = ax.bar(labels, vals, color=colors_d, edgecolor='white', width=0.5)
    for bar, val in zip(bars, vals):
        pct = val / total_depth * 100
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + total_depth * 0.005,
                f'{val:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title(f"Crossing Depth Distribution\n"
                 f"({crossing_count:,} crossing pairs, rate: {crossing_rate:.1f}%)",
                 fontsize=12)
    ax.set_ylabel("Number of crossing pairs")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "09_crossing_depth_distribution.png")

    # Crossing examples — diverse early/mid/late, one per source curve
    all_examples = examples_early + examples_mid + examples_late
    n_ex         = len(all_examples)
    section_labels = (
        ['Early crossover'] * len(examples_early) +
        ['Mid crossover']   * len(examples_mid)   +
        ['Late crossover']  * len(examples_late)
    )

    if n_ex > 0:
        ncols   = 3
        nrows   = (n_ex + ncols - 1) // ncols
        fig     = plt.figure(figsize=(18, 5.5 * nrows))
        gs      = gridspec.GridSpec(nrows, ncols, hspace=0.55, wspace=0.35)

        for plot_idx, (ex, slabel) in enumerate(
            zip(all_examples, section_labels)
        ):
            ax_main = fig.add_subplot(gs[plot_idx // ncols, plot_idx % ncols])

            ci      = ex['curve_i']
            cj      = ex['curve_j']
            tc      = ex['t_cross']
            tc_idx  = np.argmin(np.abs(t_work - tc))

            mid_before = max(0, tc_idx // 2)
            mid_after  = min(len(t_work)-1,
                             tc_idx + (len(t_work) - tc_idx) // 2)
            a_better_before = ci[mid_before] > cj[mid_before]
            a_better_after  = ci[mid_after]  > cj[mid_after]

            ax_main.plot(t_work, ci, color='#2980b9', linewidth=1.8,
                         label=f'Curve A  R(22k)={ci[-1]:.4f}')
            ax_main.plot(t_work, cj, color='#e74c3c', linewidth=1.8,
                         label=f'Curve B  R(22k)={cj[-1]:.4f}')
            ax_main.axvline(tc, color='gray', linestyle='--',
                            linewidth=1.2, label=f'Crossover: {tc:.0f}h')

            label_before = 'A better' if a_better_before else 'B better'
            label_after  = 'A better' if a_better_after  else 'B better'
            color_before = '#2980b9'  if a_better_before else '#e74c3c'
            color_after  = '#2980b9'  if a_better_after  else '#e74c3c'

            ax_main.fill_betweenx(
                [0, 1], 0, tc, alpha=0.07, color=color_before,
                label=f'{label_before} (before)'
            )
            ax_main.fill_betweenx(
                [0, 1], tc, t_work[-1], alpha=0.07, color=color_after,
                label=f'{label_after} (after)'
            )

            ax_main.set_title(
                f"{slabel}\n"
                f"Max diff: {ex['max_diff']:.3f}  |  Crossover: {tc:.0f}h",
                fontsize=9
            )
            ax_main.set_xlabel("Time (hours)", fontsize=8)
            ax_main.set_ylabel("Reliability", fontsize=8)
            ax_main.legend(fontsize=6.5, loc='lower left')
            ax_main.grid(alpha=0.25)
            ax_main.set_xlim(0, t_work[-1])
            y_lo = min(ci.min(), cj.min()) - 0.02
            y_hi = max(ci.max(), cj.max()) + 0.02
            ax_main.set_ylim(max(0, y_lo), min(1.02, y_hi))

            # Zoomed inset around crossover
            zoom_w  = 3000
            z_tmin  = max(t_work[0],  tc - zoom_w)
            z_tmax  = min(t_work[-1], tc + zoom_w)
            z_mask  = (t_work >= z_tmin) & (t_work <= z_tmax)
            axins   = ax_main.inset_axes([0.54, 0.52, 0.44, 0.40])
            axins.plot(t_work[z_mask], ci[z_mask],
                       color='#2980b9', linewidth=1.5)
            axins.plot(t_work[z_mask], cj[z_mask],
                       color='#e74c3c', linewidth=1.5)
            axins.axvline(tc, color='gray', linestyle='--', linewidth=1.0)
            z_ymin = min(ci[z_mask].min(), cj[z_mask].min()) - 0.005
            z_ymax = max(ci[z_mask].max(), cj[z_mask].max()) + 0.005
            axins.set_ylim(z_ymin, z_ymax)
            axins.set_xlim(z_tmin, z_tmax)
            axins.tick_params(labelsize=5)
            axins.set_title("Zoom", fontsize=6)
            axins.grid(alpha=0.3)

            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (z_tmin, z_ymin), z_tmax - z_tmin, z_ymax - z_ymin,
                linewidth=0.8, edgecolor='gray',
                facecolor='none', linestyle='--'
            )
            ax_main.add_patch(rect)

        fig.suptitle(
            "Deep Crossing Examples — Diverse Early, Mid, and Late Crossovers\n"
            "One example per unique source curve. "
            "Shading shows which curve is genuinely higher in each region.",
            fontsize=12, fontweight='bold'
        )
        save_fig(fig, "10_crossing_examples.png", dpi=200)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — Crossing Hub Analysis (Unique Curves Only)
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 7: Crossing Hub Analysis (Unique Curves Only)")
    log("=" * 70)
    log("  Hub = unique reliability behavior that crosses many others.")
    log("  Identical curves count as one — hubs reflect behavior, not copies.")

    n_zero   = int((cross_count_per_curve == 0).sum())
    n_nonzero = int((cross_count_per_curve > 0).sum())
    max_cross = int(cross_count_per_curve.max())

    log(f"  Curves with 0 crossings  : {n_zero:,}  ({n_zero/n_c*100:.1f}%)")
    log(f"  Curves with >0 crossings : {n_nonzero:,}  ({n_nonzero/n_c*100:.1f}%)")
    log(f"  Max crossings one curve  : {max_cross:,}")
    log(f"  Median crossings         : {np.median(cross_count_per_curve):.1f}")

    top10_idx = np.argsort(cross_count_per_curve)[::-1][:10]
    log(f"\n  Top 10 hub curves:")
    for rank, idx in enumerate(top10_idx):
        log(f"    #{rank+1:2d}  crossings={cross_count_per_curve[idx]:4d}  "
            f"R(8k)={r_at_8000[idx]:.4f}  R(22k)={Y_unique[idx,-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(cross_count_per_curve, bins=80,
                 color='steelblue', edgecolor='none', log=True)
    axes[0].axvline(np.median(cross_count_per_curve), color='orange',
                    linestyle='--',
                    label=f'Median: {np.median(cross_count_per_curve):.0f}')
    axes[0].set_title("Crossings per Unique Curve (log scale)", fontsize=11)
    axes[0].set_xlabel("Number of other unique curves crossed")
    axes[0].set_ylabel("Number of unique curves (log)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    sc2 = axes[1].scatter(r_at_8000, cross_count_per_curve,
                          c=Y_unique[:, -1], cmap='RdYlGn',
                          alpha=0.4, s=8, linewidths=0)
    plt.colorbar(sc2, ax=axes[1], label='R at t=22,000h')
    axes[1].set_title("Reliability vs Crossing Count\n"
                      "(color = R at t=22,000h)", fontsize=11)
    axes[1].set_xlabel("R at t=8,000h")
    axes[1].set_ylabel("Number of unique curves crossed")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "11_crossing_hubs.png")

    # Hub curves overlaid on low-crossing context
    fig, ax = plt.subplots(figsize=(14, 7))
    non_hub = np.argsort(cross_count_per_curve)[:200]
    for idx in non_hub:
        ax.plot(time_vals, Y_unique[idx],
                color='lightgray', linewidth=0.5, alpha=0.4)
    cmap_hub = plt.cm.tab10
    for rank, idx in enumerate(top10_idx):
        ax.plot(time_vals, Y_unique[idx],
                color=cmap_hub(rank / 10), linewidth=2.0, alpha=0.9,
                label=f'Hub #{rank+1}  '
                      f'({cross_count_per_curve[idx]} crossings, '
                      f'R(8k)={r_at_8000[idx]:.3f})')
    ax.set_title("Top 10 Hub Curves overlaid on 200 Low-Crossing Curves (gray)",
                 fontsize=13)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Reliability")
    ax.legend(fontsize=7, loc='lower left', ncol=2)
    ax.grid(alpha=0.2)
    save_fig(fig, "12_hub_curves.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — Behavioral Slope Space (Continuous)
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 8: Behavioral Slope Space (Continuous Spectrum)")
    log("=" * 70)
    log("  No forced binary families — spectrum shown as continuous.")

    early_start = 0
    early_end   = t8k_idx - 1
    late_start  = t8k_idx
    late_end    = len(time_vals) - 1
    early_dur   = time_vals[early_end] - time_vals[early_start]
    late_dur    = time_vals[late_end]  - time_vals[late_start]

    early_slope = (Y_unique[:, early_end] - Y_unique[:, early_start]) / early_dur
    late_slope  = (Y_unique[:, late_end]  - Y_unique[:, late_start])  / late_dur

    with np.errstate(divide='ignore', invalid='ignore'):
        slope_ratio = np.where(
            np.abs(early_slope) > 1e-8,
            np.abs(late_slope) / np.abs(early_slope),
            1.0
        )

    log(f"  Early slope (t=100–8000h)  mean: {early_slope.mean():.2e}  "
        f"std: {early_slope.std():.2e}")
    log(f"  Late slope (t=8000–22000h) mean: {late_slope.mean():.2e}  "
        f"std: {late_slope.std():.2e}")
    log(f"  Slope ratio (late/early)   mean: {slope_ratio.mean():.2f}  "
        f"std: {slope_ratio.std():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sc3 = axes[0].scatter(
        early_slope * 1e5, late_slope * 1e5,
        c=slope_ratio, cmap='plasma',
        alpha=0.4, s=8, linewidths=0,
        vmin=np.percentile(slope_ratio, 5),
        vmax=np.percentile(slope_ratio, 95)
    )
    plt.colorbar(sc3, ax=axes[0], label='Late/Early slope ratio')
    lim = max(abs(early_slope.min()), abs(late_slope.min())) * 1e5 * 1.1
    axes[0].plot([-lim, 0], [-lim, 0], 'k--', alpha=0.3,
                 linewidth=0.8, label='Equal slope (steady degradation)')
    axes[0].set_xlabel("Early slope (×10⁻⁵ R/hour, t=100–8000h)")
    axes[0].set_ylabel("Late slope (×10⁻⁵ R/hour, t=8000–22000h)")
    axes[0].set_title("Behavioral Slope Space\n"
                      "(color = slope ratio, dashed = equal degradation rate)",
                      fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].hist(slope_ratio, bins=100, color='mediumpurple', edgecolor='none')
    axes[1].set_title("Distribution of Late/Early Slope Ratio\n"
                      "(continuous spectrum — no forced binary families)",
                      fontsize=11)
    axes[1].set_xlabel("Late slope / Early slope (absolute values)")
    axes[1].set_ylabel("Number of unique curves")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "13_behavioral_slope_space.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9 — Architectural Correlation (Continuous)
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 9: Architectural Correlation (Continuous)")
    log("=" * 70)
    log("  Computing structural features for one rep graph per unique curve...")

    rep_indices = [hash_to_indices[h][0] for h in unique_hashes]

    task_concentration = []
    mean_switch_degree = []
    n_compute_w_tasks  = []
    graph_density      = []

    for sample_idx in tqdm(rep_indices, desc="  Arch features"):
        e_start   = int(edge_ptr[sample_idx])
        e_end     = int(edge_ptr[sample_idx + 1])
        ei        = edge_index[:, e_start:e_end]
        feats_s   = node_feats[sample_idx]
        is_task   = (feats_s[:, 3] + feats_s[:, 4]) > 0
        is_switch = feats_s[:, 1].astype(bool)
        is_compute = feats_s[:, 0].astype(bool)

        degree = np.bincount(ei[0], minlength=N_NODES).astype(np.float32)

        # Switch degree
        sw_deg = degree[is_switch].mean() if is_switch.any() else 0.0
        mean_switch_degree.append(sw_deg)

        # Task concentration
        task_nodes    = np.where(is_task)[0]
        compute_nodes = set()
        for tn in task_nodes:
            for nb in ei[1][ei[0] == tn]:
                if is_compute[nb]:
                    compute_nodes.add(int(nb))
        n_comp = len(compute_nodes) if compute_nodes else 1
        n_compute_w_tasks.append(n_comp)
        task_concentration.append(1.0 / n_comp)

        # Graph density
        n_edges = ei.shape[1]
        graph_density.append(n_edges / (N_NODES * (N_NODES - 1)))

    task_concentration = np.array(task_concentration)
    mean_switch_degree = np.array(mean_switch_degree)
    n_compute_w_tasks  = np.array(n_compute_w_tasks, dtype=float)
    graph_density      = np.array(graph_density)

    # Correlation with mean reliability at 8k
    mean_r8k = r_at_8000  # [3336]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    arch_pairs = [
        (task_concentration, "Task concentration (1/n_compute_w_tasks)",
         axes[0, 0]),
        (mean_switch_degree, "Mean switch node degree",
         axes[0, 1]),
        (n_compute_w_tasks,  "N compute nodes with tasks",
         axes[1, 0]),
        (graph_density,      "Graph density (edges / max possible)",
         axes[1, 1]),
    ]

    for feat_vals, feat_label, ax in arch_pairs:
        sc4 = ax.scatter(feat_vals, mean_r8k,
                         c=slope_ratio, cmap='plasma',
                         alpha=0.3, s=6, linewidths=0,
                         vmin=np.percentile(slope_ratio, 5),
                         vmax=np.percentile(slope_ratio, 95))
        # Trend line
        z   = np.polyfit(feat_vals, mean_r8k, 1)
        p   = np.poly1d(z)
        x_r = np.linspace(feat_vals.min(), feat_vals.max(), 100)
        ax.plot(x_r, p(x_r), 'k-', linewidth=1.5, alpha=0.6,
                label=f'Trend (slope={z[0]:.4f})')
        corr = np.corrcoef(feat_vals, mean_r8k)[0, 1]
        ax.set_title(f"{feat_label}\nPearson r = {corr:.3f}", fontsize=10)
        ax.set_xlabel(feat_label, fontsize=9)
        ax.set_ylabel("Mean R at t=8,000h", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.colorbar(sc4, ax=ax, label='Slope ratio (late/early)')

    fig.suptitle(
        "Architectural Features vs Reliability (Continuous Correlation)\n"
        "Color = late/early slope ratio (purple = accelerating decline)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    save_fig(fig, "14_architectural_correlation.png")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SUMMARY OF KEY FINDINGS")
    log("=" * 70)
    log(f"  Total samples               : {N:,}")
    log(f"  Allocations                 : {len(alloc_unique)}")
    log(f"  Unique HW layouts (WL/VF2)  : {len(unique_hw):,}  (392 proven exact)")
    log(f"  Unique curves               : {n_unique_curves:,}")
    log(f"  Curve redundancy            : {n_redundant/N*100:.1f}%")
    log(f"  Within-alloc HW uniqueness  : "
        f"{'100%' if all_unique else 'DUPLICATES EXIST'}")
    log(f"  Task-insensitive HW layouts : {n_task_insensitive:,}  (1 curve always)")
    log(f"  Crossing rate               : {crossing_rate:.1f}%")
    log(f"  Moderate+deep crossings     : {moderate_deep/total_depth*100:.1f}%")
    log(f"  Max hub crossings           : {max_cross:,}")
    log(f"  Slope ratio (continuous)    : {slope_ratio.mean():.2f} ± "
        f"{slope_ratio.std():.2f}")
    log(f"\n  Three split axes available:")
    log(f"    curve_hash  → curve-level (unseen R values)")
    log(f"    allocation  → allocation-level (unseen task strategies)")
    log(f"    hw_hash     → hardware-level (unseen physical architecture, 392 groups)")
    log(f"\n  NOTE: Splits are NOT executed here. Run 03_split.py.")
    log(f"\n✅ EDA complete. All outputs in {OUT_DIR}/")
    save_log()


if __name__ == "__main__":
    main()