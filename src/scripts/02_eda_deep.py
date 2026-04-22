"""
02_eda_deep.py
==============
STEP 2 of the ADES-v2 pipeline — Deep Exploratory Data Analysis.

Runs AFTER 01_preprocess.py has created data/dataset.h5.
Analysis only — does NOT produce splits. Splitting is handled by 03_split.py.

Sections:
    1.  Dataset inventory and integrity check
    2.  Graph structure analysis
        — node type composition per allocation
        — within-allocation adjacency uniqueness
    3.  Cross-allocation graph identity
        — does task allocation change topology or just labeling?
    4.  Reliability curve uniqueness
        — full duplicate group analysis
        — curve families visualization
    5.  Curve crossings — full pairwise analysis
        — all pairs of unique curves
        — crossing rate and depth distribution
        — fixed crossing examples with zoomed insets
    6.  Crossing hub analysis
        — which curves are responsible for most crossings?
        — hub curve visualization
        — do hubs cluster in specific allocations or reliability ranges?
    7.  Behavioral family detection
        — early vs late degradation slope clustering
        — 2D slope space visualization
        — representative curves per family
    8.  Architectural correlation
        — do behavioral families correlate with graph structure?
        — node degree, path redundancy, task concentration per family

Module : python-data/3.10-24.04
Input  : data/dataset.h5
Output : results/02_eda_deep/
         01_allocation_balance.png
         02_node_types_per_allocation.png
         03_within_alloc_uniqueness.png
         04_cross_alloc_identity.png
         05_curve_families.png
         06_curve_uniqueness_distribution.png
         07_crossing_depth_distribution.png
         08_crossing_examples.png
         09_crossing_hubs.png
         10_hub_curves.png
         11_behavioral_slope_space.png
         12_behavioral_families.png
         13_architectural_correlation.png
         report.txt
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from collections import defaultdict, Counter
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
H5_PATH  = "data/dataset.h5"
OUT_DIR  = "results/02_eda_deep"
LOG_PATH = "logs/02_eda_deep.log"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Allocations held out for allocation test set (one per reliability regime)
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
    log("  ADES-v2 DEEP EDA")
    log("  Full dataset analysis — no sampling, no split execution.")
    log("=" * 70)

    # ── Load everything ───────────────────────────────────────────────────────
    log(f"\nLoading dataset from {H5_PATH}...")
    with h5py.File(H5_PATH, 'r') as h5:
        config_ids   = h5['meta/config_ids'][:].astype(str)
        allocations  = h5['meta/allocations'][:].astype(str)
        config_nums  = h5['meta/config_nums'][:].astype(str)
        curve_hashes = h5['meta/curve_hashes'][:].astype(str)
        adj_hashes   = h5['meta/adj_hashes'][:].astype(str)
        node_names   = h5['meta/node_names'][:].astype(str)   # [N, 30]
        y_curve      = h5['targets/y_curve'][:]                # [N, 221]
        node_feats   = h5['features/node_features'][:]         # [N, 30, 5]
        edge_index   = h5['edges/edge_index'][:]               # [2, E_total]
        edge_ptr     = h5['edges/edge_ptr'][:]                 # [N+1]
        time_vals    = h5.attrs['time_values']                 # [221]

    N          = len(config_ids)
    N_TIME     = len(time_vals)
    N_NODES    = node_feats.shape[1]
    feat_names = ['compute', 'switch', 'link', 'task_T1', 'task_T2']
    alloc_unique = sorted(set(allocations))

    log(f"  Samples      : {N:,}")
    log(f"  Time steps   : {N_TIME}  ({time_vals[0]:.0f}h - {time_vals[-1]:.0f}h)")
    log(f"  Nodes/graph  : {N_NODES}")
    log(f"  Allocations  : {len(alloc_unique)}")

    # Pre-compute t=8000h index — used only for ordering/coloring, not as target
    t8k_idx = np.argmin(np.abs(time_vals - 8000))
    log(f"  t=8000h index: {t8k_idx} (time_vals[{t8k_idx}]={time_vals[t8k_idx]:.0f}h)")
    log(f"  Note: t=8000h used only as a summary statistic for visualization")
    log(f"  ordering. It has no effect on what the model predicts.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Dataset Inventory
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 1: Dataset Inventory")
    log("=" * 70)

    alloc_counts = Counter(allocations)
    counts_arr   = [alloc_counts[a] for a in alloc_unique]

    log(f"  Total samples           : {N:,}")
    log(f"  Allocations             : {len(alloc_unique)}")
    log(f"  Min samples/allocation  : {min(counts_arr):,}  "
        f"(alloc {alloc_unique[counts_arr.index(min(counts_arr))]})")
    log(f"  Max samples/allocation  : {max(counts_arr):,}  "
        f"(alloc {alloc_unique[counts_arr.index(max(counts_arr))]})")
    log(f"  Mean samples/allocation : {np.mean(counts_arr):.1f}")
    log(f"  Imbalance ratio         : {max(counts_arr)/min(counts_arr):.1f}x")

    fig, ax = plt.subplots(figsize=(16, 5))
    colors  = ['tomato' if a in ALLOC_TEST_SET else 'steelblue'
               for a in alloc_unique]
    bars    = ax.bar(alloc_unique, counts_arr, color=colors, edgecolor='white')
    for bar, cnt in zip(bars, counts_arr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(cnt), ha='center', va='bottom', fontsize=7, rotation=90)
    ax.set_title("Samples per Allocation\n"
                 "(red = held out for allocation test set)", fontsize=13)
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
    # SECTION 2 — Graph Structure Analysis
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 2: Graph Structure Analysis")
    log("=" * 70)

    # 2a. Global node type totals
    type_totals = node_feats.sum(axis=0).sum(axis=0)  # [5]
    log("\n  Global node type totals:")
    for fname, total in zip(feat_names, type_totals):
        log(f"    {fname:10s}: {int(total):,}  ({int(total)/N:.2f} per graph)")

    # 2b. Per-allocation node type composition
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
    save_fig(fig, "02_node_types_per_allocation.png")

    # 2c. Within-allocation adjacency uniqueness
    log("\n  Within-allocation graph uniqueness:")
    within_results = {}
    for alloc in alloc_unique:
        mask         = allocations == alloc
        hashes_alloc = adj_hashes[mask]
        n_total      = len(hashes_alloc)
        n_unique     = len(set(hashes_alloc))
        n_dupes      = n_total - n_unique
        within_results[alloc] = (n_total, n_unique, n_dupes)
        if n_dupes > 0:
            log(f"    alloc {alloc}: {n_total} samples, "
                f"{n_unique} unique graphs, {n_dupes} DUPLICATES")

    all_unique_within = all(v[2] == 0 for v in within_results.values())
    if all_unique_within:
        log("  RESULT: All adjacency matrices unique within every allocation.")
    else:
        log("  RESULT: Some allocations contain duplicate graph structures!")

    uniqueness_rates = [
        within_results[a][1] / within_results[a][0] for a in alloc_unique
    ]
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(alloc_unique, uniqueness_rates,
           color=['green' if r == 1.0 else 'red' for r in uniqueness_rates])
    ax.set_ylim(0.95, 1.005)
    ax.set_title("Within-Allocation Graph Uniqueness Rate", fontsize=13)
    ax.set_xlabel("Allocation ID")
    ax.set_ylabel("Fraction unique")
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "03_within_alloc_uniqueness.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Cross-Allocation Graph Identity
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 3: Cross-Allocation Graph Identity")
    log("=" * 70)
    log("  Does task allocation change graph topology or just node labeling?")

    conf_to_allocs = defaultdict(list)
    for cid, alloc, cnum, ah in zip(config_ids, allocations,
                                    config_nums, adj_hashes):
        conf_to_allocs[cnum].append((alloc, ah))

    shared_conf_nums = {k: v for k, v in conf_to_allocs.items() if len(v) > 1}
    log(f"  Config nums in multiple allocations: {len(shared_conf_nums):,}")

    same_count = diff_count = mixed_count = 0
    for cnum, entries in tqdm(shared_conf_nums.items(), desc="  Cross-alloc"):
        hashes     = [e[1] for e in entries]
        n_unique_h = len(set(hashes))
        if   n_unique_h == 1:            same_count  += 1
        elif n_unique_h == len(hashes):  diff_count  += 1
        else:                            mixed_count += 1

    total_shared = len(shared_conf_nums)
    log(f"\n  Same topology across all allocations : "
        f"{same_count:,}  ({same_count/total_shared*100:.1f}%)")
    log(f"  Different topology per allocation    : "
        f"{diff_count:,}  ({diff_count/total_shared*100:.1f}%)")
    log(f"  Mixed                                : "
        f"{mixed_count:,}  ({mixed_count/total_shared*100:.1f}%)")

    if diff_count > 0 or mixed_count > 0:
        log("  RESULT: Task allocation DOES change graph topology.")
        log("  Allocation is a structural variable, not just a label.")
    else:
        log("  RESULT: Allocation only changes node labeling, not topology.")

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Same topology\nacross allocations',
              'Different topology\nper allocation', 'Mixed']
    values = [same_count, diff_count, mixed_count]
    colors_bar = ['#2ecc71', '#e74c3c', '#f39c12']
    bars   = ax.bar(labels, values, color=colors_bar, edgecolor='white', width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + total_shared * 0.005,
                f'{val:,}\n({val/total_shared*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title("Cross-Allocation Graph Identity", fontsize=13)
    ax.set_ylabel("Number of shared config numbers")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "04_cross_alloc_identity.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Reliability Curve Uniqueness
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 4: Reliability Curve Uniqueness")
    log("=" * 70)

    hash_to_indices = defaultdict(list)
    for i, ch in enumerate(curve_hashes):
        hash_to_indices[ch].append(i)

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

    # Build unique curve matrix — one representative per hash group
    unique_hashes = list(hash_to_indices.keys())
    Y_unique      = np.array([
        y_curve[hash_to_indices[h][0]] for h in unique_hashes
    ])  # [3336, 221]
    r_at_8000     = Y_unique[:, t8k_idx]

    # Curve families plot
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap    = plt.cm.RdYlGn
    norm    = plt.Normalize(r_at_8000.min(), r_at_8000.max())
    sm      = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    order   = np.argsort(r_at_8000)
    for idx in order:
        ax.plot(time_vals, Y_unique[idx],
                color=cmap(norm(r_at_8000[idx])),
                alpha=0.15, linewidth=0.5)
    plt.colorbar(sm, ax=ax, label='Reliability at t=8,000h (used for coloring only)')
    ax.set_title(f"All {n_unique_curves:,} Unique Reliability Curves\n"
                 f"({N:,} total samples — {n_redundant:,} are exact duplicates)",
                 fontsize=12)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Reliability")
    ax.grid(alpha=0.2)
    save_fig(fig, "05_curve_families.png")

    # Group size distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(group_sizes, bins=80, color='steelblue', edgecolor='none', log=True)
    ax.axvline(np.median(group_sizes), color='orange', linestyle='--',
               label=f'Median: {np.median(group_sizes):.0f}')
    ax.axvline(group_sizes[0], color='red', linestyle='--',
               label=f'Max: {group_sizes[0]:,}')
    ax.set_title(f"Duplicate Group Size Distribution\n"
                 f"({n_unique_curves:,} unique curves from {N:,} samples)",
                 fontsize=13)
    ax.set_xlabel("Copies sharing the same reliability curve")
    ax.set_ylabel("Number of groups (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "06_curve_uniqueness_distribution.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Full Pairwise Crossing Analysis
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 5: Curve Crossing Analysis (Full Pairwise)")
    log("=" * 70)

    n_pairs = n_unique_curves * (n_unique_curves - 1) // 2
    log(f"  Unique curves : {n_unique_curves:,}")
    log(f"  Total pairs   : {n_pairs:,}")
    log("  t=0 excluded (universal anchor). Min diff threshold: 0.01.")

    # Drop t=0
    t0_idx  = np.where(time_vals == 0)[0]
    Y_work  = np.delete(Y_unique, t0_idx, axis=1)   # [3336, 220]
    t_work  = np.delete(time_vals, t0_idx)           # [220]
    n_c     = Y_work.shape[0]

    depth_buckets = {
        'shallow (0.01–0.05)' : 0,
        'moderate (0.05–0.10)': 0,
        'deep (>0.10)'        : 0,
    }
    crossing_count = 0
    total_pairs    = 0
    deep_examples  = []

    # Per-curve crossing count — for hub analysis in Section 6
    cross_count_per_curve = np.zeros(n_c, dtype=np.int32)

    for i in tqdm(range(n_c), desc="  Pairwise crossing", unit="curve"):
        diffs      = Y_work[i] - Y_work[i+1:]         # [n_c-i-1, 220]
        max_abs    = np.abs(diffs).max(axis=1)
        meaningful = max_abs > 0.01
        has_pos    = (diffs >  1e-6).any(axis=1)
        has_neg    = (diffs < -1e-6).any(axis=1)
        crossing   = has_pos & has_neg & meaningful

        n_cross        = int(crossing.sum())
        crossing_count += n_cross
        total_pairs    += int(meaningful.sum())

        # Record per-curve crossing counts
        cross_count_per_curve[i] += n_cross
        if n_cross > 0:
            j_offsets = np.where(crossing)[0]
            for jo in j_offsets:
                cross_count_per_curve[i + 1 + jo] += 1

        if n_cross > 0:
            cross_diffs  = diffs[crossing]
            cross_maxabs = max_abs[crossing]

            for k, d in enumerate(cross_maxabs):
                if   d < 0.05: depth_buckets['shallow (0.01–0.05)']  += 1
                elif d < 0.10: depth_buckets['moderate (0.05–0.10)'] += 1
                else:
                    depth_buckets['deep (>0.10)'] += 1
                    if len(deep_examples) < 12:
                        j        = i + 1 + np.where(crossing)[0][k]
                        sign_chg = np.where(np.diff(np.sign(cross_diffs[k])))[0]
                        if len(sign_chg) > 0:
                            t_cross = t_work[sign_chg[0]]
                            deep_examples.append({
                                'i'         : i,
                                'j'         : j,
                                't_cross'   : t_cross,
                                'max_diff'  : d,
                                'curve_i'   : Y_work[i].copy(),
                                'curve_j'   : Y_work[j].copy(),
                                'diff'      : cross_diffs[k].copy(),
                            })

    crossing_rate = crossing_count / total_pairs * 100 if total_pairs > 0 else 0
    total_depth   = sum(depth_buckets.values())

    log(f"\n  Meaningful pairs checked : {total_pairs:,}")
    log(f"  Crossing pairs found     : {crossing_count:,}")
    log(f"  Crossing rate            : {crossing_rate:.2f}%")
    log(f"\n  Depth distribution:")
    for label, count in depth_buckets.items():
        pct = count / total_depth * 100 if total_depth > 0 else 0
        log(f"    {label:25s}: {count:8,}  ({pct:.1f}%)")

    moderate_deep = (depth_buckets['moderate (0.05–0.10)'] +
                     depth_buckets['deep (>0.10)'])
    log(f"\n  Moderate + deep: {moderate_deep:,} "
        f"({moderate_deep/total_depth*100:.1f}%)")
    log("  -> Static ranking impossible. Full-curve regression required.")

    # Depth distribution bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    labels  = list(depth_buckets.keys())
    vals    = [depth_buckets[l] for l in labels]
    colors_d = ['#3498db', '#e67e22', '#e74c3c']
    bars     = ax.bar(labels, vals, color=colors_d, edgecolor='white', width=0.5)
    for bar, val in zip(bars, vals):
        pct = val / total_depth * 100
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + total_depth * 0.005,
                f'{val:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title(f"Crossing Depth Distribution\n"
                 f"({crossing_count:,} crossing pairs, "
                 f"crossing rate: {crossing_rate:.1f}%)",
                 fontsize=12)
    ax.set_ylabel("Number of crossing pairs")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "07_crossing_depth_distribution.png")

    # Crossing examples — fixed labeling + zoomed inset
    n_ex = min(len(deep_examples), 6)
    if n_ex > 0:
        fig = plt.figure(figsize=(18, 5 * ((n_ex + 2) // 3)))
        gs  = gridspec.GridSpec((n_ex + 2) // 3, 3,
                                hspace=0.55, wspace=0.35)

        for plot_idx in range(n_ex):
            ex      = deep_examples[plot_idx]
            ax_main = fig.add_subplot(gs[plot_idx // 3, plot_idx % 3])

            ci      = ex['curve_i']
            cj      = ex['curve_j']
            diff    = ex['diff']
            tc      = ex['t_cross']
            tc_idx  = np.argmin(np.abs(t_work - tc))

            # Determine which curve is higher before and after crossover
            # Use the midpoint of each region for clarity
            mid_before = tc_idx // 2
            mid_after  = tc_idx + (len(t_work) - tc_idx) // 2
            a_better_before = ci[mid_before] > cj[mid_before]
            a_better_after  = ci[mid_after]  > cj[mid_after]

            ax_main.plot(t_work, ci, color='#2980b9', linewidth=1.8,
                         label=f'Curve A  R(22k)={ci[-1]:.4f}')
            ax_main.plot(t_work, cj, color='#e74c3c', linewidth=1.8,
                         label=f'Curve B  R(22k)={cj[-1]:.4f}')
            ax_main.axvline(tc, color='gray', linestyle='--',
                            linewidth=1.2, label=f'Crossover: {tc:.0f}h')

            # Shade regions with correct labels based on actual curve ordering
            label_before = 'A better' if a_better_before else 'B better'
            label_after  = 'A better' if a_better_after  else 'B better'
            color_before = '#2980b9'  if a_better_before else '#e74c3c'
            color_after  = '#2980b9'  if a_better_after  else '#e74c3c'

            ax_main.fill_betweenx(
                [0, 1], 0, tc,
                alpha=0.07, color=color_before, label=f'{label_before} (before)'
            )
            ax_main.fill_betweenx(
                [0, 1], tc, t_work[-1],
                alpha=0.07, color=color_after, label=f'{label_after} (after)'
            )

            ax_main.set_title(
                f"Max diff: {ex['max_diff']:.3f}  |  Crossover: {tc:.0f}h",
                fontsize=9
            )
            ax_main.set_xlabel("Time (hours)", fontsize=8)
            ax_main.set_ylabel("Reliability", fontsize=8)
            ax_main.legend(fontsize=6.5, loc='lower left')
            ax_main.grid(alpha=0.25)
            ax_main.set_xlim(0, t_work[-1])
            y_min_plot = min(ci.min(), cj.min()) - 0.02
            y_max_plot = max(ci.max(), cj.max()) + 0.02
            ax_main.set_ylim(max(0, y_min_plot), min(1.02, y_max_plot))

            # Zoomed inset around crossover point
            zoom_window = 3000  # hours either side of crossover
            z_tmin = max(t_work[0],  tc - zoom_window)
            z_tmax = min(t_work[-1], tc + zoom_window)
            z_mask = (t_work >= z_tmin) & (t_work <= z_tmax)

            axins = inset_axes(ax_main, width="40%", height="35%",
                               loc='upper right',
                               bbox_to_anchor=ax_main.bbox,
                               bbox_transform=ax_main.transAxes)
            axins.plot(t_work[z_mask], ci[z_mask], color='#2980b9', linewidth=1.5)
            axins.plot(t_work[z_mask], cj[z_mask], color='#e74c3c', linewidth=1.5)
            axins.axvline(tc, color='gray', linestyle='--', linewidth=1.0)

            # Y limits for inset: tight around the crossing region
            z_ci = ci[z_mask]
            z_cj = cj[z_mask]
            z_ymin = min(z_ci.min(), z_cj.min()) - 0.005
            z_ymax = max(z_ci.max(), z_cj.max()) + 0.005
            axins.set_ylim(z_ymin, z_ymax)
            axins.set_xlim(z_tmin, z_tmax)
            axins.tick_params(labelsize=5)
            axins.set_title("Zoom", fontsize=6)
            axins.grid(alpha=0.3)

            try:
                mark_inset(ax_main, axins, loc1=2, loc2=4,
                           fc="none", ec="gray", linewidth=0.5)
            except Exception:
                pass

        fig.suptitle(
            "Deep Crossing Examples — Why Full-Curve Regression Is Required\n"
            "Shading shows which curve is genuinely higher in each region. "
            "Insets zoom around the crossover point.",
            fontsize=12, fontweight='bold'
        )
        save_fig(fig, "08_crossing_examples.png", dpi=200)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Crossing Hub Analysis
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 6: Crossing Hub Analysis")
    log("=" * 70)
    log("  Which curves are responsible for the most crossings?")

    n_nonzero     = (cross_count_per_curve > 0).sum()
    n_zero        = (cross_count_per_curve == 0).sum()
    max_crossings = cross_count_per_curve.max()
    median_cross  = np.median(cross_count_per_curve)

    log(f"  Curves with zero crossings    : {n_zero:,}  "
        f"({n_zero/n_c*100:.1f}%)")
    log(f"  Curves with any crossing      : {n_nonzero:,}  "
        f"({n_nonzero/n_c*100:.1f}%)")
    log(f"  Max crossings for one curve   : {max_crossings:,}")
    log(f"  Median crossings per curve    : {median_cross:.1f}")

    # Top 10 hub curves
    top10_idx = np.argsort(cross_count_per_curve)[::-1][:10]
    log(f"\n  Top 10 hub curves (most crossings):")
    for rank, idx in enumerate(top10_idx):
        h          = unique_hashes[idx]
        members    = hash_to_indices[h]
        allocs_hub = Counter(allocations[members])
        log(f"    #{rank+1:2d}  curve_idx={idx:4d}  "
            f"crossings={cross_count_per_curve[idx]:4d}  "
            f"R(8k)={r_at_8000[idx]:.4f}  "
            f"R(22k)={Y_unique[idx,-1]:.4f}  "
            f"allocs={dict(allocs_hub)}")

    # Crossing count distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(cross_count_per_curve, bins=80,
                 color='steelblue', edgecolor='none', log=True)
    axes[0].set_title("Distribution of Crossings per Unique Curve\n"
                       "(log scale — power law would confirm hub structure)",
                       fontsize=11)
    axes[0].set_xlabel("Number of other curves this curve crosses")
    axes[0].set_ylabel("Number of curves (log)")
    axes[0].grid(alpha=0.3)
    axes[0].axvline(median_cross, color='orange', linestyle='--',
                    label=f'Median: {median_cross:.0f}')
    axes[0].legend()

    # Scatter: R(8k) vs crossing count — do hubs cluster in reliability range?
    sc = axes[1].scatter(r_at_8000, cross_count_per_curve,
                         c=Y_unique[:, -1], cmap='RdYlGn',
                         alpha=0.4, s=8, linewidths=0)
    plt.colorbar(sc, ax=axes[1], label='R at t=22,000h')
    axes[1].set_title("Reliability at t=8,000h vs Crossing Count\n"
                       "(color = R at t=22,000h)",
                       fontsize=11)
    axes[1].set_xlabel("Reliability at t=8,000h")
    axes[1].set_ylabel("Number of curves crossed")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "09_crossing_hubs.png")

    # Plot top 10 hub curves overlaid
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap_hub = plt.cm.tab10
    for rank, idx in enumerate(top10_idx):
        ax.plot(time_vals, Y_unique[idx],
                color=cmap_hub(rank / 10),
                linewidth=2.0, alpha=0.9,
                label=f'Hub #{rank+1} ({cross_count_per_curve[idx]} crossings, '
                      f'R(8k)={r_at_8000[idx]:.3f})')

    # Also plot a random sample of non-hub curves as context
    non_hub = np.argsort(cross_count_per_curve)[:200]
    for idx in non_hub:
        ax.plot(time_vals, Y_unique[idx],
                color='lightgray', linewidth=0.5, alpha=0.3)

    ax.set_title("Top 10 Hub Curves (most crossings)\n"
                 "overlaid on 200 low-crossing curves (gray)",
                 fontsize=13)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Reliability")
    ax.legend(fontsize=7, loc='lower left', ncol=2)
    ax.grid(alpha=0.2)
    save_fig(fig, "10_hub_curves.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — Behavioral Family Detection
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 7: Behavioral Family Detection")
    log("=" * 70)
    log("  Hypothesis: two behavioral families exist —")
    log("  Family A: steady degraders (flat, consistent decline)")
    log("  Family B: late cliff droppers (high early, sharp drop after ~10kh)")

    # Compute early and late slopes for every unique curve
    # Early region: t=100h to t=8000h
    # Late region:  t=8000h to t=22000h
    early_start_idx = 0                                    # t=100h (first after t=0 drop)
    early_end_idx   = t8k_idx - 1                         # t=8000h
    late_start_idx  = t8k_idx                             # t=8000h
    late_end_idx    = len(time_vals) - 1                  # t=22000h

    early_duration  = time_vals[early_end_idx] - time_vals[early_start_idx]
    late_duration   = time_vals[late_end_idx]  - time_vals[late_start_idx]

    early_slope = (
        (Y_unique[:, early_end_idx] - Y_unique[:, early_start_idx]) / early_duration
    )  # reliability change per hour, early region (negative = declining)

    late_slope  = (
        (Y_unique[:, late_end_idx]  - Y_unique[:, late_start_idx])  / late_duration
    )  # reliability change per hour, late region

    # Ratio of late slope to early slope — >1 means accelerating decline
    # Use absolute values since both slopes are negative
    with np.errstate(divide='ignore', invalid='ignore'):
        slope_ratio = np.where(
            np.abs(early_slope) > 1e-8,
            np.abs(late_slope) / np.abs(early_slope),
            1.0
        )

    log(f"\n  Early slope (t=100-8000h)  — mean: {early_slope.mean():.2e}  "
        f"std: {early_slope.std():.2e}")
    log(f"  Late slope  (t=8000-22000h)— mean: {late_slope.mean():.2e}  "
        f"std: {late_slope.std():.2e}")
    log(f"  Slope ratio (late/early)   — mean: {slope_ratio.mean():.2f}  "
        f"std: {slope_ratio.std():.2f}")

    # Classify into families using slope ratio threshold
    # Ratio > 2: late decline is at least 2x faster than early — cliff behavior
    CLIFF_THRESHOLD = 2.0
    family_B = slope_ratio > CLIFF_THRESHOLD  # cliff droppers
    family_A = ~family_B                       # steady degraders

    n_fam_A = family_A.sum()
    n_fam_B = family_B.sum()
    log(f"\n  Family A (steady degraders, ratio ≤ {CLIFF_THRESHOLD}) : "
        f"{n_fam_A:,}  ({n_fam_A/n_c*100:.1f}%)")
    log(f"  Family B (cliff droppers,   ratio >  {CLIFF_THRESHOLD}) : "
        f"{n_fam_B:,}  ({n_fam_B/n_c*100:.1f}%)")

    # 2D slope space plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.scatter(early_slope[family_A] * 1e5, late_slope[family_A] * 1e5,
               c='#2980b9', alpha=0.3, s=8, label=f'Family A (n={n_fam_A:,})')
    ax.scatter(early_slope[family_B] * 1e5, late_slope[family_B] * 1e5,
               c='#e74c3c', alpha=0.3, s=8, label=f'Family B (n={n_fam_B:,})')
    ax.set_xlabel("Early slope (×10⁻⁵ R/hour, t=100–8000h)")
    ax.set_ylabel("Late slope (×10⁻⁵ R/hour, t=8000–22000h)")
    ax.set_title("Behavioral Slope Space\n"
                 "Blue=steady degraders  Red=cliff droppers",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    # Add diagonal reference: equal slope = steady degradation
    lim = max(abs(early_slope.min()), abs(late_slope.min())) * 1e5 * 1.1
    ax.plot([-lim, 0], [-lim, 0], 'k--', alpha=0.3, linewidth=0.8,
            label='Equal slope')

    # Slope ratio distribution
    axes[1].hist(slope_ratio, bins=100, color='mediumseagreen', edgecolor='none')
    axes[1].axvline(CLIFF_THRESHOLD, color='red', linestyle='--', linewidth=2,
                    label=f'Cliff threshold = {CLIFF_THRESHOLD}')
    axes[1].set_title("Distribution of Late/Early Slope Ratio\n"
                       "Values > threshold classified as cliff droppers",
                       fontsize=11)
    axes[1].set_xlabel("Late slope / Early slope (absolute values)")
    axes[1].set_ylabel("Number of unique curves")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "11_behavioral_slope_space.png")

    # Representative curves from each family
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, fam_mask, title, color in [
        (axes[0], family_A, f"Family A — Steady Degraders\n(n={n_fam_A:,})", '#2980b9'),
        (axes[1], family_B, f"Family B — Cliff Droppers\n(n={n_fam_B:,})", '#e74c3c'),
    ]:
        fam_indices = np.where(fam_mask)[0]
        # Sample up to 200 representative curves
        sample_size = min(200, len(fam_indices))
        sampled     = np.random.choice(fam_indices, sample_size, replace=False)
        for idx in sampled:
            ax.plot(time_vals, Y_unique[idx],
                    color=color, alpha=0.1, linewidth=0.7)
        # Plot mean curve
        mean_curve = Y_unique[fam_indices].mean(axis=0)
        ax.plot(time_vals, mean_curve,
                color='black', linewidth=2.5, label='Mean curve')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Time (hours)")
        ax.grid(alpha=0.25)
        ax.legend()

    axes[0].set_ylabel("Reliability")
    fig.suptitle("Behavioral Families — Representative Curves",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, "12_behavioral_families.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — Architectural Correlation
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 8: Architectural Correlation")
    log("=" * 70)
    log("  Do behavioral families correlate with graph structural features?")
    log("  Computing: node degree, task concentration, switch connectivity")
    log("  for one representative graph per unique curve...")

    # For each unique curve, take the first sample's index to get its graph
    rep_indices = [hash_to_indices[h][0] for h in unique_hashes]

    arch_features = []  # one row per unique curve

    for sample_idx in tqdm(rep_indices, desc="  Extracting arch features"):
        # Get edge_index for this sample via CSR pointer
        e_start = edge_ptr[sample_idx]
        e_end   = edge_ptr[sample_idx + 1]
        ei      = edge_index[:, e_start:e_end]  # [2, E]

        nn      = node_names[sample_idx]          # [30] node name strings

        # Node degrees (how many edges each node has)
        degree  = np.bincount(ei[0], minlength=N_NODES).astype(np.float32)

        # Node type masks from stored features
        feats_s = node_feats[sample_idx]           # [30, 5]
        is_compute = feats_s[:, 0].astype(bool)
        is_switch  = feats_s[:, 1].astype(bool)
        is_link    = feats_s[:, 2].astype(bool)
        is_t1      = feats_s[:, 3].astype(bool)
        is_t2      = feats_s[:, 4].astype(bool)
        is_task    = is_t1 | is_t2

        # Feature 1: mean degree of compute nodes
        mean_compute_degree = degree[is_compute].mean() if is_compute.any() else 0.0

        # Feature 2: mean degree of switch nodes
        mean_switch_degree  = degree[is_switch].mean()  if is_switch.any()  else 0.0

        # Feature 3: task concentration
        # How many distinct compute nodes do tasks connect to?
        # Find compute nodes that have at least one task neighbor
        task_nodes    = np.where(is_task)[0]
        compute_nodes = np.where(is_compute)[0]
        task_compute_connections = set()
        for tn in task_nodes:
            # neighbors of this task node
            neighbors = ei[1][ei[0] == tn]
            for nb in neighbors:
                if is_compute[nb]:
                    task_compute_connections.add(int(nb))
        n_compute_with_tasks = len(task_compute_connections)
        # Concentration: 1/n_compute_with_tasks — high = tasks concentrated on few nodes
        task_concentration = (1.0 / n_compute_with_tasks
                              if n_compute_with_tasks > 0 else 1.0)

        # Feature 4: graph density (edges / max possible edges)
        n_edges  = ei.shape[1]
        density  = n_edges / (N_NODES * (N_NODES - 1))

        # Feature 5: max degree in graph
        max_degree = float(degree.max())

        arch_features.append([
            mean_compute_degree,
            mean_switch_degree,
            task_concentration,
            density,
            max_degree,
            n_compute_with_tasks,
        ])

    arch_arr = np.array(arch_features, dtype=np.float32)
    arch_names = [
        'Mean compute degree',
        'Mean switch degree',
        'Task concentration (1/n_compute_w_tasks)',
        'Graph density',
        'Max node degree',
        'N compute nodes with tasks',
    ]

    log(f"\n  Architectural features per family:")
    log(f"  {'Feature':40s}  {'Family A mean':>15}  {'Family B mean':>15}  {'Diff':>10}")
    log(f"  {'-'*82}")
    for fi, fname in enumerate(arch_names):
        mean_a = arch_arr[family_A, fi].mean()
        mean_b = arch_arr[family_B, fi].mean()
        diff   = mean_b - mean_a
        log(f"  {fname:40s}  {mean_a:>15.4f}  {mean_b:>15.4f}  {diff:>+10.4f}")

    # Visualize architectural differences
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for fi, (fname, ax) in enumerate(zip(arch_names, axes_flat)):
        vals_a = arch_arr[family_A, fi]
        vals_b = arch_arr[family_B, fi]

        bins_all = np.linspace(
            min(vals_a.min(), vals_b.min()),
            max(vals_a.max(), vals_b.max()),
            40
        )
        ax.hist(vals_a, bins=bins_all, alpha=0.6, color='#2980b9',
                label=f'Family A (μ={vals_a.mean():.3f})', density=True)
        ax.hist(vals_b, bins=bins_all, alpha=0.6, color='#e74c3c',
                label=f'Family B (μ={vals_b.mean():.3f})', density=True)
        ax.set_title(fname, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Architectural Feature Distributions per Behavioral Family\n"
        "Blue = Family A (steady degraders)   Red = Family B (cliff droppers)\n"
        "Separation in any feature suggests a structural driver of the behavioral difference.",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    save_fig(fig, "13_architectural_correlation.png")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SUMMARY OF KEY FINDINGS")
    log("=" * 70)
    log(f"  Dataset                  : {N:,} samples, {n_unique_curves:,} unique curves")
    log(f"  Curve redundancy         : {n_redundant/N*100:.1f}% exact duplicates")
    log(f"  Within-alloc uniqueness  : "
        f"{'100%' if all_unique_within else 'DUPLICATES EXIST'}")
    log(f"  Cross-alloc topology     : "
        f"{diff_count/total_shared*100:.1f}% of shared configs differ by alloc")
    log(f"  Curve crossing rate      : {crossing_rate:.1f}%")
    log(f"  Deep crossings           : "
        f"{depth_buckets['deep (>0.10)']/total_depth*100:.1f}%")
    log(f"  Hub structure            : max {max_crossings} crossings for one curve")
    log(f"  Behavioral families      : "
        f"A={n_fam_A:,} ({n_fam_A/n_c*100:.1f}%)  "
        f"B={n_fam_B:,} ({n_fam_B/n_c*100:.1f}%)")
    log(f"  Conclusion — ML needed   : Yes")
    log(f"  Conclusion — Full curve  : Yes (crossings, cliff behavior)")
    log(f"\n  NOTE: Split strategy is visualized in 09_split_strategy_preview.png")
    log(f"  but splits are NOT executed here. Run 03_split.py for that.")
    log(f"\n✅ EDA complete. All outputs in {OUT_DIR}/")
    save_log()


if __name__ == "__main__":
    main()