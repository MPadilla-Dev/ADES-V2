"""
02_eda_deep.py
==============
STEP 2 of the ADES-v2 pipeline — Deep Exploratory Data Analysis.

Runs AFTER 01_preprocess.py has created data/dataset.h5.
Analysis only — does NOT produce splits. Splitting is in 03_split.py.

Sections:
    1.  Dataset inventory and allocation balance
    2.  Two-axis dataset structure
        2a. Allocation-first view
        2b. Hardware hash summary (MD5 11,903 vs WL 392)
    3.  Graph node composition and within-allocation uniqueness
    4.  Cross-allocation hardware identity
        4b. Config comparison: 0000_0348 vs 0000_0495
            (isomorphic hardware, same allocation, different curve)
    5.  Reliability curve uniqueness
    6.  Curve crossings — full pairwise on unique curves
        — diverse early/mid/late examples, one per source curve
    7.  Crossing hub analysis (unique curves only)
    8.  Behavioral slope space (continuous, no forced families)
    9.  Architectural correlation (continuous scatter plots)

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

# Two specific configs used in the hardware identity analysis
CONFIG_A = '0000_0348'
CONFIG_B = '0000_0495'

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
    log("  ADES-v2 DEEP EDA (final)")
    log("  Full dataset — no sampling, no split execution.")
    log("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────────────
    log(f"\nLoading dataset from {H5_PATH}...")
    with h5py.File(H5_PATH, 'r') as h5:
        config_ids      = h5['meta/config_ids'][:].astype(str)
        allocations     = h5['meta/allocations'][:].astype(str)
        config_nums     = h5['meta/config_nums'][:].astype(str)
        curve_hashes    = h5['meta/curve_hashes'][:].astype(str)
        adj_hashes      = h5['meta/adj_hashes'][:].astype(str)
        hw_md5_hashes   = h5['meta/hw_md5_hashes'][:].astype(str)
        hw_wl_hashes    = h5['meta/hw_wl_hashes'][:].astype(str)
        node_names      = h5['meta/node_names'][:].astype(str)
        y_curve         = h5['targets/y_curve'][:]
        node_feats      = h5['features/node_features'][:]
        edge_index      = h5['edges/edge_index'][:]
        edge_ptr        = h5['edges/edge_ptr'][:]
        time_vals       = h5.attrs['time_values']

    N            = len(config_ids)
    N_TIME       = len(time_vals)
    N_NODES      = node_feats.shape[1]
    feat_names   = ['compute', 'switch', 'link', 'task_T1', 'task_T2']
    alloc_unique = sorted(set(allocations))
    t8k_idx      = np.argmin(np.abs(time_vals - 8000))

    log(f"  Samples         : {N:,}")
    log(f"  Time steps      : {N_TIME}  ({time_vals[0]:.0f}h - {time_vals[-1]:.0f}h)")
    log(f"  Allocations     : {len(alloc_unique)}")
    log(f"  Unique curves   : {len(set(curve_hashes)):,}")
    log(f"  Unique hw MD5   : {len(set(hw_md5_hashes)):,}")
    log(f"  Unique hw WL    : {len(set(hw_wl_hashes)):,}")

    # Pre-build index structures
    alloc_counts     = Counter(allocations)
    hash_to_indices  = defaultdict(list)
    for i, ch in enumerate(curve_hashes):
        hash_to_indices[ch].append(i)
    unique_hashes = list(hash_to_indices.keys())
    Y_unique      = np.array([y_curve[hash_to_indices[h][0]]
                               for h in unique_hashes])
    r_at_8000     = Y_unique[:, t8k_idx]

    hw_md5_to_indices = defaultdict(list)
    hw_wl_to_indices  = defaultdict(list)
    for i in range(N):
        hw_md5_to_indices[hw_md5_hashes[i]].append(i)
        hw_wl_to_indices[hw_wl_hashes[i]].append(i)

    unique_md5 = list(hw_md5_to_indices.keys())
    unique_wl  = list(hw_wl_to_indices.keys())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Dataset Inventory
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 1: Dataset Inventory")
    log("=" * 70)

    counts_arr = [alloc_counts[a] for a in alloc_unique]
    log(f"  Total samples           : {N:,}")
    log(f"  Allocations             : {len(alloc_unique)}")
    log(f"  Min/Max samples/alloc   : {min(counts_arr):,} / {max(counts_arr):,}")
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
    # SECTION 2a — Allocation-First View
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 2a: Allocation-First View")
    log("=" * 70)

    log(f"\n  {'Alloc':>6}  {'Samples':>8}  {'HW MD5':>8}  "
        f"{'HW WL':>7}  {'Curves':>7}")
    log(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")

    alloc_hw_md5_counts = []
    alloc_hw_wl_counts  = []
    alloc_curve_counts  = []

    for alloc in alloc_unique:
        mask    = allocations == alloc
        n_samp  = mask.sum()
        n_md5   = len(set(hw_md5_hashes[mask]))
        n_wl    = len(set(hw_wl_hashes[mask]))
        n_curv  = len(set(curve_hashes[mask]))
        alloc_hw_md5_counts.append(n_md5)
        alloc_hw_wl_counts.append(n_wl)
        alloc_curve_counts.append(n_curv)
        log(f"  {alloc:>6}  {n_samp:>8,}  {n_md5:>8,}  {n_wl:>7,}  {n_curv:>7,}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)
    x = np.arange(len(alloc_unique))
    for ax, vals, title, color in [
        (axes[0], alloc_hw_md5_counts, 'HW Wirings (MD5)\nper Allocation', 'steelblue'),
        (axes[1], alloc_hw_wl_counts,  'HW Topology Groups (WL)\nper Allocation', 'darkorange'),
        (axes[2], alloc_curve_counts,  'Unique Curves\nper Allocation', 'mediumseagreen'),
    ]:
        ax.bar(alloc_unique, vals, color=color, edgecolor='white')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Allocation ID")
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel("Count")
    plt.suptitle("Allocation-First View: Hardware and Curve Counts per Allocation",
                 fontsize=13)
    plt.tight_layout()
    save_fig(fig, "02_allocation_first_view.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2b — Hardware Hash Summary (MD5 vs WL)
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 2b: Hardware Hash Summary — MD5 (11,903) vs WL (392)")
    log("=" * 70)
    log("""
  Two levels of hardware equivalence are stored in the dataset:

  hw_md5_hashes (11,903 unique groups):
    Definition : MD5 of sorted hardware-only edge set (task edges stripped)
    Meaning    : Same physical wiring — identical compute-to-switch and
                 switch-to-switch connections regardless of task placement
    Variation  : Curves differ within a group ONLY because different task
                 allocations place tasks on differently-connected nodes

  hw_wl_hashes (392 unique groups, VF2-proven):
    Definition : Weisfeiler-Lehman hash of hardware subgraph (node type attr)
    Meaning    : Same abstract hardware topology — same degree sequence per
                 node type, even if different specific nodes fill each role
    Variation  : Curves can differ within a group because different labeled
                 nodes (e.g. N6 vs N3) may have different switch connectivity
                 despite being the same node type
    Key finding: Within a WL group, the same allocation can produce different
                 curves if the task-hosting node has different degree in
                 different physical wirings (see Section 4b)
    Open question: Are all compute nodes physically identical in the CTMC
                   model? If yes, WL is the correct hardware identity.
                   If no, MD5 is. We test both empirically.
    """)

    # Curves per MD5 group
    md5_curves = [len(set(curve_hashes[hw_md5_to_indices[h]]))
                  for h in unique_md5]
    wl_curves  = [len(set(curve_hashes[hw_wl_to_indices[h]]))
                  for h in unique_wl]

    # Allocations per MD5 and WL group
    md5_allocs = [len(set(allocations[hw_md5_to_indices[h]]))
                  for h in unique_md5]
    wl_allocs  = [len(set(allocations[hw_wl_to_indices[h]]))
                  for h in unique_wl]

    log(f"  MD5 groups — curves per group: "
        f"min={min(md5_curves)}  max={max(md5_curves)}  "
        f"mean={np.mean(md5_curves):.2f}")
    log(f"  WL  groups — curves per group: "
        f"min={min(wl_curves)}  max={max(wl_curves)}  "
        f"mean={np.mean(wl_curves):.2f}")
    log(f"  MD5 groups — allocs per group: "
        f"min={min(md5_allocs)}  max={max(md5_allocs)}  "
        f"mean={np.mean(md5_allocs):.2f}")
    log(f"  WL  groups — allocs per group: "
        f"min={min(wl_allocs)}  max={max(wl_allocs)}  "
        f"mean={np.mean(wl_allocs):.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # MD5 curves per group
    md5_curve_dist = Counter(md5_curves)
    axes[0,0].bar(sorted(md5_curve_dist.keys()),
                  [md5_curve_dist[k] for k in sorted(md5_curve_dist.keys())],
                  color='steelblue', edgecolor='white')
    axes[0,0].set_title(f"MD5 Hardware Groups (11,903)\n"
                        f"Unique Curves per Group\n"
                        f"(variation = task allocation only)",
                        fontsize=10)
    axes[0,0].set_xlabel("Unique curves produced by this hardware wiring")
    axes[0,0].set_ylabel("Number of hardware wirings")
    axes[0,0].grid(alpha=0.3)

    # WL curves per group
    wl_curve_dist = Counter(wl_curves)
    axes[0,1].bar(sorted(wl_curve_dist.keys()),
                  [wl_curve_dist[k] for k in sorted(wl_curve_dist.keys())],
                  color='darkorange', edgecolor='white')
    axes[0,1].set_title(f"WL Hardware Groups (392)\n"
                        f"Unique Curves per Group\n"
                        f"(variation = task allocation + wiring differences)",
                        fontsize=10)
    axes[0,1].set_xlabel("Unique curves produced by this topology class")
    axes[0,1].set_ylabel("Number of topology classes")
    axes[0,1].grid(alpha=0.3)

    # MD5 allocations per group
    md5_alloc_dist = Counter(md5_allocs)
    axes[1,0].bar(sorted(md5_alloc_dist.keys()),
                  [md5_alloc_dist[k] for k in sorted(md5_alloc_dist.keys())],
                  color='steelblue', edgecolor='white', alpha=0.7)
    axes[1,0].set_title("MD5 Hardware Groups\nAllocations Tested per Wiring",
                        fontsize=10)
    axes[1,0].set_xlabel("Number of allocations this wiring appears in")
    axes[1,0].set_ylabel("Number of hardware wirings")
    axes[1,0].grid(alpha=0.3)

    # WL allocations per group
    wl_alloc_dist = Counter(wl_allocs)
    axes[1,1].bar(sorted(wl_alloc_dist.keys()),
                  [wl_alloc_dist[k] for k in sorted(wl_alloc_dist.keys())],
                  color='darkorange', edgecolor='white', alpha=0.7)
    axes[1,1].set_title("WL Hardware Groups\nAllocations Tested per Topology Class",
                        fontsize=10)
    axes[1,1].set_xlabel("Number of allocations this topology class appears in")
    axes[1,1].set_ylabel("Number of topology classes")
    axes[1,1].grid(alpha=0.3)

    fig.suptitle(
        "Hardware Hash Comparison: MD5 Physical Wiring (11,903) vs "
        "WL Topology Class (392)\n"
        "MD5: curves vary only from task allocation  |  "
        "WL: curves vary from task allocation AND wiring differences within class",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    save_fig(fig, "03_hardware_hash_summary.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Graph Node Composition
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 3: Graph Node Composition")
    log("=" * 70)

    type_totals = node_feats.sum(axis=0).sum(axis=0)
    log("\n  Global node type totals:")
    for fname, total in zip(feat_names, type_totals):
        log(f"    {fname:10s}: {int(total):,}  ({int(total)/N:.2f} per graph)")

    all_unique_within = True
    for alloc in alloc_unique:
        mask    = allocations == alloc
        hw_alloc = hw_md5_hashes[mask]
        if len(set(hw_alloc)) < len(hw_alloc):
            all_unique_within = False
            log(f"  DUPLICATE hw_md5 within alloc {alloc}!")
    if all_unique_within:
        log("  Within-allocation HW MD5 uniqueness: 100% confirmed.")

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
    save_fig(fig, "04_node_types_per_allocation.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Cross-Allocation Hardware Identity
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 4: Cross-Allocation Hardware Identity")
    log("=" * 70)

    n_shared  = sum(1 for h in unique_md5
                    if len(set(allocations[hw_md5_to_indices[h]])) > 1)
    n_single  = len(unique_md5) - n_shared
    log(f"  MD5 wirings in exactly 1 allocation : {n_single:,}")
    log(f"  MD5 wirings shared across allocations: {n_shared:,}")

    # Allocation symmetry groups
    log("\n  Allocation symmetry groups (same unique curve count):")
    curve_count_to_allocs = defaultdict(list)
    for alloc, n_curves in zip(alloc_unique, alloc_curve_counts):
        curve_count_to_allocs[n_curves].append(alloc)
    for n_curves, allocs in sorted(curve_count_to_allocs.items()):
        if len(allocs) > 1:
            log(f"    {n_curves:4d} unique curves: {allocs}")
        else:
            log(f"    {n_curves:4d} unique curves: {allocs[0]}  (unique)")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(['In 1 allocation\nonly', 'Shared across\nallocations'],
           [n_single, n_shared],
           color=['#3498db', '#e74c3c'], edgecolor='white', width=0.5)
    for bar, val in zip(ax.patches, [n_single, n_shared]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 50,
                f'{val:,}\n({val/len(unique_md5)*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title("Hardware Wiring Sharing Across Allocations\n"
                 "(MD5 physical wiring level)", fontsize=12)
    ax.set_ylabel("Number of unique hardware wirings")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "05_cross_alloc_identity.png")

    # ── Section 4b — Config Comparison ────────────────────────────────────────
    log("\n  SECTION 4b: Config Comparison — Isomorphic Hardware, Different Curves")
    log(f"  Comparing {CONFIG_A} vs {CONFIG_B}")
    log("  Both: same WL topology group, same allocation (0000)")
    log("  Both: same node names, same type-degree signature")
    log("  Different: physical wiring (hw_md5 differs)")
    log("  Different: reliability curves")

    type_colors = {
        'compute': '#2980b9',
        'switch' : '#27ae60',
        'link'   : '#e67e22',
        'task_T1': '#8e44ad',
        'task_T2': '#c0392b',
    }
    type_name_list = ['compute', 'switch', 'link', 'task_T1', 'task_T2']

    def get_named_edges(idx):
        e_start = int(edge_ptr[idx])
        e_end   = int(edge_ptr[idx + 1])
        ei = edge_index[:, e_start:e_end]
        nn = node_names[idx]
        return set(tuple(sorted([nn[int(ei[0,e])], nn[int(ei[1,e])]]))
                   for e in range(ei.shape[1]))

    idx_a = np.where(config_ids == CONFIG_A)[0][0]
    idx_b = np.where(config_ids == CONFIG_B)[0][0]

    edges_a   = get_named_edges(idx_a)
    edges_b   = get_named_edges(idx_b)
    shared    = edges_a & edges_b
    only_a    = edges_a - edges_b
    only_b    = edges_b - edges_a

    log(f"  Shared edges   : {len(shared)}")
    log(f"  Only in A      : {len(only_a)}")
    log(f"  Only in B      : {len(only_b)}")
    log(f"  Curve diff R(8k): "
        f"{abs(y_curve[idx_a,80]-y_curve[idx_b,80]):.6f}")
    log(f"  hw_md5 same    : {hw_md5_hashes[idx_a]==hw_md5_hashes[idx_b]}")
    log(f"  hw_wl same     : {hw_wl_hashes[idx_a]==hw_wl_hashes[idx_b]}")

    import networkx as nx

    # Build supergraph for consistent layout
    G_super = nx.Graph()
    for idx in [idx_a, idx_b]:
        for n in range(N_NODES):
            ntype = type_name_list[int(node_feats[idx, n].argmax())]
            G_super.add_node(node_names[idx][n], ntype=ntype)
    for e in (edges_a | edges_b):
        G_super.add_edge(e[0], e[1])

    pos = nx.spring_layout(G_super, seed=42, k=2.5)

    fig, axes = plt.subplots(1, 3, figsize=(26, 10))

    for col, (cid, idx, only_this, only_other) in enumerate([
        (CONFIG_A, idx_a, only_a, only_b),
        (CONFIG_B, idx_b, only_b, only_a),
    ]):
        ax = axes[col]
        node_colors = [type_colors[G_super.nodes[n]['ntype']]
                       for n in G_super.nodes]

        # Shared edges — gray
        nx.draw_networkx_edges(
            G_super, pos, ax=ax,
            edgelist=list(shared),
            edge_color='#aaaaaa', width=1.0, alpha=0.5
        )
        # Unique to this config — red solid thick
        if only_this:
            nx.draw_networkx_edges(
                G_super, pos, ax=ax,
                edgelist=list(only_this),
                edge_color='#e74c3c', width=3.5, alpha=1.0
            )
        # Unique to other config — blue dashed
        if only_other:
            nx.draw_networkx_edges(
                G_super, pos, ax=ax,
                edgelist=list(only_other),
                edge_color='#2980b9', width=2.0, alpha=0.6, style='dashed'
            )

        diff_nodes = set()
        for u, v in (only_this | only_other):
            diff_nodes.add(u)
            diff_nodes.add(v)

        nx.draw_networkx_nodes(G_super, pos, ax=ax,
                               node_color=node_colors,
                               node_size=500, alpha=0.85)
        if diff_nodes:
            nx.draw_networkx_nodes(G_super, pos, ax=ax,
                                   nodelist=list(diff_nodes),
                                   node_color='none', node_size=700,
                                   linewidths=3.0, edgecolors='black')
        nx.draw_networkx_labels(G_super, pos, ax=ax,
                                font_size=5.5, font_color='white',
                                font_weight='bold')

        r8k = y_curve[idx, t8k_idx]
        wl_same = hw_wl_hashes[idx_a] == hw_wl_hashes[idx_b]
        ax.set_title(
            f"Config {cid}\n"
            f"R(8,000h) = {r8k:.6f}\n"
            f"Red solid = edges only here  |  Blue dashed = edges only in other",
            fontsize=9
        )
        ax.axis('off')

    # Third panel: curve comparison
    ax3 = axes[2]
    colors_c = ['#e74c3c', '#2980b9']
    for i, (cid, idx) in enumerate([(CONFIG_A, idx_a), (CONFIG_B, idx_b)]):
        ax3.plot(time_vals, y_curve[idx],
                 color=colors_c[i], linewidth=2.2, label=cid)
    ax3.fill_between(time_vals, y_curve[idx_a], y_curve[idx_b],
                     alpha=0.2, color='gray',
                     label=f'Diff (max={np.abs(y_curve[idx_a]-y_curve[idx_b]).max():.5f})')
    ax3.set_title("Reliability Curves\n(shaded = difference)", fontsize=11)
    ax3.set_xlabel("Time (hours)")
    ax3.set_ylabel("Reliability")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(color=type_colors['compute'],  label='Compute (N1-N6)'),
        Patch(color=type_colors['switch'],   label='Switch  (S1-S3)'),
        Patch(color=type_colors['link'],     label='Link    (NxSy)'),
        Patch(color=type_colors['task_T1'],  label='Task T1'),
        Patch(color=type_colors['task_T2'],  label='Task T2'),
        Line2D([0],[0], color='#e74c3c', linewidth=3,
               label='Edge only in this config'),
        Line2D([0],[0], color='#2980b9', linewidth=2, linestyle='dashed',
               label='Edge only in other config'),
        Patch(color='none', edgecolor='black', linewidth=2,
              label='Node involved in difference'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        f"Hardware Identity Question: {CONFIG_A} vs {CONFIG_B}\n"
        f"Same WL topology group (hw_wl same: {hw_wl_hashes[idx_a]==hw_wl_hashes[idx_b]})  |  "
        f"Same allocation (0000)  |  Same node names  |  Same type-degree signature\n"
        f"Different physical wiring (hw_md5 same: "
        f"{hw_md5_hashes[idx_a]==hw_md5_hashes[idx_b]})  |  "
        f"Different reliability curves  |  Black rings = nodes in differing edges",
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, "06_config_comparison.png", dpi=200)

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

    log(f"  Total samples    : {N:,}")
    log(f"  Unique curves    : {n_unique_curves:,}")
    log(f"  Redundant copies : {n_redundant:,} ({n_redundant/N*100:.1f}%)")
    log(f"  Largest group    : {group_sizes[0]:,}")

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

    t0_idx = np.where(time_vals == 0)[0]
    Y_work = np.delete(Y_unique, t0_idx, axis=1)
    t_work = np.delete(time_vals, t0_idx)
    n_c    = Y_work.shape[0]

    depth_buckets = {
        'shallow (0.01–0.05)' : 0,
        'moderate (0.05–0.10)': 0,
        'deep (>0.10)'        : 0,
    }
    crossing_count        = 0
    total_pairs           = 0
    cross_count_per_curve = np.zeros(n_c, dtype=np.int32)

    t_early_max = 5000
    t_mid_max   = 12000
    examples_early = []
    examples_mid   = []
    examples_late  = []
    seen_source    = set()

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
                    if i not in seen_source and d > 0.10:
                        j        = i + 1 + np.where(crossing)[0][k]
                        sign_chg = np.where(
                            np.diff(np.sign(cross_diffs[k]))
                        )[0]
                        if len(sign_chg) > 0:
                            t_cross = t_work[sign_chg[0]]
                            ex = {
                                'i': i, 'j': j, 't_cross': t_cross,
                                'max_diff': d,
                                'curve_i': Y_work[i].copy(),
                                'curve_j': Y_work[j].copy(),
                                'diff': cross_diffs[k].copy(),
                            }
                            if t_cross < t_early_max and len(examples_early) < 2:
                                examples_early.append(ex)
                                seen_source.add(i)
                            elif (t_early_max <= t_cross < t_mid_max
                                  and len(examples_mid) < 2):
                                examples_mid.append(ex)
                                seen_source.add(i)
                            elif t_cross >= t_mid_max and len(examples_late) < 2:
                                examples_late.append(ex)
                                seen_source.add(i)

    crossing_rate = crossing_count / total_pairs * 100 if total_pairs > 0 else 0
    total_depth   = sum(depth_buckets.values())

    log(f"  Meaningful pairs  : {total_pairs:,}")
    log(f"  Crossing pairs    : {crossing_count:,}")
    log(f"  Crossing rate     : {crossing_rate:.2f}%")
    for label, count in depth_buckets.items():
        pct = count / total_depth * 100 if total_depth > 0 else 0
        log(f"    {label:25s}: {count:8,}  ({pct:.1f}%)")

    moderate_deep = (depth_buckets['moderate (0.05–0.10)'] +
                     depth_buckets['deep (>0.10)'])
    log(f"  Moderate+deep: {moderate_deep/total_depth*100:.1f}%")

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
                 f"({crossing_count:,} pairs, rate: {crossing_rate:.1f}%)",
                 fontsize=12)
    ax.set_ylabel("Number of crossing pairs")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "09_crossing_depth_distribution.png")

    all_examples   = examples_early + examples_mid + examples_late
    section_labels = (
        ['Early crossover'] * len(examples_early) +
        ['Mid crossover']   * len(examples_mid)   +
        ['Late crossover']  * len(examples_late)
    )
    n_ex = len(all_examples)

    if n_ex > 0:
        ncols = 3
        nrows = (n_ex + ncols - 1) // ncols
        fig   = plt.figure(figsize=(18, 5.5 * nrows))
        gs    = gridspec.GridSpec(nrows, ncols, hspace=0.55, wspace=0.35)

        for plot_idx, (ex, slabel) in enumerate(
            zip(all_examples, section_labels)
        ):
            ax_main = fig.add_subplot(gs[plot_idx // ncols, plot_idx % ncols])
            ci, cj  = ex['curve_i'], ex['curve_j']
            tc      = ex['t_cross']
            tc_idx  = np.argmin(np.abs(t_work - tc))
            mb      = max(0, tc_idx // 2)
            ma      = min(len(t_work)-1, tc_idx + (len(t_work)-tc_idx)//2)
            a_bef   = ci[mb] > cj[mb]
            a_aft   = ci[ma] > cj[ma]

            ax_main.plot(t_work, ci, color='#2980b9', linewidth=1.8,
                         label=f'Curve A  R(22k)={ci[-1]:.4f}')
            ax_main.plot(t_work, cj, color='#e74c3c', linewidth=1.8,
                         label=f'Curve B  R(22k)={cj[-1]:.4f}')
            ax_main.axvline(tc, color='gray', linestyle='--',
                            linewidth=1.2, label=f'Crossover: {tc:.0f}h')
            ax_main.fill_betweenx([0,1], 0, tc,
                alpha=0.07, color='#2980b9' if a_bef else '#e74c3c',
                label=f'{"A" if a_bef else "B"} better (before)')
            ax_main.fill_betweenx([0,1], tc, t_work[-1],
                alpha=0.07, color='#2980b9' if a_aft else '#e74c3c',
                label=f'{"A" if a_aft else "B"} better (after)')
            ax_main.set_title(f"{slabel}\n"
                              f"Max diff: {ex['max_diff']:.3f}  |  "
                              f"Crossover: {tc:.0f}h", fontsize=9)
            ax_main.set_xlabel("Time (hours)", fontsize=8)
            ax_main.set_ylabel("Reliability", fontsize=8)
            ax_main.legend(fontsize=6.5, loc='lower left')
            ax_main.grid(alpha=0.25)
            ax_main.set_xlim(0, t_work[-1])
            y_lo = min(ci.min(), cj.min()) - 0.02
            y_hi = max(ci.max(), cj.max()) + 0.02
            ax_main.set_ylim(max(0, y_lo), min(1.02, y_hi))

            zoom_w = 3000
            z_tmin = max(t_work[0],  tc - zoom_w)
            z_tmax = min(t_work[-1], tc + zoom_w)
            z_mask = (t_work >= z_tmin) & (t_work <= z_tmax)
            axins  = ax_main.inset_axes([0.54, 0.52, 0.44, 0.40])
            axins.plot(t_work[z_mask], ci[z_mask], color='#2980b9', linewidth=1.5)
            axins.plot(t_work[z_mask], cj[z_mask], color='#e74c3c', linewidth=1.5)
            axins.axvline(tc, color='gray', linestyle='--', linewidth=1.0)
            z_ymin = min(ci[z_mask].min(), cj[z_mask].min()) - 0.005
            z_ymax = max(ci[z_mask].max(), cj[z_mask].max()) + 0.005
            axins.set_ylim(z_ymin, z_ymax)
            axins.set_xlim(z_tmin, z_tmax)
            axins.tick_params(labelsize=5)
            axins.set_title("Zoom", fontsize=6)
            axins.grid(alpha=0.3)

            from matplotlib.patches import Rectangle
            rect = Rectangle((z_tmin, z_ymin), z_tmax-z_tmin, z_ymax-z_ymin,
                             linewidth=0.8, edgecolor='gray',
                             facecolor='none', linestyle='--')
            ax_main.add_patch(rect)

        fig.suptitle(
            "Deep Crossing Examples — Diverse Early, Mid, and Late Crossovers\n"
            "One example per unique source curve. "
            "Shading shows which curve is genuinely higher in each region.",
            fontsize=12, fontweight='bold'
        )
        save_fig(fig, "10_crossing_examples.png", dpi=200)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — Crossing Hub Analysis
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 7: Crossing Hub Analysis (Unique Curves Only)")
    log("=" * 70)

    n_zero    = int((cross_count_per_curve == 0).sum())
    max_cross = int(cross_count_per_curve.max())
    log(f"  Curves with 0 crossings : {n_zero:,}  ({n_zero/n_c*100:.1f}%)")
    log(f"  Max crossings one curve : {max_cross:,}")
    log(f"  Median crossings        : {np.median(cross_count_per_curve):.1f}")

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
    axes[1].set_title("R(8k) vs Crossing Count", fontsize=11)
    axes[1].set_xlabel("R at t=8,000h")
    axes[1].set_ylabel("Number of unique curves crossed")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "11_crossing_hubs.png")

    fig, ax = plt.subplots(figsize=(14, 7))
    for idx in np.argsort(cross_count_per_curve)[:200]:
        ax.plot(time_vals, Y_unique[idx],
                color='lightgray', linewidth=0.5, alpha=0.4)
    cmap_hub = plt.cm.tab10
    for rank, idx in enumerate(top10_idx):
        ax.plot(time_vals, Y_unique[idx],
                color=cmap_hub(rank/10), linewidth=2.0, alpha=0.9,
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
    # SECTION 8 — Behavioral Slope Space
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 8: Behavioral Slope Space (Continuous)")
    log("=" * 70)

    early_slope = (Y_unique[:, t8k_idx-1] - Y_unique[:, 0]) / (time_vals[t8k_idx-1] - time_vals[0])
    late_slope  = (Y_unique[:, -1] - Y_unique[:, t8k_idx]) / (time_vals[-1] - time_vals[t8k_idx])

    with np.errstate(divide='ignore', invalid='ignore'):
        slope_ratio = np.where(
            np.abs(early_slope) > 1e-8,
            np.abs(late_slope) / np.abs(early_slope), 1.0
        )

    log(f"  Early slope mean: {early_slope.mean():.2e}")
    log(f"  Late slope mean : {late_slope.mean():.2e}")
    log(f"  Slope ratio mean: {slope_ratio.mean():.2f}  std: {slope_ratio.std():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sc3 = axes[0].scatter(
        early_slope*1e5, late_slope*1e5,
        c=slope_ratio, cmap='plasma', alpha=0.4, s=8, linewidths=0,
        vmin=np.percentile(slope_ratio, 5),
        vmax=np.percentile(slope_ratio, 95)
    )
    plt.colorbar(sc3, ax=axes[0], label='Late/Early slope ratio')
    lim = max(abs(early_slope.min()), abs(late_slope.min()))*1e5*1.1
    axes[0].plot([-lim,0],[-lim,0],'k--',alpha=0.3,linewidth=0.8,
                 label='Equal slope')
    axes[0].set_xlabel("Early slope (×10⁻⁵ R/hour, t=100–8000h)")
    axes[0].set_ylabel("Late slope (×10⁻⁵ R/hour, t=8000–22000h)")
    axes[0].set_title("Behavioral Slope Space\n"
                      "(continuous — no forced families)", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].hist(slope_ratio, bins=100, color='mediumpurple', edgecolor='none')
    axes[1].set_title("Slope Ratio Distribution\n"
                      "(late/early degradation rate)", fontsize=11)
    axes[1].set_xlabel("Late / Early slope ratio")
    axes[1].set_ylabel("Number of unique curves")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "13_behavioral_slope_space.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9 — Architectural Correlation
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SECTION 9: Architectural Correlation (Continuous)")
    log("=" * 70)

    rep_indices        = [hash_to_indices[h][0] for h in unique_hashes]
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
        degree    = np.bincount(ei[0], minlength=N_NODES).astype(np.float32)

        sw_deg = degree[is_switch].mean() if is_switch.any() else 0.0
        mean_switch_degree.append(sw_deg)

        task_nodes    = np.where(is_task)[0]
        compute_nodes = set()
        for tn in task_nodes:
            for nb in ei[1][ei[0] == tn]:
                if is_compute[nb]:
                    compute_nodes.add(int(nb))
        n_comp = len(compute_nodes) if compute_nodes else 1
        n_compute_w_tasks.append(n_comp)
        task_concentration.append(1.0 / n_comp)
        graph_density.append(ei.shape[1] / (N_NODES*(N_NODES-1)))

    task_concentration = np.array(task_concentration)
    mean_switch_degree = np.array(mean_switch_degree)
    n_compute_w_tasks  = np.array(n_compute_w_tasks, dtype=float)
    graph_density      = np.array(graph_density)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    arch_pairs = [
        (task_concentration, "Task concentration (1/n_compute_w_tasks)", axes[0,0]),
        (mean_switch_degree, "Mean switch node degree",                  axes[0,1]),
        (n_compute_w_tasks,  "N compute nodes with tasks",               axes[1,0]),
        (graph_density,      "Graph density",                            axes[1,1]),
    ]
    for feat_vals, feat_label, ax in arch_pairs:
        sc4 = ax.scatter(feat_vals, r_at_8000,
                         c=slope_ratio, cmap='plasma', alpha=0.3,
                         s=6, linewidths=0,
                         vmin=np.percentile(slope_ratio, 5),
                         vmax=np.percentile(slope_ratio, 95))
        z   = np.polyfit(feat_vals, r_at_8000, 1)
        x_r = np.linspace(feat_vals.min(), feat_vals.max(), 100)
        ax.plot(x_r, np.poly1d(z)(x_r), 'k-', linewidth=1.5, alpha=0.6,
                label=f'Trend (slope={z[0]:.4f})')
        corr = np.corrcoef(feat_vals, r_at_8000)[0, 1]
        ax.set_title(f"{feat_label}\nPearson r = {corr:.3f}", fontsize=10)
        ax.set_xlabel(feat_label, fontsize=9)
        ax.set_ylabel("Mean R at t=8,000h", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.colorbar(sc4, ax=ax, label='Slope ratio')

    fig.suptitle(
        "Architectural Features vs Reliability (Continuous Correlation)\n"
        "Color = late/early slope ratio",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    save_fig(fig, "14_architectural_correlation.png")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("  SUMMARY")
    log("=" * 70)
    log(f"  Samples                  : {N:,}")
    log(f"  Unique HW wirings (MD5)  : {len(unique_md5):,}")
    log(f"  Unique HW topology (WL)  : {len(unique_wl):,}  (VF2-proven)")
    log(f"  Unique curves            : {n_unique_curves:,}")
    log(f"  Curve redundancy         : {n_redundant/N*100:.1f}%")
    log(f"  Crossing rate            : {crossing_rate:.1f}%")
    log(f"  Moderate+deep crossings  : {moderate_deep/total_depth*100:.1f}%")
    log(f"  Max hub crossings        : {max_cross:,}")
    log(f"\n  Split axes available:")
    log(f"    curve_hashes    : 3,336 groups")
    log(f"    allocations     : 31 groups (8 functional)")
    log(f"    hw_md5_hashes   : 11,903 groups")
    log(f"    hw_wl_hashes    : 392 groups")
    log(f"\n✅ EDA complete. All outputs in {OUT_DIR}/")
    save_log()


if __name__ == "__main__":
    main()