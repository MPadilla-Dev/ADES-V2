"""
00_verify_data.py
=================
STEP 0 of the clean preprocessing pipeline.

Purpose:
  - Confirm dataset integrity (zip <-> CSV matching)
  - Decode and validate node naming conventions
  - Detect duplicate / symmetric graphs
  - Profile the reliability curve distribution
  - Report allocation balance and potential biases

Usage (on Puhti):
  python src/scripts/00_verify_data.py

Outputs:
  results/00_verify/
    ├── report.txt                  ← full text report
    ├── curve_distribution.png      ← reliability value histograms
    ├── allocation_balance.png      ← samples per allocation
    ├── curve_spread_per_alloc.png  ← mean ± std curve per allocation
    └── node_type_distribution.png  ← node type counts across dataset
"""

import os
import sys
import zipfile
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # No display needed on Puhti
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT          = "data/new_raw"
CSV_FILE      = os.path.join(ROOT, "config_all_0_22000_100.csv")
ZIP_FILE      = os.path.join(ROOT, "matrices.zip")
OUT_DIR       = "results/00_verify"
os.makedirs(OUT_DIR, exist_ok=True)

# ── NODE TYPE PARSING ─────────────────────────────────────────────────────────
def classify_node(name: str) -> str:
    """
    Clean, single-source-of-truth node classifier based on observed naming:
      N1, N2 ...          → 'compute'   (slave/processing node)
      S1, S2 ...          → 'switch'
      N1S1, S1S2 ...      → 'link'      (two device names fused = link between them)
      T1_1, T2_4 ...      → 'task'      (underscore present)
    """
    if '_' in name:
        return 'task'
    # Link: name contains two uppercase-starting segments fused (e.g. N1S2, S1S3)
    # Heuristic: contains both a letter from {N,S} after position 0 that starts a new word
    # More robust: if stripping digits still gives 2+ capital letters
    stripped = ''.join(c for c in name if not c.isdigit())  # e.g. "NS", "SS", "N", "S"
    if len(stripped) >= 2:
        return 'link'
    if name.startswith('N'):
        return 'compute'
    if name.startswith('S'):
        return 'switch'
    return 'unknown'


def parse_graph_file(f_obj):
    """
    Parse a single graph text file.
    Returns (node_names, adj_matrix) or raises on failure.
    """
    lines = f_obj.read().decode('utf-8').splitlines()
    # Header: # ['N1', 'N1S1', ...]
    header = lines[0].replace('#', '').strip()
    node_names = eval(header)
    n = len(node_names)

    # Matrix: remaining lines, strip brackets
    matrix_str = ' '.join(lines[1:]).replace('[', ' ').replace(']', ' ')
    values = np.fromstring(matrix_str, sep=' ')
    adj = values.reshape(n, n)
    return node_names, adj


def adj_hash(adj: np.ndarray) -> str:
    """Fast hash of an adjacency matrix for duplicate detection."""
    return hashlib.md5(adj.tobytes()).hexdigest()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    report_lines = []
    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=" * 60)
    log("  DATASET VERIFICATION REPORT")
    log("=" * 60)

    # ── 1. Load CSV ────────────────────────────────────────────────────────
    log("\n[1] Loading CSV...")
    df = pd.read_csv(CSV_FILE)
    df.set_index('CONFIG', inplace=True)

    time_cols = [c for c in df.columns if c.replace('.', '').isdigit()]
    time_vals = np.array([float(c) for c in time_cols])

    log(f"  CSV rows       : {len(df):,}")
    log(f"  Time columns   : {len(time_cols)}  ({time_vals[0]:.0f}h – {time_vals[-1]:.0f}h, step {time_vals[1]-time_vals[0]:.0f}h)")
    log(f"  Reliability min: {df[time_cols].values.min():.6f}")
    log(f"  Reliability max: {df[time_cols].values.max():.6f}")
    log(f"  Any NaN in CSV : {df[time_cols].isnull().any().any()}")

    csv_ids = set(df.index.astype(str))

    # ── 2. Scan ZIP ────────────────────────────────────────────────────────
    log("\n[2] Scanning ZIP...")
    zip_id_to_path = {}
    with zipfile.ZipFile(ZIP_FILE, 'r') as z:
        for fname in z.namelist():
            if fname.endswith('.txt') and 'matrix' in fname:
                try:
                    parts = fname.split('/')
                    alloc = parts[0]
                    conf  = parts[-1].replace('config_', '').replace('.txt', '')
                    cid   = f"{alloc}_{conf}"
                    zip_id_to_path[cid] = fname
                except Exception:
                    continue

    zip_ids = set(zip_id_to_path.keys())
    in_zip_not_csv = zip_ids - csv_ids
    in_csv_not_zip = csv_ids - zip_ids
    matched         = zip_ids & csv_ids

    log(f"  ZIP graph files : {len(zip_ids):,}")
    log(f"  Matched (usable): {len(matched):,}")
    log(f"  In ZIP, not CSV : {len(in_zip_not_csv)}  ← will be skipped")
    log(f"  In CSV, not ZIP : {len(in_csv_not_zip)}  ← missing inputs")

    if in_zip_not_csv:
        log(f"  Skipped IDs     : {sorted(in_zip_not_csv)[:10]} ...")
    if in_csv_not_zip:
        log(f"  ⚠️  MISSING INPUTS: {sorted(in_csv_not_zip)[:10]}")

    # ── 3. Allocation balance ──────────────────────────────────────────────
    log("\n[3] Allocation balance...")
    alloc_counts = Counter(cid.split('_')[0] for cid in matched)
    allocs_sorted = sorted(alloc_counts.keys())
    counts_sorted = [alloc_counts[a] for a in allocs_sorted]

    log(f"  Number of allocations : {len(alloc_counts)}")
    log(f"  Min samples/alloc     : {min(counts_sorted):,}  (alloc {allocs_sorted[counts_sorted.index(min(counts_sorted))]})")
    log(f"  Max samples/alloc     : {max(counts_sorted):,}  (alloc {allocs_sorted[counts_sorted.index(max(counts_sorted))]})")
    log(f"  Mean samples/alloc    : {np.mean(counts_sorted):.1f}")

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(allocs_sorted, counts_sorted, color='steelblue', edgecolor='white')
    for bar, cnt in zip(bars, counts_sorted):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(cnt), ha='center', va='bottom', fontsize=7, rotation=90)
    ax.set_title(f"Samples per Allocation (Total matched: {len(matched):,})")
    ax.set_xlabel("Allocation ID")
    ax.set_ylabel("Count")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/allocation_balance.png", dpi=150)
    plt.close()
    log(f"  → Saved allocation_balance.png")

    # ── 4. Reliability curve profiling ────────────────────────────────────
    log("\n[4] Reliability curve profiling...")

    Y = df.loc[list(matched), time_cols].values.astype(np.float32)

    # Final reliability (last time point)
    r_final = Y[:, -1]
    log(f"  R at t=0 (should all be 1.0) — unique values: {np.unique(Y[:,0])}")
    log(f"  R at t={time_vals[-1]:.0f}h:")
    log(f"    mean  = {r_final.mean():.6f}")
    log(f"    std   = {r_final.std():.6f}")
    log(f"    min   = {r_final.min():.6f}")
    log(f"    max   = {r_final.max():.6f}")

    # Distribution of final R — key for understanding classification bins
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(r_final, bins=100, color='steelblue', edgecolor='none')
    axes[0].set_title(f"Distribution of R(t={time_vals[-1]:.0f}h)")
    axes[0].set_xlabel("Reliability")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.3)

    # Log scale version to see tail structure
    axes[1].hist(r_final, bins=100, color='darkorange', edgecolor='none', log=True)
    axes[1].set_title(f"Distribution of R(t={time_vals[-1]:.0f}h) — log scale")
    axes[1].set_xlabel("Reliability")
    axes[1].set_ylabel("Count (log)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/curve_distribution.png", dpi=150)
    plt.close()
    log(f"  → Saved curve_distribution.png")

    # Count samples by number-of-nines (classification view)
    log("\n  'Number of 9s' breakdown at final timestep:")
    thresholds = [
        ("< 0.9",      r_final <  0.9),
        ("0.9–0.99",   (r_final >= 0.9)    & (r_final < 0.99)),
        ("0.99–0.999", (r_final >= 0.99)   & (r_final < 0.999)),
        ("0.999–0.9999",(r_final >= 0.999)  & (r_final < 0.9999)),
        (">= 0.9999",  r_final >= 0.9999),
    ]
    for label, mask in thresholds:
        pct = mask.sum() / len(r_final) * 100
        log(f"    {label:20s}: {mask.sum():6,}  ({pct:.1f}%)")

    # ── 5. Mean curve per allocation ───────────────────────────────────────
    log("\n[5] Computing mean curves per allocation...")
    matched_list = list(matched)
    alloc_of = {cid: cid.split('_')[0] for cid in matched_list}
    Y_df = pd.DataFrame(Y, index=matched_list, columns=time_cols)

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.cm.nipy_spectral(np.linspace(0, 1, len(allocs_sorted)))
    for alloc, color in zip(allocs_sorted, cmap):
        ids_a = [cid for cid in matched_list if alloc_of[cid] == alloc]
        mu  = Y_df.loc[ids_a].mean(axis=0).values
        std = Y_df.loc[ids_a].std(axis=0).values
        ax.plot(time_vals, mu, color=color, linewidth=1.2, alpha=0.8, label=alloc)
        ax.fill_between(time_vals, mu - std, mu + std, color=color, alpha=0.07)

    ax.set_title("Mean ± Std Reliability Curve per Allocation")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Reliability")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7,
              ncol=2, title="Alloc")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/curve_spread_per_alloc.png", dpi=150)
    plt.close()
    log(f"  → Saved curve_spread_per_alloc.png")

    # ── 6. Node type analysis (sample 500 graphs) ─────────────────────────
    log("\n[6] Node type analysis (sampling 500 graphs)...")
    sample_ids = list(matched)[:500]
    type_counter = Counter()
    node_count_per_graph = []
    parse_errors = 0

    with zipfile.ZipFile(ZIP_FILE, 'r') as z:
        for cid in tqdm(sample_ids, desc="  Sampling graphs"):
            try:
                with z.open(zip_id_to_path[cid]) as f:
                    node_names, adj = parse_graph_file(f)
                node_count_per_graph.append(len(node_names))
                for name in node_names:
                    type_counter[classify_node(name)] += 1
            except Exception:
                parse_errors += 1

    log(f"  Parse errors in sample: {parse_errors}")
    log(f"  Node counts — min: {min(node_count_per_graph)}, "
        f"max: {max(node_count_per_graph)}, "
        f"mean: {np.mean(node_count_per_graph):.1f}")
    log(f"  Node type totals across 500 graphs:")
    for ntype, cnt in type_counter.most_common():
        log(f"    {ntype:10s}: {cnt:,}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(type_counter.keys(), type_counter.values(), color='mediumseagreen')
    axes[0].set_title("Node Type Counts (500-graph sample)")
    axes[0].set_xlabel("Node Type")
    axes[0].set_ylabel("Total Count")
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].hist(node_count_per_graph, bins=20, color='mediumseagreen', edgecolor='white')
    axes[1].set_title("Nodes per Graph (500-graph sample)")
    axes[1].set_xlabel("Node Count")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/node_type_distribution.png", dpi=150)
    plt.close()
    log(f"  → Saved node_type_distribution.png")

    # ── 7. Duplicate / symmetry detection (sample 2000 graphs) ───────────
    log("\n[7] Duplicate/symmetric graph detection (sampling 2,000 graphs)...")
    sample_ids_dup = list(matched)[:2000]
    hash_to_ids = defaultdict(list)
    dup_errors = 0

    with zipfile.ZipFile(ZIP_FILE, 'r') as z:
        for cid in tqdm(sample_ids_dup, desc="  Hashing graphs"):
            try:
                with z.open(zip_id_to_path[cid]) as f:
                    _, adj = parse_graph_file(f)
                # Also hash the transpose to catch symmetric duplicates
                h     = adj_hash(adj)
                h_sym = adj_hash(adj.T)
                canonical = min(h, h_sym)  # same hash regardless of direction
                hash_to_ids[canonical].append(cid)
            except Exception:
                dup_errors += 1

    dup_groups = {h: ids for h, ids in hash_to_ids.items() if len(ids) > 1}
    total_dups = sum(len(v) - 1 for v in dup_groups.values())

    log(f"  Graphs sampled          : {len(sample_ids_dup)}")
    log(f"  Unique graph structures : {len(hash_to_ids)}")
    log(f"  Duplicate groups found  : {len(dup_groups)}")
    log(f"  Total duplicate samples : {total_dups}")

    if dup_groups:
        log(f"\n  Example duplicate groups (first 5):")
        for i, (h, ids) in enumerate(list(dup_groups.items())[:5]):
            log(f"    Group {i+1}: {ids}")

    # ── 8. Write report ───────────────────────────────────────────────────
    report_path = f"{OUT_DIR}/report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    log(f"\n✅ Report saved to {report_path}")
    log(f"✅ All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()