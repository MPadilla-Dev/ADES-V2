"""
01_preprocess.py
================
STEP 1 of the ADES-v2 pipeline.

Reads ALL raw data (matrices.zip + CSV) without sampling or shortcuts.
Writes a single HDF5 file with full fidelity.

HDF5 Layout:
    meta/
        config_ids      [N]        str   — "0000_0736"
        allocations     [N]        str   — "0000"
        config_nums     [N]        str   — "0736"
        curve_hashes    [N]        str   — md5 of y_curve
                                           used for curve-level split
        adj_hashes      [N]        str   — md5 of full 30-node adjacency
                                           matrix (unique per sample)
        hw_md5_hashes   [N]        str   — md5 of hardware-only edge set
                                           (task edges stripped)
                                           11,903 unique values
                                           used for hardware-wiring split
        hw_wl_hashes    [N]        str   — Weisfeiler-Lehman hash of
                                           hardware-only graph (task edges
                                           stripped, node type attribute)
                                           392 unique values
                                           VF2-verified exhaustively
                                           used for topology-level split
        node_names      [N, 30]    str   — preserved for re-featurization
    features/
        node_features   [N, 30, 5] f32   — one-hot type encoding
    edges/
        edge_index      [2, E_tot] i32   — all edges concatenated (CSR)
        edge_ptr        [N+1]      i64   — ptr[i]:ptr[i+1] = edges for i
    targets/
        y_curve         [N, 221]   f32   — full reliability curve 0h-22000h
    attrs:
        time_values, n_samples, n_time_steps, n_nodes, n_features

--- UNDERSTANDING THE TWO HARDWARE HASHES ---

hw_md5_hashes (11,903 unique groups):
    Definition: MD5 of the sorted hardware-only edge index after
                stripping all edges touching task nodes.
    Meaning: Two samples with the same hw_md5_hash have IDENTICAL
             physical wiring — same compute-to-switch connections,
             same switch-to-switch connections. They differ only in
             task placement (which compute node hosts which task).
    Split use: Hardware-wiring split. Holding out hw_md5 groups
               means the model never sees a particular physical
               wiring in any allocation during training.
    Key fact:  Within a hw_md5 group, curves differ only because
               task allocation changes which compute node the tasks
               connect to. The curve difference is PURELY from
               task placement variation.

hw_wl_hashes (392 unique groups, VF2-proven):
    Definition: Weisfeiler-Lehman graph hash (iterations=4,
                node_attr='ntype') on the hardware-only subgraph.
    Meaning: Two samples with the same hw_wl_hash have hardware
             subgraphs that are ISOMORPHIC at the node-type level —
             same abstract topology (same degree sequence per type)
             but possibly different specific node assignments.
    Split use: Topology-level split. Coarser than hw_md5.
    Key fact:  Within a hw_wl group, different hw_md5 groups can
               exist. Their curve differences reflect genuinely
               different wirings — e.g., the task-hosting compute
               node (N6) may have degree 2 in one wiring and
               degree 1 in another, producing different reliability
               even though the hardware topology class is the same.
    Open question: Whether hw_md5 or hw_wl is the right split
                   level is a domain question sent to the expert.
                   Both are stored so the decision can be made
                   without rerunning preprocessing.

--- THREE SPLIT AXES (for 03_split.py) ---

    curve_hashes  → 3,336 groups  — unseen reliability values
    allocations   → 31 groups     — unseen task strategies
    hw_md5_hashes → 11,903 groups — unseen hardware wirings
    hw_wl_hashes  → 392 groups    — unseen hardware topologies

Module : python-data/3.10-24.04
Usage  : python src/scripts/01_preprocess.py
Output : data/dataset.h5
"""

import os
import sys
import zipfile
import hashlib
import numpy as np
import pandas as pd
import h5py
import networkx as nx
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH      = "data/new_raw/config_all_0_22000_100.csv"
ZIP_PATH      = "data/new_raw/matrices.zip"
OUTPUT_PATH   = "data/dataset.h5"
LOG_PATH      = "logs/01_preprocess.log"
N_NODES       = 30
N_FEATURES    = 5
WL_ITERATIONS = 4

# Expected counts from prior analysis — used for verification
EXPECTED_UNIQUE_CURVES   = 3336
EXPECTED_UNIQUE_HW_MD5   = 11903
EXPECTED_UNIQUE_HW_WL    = 392

# ── LOGGING ───────────────────────────────────────────────────────────────────
log_lines = []
def log(msg=""):
    print(msg, flush=True)
    log_lines.append(msg)

def save_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w') as f:
        f.write('\n'.join(log_lines))

# ── NODE CLASSIFIER ───────────────────────────────────────────────────────────
def classify_node(name: str) -> str:
    """
    Canonical node classifier — single source of truth for all scripts.
      task    : underscore present          T1_1, T2_4
      link    : two device prefixes fused   N1S1, S1S2
      compute : single N prefix             N1, N2
      switch  : single S prefix             S1, S2
    """
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


def node_to_feature(name: str) -> list:
    """5-dim one-hot from node name."""
    feat  = [0, 0, 0, 0, 0]
    ntype = classify_node(name)
    if   ntype == 'compute': feat[0] = 1
    elif ntype == 'switch':  feat[1] = 1
    elif ntype == 'link':    feat[2] = 1
    elif ntype == 'task':
        if name.startswith('T1'): feat[3] = 1
        else:                     feat[4] = 1
    return feat


# ── PARSING ───────────────────────────────────────────────────────────────────
def parse_graph_file(f_obj):
    """
    Parse one graph txt file completely.
    Returns (node_names: list[str], adj_matrix: ndarray[N,N])
    Raises ValueError on any format issue.
    """
    content    = f_obj.read().decode('utf-8')
    lines      = content.splitlines()
    if not lines:
        raise ValueError("Empty file")
    header     = lines[0].replace('#', '').strip()
    node_names = eval(header)
    n          = len(node_names)
    if n != N_NODES:
        raise ValueError(f"Expected {N_NODES} nodes, got {n}")
    matrix_str = ' '.join(lines[1:]).replace('[', ' ').replace(']', ' ')
    values     = np.fromstring(matrix_str, sep=' ')
    if values.size != n * n:
        raise ValueError(f"Matrix mismatch: expected {n*n}, got {values.size}")
    return node_names, values.reshape(n, n).astype(np.float32)


def md5_f32(arr: np.ndarray) -> str:
    return hashlib.md5(arr.astype(np.float32).tobytes()).hexdigest()


def compute_hw_md5(edge_idx: np.ndarray,
                   node_feats: np.ndarray) -> str:
    """
    MD5 of hardware-only edge set (task edges stripped).

    Identical for two samples with the same physical wiring
    regardless of task placement. Different for any change in
    compute-to-switch or switch-to-switch connectivity.

    11,903 unique values in this dataset.
    """
    is_task = (node_feats[:, 3] + node_feats[:, 4]) > 0
    mask    = ~is_task[edge_idx[0]] & ~is_task[edge_idx[1]]
    hw_ei   = edge_idx[:, mask]
    if hw_ei.shape[1] == 0:
        return 'empty'
    sorted_edges = hw_ei[:, np.lexsort((hw_ei[1], hw_ei[0]))]
    return hashlib.md5(sorted_edges.astype(np.int32).tobytes()).hexdigest()


def build_hw_nx_graph(edge_idx: np.ndarray,
                      node_feats: np.ndarray) -> nx.Graph:
    """
    Build hardware-only NetworkX graph (task nodes/edges excluded).
    Node attribute 'ntype' = '0' (compute), '1' (switch), '2' (link).
    Used for WL hash computation.
    """
    is_task = (node_feats[:, 3] + node_feats[:, 4]) > 0
    mask    = ~is_task[edge_idx[0]] & ~is_task[edge_idx[1]]
    hw_ei   = edge_idx[:, mask]
    G = nx.Graph()
    for n in range(len(node_feats)):
        if is_task[n]:
            continue
        ntype = str(int(node_feats[n, :3].argmax()))
        G.add_node(n, ntype=ntype)
    for e in range(hw_ei.shape[1]):
        G.add_edge(int(hw_ei[0, e]), int(hw_ei[1, e]))
    return G


def compute_hw_wl(G: nx.Graph) -> str:
    """
    Weisfeiler-Lehman hash of hardware-only graph.

    Groups isomorphic hardware topologies together.
    392 unique values confirmed by exhaustive VF2 verification
    (356,589 pairs checked, 0 false positives).

    NOTE: Two samples with the same WL hash may have different
    physical wirings (different hw_md5) if their hardware subgraphs
    are isomorphic but assign different structural roles to different
    labeled nodes (e.g. N6 connecting to 2 switches vs 1 switch).
    Whether WL or MD5 grouping is the right split level is an open
    domain question — both hashes are stored.
    """
    return nx.weisfeiler_lehman_graph_hash(
        G, node_attr='ntype', iterations=WL_ITERATIONS
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("  STEP 1: Full Preprocessing -> HDF5")
    log("  Stores BOTH hw_md5_hashes and hw_wl_hashes.")
    log("  See docstring for explanation of each hash.")
    log("=" * 60)

    # 1. Load CSV
    log(f"\n[1] Loading CSV: {CSV_PATH}")
    df        = pd.read_csv(CSV_PATH)
    df.set_index('CONFIG', inplace=True)
    time_cols = [c for c in df.columns if c.replace('.', '').isdigit()]
    time_vals = np.array([float(c) for c in time_cols])
    N_TIME    = len(time_cols)

    log(f"  Rows        : {len(df):,}")
    log(f"  Time steps  : {N_TIME}  ({time_vals[0]:.0f}h - {time_vals[-1]:.0f}h)")
    log(f"  R min/max   : {df[time_cols].values.min():.6f} / "
        f"{df[time_cols].values.max():.6f}")
    log(f"  NaN         : {df[time_cols].isnull().any().any()}")

    log("  Building target lookup...")
    target_map = {
        str(idx): row[time_cols].values.astype(np.float32)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Indexing")
    }

    # 2. Scan ZIP
    log(f"\n[2] Scanning ZIP: {ZIP_PATH}")
    matched_files     = []
    skipped_no_target = []

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        all_names = z.namelist()
        log(f"  Total ZIP entries: {len(all_names):,}")
        for fname in all_names:
            if not fname.endswith('.txt') or 'matrix' not in fname:
                continue
            try:
                parts    = fname.split('/')
                alloc    = parts[0]
                conf_num = parts[-1].replace('config_', '').replace('.txt', '')
                cid      = f"{alloc}_{conf_num}"
            except Exception:
                continue
            if cid in target_map:
                matched_files.append((cid, alloc, conf_num, fname))
            else:
                skipped_no_target.append(cid)

    log(f"  Matched : {len(matched_files):,}")
    log(f"  Skipped : {len(skipped_no_target):,}")
    if skipped_no_target:
        log(f"  Skipped IDs: {sorted(skipped_no_target)}")

    # 3. Parse ALL graphs — collect MD5 hw hashes
    log(f"\n[3] Parsing all {len(matched_files):,} graph files...")
    records      = []
    parse_errors = []

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for cid, alloc, conf_num, fpath in tqdm(
            matched_files, desc="  Parsing", unit="graph"
        ):
            try:
                with z.open(fpath) as f:
                    node_names, adj = parse_graph_file(f)
            except Exception as e:
                parse_errors.append((cid, str(e)))
                continue

            node_feats = np.array(
                [node_to_feature(nm) for nm in node_names],
                dtype=np.float32
            )

            row_sums = node_feats.sum(axis=1)
            if not np.all(row_sums == 1):
                bad = [node_names[i] for i in np.where(row_sums != 1)[0]]
                parse_errors.append((cid, f"Bad one-hot: {bad}"))
                continue

            rows, cols = np.nonzero(adj)
            edge_idx   = np.vstack([rows, cols]).astype(np.int32)
            curve      = target_map[cid]
            hw_md5     = compute_hw_md5(edge_idx, node_feats)

            records.append({
                'config_id'    : cid,
                'allocation'   : alloc,
                'config_num'   : conf_num,
                'curve_hash'   : md5_f32(curve),
                'adj_hash'     : md5_f32(adj),
                'hw_md5'       : hw_md5,
                'node_names'   : node_names,
                'node_features': node_feats,
                'edge_index'   : edge_idx,
                'y_curve'      : curve,
            })

    N_valid = len(records)
    log(f"\n  Parsed OK : {N_valid:,}")
    log(f"  Errors    : {len(parse_errors):,}")
    if parse_errors:
        for cid, err in parse_errors[:20]:
            log(f"    {cid}: {err}")
    if N_valid == 0:
        log("ERROR: No valid samples. Aborting.")
        save_log()
        sys.exit(1)

    # 4. Compute WL hash — once per unique MD5 hw hash (12x speedup)
    log(f"\n[4] Computing WL hashes (optimized by MD5 hw hash)...")

    md5_to_record = {}
    for r in records:
        if r['hw_md5'] not in md5_to_record:
            md5_to_record[r['hw_md5']] = r

    n_unique_md5 = len(md5_to_record)
    log(f"  Unique MD5 hw hashes : {n_unique_md5:,}  "
        f"(expected {EXPECTED_UNIQUE_HW_MD5:,})")
    log(f"  WL computations      : {n_unique_md5:,}  "
        f"(vs {N_valid:,} without optimization, {N_valid/n_unique_md5:.1f}x speedup)")

    md5_to_wl = {}
    for md5, r in tqdm(md5_to_record.items(),
                        desc="  WL hashing", unit="wiring"):
        G             = build_hw_nx_graph(r['edge_index'], r['node_features'])
        md5_to_wl[md5] = compute_hw_wl(G)

    for r in records:
        r['hw_wl'] = md5_to_wl[r['hw_md5']]

    # Report unique counts
    n_uc  = len(set(r['curve_hash'] for r in records))
    n_umd = len(set(r['hw_md5']    for r in records))
    n_uwl = len(set(r['hw_wl']     for r in records))
    n_ua  = len(set(r['adj_hash']  for r in records))

    log(f"\n  Unique curve hashes   : {n_uc:,}  (expected {EXPECTED_UNIQUE_CURVES:,})")
    log(f"  Unique hw MD5 hashes  : {n_umd:,}  (expected {EXPECTED_UNIQUE_HW_MD5:,})")
    log(f"  Unique hw WL hashes   : {n_uwl:,}  (expected {EXPECTED_UNIQUE_HW_WL:,})")
    log(f"  Unique adj hashes     : {n_ua:,}")

    # Warn if counts deviate from expected
    for label, got, exp in [
        ("curve hashes", n_uc,  EXPECTED_UNIQUE_CURVES),
        ("hw MD5 hashes", n_umd, EXPECTED_UNIQUE_HW_MD5),
        ("hw WL hashes",  n_uwl, EXPECTED_UNIQUE_HW_WL),
    ]:
        if got != exp:
            log(f"  WARNING: Expected {exp:,} unique {label}, got {got:,}")
        else:
            log(f"  OK: {label} count matches expected. ✅")

    # 5. Write HDF5
    log(f"\n[5] Writing HDF5: {OUTPUT_PATH}")
    os.makedirs('data', exist_ok=True)
    str_dt = h5py.special_dtype(vlen=str)

    with h5py.File(OUTPUT_PATH, 'w') as h5:

        meta = h5.create_group('meta')
        meta.create_dataset('config_ids',    (N_valid,),         dtype=str_dt)
        meta.create_dataset('allocations',   (N_valid,),         dtype=str_dt)
        meta.create_dataset('config_nums',   (N_valid,),         dtype=str_dt)
        meta.create_dataset('curve_hashes',  (N_valid,),         dtype=str_dt)
        meta.create_dataset('adj_hashes',    (N_valid,),         dtype=str_dt)
        meta.create_dataset('hw_md5_hashes', (N_valid,),         dtype=str_dt)
        meta.create_dataset('hw_wl_hashes',  (N_valid,),         dtype=str_dt)
        meta.create_dataset('node_names',    (N_valid, N_NODES), dtype=str_dt)

        feat_ds = h5.create_group('features').create_dataset(
            'node_features',
            shape=(N_valid, N_NODES, N_FEATURES),
            dtype=np.float32,
            chunks=(512, N_NODES, N_FEATURES),
            compression='gzip', compression_opts=4
        )

        tgt_ds = h5.create_group('targets').create_dataset(
            'y_curve',
            shape=(N_valid, N_TIME),
            dtype=np.float32,
            chunks=(512, N_TIME),
            compression='gzip', compression_opts=4
        )

        log("  Assembling CSR edge structure...")
        all_edges = np.concatenate(
            [r['edge_index'] for r in records], axis=1
        ).astype(np.int32)

        edge_ptr = np.zeros(N_valid + 1, dtype=np.int64)
        for i, r in enumerate(records):
            edge_ptr[i + 1] = edge_ptr[i] + r['edge_index'].shape[1]

        edge_grp = h5.create_group('edges')
        edge_grp.create_dataset('edge_index', data=all_edges,
                                compression='gzip', compression_opts=4)
        edge_grp.create_dataset('edge_ptr', data=edge_ptr)

        log(f"  Total edges : {all_edges.shape[1]:,}")
        log(f"  Mean E/graph: {all_edges.shape[1]/N_valid:.1f}")

        log("  Writing samples...")
        for i, r in enumerate(tqdm(records, desc="  Writing", unit="sample")):
            meta['config_ids'][i]    = r['config_id']
            meta['allocations'][i]   = r['allocation']
            meta['config_nums'][i]   = r['config_num']
            meta['curve_hashes'][i]  = r['curve_hash']
            meta['adj_hashes'][i]    = r['adj_hash']
            meta['hw_md5_hashes'][i] = r['hw_md5']
            meta['hw_wl_hashes'][i]  = r['hw_wl']
            meta['node_names'][i]    = r['node_names']
            feat_ds[i]               = r['node_features']
            tgt_ds[i]                = r['y_curve']

        h5.attrs['time_values']          = time_vals
        h5.attrs['n_samples']            = N_valid
        h5.attrs['n_time_steps']         = N_TIME
        h5.attrs['n_nodes']              = N_NODES
        h5.attrs['n_features']           = N_FEATURES
        h5.attrs['n_unique_curves']      = n_uc
        h5.attrs['n_unique_hw_md5']      = n_umd
        h5.attrs['n_unique_hw_wl']       = n_uwl

    # 6. Verify
    log(f"\n[6] Verifying HDF5 integrity...")
    with h5py.File(OUTPUT_PATH, 'r') as h5:
        log(f"  node_features : {h5['features/node_features'].shape}")
        log(f"  y_curve       : {h5['targets/y_curve'].shape}")
        log(f"  edge_index    : {h5['edges/edge_index'].shape}")
        log(f"  config_ids[0] : {h5['meta/config_ids'][0]}")
        log(f"  curve_hash[0] : {h5['meta/curve_hashes'][0]}")
        log(f"  hw_md5[0]     : {h5['meta/hw_md5_hashes'][0]}")
        log(f"  hw_wl[0]      : {h5['meta/hw_wl_hashes'][0]}")
        log(f"  y_curve[0,:5] : {h5['targets/y_curve'][0, :5]}")

        ep = h5['edges/edge_ptr'][:]
        assert np.all(np.diff(ep) >= 0), "edge_ptr not monotonic!"
        assert ep[0] == 0
        assert ep[-1] == h5['edges/edge_index'].shape[1]
        log("  edge_ptr      : PASSED")

        y_min = h5['targets/y_curve'][:, 1:].min()
        y_max = h5['targets/y_curve'][:].max()
        assert y_min >= 0.0 and y_max <= 1.0
        log(f"  y_curve range : [{y_min:.6f}, {y_max:.6f}]  PASSED")

        n_uc_v  = len(set(h5['meta/curve_hashes'][:].astype(str)))
        n_umd_v = len(set(h5['meta/hw_md5_hashes'][:].astype(str)))
        n_uwl_v = len(set(h5['meta/hw_wl_hashes'][:].astype(str)))
        log(f"  Unique curves  : {n_uc_v:,}   (expected {EXPECTED_UNIQUE_CURVES:,})")
        log(f"  Unique hw MD5  : {n_umd_v:,}  (expected {EXPECTED_UNIQUE_HW_MD5:,})")
        log(f"  Unique hw WL   : {n_uwl_v:,}    (expected {EXPECTED_UNIQUE_HW_WL:,})")

        assert n_uc_v  == EXPECTED_UNIQUE_CURVES, f"curve count mismatch"
        assert n_umd_v == EXPECTED_UNIQUE_HW_MD5, f"hw md5 count mismatch"
        assert n_uwl_v == EXPECTED_UNIQUE_HW_WL,  f"hw wl count mismatch"
        log("  All counts     : PASSED ✅")

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    log(f"\n  File size : {size_mb:.1f} MB")
    log(f"  Samples   : {N_valid:,}")
    log(f"  Errors    : {len(parse_errors):,}")
    log(f"\n✅ Preprocessing complete: {OUTPUT_PATH}")
    save_log()


if __name__ == "__main__":
    main()