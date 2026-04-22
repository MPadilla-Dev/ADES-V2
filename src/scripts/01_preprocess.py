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
        adj_hashes      [N]        str   — md5 of full adjacency matrix
        hw_hashes       [N]        str   — Weisfeiler-Lehman graph hash
                                           of hardware-only graph
                                           (task edges stripped, node
                                           type attributes included)
                                           used for hardware-level split
                                           392 unique values confirmed
                                           by exhaustive VF2 verification
        node_names      [N, 30]    str   — preserved for re-featurization
    features/
        node_features   [N, 30, 5] f32   — one-hot type encoding
    edges/
        edge_index      [2, E_tot] i32   — all edges concatenated (CSR)
        edge_ptr        [N+1]      i64   — ptr[i]:ptr[i+1] = edges for i
    targets/
        y_curve         [N, 221]   f32   — full reliability curve 0h-22000h
    attrs:
        time_values     [221]      f64   — actual hour values
        n_samples, n_time_steps, n_nodes, n_features

Node features (5-dim one-hot):
    [0] compute   N1, N2 ...
    [1] switch    S1, S2 ...
    [2] link      N1S1, S1S2 ...
    [3] task_T1   T1_1, T1_2 ...
    [4] task_T2   T2_4, T2_5 ...

Hardware hash definition:
    Step 1: Strip all edges where either endpoint is a task node.
    Step 2: Build a NetworkX graph on the remaining 24 hardware nodes,
            with node type (compute/switch/link) as a node attribute.
    Step 3: Compute Weisfeiler-Lehman graph hash with node_attr='ntype'
            at iterations=4.
    Two samples with the same hw_hash are isomorphic hardware layouts —
    same physical structure with different node numbering.

Optimization:
    WL hash is expensive (requires NetworkX graph construction).
    To avoid redundant computation, we first compute the cheap MD5
    adjacency hash. Two graphs with different MD5 hashes cannot be
    isomorphic, so they get different WL hashes trivially.
    WL is only computed once per unique MD5 hash, then propagated
    to all samples sharing that MD5 hash.
    This reduces WL computations from 144,255 to ~11,903.

Key facts confirmed by EDA and exhaustive VF2 verification:
    - 392 truly unique hardware layouts (VF2-proven, 356,589 pairs checked)
    - 3,336 unique reliability curves
    - 11,903 unique MD5 hw hashes (isomorphism-unaware, ~30x overcount)
    - Task allocation explains 73% of overall reliability variance
    - Hardware layout explains 27%

Three split axes available via stored hashes:
    curve_hash  → curve-level split (unseen reliability values)
    allocation  → allocation-level split (unseen task strategies)
    hw_hash     → hardware-level split (unseen physical architectures)

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
CSV_PATH    = "data/new_raw/config_all_0_22000_100.csv"
ZIP_PATH    = "data/new_raw/matrices.zip"
OUTPUT_PATH = "data/dataset.h5"
LOG_PATH    = "logs/01_preprocess.log"
N_NODES     = 30
N_FEATURES  = 5
WL_ITERATIONS = 4   # WL iterations — 4 proven sufficient for this graph family

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
        raise ValueError(
            f"Matrix size mismatch: expected {n*n}, got {values.size}"
        )

    return node_names, values.reshape(n, n).astype(np.float32)


def md5_f32(arr: np.ndarray) -> str:
    """MD5 hash of a float32 array."""
    return hashlib.md5(arr.astype(np.float32).tobytes()).hexdigest()


def compute_md5_hw_hash(edge_idx: np.ndarray,
                        node_feats: np.ndarray) -> str:
    """
    Fast MD5 hash of hardware-only edge set.
    Used as a cheap pre-filter before WL computation.
    Two graphs with different MD5 hw hashes cannot be isomorphic.
    Two graphs with the same MD5 hw hash may or may not be isomorphic
    — WL hash is needed to distinguish them.
    """
    is_task = (node_feats[:, 3] + node_feats[:, 4]) > 0
    mask    = ~is_task[edge_idx[0]] & ~is_task[edge_idx[1]]
    hw_ei   = edge_idx[:, mask]

    if hw_ei.shape[1] == 0:
        return 'empty'

    sorted_edges = hw_ei[:, np.lexsort((hw_ei[1], hw_ei[0]))]
    return hashlib.md5(sorted_edges.astype(np.int32).tobytes()).hexdigest()


def build_hw_networkx_graph(edge_idx: np.ndarray,
                             node_feats: np.ndarray) -> nx.Graph:
    """
    Build a NetworkX graph from hardware-only edges.
    Task nodes and their edges are excluded.
    Node type (compute/switch/link) stored as 'ntype' attribute
    so WL hash is type-aware.
    """
    is_task = (node_feats[:, 3] + node_feats[:, 4]) > 0
    mask    = ~is_task[edge_idx[0]] & ~is_task[edge_idx[1]]
    hw_ei   = edge_idx[:, mask]

    G = nx.Graph()
    for node_idx in range(len(node_feats)):
        if is_task[node_idx]:
            continue
        # 0=compute, 1=switch, 2=link
        ntype = str(int(node_feats[node_idx, :3].argmax()))
        G.add_node(node_idx, ntype=ntype)

    for e in range(hw_ei.shape[1]):
        src, dst = int(hw_ei[0, e]), int(hw_ei[1, e])
        G.add_edge(src, dst)

    return G


def compute_wl_hash(G: nx.Graph) -> str:
    """
    Weisfeiler-Lehman graph hash.
    Isomorphic graphs always produce the same WL hash.
    Proven reliable for this dataset by exhaustive VF2 verification
    of all 356,589 pairs across 390 isomorphic groups — 0 collisions.
    """
    return nx.weisfeiler_lehman_graph_hash(
        G, node_attr='ntype', iterations=WL_ITERATIONS
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("  STEP 1: Full Preprocessing -> HDF5")
    log("  No sampling. All 144,255 graphs processed completely.")
    log("  WL hash computed per unique MD5 hw hash (optimized).")
    log("=" * 60)

    # 1. Load full CSV
    log(f"\n[1] Loading CSV: {CSV_PATH}")
    df        = pd.read_csv(CSV_PATH)
    df.set_index('CONFIG', inplace=True)
    time_cols = [c for c in df.columns if c.replace('.', '').isdigit()]
    time_vals = np.array([float(c) for c in time_cols])
    N_TIME    = len(time_cols)

    log(f"  Rows        : {len(df):,}")
    log(f"  Time steps  : {N_TIME}  ({time_vals[0]:.0f}h - {time_vals[-1]:.0f}h)")
    log(f"  R min / max : {df[time_cols].values.min():.6f} / "
        f"{df[time_cols].values.max():.6f}")
    log(f"  NaN present : {df[time_cols].isnull().any().any()}")

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

    log(f"  Matched (have CSV target) : {len(matched_files):,}")
    log(f"  Skipped (no CSV target)   : {len(skipped_no_target):,}")
    if skipped_no_target:
        log(f"  Skipped IDs: {sorted(skipped_no_target)}")

    # 3. Parse ALL graphs
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
            )  # [30, 5]

            # Validate one-hot encoding
            row_sums = node_feats.sum(axis=1)
            if not np.all(row_sums == 1):
                bad = [node_names[i] for i in np.where(row_sums != 1)[0]]
                parse_errors.append((cid, f"Bad one-hot for: {bad}"))
                continue

            rows, cols = np.nonzero(adj)
            edge_idx   = np.vstack([rows, cols]).astype(np.int32)  # [2, E]

            curve       = target_map[cid]
            md5_hw      = compute_md5_hw_hash(edge_idx, node_feats)

            records.append({
                'config_id'    : cid,
                'allocation'   : alloc,
                'config_num'   : conf_num,
                'curve_hash'   : md5_f32(curve),
                'adj_hash'     : md5_f32(adj),
                'md5_hw'       : md5_hw,       # temporary — used to compute WL
                'node_names'   : node_names,
                'node_features': node_feats,
                'edge_index'   : edge_idx,
                'y_curve'      : curve,
            })

    N_valid = len(records)
    log(f"\n  Successfully parsed : {N_valid:,}")
    log(f"  Parse errors        : {len(parse_errors):,}")
    if parse_errors:
        log("  Error details (first 20):")
        for cid, err in parse_errors[:20]:
            log(f"    {cid}: {err}")

    if N_valid == 0:
        log("ERROR: No valid samples. Aborting.")
        save_log()
        sys.exit(1)

    # 4. Compute WL hashes — one per unique MD5 hw hash
    log(f"\n[4] Computing WL hardware hashes (optimized by MD5)...")

    # Collect one representative record per unique MD5 hw hash
    md5_to_record = {}
    for r in records:
        md5 = r['md5_hw']
        if md5 not in md5_to_record:
            md5_to_record[md5] = r

    n_unique_md5 = len(md5_to_record)
    log(f"  Unique MD5 hw hashes   : {n_unique_md5:,}")
    log(f"  WL computations needed : {n_unique_md5:,}  "
        f"(vs {N_valid:,} without optimization)")
    log(f"  Speedup factor         : {N_valid / n_unique_md5:.1f}x")

    # Compute WL hash for each unique MD5
    md5_to_wl = {}
    for md5, r in tqdm(md5_to_record.items(),
                        desc="  Computing WL", unit="hw_config"):
        G = build_hw_networkx_graph(r['edge_index'], r['node_features'])
        md5_to_wl[md5] = compute_wl_hash(G)

    # Propagate WL hash to all records
    for r in records:
        r['hw_hash'] = md5_to_wl[r['md5_hw']]

    # Report unique counts
    n_unique_curves = len(set(r['curve_hash'] for r in records))
    n_unique_wl     = len(set(r['hw_hash']    for r in records))
    n_unique_adj    = len(set(r['adj_hash']   for r in records))

    log(f"\n  Unique curve hashes (reliability behaviors) : {n_unique_curves:,}")
    log(f"  Unique WL hw hashes (hardware layouts)      : {n_unique_wl:,}")
    log(f"  Unique adj hashes   (full graph structures) : {n_unique_adj:,}")
    log(f"  Expected WL unique  : 392 (VF2-proven)")

    if n_unique_wl != 392:
        log(f"  WARNING: Expected 392 unique WL hashes, got {n_unique_wl}.")
        log(f"  This may indicate a WL iteration count issue or data change.")
    else:
        log(f"  WL unique count matches expected value. ✅")

    # 5. Write HDF5
    log(f"\n[5] Writing HDF5: {OUTPUT_PATH}")
    os.makedirs('data', exist_ok=True)
    str_dt = h5py.special_dtype(vlen=str)

    with h5py.File(OUTPUT_PATH, 'w') as h5:

        # meta
        meta = h5.create_group('meta')
        meta.create_dataset('config_ids',   (N_valid,),         dtype=str_dt)
        meta.create_dataset('allocations',  (N_valid,),         dtype=str_dt)
        meta.create_dataset('config_nums',  (N_valid,),         dtype=str_dt)
        meta.create_dataset('curve_hashes', (N_valid,),         dtype=str_dt)
        meta.create_dataset('adj_hashes',   (N_valid,),         dtype=str_dt)
        meta.create_dataset('hw_hashes',    (N_valid,),         dtype=str_dt)
        meta.create_dataset('node_names',   (N_valid, N_NODES), dtype=str_dt)

        # features
        feat_ds = h5.create_group('features').create_dataset(
            'node_features',
            shape=(N_valid, N_NODES, N_FEATURES),
            dtype=np.float32,
            chunks=(512, N_NODES, N_FEATURES),
            compression='gzip', compression_opts=4
        )

        # targets
        tgt_ds = h5.create_group('targets').create_dataset(
            'y_curve',
            shape=(N_valid, N_TIME),
            dtype=np.float32,
            chunks=(512, N_TIME),
            compression='gzip', compression_opts=4
        )

        # edges — CSR format
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
        log(f"  Mean E/graph: {all_edges.shape[1] / N_valid:.1f}")

        # fill per-sample datasets
        log("  Writing all samples to HDF5...")
        for i, r in enumerate(tqdm(records, desc="  Writing", unit="sample")):
            meta['config_ids'][i]   = r['config_id']
            meta['allocations'][i]  = r['allocation']
            meta['config_nums'][i]  = r['config_num']
            meta['curve_hashes'][i] = r['curve_hash']
            meta['adj_hashes'][i]   = r['adj_hash']
            meta['hw_hashes'][i]    = r['hw_hash']
            meta['node_names'][i]   = r['node_names']
            feat_ds[i]              = r['node_features']
            tgt_ds[i]               = r['y_curve']

        # file-level attributes
        h5.attrs['time_values']   = time_vals
        h5.attrs['n_samples']     = N_valid
        h5.attrs['n_time_steps']  = N_TIME
        h5.attrs['n_nodes']       = N_NODES
        h5.attrs['n_features']    = N_FEATURES
        h5.attrs['n_unique_hw']   = n_unique_wl
        h5.attrs['n_unique_curves'] = n_unique_curves

    # 6. Verify
    log(f"\n[6] Verifying HDF5 integrity...")
    with h5py.File(OUTPUT_PATH, 'r') as h5:
        log(f"  node_features shape  : {h5['features/node_features'].shape}")
        log(f"  y_curve shape        : {h5['targets/y_curve'].shape}")
        log(f"  edge_index shape     : {h5['edges/edge_index'].shape}")
        log(f"  edge_ptr shape       : {h5['edges/edge_ptr'].shape}")
        log(f"  config_ids[0]        : {h5['meta/config_ids'][0]}")
        log(f"  allocation[0]        : {h5['meta/allocations'][0]}")
        log(f"  curve_hash[0]        : {h5['meta/curve_hashes'][0]}")
        log(f"  adj_hash[0]          : {h5['meta/adj_hashes'][0]}")
        log(f"  hw_hash[0]           : {h5['meta/hw_hashes'][0]}")
        log(f"  y_curve[0, :5]       : {h5['targets/y_curve'][0, :5]}")
        log(f"  node_features[0,0,:] : {h5['features/node_features'][0,0,:]}")

        # Sanity checks
        ep = h5['edges/edge_ptr'][:]
        assert np.all(np.diff(ep) >= 0), "edge_ptr not monotonic!"
        assert ep[0] == 0,               "edge_ptr does not start at 0!"
        assert ep[-1] == h5['edges/edge_index'].shape[1], \
            "edge_ptr end != total edges!"
        log("  edge_ptr sanity      : PASSED")

        y_min = h5['targets/y_curve'][:, 1:].min()
        y_max = h5['targets/y_curve'][:].max()
        assert y_min >= 0.0 and y_max <= 1.0, "Reliability out of [0,1]!"
        log(f"  y_curve range        : [{y_min:.6f}, {y_max:.6f}]  PASSED")

        n_uc = len(set(h5['meta/curve_hashes'][:].astype(str)))
        n_uh = len(set(h5['meta/hw_hashes'][:].astype(str)))
        n_ua = len(set(h5['meta/adj_hashes'][:].astype(str)))
        log(f"  Unique curve hashes  : {n_uc:,}  (expected 3,336)")
        log(f"  Unique WL hw hashes  : {n_uh:,}  (expected 392)")
        log(f"  Unique adj hashes    : {n_ua:,}  (expected 144,255)")

        assert n_uc == 3336, f"Expected 3336 unique curves, got {n_uc}"
        assert n_uh == 392,  f"Expected 392 unique hw layouts, got {n_uh}"
        log("  All hash counts      : PASSED")

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    log(f"\n  HDF5 file size : {size_mb:.1f} MB")
    log(f"  Valid samples  : {N_valid:,}")
    log(f"  Parse errors   : {len(parse_errors):,}")
    log(f"\n✅ Preprocessing complete. Dataset saved to: {OUTPUT_PATH}")
    save_log()


if __name__ == "__main__":
    main()