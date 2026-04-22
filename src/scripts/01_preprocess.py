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
        adj_hashes      [N]        str   — md5 of full adjacency matrix
        hw_hashes       [N]        str   — md5 of hardware-only edges
                                           (task edges stripped)
                                           used for hardware-config split
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
    All edges where neither endpoint is a task node are retained.
    The remaining edge set is sorted canonically and md5-hashed.
    Two samples with the same hw_hash have identical physical hardware
    connectivity, differing only in task node placement.

Key findings from EDA that motivate this design:
    - 11,903 unique hardware configurations across 144,255 samples
    - Task allocation explains 73% of overall reliability variance
    - Hardware configuration explains only 27%
    - Three split strategies use hw_hash, curve_hash, and allocation
      respectively — all stored here for 03_split.py

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
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH    = "data/new_raw/config_all_0_22000_100.csv"
ZIP_PATH    = "data/new_raw/matrices.zip"
OUTPUT_PATH = "data/dataset.h5"
LOG_PATH    = "logs/01_preprocess.log"
N_NODES     = 30
N_FEATURES  = 5

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
    return hashlib.md5(arr.astype(np.float32).tobytes()).hexdigest()


def compute_hw_hash(edge_idx: np.ndarray,
                    node_feats: np.ndarray) -> str:
    """
    Hardware-only graph hash.

    Strips all edges where either endpoint is a task node (feature
    indices 3 or 4), then sorts the remaining edges canonically and
    returns their md5 hash. Two samples with the same hw_hash have
    identical physical hardware connectivity regardless of task placement.

    Parameters
    ----------
    edge_idx   : [2, E]  full edge index for this sample
    node_feats : [30, 5] one-hot node features for this sample
    """
    is_task = (node_feats[:, 3] + node_feats[:, 4]) > 0  # [30] bool
    mask    = ~is_task[edge_idx[0]] & ~is_task[edge_idx[1]]
    hw_ei   = edge_idx[:, mask]

    if hw_ei.shape[1] == 0:
        return 'empty'

    # Canonical sort: primary by source node, secondary by dest node
    sorted_edges = hw_ei[:, np.lexsort((hw_ei[1], hw_ei[0]))]
    return hashlib.md5(sorted_edges.astype(np.int32).tobytes()).hexdigest()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("  STEP 1: Full Preprocessing -> HDF5")
    log("  No sampling. All 144,255 graphs processed completely.")
    log("  Includes hardware-only hash (hw_hash) for config-level split.")
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

            curve   = target_map[cid]
            hw_hash = compute_hw_hash(edge_idx, node_feats)

            records.append({
                'config_id'    : cid,
                'allocation'   : alloc,
                'config_num'   : conf_num,
                'curve_hash'   : md5_f32(curve),
                'adj_hash'     : md5_f32(adj),
                'hw_hash'      : hw_hash,
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

    # Report unique hash counts
    n_unique_curves = len(set(r['curve_hash'] for r in records))
    n_unique_hw     = len(set(r['hw_hash']    for r in records))
    n_unique_adj    = len(set(r['adj_hash']   for r in records))
    log(f"\n  Unique curve hashes (reliability behaviors) : {n_unique_curves:,}")
    log(f"  Unique hw hashes    (hardware configs)      : {n_unique_hw:,}")
    log(f"  Unique adj hashes   (full graph structures) : {n_unique_adj:,}")

    # 4. Write HDF5
    log(f"\n[4] Writing HDF5: {OUTPUT_PATH}")
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
        h5.attrs['time_values']  = time_vals
        h5.attrs['n_samples']    = N_valid
        h5.attrs['n_time_steps'] = N_TIME
        h5.attrs['n_nodes']      = N_NODES
        h5.attrs['n_features']   = N_FEATURES

    # 5. Verify
    log(f"\n[5] Verifying HDF5 integrity...")
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

        # Report unique counts from stored hashes
        n_uc = len(set(h5['meta/curve_hashes'][:].astype(str)))
        n_uh = len(set(h5['meta/hw_hashes'][:].astype(str)))
        n_ua = len(set(h5['meta/adj_hashes'][:].astype(str)))
        log(f"  Unique curve hashes  : {n_uc:,}")
        log(f"  Unique hw hashes     : {n_uh:,}")
        log(f"  Unique adj hashes    : {n_ua:,}")

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    log(f"\n  HDF5 file size : {size_mb:.1f} MB")
    log(f"  Valid samples  : {N_valid:,}")
    log(f"  Parse errors   : {len(parse_errors):,}")
    log(f"\n✅ Preprocessing complete. Dataset saved to: {OUTPUT_PATH}")
    save_log()


if __name__ == "__main__":
    main()