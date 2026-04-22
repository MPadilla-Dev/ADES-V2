"""
01_preprocess.py
================
STEP 1 of the ADES-v2 pipeline.

Reads raw data (matrices.zip + CSV) and writes a single HDF5 file
with full fidelity — all 221 time points, all 144,255 samples.

HDF5 Layout:
  meta/
    config_ids    [N]        string  — e.g. "0000_0736"
    allocations   [N]        string  — e.g. "0000"
    config_nums   [N]        string  — e.g. "0736"
    curve_hashes  [N]        string  — md5 of y_curve (for split grouping)
    adj_hashes    [N]        string  — md5 of adjacency matrix
    node_names    [N, 30]    string  — preserved for re-featurization
  features/
    node_features [N, 30, 5] float32 — one-hot type encoding
  edges/
    edge_index    [2, E_tot] int32   — all edge indices concatenated
    edge_ptr      [N+1]      int64   — CSR pointer: sample i uses
                                       edge_index[:, ptr[i]:ptr[i+1]]
  targets/
    y_curve       [N, 221]   float32 — full reliability curve 0h-22000h

Node features (5-dim one-hot):
  [0] is_compute  (N1, N2 ...)
  [1] is_switch   (S1, S2 ...)
  [2] is_link     (N1S1, S1S2 ...)
  [3] is_task_T1  (T1_1, T1_2 ...)
  [4] is_task_T2  (T2_4, T2_5 ...)

Usage:
  python src/scripts/01_preprocess.py

Output:
  data/dataset.h5   (~500MB estimated)
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
N_NODES     = 30   # Fixed for all graphs in this dataset
N_FEATURES  = 5    # One-hot node type
CHUNK_SIZE  = 1000 # Write to HDF5 every N samples (memory safety)

# ── NODE CLASSIFIER ───────────────────────────────────────────────────────────
def classify_node(name: str) -> str:
    """
    Canonical node type classifier.
    Single source of truth — do not duplicate this logic elsewhere.
      task    : contains underscore  (T1_1, T2_4 ...)
      link    : two device prefixes fused (N1S1, S1S2 ...)
      compute : starts with N, single device (N1, N2 ...)
      switch  : starts with S, single device (S1, S2 ...)
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
    """5-dim one-hot encoding from node name."""
    feat = [0, 0, 0, 0, 0]
    ntype = classify_node(name)
    if ntype == 'compute':  feat[0] = 1
    elif ntype == 'switch': feat[1] = 1
    elif ntype == 'link':   feat[2] = 1
    elif ntype == 'task':
        if name.startswith('T1'): feat[3] = 1
        else:                     feat[4] = 1
    return feat


# ── PARSING ───────────────────────────────────────────────────────────────────
def parse_graph_file(f_obj):
    """
    Parse a single graph txt file.
    Returns (node_names: list[str], adj_matrix: np.ndarray[N,N])
    Raises ValueError on bad format.
    """
    lines = f_obj.read().decode('utf-8').splitlines()
    if not lines:
        raise ValueError("Empty file")

    # Header: # ['N1', 'N1S1', ...]
    header = lines[0].replace('#', '').strip()
    node_names = eval(header)
    n = len(node_names)

    if n != N_NODES:
        raise ValueError(f"Expected {N_NODES} nodes, got {n}")

    # Adjacency matrix
    matrix_str = ' '.join(lines[1:]).replace('[', ' ').replace(']', ' ')
    values = np.fromstring(matrix_str, sep=' ')
    if values.size != n * n:
        raise ValueError(f"Matrix size mismatch: expected {n*n}, got {values.size}")

    adj = values.reshape(n, n)
    return node_names, adj


def md5(arr: np.ndarray) -> str:
    return hashlib.md5(arr.astype(np.float32).tobytes()).hexdigest()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  STEP 1: Preprocessing → HDF5")
    print("=" * 60)

    # 1. Load CSV
    print(f"\n[1] Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df.set_index('CONFIG', inplace=True)

    time_cols = [c for c in df.columns if c.replace('.', '').isdigit()]
    time_vals = np.array([float(c) for c in time_cols])
    N_TIME    = len(time_cols)

    print(f"  Samples    : {len(df):,}")
    print(f"  Time steps : {N_TIME}  ({time_vals[0]:.0f}h – {time_vals[-1]:.0f}h)")

    # Build fast lookup: config_id → float32 curve array
    print("  Building target lookup...")
    target_map = {
        str(idx): row[time_cols].values.astype(np.float32)
        for idx, row in df.iterrows()
    }

    # 2. Scan ZIP for matched files
    print(f"\n[2] Scanning ZIP: {ZIP_PATH}")
    matched_files = []
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for fname in z.namelist():
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

    N = len(matched_files)
    print(f"  Matched files: {N:,}")

    # 3. Create HDF5 file
    print(f"\n[3] Creating HDF5: {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH) if os.path.dirname(OUTPUT_PATH) else '.', exist_ok=True)

    with h5py.File(OUTPUT_PATH, 'w') as h5:

        # Pre-allocate datasets whose shapes are known
        # meta — variable-length strings
        str_dt = h5py.special_dtype(vlen=str)

        meta = h5.create_group('meta')
        meta.create_dataset('config_ids',   shape=(N,),        dtype=str_dt)
        meta.create_dataset('allocations',  shape=(N,),        dtype=str_dt)
        meta.create_dataset('config_nums',  shape=(N,),        dtype=str_dt)
        meta.create_dataset('curve_hashes', shape=(N,),        dtype=str_dt)
        meta.create_dataset('adj_hashes',   shape=(N,),        dtype=str_dt)
        meta.create_dataset('node_names',   shape=(N, N_NODES),dtype=str_dt)

        # features — fixed shape
        feat_grp = h5.create_group('features')
        feat_ds  = feat_grp.create_dataset(
            'node_features',
            shape=(N, N_NODES, N_FEATURES),
            dtype=np.float32,
            chunks=(CHUNK_SIZE, N_NODES, N_FEATURES),
            compression='gzip', compression_opts=4
        )

        # targets — fixed shape
        tgt_grp = h5.create_group('targets')
        tgt_ds  = tgt_grp.create_dataset(
            'y_curve',
            shape=(N, N_TIME),
            dtype=np.float32,
            chunks=(CHUNK_SIZE, N_TIME),
            compression='gzip', compression_opts=4
        )

        # edges — variable length, use CSR format
        # We collect all edges first, then write
        edge_grp = h5.create_group('edges')

        # 4. Stream ZIP and fill HDF5
        print(f"\n[4] Processing {N:,} graphs...")
        processed  = 0
        skipped    = 0

        # Buffers for CSR edge storage
        all_edge_indices = []   # list of [2, E_i] arrays
        edge_ptr         = [0]  # cumulative edge counts

        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            for i, (cid, alloc, conf_num, fpath) in enumerate(
                tqdm(matched_files, desc="  Parsing graphs")
            ):
                try:
                    with z.open(fpath) as f:
                        node_names, adj = parse_graph_file(f)
                except Exception as e:
                    tqdm.write(f"  SKIP {cid}: {e}")
                    skipped += 1
                    # Write placeholder zeros so indices stay aligned
                    node_names = [f'node_{k}' for k in range(N_NODES)]
                    adj = np.zeros((N_NODES, N_NODES), dtype=np.float32)

                # Node features
                node_feats = np.array(
                    [node_to_feature(name) for name in node_names],
                    dtype=np.float32
                )  # [30, 5]

                # Edge index from adjacency
                rows, cols = np.nonzero(adj)
                edge_idx   = np.vstack([rows, cols]).astype(np.int32)  # [2, E]

                # Hashes
                c_hash = hashlib.md5(
                    target_map[cid].tobytes()
                ).hexdigest()
                a_hash = md5(adj)

                # Write scalars and arrays to HDF5
                meta['config_ids'][i]    = cid
                meta['allocations'][i]   = alloc
                meta['config_nums'][i]   = conf_num
                meta['curve_hashes'][i]  = c_hash
                meta['adj_hashes'][i]    = a_hash
                meta['node_names'][i]    = node_names

                feat_ds[i]  = node_feats
                tgt_ds[i]   = target_map[cid]

                # Accumulate edges for CSR
                all_edge_indices.append(edge_idx)
                edge_ptr.append(edge_ptr[-1] + edge_idx.shape[1])

                processed += 1

        # Write edges as CSR
        print("\n[5] Writing edge index (CSR format)...")
        all_edges = np.concatenate(all_edge_indices, axis=1).astype(np.int32)
        edge_grp.create_dataset('edge_index', data=all_edges,
                                compression='gzip', compression_opts=4)
        edge_grp.create_dataset('edge_ptr',
                                data=np.array(edge_ptr, dtype=np.int64))

        # Store time axis as metadata
        h5.attrs['time_values']  = time_vals
        h5.attrs['n_samples']    = N
        h5.attrs['n_time_steps'] = N_TIME
        h5.attrs['n_nodes']      = N_NODES
        h5.attrs['n_features']   = N_FEATURES

    # 5. Verify output
    print(f"\n[6] Verifying output...")
    with h5py.File(OUTPUT_PATH, 'r') as h5:
        print(f"  node_features : {h5['features/node_features'].shape}")
        print(f"  y_curve       : {h5['targets/y_curve'].shape}")
        print(f"  edge_index    : {h5['edges/edge_index'].shape}")
        print(f"  edge_ptr      : {h5['edges/edge_ptr'].shape}")
        print(f"  config_ids[0] : {h5['meta/config_ids'][0]}")
        print(f"  y_curve[0,:5] : {h5['targets/y_curve'][0, :5]}")
        print(f"  curve_hash[0] : {h5['meta/curve_hashes'][0]}")

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\n  File size : {size_mb:.1f} MB")
    print(f"  Processed : {processed:,}")
    print(f"  Skipped   : {skipped:,}")
    print(f"\n✅ Done. Dataset saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()