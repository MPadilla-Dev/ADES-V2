"""
03_split.py
===========
STEP 3 of the ADES-v2 pipeline — Dataset Splitting.

Reads metadata from dataset.h5 and produces splits.json containing
train/val/test index lists for all four split strategies.

All four splits are computed in one run and stored in one file.
No data is copied — splits.json contains only integer indices into
the HDF5 file. The training script selects which split to use via
a --split argument.

Split strategies:
─────────────────────────────────────────────────────────────────

  curve_hash  (3,336 curve groups)
    Grouping unit : all samples sharing a curve_hash go to same side
    Stratification: groups sorted by mean R(8000h), every 5th → val,
                    every 5th of remainder → test, rest → train
    Guarantees    : no reliability value seen in train appears in test
    Tests         : can model predict unseen reliability values?
    Approx ratio  : 70% train / 15% val / 15% test

  allocation  (31 allocations, 8 functional groups)
    Grouping unit : entire allocations held out
    Test allocs   : 0009, 0013, 0025, 0019  (one per reliability regime)
    Val allocs    : 0001, 0008              (same functional group as test)
    Train allocs  : remaining 25 allocations
    Guarantees    : model never sees any task strategy from test allocs
    Tests         : can model generalise to unseen task strategies?

  hw_md5  (11,903 hardware wiring groups)
    Grouping unit : all samples sharing a hw_md5_hash go to same side
    Stratification: groups sorted by mean R(8000h), every 5th → val,
                    every 5th of remainder → test, rest → train
    Guarantees    : no physical wiring in test seen in any allocation
                    during training
    Tests         : can model generalise to unseen physical wirings?
    Approx ratio  : 70% train / 15% val / 15% test

  hw_wl   (392 hardware topology groups)
    Grouping unit : all samples sharing a hw_wl_hash go to same side
    Stratification: groups sorted by mean R(8000h), every 5th → val,
                    every 5th of remainder → test, rest → train
    Guarantees    : no hardware topology class in test seen in training
    Tests         : can model generalise to unseen topology classes?
    Approx ratio  : 70% train / 15% val / 15% test

─────────────────────────────────────────────────────────────────

Output splits.json structure:
  {
    "curve_hash": {
      "train": [int, ...],
      "val"  : [int, ...],
      "test" : [int, ...]
    },
    "allocation": { ... },
    "hw_md5"    : { ... },
    "hw_wl"     : { ... },
    "metadata": {
      "n_samples"    : int,
      "time_values"  : [float, ...],
      "split_summary": { ... }
    }
  }

Module : python-data/3.10-24.04
Input  : data/dataset.h5
Output : data/splits.json
"""

import os
import json
import numpy as np
import h5py
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
H5_PATH     = "data/dataset.h5"
OUTPUT_PATH = "data/splits.json"
LOG_PATH    = "logs/03_split.log"

# Allocation split — held-out allocations
# Test: one per reliability regime (high / medium-wide / wide / low)
ALLOC_TEST = ['0009', '0013', '0025', '0019']
# Val: same functional groups as test allocs to ensure similar distribution
ALLOC_VAL  = ['0001', '0008']

# Stratified split fractions for group-based splits
# Every VAL_EVERY-th group (sorted by mean R) → val
# Every TEST_EVERY-th of remainder → test
VAL_EVERY  = 7   # ~14% val
TEST_EVERY = 5   # ~17% of remainder → test → ~14% overall

RANDOM_SEED = 42

# ── LOGGING ───────────────────────────────────────────────────────────────────
log_lines = []
def log(msg=""):
    print(msg, flush=True)
    log_lines.append(msg)

def save_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w') as f:
        f.write('\n'.join(log_lines))


# ── HELPERS ───────────────────────────────────────────────────────────────────
def stratified_group_split(group_to_indices: dict,
                           group_mean_r: dict,
                           val_every: int = VAL_EVERY,
                           test_every: int = TEST_EVERY):
    """
    Stratified systematic split of hash groups.

    Groups are sorted by mean reliability at t=8000h so that
    systematic sampling gives balanced reliability distributions
    across train / val / test.

    val_every  : every val_every-th group (0-indexed) → val
    test_every : every test_every-th remaining group → test
    rest       → train

    Returns (train_indices, val_indices, test_indices)
    all as sorted lists of integer sample indices.
    """
    # Sort groups by mean reliability
    sorted_groups = sorted(
        group_to_indices.keys(),
        key=lambda h: group_mean_r[h]
    )

    val_indices   = []
    rem_groups    = []

    for gi, h in enumerate(sorted_groups):
        if gi % val_every == 0:
            val_indices.extend(group_to_indices[h])
        else:
            rem_groups.append(h)

    train_indices = []
    test_indices  = []

    for gi, h in enumerate(rem_groups):
        if gi % test_every == 0:
            test_indices.extend(group_to_indices[h])
        else:
            train_indices.extend(group_to_indices[h])

    return sorted(train_indices), sorted(val_indices), sorted(test_indices)


def split_summary(train, val, test, n_total):
    return {
        "n_train" : len(train),
        "n_val"   : len(val),
        "n_test"  : len(test),
        "pct_train": round(len(train) / n_total * 100, 1),
        "pct_val"  : round(len(val)   / n_total * 100, 1),
        "pct_test" : round(len(test)  / n_total * 100, 1),
    }


def verify_no_overlap(train, val, test, name):
    t = set(train)
    v = set(val)
    te = set(test)
    tv  = len(t & v)
    tt  = len(t & te)
    vt  = len(v & te)
    if tv + tt + vt == 0:
        log(f"  {name}: no overlap between train/val/test ✅")
    else:
        log(f"  {name}: OVERLAP DETECTED — "
            f"train∩val={tv}  train∩test={tt}  val∩test={vt} ❌")


def group_mean_reliability(group_to_indices, y_curve, t8k_idx):
    """Mean R(8000h) for each group — used for stratification."""
    return {
        h: float(y_curve[indices, t8k_idx].mean())
        for h, indices in group_to_indices.items()
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("  STEP 3: Dataset Splitting")
    log("  Four split strategies → splits.json")
    log("=" * 60)

    # ── Load metadata ──────────────────────────────────────────────────────
    log(f"\nLoading metadata from {H5_PATH}...")
    with h5py.File(H5_PATH, 'r') as h5:
        curve_hashes  = h5['meta/curve_hashes'][:].astype(str)
        allocations   = h5['meta/allocations'][:].astype(str)
        hw_md5_hashes = h5['meta/hw_md5_hashes'][:].astype(str)
        hw_wl_hashes  = h5['meta/hw_wl_hashes'][:].astype(str)
        y_curve       = h5['targets/y_curve'][:]
        time_vals     = h5.attrs['time_values']

    N       = len(curve_hashes)
    t8k_idx = int(np.argmin(np.abs(time_vals - 8000)))

    log(f"  Samples        : {N:,}")
    log(f"  Unique curves  : {len(set(curve_hashes)):,}")
    log(f"  Unique hw_md5  : {len(set(hw_md5_hashes)):,}")
    log(f"  Unique hw_wl   : {len(set(hw_wl_hashes)):,}")
    log(f"  Allocations    : {len(set(allocations))}")
    log(f"  t=8000h index  : {t8k_idx}")

    splits = {}

    # ══════════════════════════════════════════════════════════════════════
    # SPLIT 1 — curve_hash
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "─" * 60)
    log("  SPLIT 1: curve_hash")
    log("─" * 60)
    log("  Grouping: all samples sharing a curve hash → same side")
    log("  Guarantees: no reliability value in test seen in train")

    curve_groups = defaultdict(list)
    for i, ch in enumerate(curve_hashes):
        curve_groups[ch].append(i)

    curve_mean_r = group_mean_reliability(curve_groups, y_curve, t8k_idx)

    train_c, val_c, test_c = stratified_group_split(
        curve_groups, curve_mean_r
    )

    log(f"  Unique curve groups : {len(curve_groups):,}")
    log(f"  Train samples : {len(train_c):,}  "
        f"({len(train_c)/N*100:.1f}%)")
    log(f"  Val samples   : {len(val_c):,}  "
        f"({len(val_c)/N*100:.1f}%)")
    log(f"  Test samples  : {len(test_c):,}  "
        f"({len(test_c)/N*100:.1f}%)")

    # Check unique curve groups per split
    train_c_hashes = set(curve_hashes[train_c])
    val_c_hashes   = set(curve_hashes[val_c])
    test_c_hashes  = set(curve_hashes[test_c])
    log(f"  Unique curves in train : {len(train_c_hashes):,}")
    log(f"  Unique curves in val   : {len(val_c_hashes):,}")
    log(f"  Unique curves in test  : {len(test_c_hashes):,}")

    # Leakage check
    train_test_leak = len(train_c_hashes & test_c_hashes)
    train_val_leak  = len(train_c_hashes & val_c_hashes)
    val_test_leak   = len(val_c_hashes   & test_c_hashes)
    log(f"  Curve hash leakage — "
        f"train∩test={train_test_leak}  "
        f"train∩val={train_val_leak}  "
        f"val∩test={val_test_leak}")
    assert train_test_leak == 0, "curve_hash: train/test leakage!"
    assert train_val_leak  == 0, "curve_hash: train/val leakage!"
    assert val_test_leak   == 0, "curve_hash: val/test leakage!"
    log("  Leakage check: PASSED ✅")

    verify_no_overlap(train_c, val_c, test_c, "curve_hash sample indices")

    # Reliability distribution check
    log(f"  Mean R(8k) train: {y_curve[train_c, t8k_idx].mean():.4f}")
    log(f"  Mean R(8k) val  : {y_curve[val_c,   t8k_idx].mean():.4f}")
    log(f"  Mean R(8k) test : {y_curve[test_c,  t8k_idx].mean():.4f}")

    splits['curve_hash'] = {
        'train': train_c,
        'val'  : val_c,
        'test' : test_c,
    }

    # ══════════════════════════════════════════════════════════════════════
    # SPLIT 2 — allocation
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "─" * 60)
    log("  SPLIT 2: allocation")
    log("─" * 60)
    log(f"  Test  allocations: {ALLOC_TEST}")
    log(f"  Val   allocations: {ALLOC_VAL}")
    log(f"  Train allocations: all others")

    train_a, val_a, test_a = [], [], []
    for i, alloc in enumerate(allocations):
        if   alloc in ALLOC_TEST: test_a.append(i)
        elif alloc in ALLOC_VAL:  val_a.append(i)
        else:                      train_a.append(i)

    log(f"  Train samples : {len(train_a):,}  "
        f"({len(train_a)/N*100:.1f}%)")
    log(f"  Val samples   : {len(val_a):,}  "
        f"({len(val_a)/N*100:.1f}%)")
    log(f"  Test samples  : {len(test_a):,}  "
        f"({len(test_a)/N*100:.1f}%)")

    # Show allocations in each split
    train_allocs = sorted(set(allocations[train_a]))
    val_allocs   = sorted(set(allocations[val_a]))
    test_allocs  = sorted(set(allocations[test_a]))
    log(f"  Train allocs ({len(train_allocs)}): {train_allocs}")
    log(f"  Val   allocs ({len(val_allocs)}): {val_allocs}")
    log(f"  Test  allocs ({len(test_allocs)}): {test_allocs}")

    # Note: curve hash overlap between splits is expected here
    # (same curve can appear in multiple allocations)
    train_a_curves = set(curve_hashes[train_a])
    test_a_curves  = set(curve_hashes[test_a])
    overlap = len(train_a_curves & test_a_curves)
    log(f"  Curve hash overlap train∩test: {overlap:,}  "
        f"(expected due to allocation symmetry groups)")

    verify_no_overlap(train_a, val_a, test_a, "allocation sample indices")

    log(f"  Mean R(8k) train: {y_curve[train_a, t8k_idx].mean():.4f}")
    log(f"  Mean R(8k) val  : {y_curve[val_a,   t8k_idx].mean():.4f}")
    log(f"  Mean R(8k) test : {y_curve[test_a,  t8k_idx].mean():.4f}")

    splits['allocation'] = {
        'train'            : sorted(train_a),
        'val'              : sorted(val_a),
        'test'             : sorted(test_a),
        'train_allocations': train_allocs,
        'val_allocations'  : val_allocs,
        'test_allocations' : test_allocs,
    }

    # ══════════════════════════════════════════════════════════════════════
    # SPLIT 3 — hw_md5
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "─" * 60)
    log("  SPLIT 3: hw_md5 (physical wiring, 11,903 groups)")
    log("─" * 60)
    log("  Grouping: all samples sharing a hw_md5_hash → same side")
    log("  Guarantees: no physical wiring in test seen in training")

    md5_groups = defaultdict(list)
    for i, hw in enumerate(hw_md5_hashes):
        md5_groups[hw].append(i)

    md5_mean_r = group_mean_reliability(md5_groups, y_curve, t8k_idx)

    train_m, val_m, test_m = stratified_group_split(
        md5_groups, md5_mean_r
    )

    log(f"  Unique MD5 groups   : {len(md5_groups):,}")
    log(f"  Train samples : {len(train_m):,}  "
        f"({len(train_m)/N*100:.1f}%)")
    log(f"  Val samples   : {len(val_m):,}  "
        f"({len(val_m)/N*100:.1f}%)")
    log(f"  Test samples  : {len(test_m):,}  "
        f"({len(test_m)/N*100:.1f}%)")

    # Unique hw groups per split
    train_m_hws = set(hw_md5_hashes[train_m])
    val_m_hws   = set(hw_md5_hashes[val_m])
    test_m_hws  = set(hw_md5_hashes[test_m])
    log(f"  HW wirings in train : {len(train_m_hws):,}")
    log(f"  HW wirings in val   : {len(val_m_hws):,}")
    log(f"  HW wirings in test  : {len(test_m_hws):,}")

    hw_train_test_leak = len(train_m_hws & test_m_hws)
    hw_train_val_leak  = len(train_m_hws & val_m_hws)
    hw_val_test_leak   = len(val_m_hws   & test_m_hws)
    log(f"  HW MD5 leakage — "
        f"train∩test={hw_train_test_leak}  "
        f"train∩val={hw_train_val_leak}  "
        f"val∩test={hw_val_test_leak}")
    assert hw_train_test_leak == 0, "hw_md5: train/test leakage!"
    assert hw_train_val_leak  == 0, "hw_md5: train/val leakage!"
    assert hw_val_test_leak   == 0, "hw_md5: val/test leakage!"
    log("  HW MD5 leakage check: PASSED ✅")

    # Curve overlap is expected (same curve from different wirings)
    curve_m_train = set(curve_hashes[train_m])
    curve_m_test  = set(curve_hashes[test_m])
    log(f"  Curve overlap train∩test: {len(curve_m_train & curve_m_test):,}  "
        f"(expected — same curve achievable by different wirings)")

    verify_no_overlap(train_m, val_m, test_m, "hw_md5 sample indices")
    log(f"  Mean R(8k) train: {y_curve[train_m, t8k_idx].mean():.4f}")
    log(f"  Mean R(8k) val  : {y_curve[val_m,   t8k_idx].mean():.4f}")
    log(f"  Mean R(8k) test : {y_curve[test_m,  t8k_idx].mean():.4f}")

    splits['hw_md5'] = {
        'train': train_m,
        'val'  : val_m,
        'test' : test_m,
    }

    # ══════════════════════════════════════════════════════════════════════
    # SPLIT 4 — hw_wl
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "─" * 60)
    log("  SPLIT 4: hw_wl (topology class, 392 groups, VF2-verified)")
    log("─" * 60)
    log("  Grouping: all samples sharing a hw_wl_hash → same side")
    log("  Guarantees: no hardware topology class in test seen in training")

    wl_groups = defaultdict(list)
    for i, wl in enumerate(hw_wl_hashes):
        wl_groups[wl].append(i)

    wl_mean_r = group_mean_reliability(wl_groups, y_curve, t8k_idx)

    train_w, val_w, test_w = stratified_group_split(
        wl_groups, wl_mean_r
    )

    log(f"  Unique WL groups    : {len(wl_groups):,}")
    log(f"  Train samples : {len(train_w):,}  "
        f"({len(train_w)/N*100:.1f}%)")
    log(f"  Val samples   : {len(val_w):,}  "
        f"({len(val_w)/N*100:.1f}%)")
    log(f"  Test samples  : {len(test_w):,}  "
        f"({len(test_w)/N*100:.1f}%)")

    train_w_wls = set(hw_wl_hashes[train_w])
    val_w_wls   = set(hw_wl_hashes[val_w])
    test_w_wls  = set(hw_wl_hashes[test_w])
    log(f"  WL groups in train  : {len(train_w_wls):,}")
    log(f"  WL groups in val    : {len(val_w_wls):,}")
    log(f"  WL groups in test   : {len(test_w_wls):,}")

    wl_train_test = len(train_w_wls & test_w_wls)
    wl_train_val  = len(train_w_wls & val_w_wls)
    wl_val_test   = len(val_w_wls   & test_w_wls)
    log(f"  WL leakage — "
        f"train∩test={wl_train_test}  "
        f"train∩val={wl_train_val}  "
        f"val∩test={wl_val_test}")
    assert wl_train_test == 0, "hw_wl: train/test leakage!"
    assert wl_train_val  == 0, "hw_wl: train/val leakage!"
    assert wl_val_test   == 0, "hw_wl: val/test leakage!"
    log("  WL leakage check    : PASSED ✅")

    verify_no_overlap(train_w, val_w, test_w, "hw_wl sample indices")
    log(f"  Mean R(8k) train: {y_curve[train_w, t8k_idx].mean():.4f}")
    log(f"  Mean R(8k) val  : {y_curve[val_w,   t8k_idx].mean():.4f}")
    log(f"  Mean R(8k) test : {y_curve[test_w,  t8k_idx].mean():.4f}")

    splits['hw_wl'] = {
        'train': train_w,
        'val'  : val_w,
        'test' : test_w,
    }

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("  SPLIT SUMMARY")
    log("=" * 60)
    log(f"  {'Split':12s}  {'Train':>8}  {'Val':>8}  {'Test':>8}  "
        f"{'Train%':>7}  {'Val%':>6}  {'Test%':>6}")
    log(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  "
        f"{'-'*7}  {'-'*6}  {'-'*6}")
    for name, (tr, va, te) in [
        ('curve_hash', (train_c, val_c, test_c)),
        ('allocation', (train_a, val_a, test_a)),
        ('hw_md5',     (train_m, val_m, test_m)),
        ('hw_wl',      (train_w, val_w, test_w)),
    ]:
        log(f"  {name:12s}  {len(tr):>8,}  {len(va):>8,}  {len(te):>8,}  "
            f"{len(tr)/N*100:>6.1f}%  {len(va)/N*100:>5.1f}%  "
            f"{len(te)/N*100:>5.1f}%")

    # ── Write splits.json ──────────────────────────────────────────────────
    log(f"\nWriting splits to {OUTPUT_PATH}...")

    output = {
        'metadata': {
            'n_samples'   : N,
            'time_values' : time_vals.tolist(),
            't8k_idx'     : t8k_idx,
            'split_summary': {
                name: split_summary(
                    splits[name]['train'],
                    splits[name]['val'],
                    splits[name]['test'],
                    N
                )
                for name in splits
            }
        }
    }

    # Add splits — convert numpy arrays to plain Python lists for JSON
    for name, s in splits.items():
        output[name] = {
            k: (v if isinstance(v, list) and len(v) > 0
                  and isinstance(v[0], str)
                else [int(x) for x in v])
            for k, v in s.items()
        }

    os.makedirs(os.path.dirname(OUTPUT_PATH)
                if os.path.dirname(OUTPUT_PATH) else '.', exist_ok=True)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    log(f"  File size : {size_kb:.1f} KB")
    log(f"\n✅ splits.json written to {OUTPUT_PATH}")
    save_log()


if __name__ == "__main__":
    main()