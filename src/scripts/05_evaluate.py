"""
05_evaluate.py
==============
STEP 5 of the ADES-v2 pipeline — Evaluation.

Evaluates all trained models on their respective test sets and produces
a comprehensive results report with cross-experiment comparison.

Usage:
  # Evaluate one experiment
  python src/scripts/05_evaluate.py \
      --split curve_hash \
      --formulation time_conditioned

  # Evaluate all experiments at once
  python src/scripts/05_evaluate.py --all

Metrics computed for each experiment:
  Standard:
    MSE, MAE, R² on test set
    Per-timestep MSE across 0h–22,000h (time-conditioned only)

  Crossing accuracy (key metric):
    For pairs of test samples whose true curves cross, what fraction
    does the model correctly rank at queried times t?
    Reported at t = 2000h, 4000h, 8000h, 12000h, 16000h, 22000h

Outputs:
  results/05_evaluate/
    {split}_{formulation}_metrics.json    ← per-experiment metrics
    {split}_{formulation}_per_timestep.png ← per-timestep MSE curve
    comparison_table.txt                  ← all experiments side by side
    crossing_accuracy.png                 ← crossing accuracy comparison
    summary_report.txt                    ← human-readable summary

Module: pytorch/2.4
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from collections import defaultdict
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
H5_PATH     = "data/dataset.h5"
SPLITS_PATH = "data/splits.json"
MODELS_DIR  = "models"
OUT_DIR     = "results/05_evaluate"
os.makedirs(OUT_DIR, exist_ok=True)

# All experiments in the experiment table
ALL_EXPERIMENTS = [
    ("curve_hash",  "time_conditioned"),
    ("curve_hash",  "full_curve"),
    ("curve_hash",  "single_timestep"),
    ("allocation",  "time_conditioned"),
    ("allocation",  "full_curve"),
    ("hw_md5",      "time_conditioned"),
    ("hw_wl",       "time_conditioned"),
]

# Timesteps to query for crossing accuracy
CROSSING_QUERY_TIMES = [2000, 4000, 8000, 12000, 16000, 22000]

# Fixed timestep for single_timestep formulation
T_FIXED_HOURS = 8000

# ── MODEL ─────────────────────────────────────────────────────────────────────
class GAT_LN_HEAD(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 dropout_rate, num_heads=8):
        super().__init__()
        self.conv1 = GATConv(input_dim,              hidden_dim,     heads=num_heads)
        self.ln1   = torch.nn.LayerNorm(hidden_dim * num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim // 2, heads=num_heads)
        self.ln2   = torch.nn.LayerNorm((hidden_dim // 2) * num_heads)
        self.drop  = torch.nn.Dropout(p=dropout_rate)
        self.fc    = torch.nn.Linear((hidden_dim // 2) * num_heads, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.nn.functional.relu(self.ln1(self.conv1(x, edge_index)))
        x = self.drop(torch.nn.functional.relu(self.ln2(self.conv2(x, edge_index))))
        x = global_mean_pool(x, batch)
        return self.fc(x)


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_test_data(h5_path, test_indices, time_vals):
    """
    Load all test samples into memory as numpy arrays.
    Returns:
        node_feats  : [N, 30, 5]
        edge_indices: list of [2, E_i] arrays
        y_curves    : [N, 221]
        curve_hashes: [N] str
    """
    N = len(test_indices)
    print(f"  Loading {N:,} test samples into memory...")

    with h5py.File(h5_path, 'r') as h5:
        y_curves     = h5['targets/y_curve'][test_indices]        # [N, 221]
        node_feats   = h5['features/node_features'][test_indices]  # [N, 30, 5]
        curve_hashes = h5['meta/curve_hashes'][test_indices].astype(str)
        edge_ptr_all = h5['edges/edge_ptr'][:]
        edge_idx_all = h5['edges/edge_index'][:]

    edge_indices = []
    for idx in test_indices:
        e_start = int(edge_ptr_all[idx])
        e_end   = int(edge_ptr_all[idx + 1])
        edge_indices.append(edge_idx_all[:, e_start:e_end])

    return node_feats, edge_indices, y_curves, curve_hashes


def predict_at_time(model, node_feats, edge_indices, t_norm,
                    formulation, time_vals, device, batch_size=256):
    """
    Run inference for all test samples at a specific normalised time t_norm.

    For time_conditioned: appends t_norm to node features, predicts scalar R(t)
    For full_curve      : predicts all 221 values, slices at t_idx
    For single_timestep : predicts scalar R(t_fixed) — t_norm ignored

    Returns predictions array [N] as numpy.
    """
    from torch_geometric.data import Data, Batch

    model.eval()
    predictions = []
    N = len(node_feats)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end        = min(start + batch_size, N)
            batch_nf   = node_feats[start:end]    # [B, 30, 5]
            batch_ei   = edge_indices[start:end]

            data_list = []
            for i in range(end - start):
                nf = torch.tensor(batch_nf[i], dtype=torch.float32)

                if formulation == 'time_conditioned':
                    t_feat = torch.full((nf.shape[0], 1), t_norm)
                    nf     = torch.cat([nf, t_feat], dim=1)  # [30, 6]

                ei = torch.tensor(batch_ei[i], dtype=torch.long)
                data_list.append(Data(x=nf, edge_index=ei))

            batch = Batch.from_data_list(data_list).to(device)
            pred  = model(batch)  # [B, output_dim]

            if formulation == 'full_curve':
                # Find the time index closest to t_norm * t_max
                t_hours = t_norm * time_vals[-1]
                t_idx   = int(np.argmin(np.abs(time_vals - t_hours)))
                pred    = pred[:, t_idx]  # [B]
            else:
                pred = pred.squeeze(-1)   # [B]

            predictions.extend(pred.cpu().numpy().tolist())

    return np.array(predictions)


def predict_full_curve(model, node_feats, edge_indices, formulation,
                       time_vals, device, batch_size=256):
    """
    Predict the full reliability curve for all test samples.

    For time_conditioned: runs 221 forward passes (one per timestep)
    For full_curve      : runs one forward pass
    For single_timestep : runs one forward pass, broadcasts to [N, 221]

    Returns [N, 221] array of predicted curves.
    """
    from torch_geometric.data import Data, Batch

    model.eval()
    N      = len(node_feats)
    N_TIME = len(time_vals)
    t_max  = float(time_vals[-1])

    if formulation == 'full_curve':
        # One pass predicts all 221 timesteps
        all_preds = []
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end      = min(start + batch_size, N)
                batch_nf = node_feats[start:end]
                batch_ei = edge_indices[start:end]
                data_list = []
                for i in range(end - start):
                    nf = torch.tensor(batch_nf[i], dtype=torch.float32)
                    ei = torch.tensor(batch_ei[i], dtype=torch.long)
                    data_list.append(Data(x=nf, edge_index=ei))
                batch = Batch.from_data_list(data_list).to(device)
                pred  = model(batch)  # [B, 221]
                all_preds.append(pred.cpu().numpy())
        return np.concatenate(all_preds, axis=0)  # [N, 221]

    elif formulation == 'single_timestep':
        # One pass predicts R(8000h), broadcast to full curve shape
        # We only use this for crossing analysis at t=8000h
        t8k_pred = predict_at_time(
            model, node_feats, edge_indices,
            T_FIXED_HOURS / t_max, formulation, time_vals, device, batch_size
        )  # [N]
        # Return as [N, 1] — only valid at t=8000h
        return t8k_pred[:, np.newaxis]

    else:  # time_conditioned — 221 forward passes
        print(f"  Running 221 forward passes for full curve reconstruction...")
        all_curves = np.zeros((N, N_TIME), dtype=np.float32)
        for t_idx, t_val in enumerate(tqdm(time_vals, desc="  Timesteps")):
            t_norm = float(t_val) / t_max
            preds  = predict_at_time(
                model, node_feats, edge_indices,
                t_norm, formulation, time_vals, device, batch_size
            )
            all_curves[:, t_idx] = preds
        return all_curves  # [N, 221]


# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_standard_metrics(y_true, y_pred):
    """MSE, MAE, R² — works for any shape if both are [N, *]."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mse    = float(np.mean((y_true - y_pred) ** 2))
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return mse, mae, r2


def compute_crossing_accuracy(y_true, y_pred_curves, time_vals,
                               query_times, curve_hashes,
                               max_pairs=50000):
    """
    Crossing accuracy: for pairs whose true curves cross, what fraction
    does the model correctly rank at each query time?

    y_true        : [N, 221] true curves
    y_pred_curves : [N, 221] predicted curves (or [N,1] for single_timestep)
    query_times   : list of hours to query (e.g. [2000, 4000, 8000, ...])
    curve_hashes  : [N] str — we only compare pairs with different hashes
    max_pairs     : cap on pairs to check (for speed)

    Returns dict: {t_hours: crossing_accuracy_float}
    """
    N      = len(y_true)
    t_max  = float(time_vals[-1])

    # Find crossing pairs among unique curves in test set
    # Two samples cross if their difference changes sign
    # Only compare samples with different curve hashes (avoid trivial same-curve pairs)
    hash_to_idx = defaultdict(list)
    for i, ch in enumerate(curve_hashes):
        hash_to_idx[ch].append(i)
    unique_hashes = list(hash_to_idx.keys())
    n_unique      = len(unique_hashes)

    print(f"  Finding crossing pairs among {n_unique} unique curves in test set...")

    # Drop t=0 for crossing detection
    t0_mask  = time_vals > 0
    t_work   = time_vals[t0_mask]
    Y_work   = y_true[:, t0_mask]  # [N, 220]

    crossing_pairs = []  # list of (i, j) sample indices

    # Use one representative per unique curve hash
    rep_indices = [hash_to_idx[h][0] for h in unique_hashes]

    np.random.seed(42)
    if n_unique * (n_unique - 1) // 2 > max_pairs:
        # Sample random pairs
        sampled_i = np.random.randint(0, n_unique, max_pairs * 3)
        sampled_j = np.random.randint(0, n_unique, max_pairs * 3)
        pairs_to_check = [(i, j) for i, j in zip(sampled_i, sampled_j)
                         if i != j][:max_pairs]
    else:
        pairs_to_check = [(i, j) for i in range(n_unique)
                         for j in range(i+1, n_unique)]

    for pi, pj in pairs_to_check:
        ri, rj = rep_indices[pi], rep_indices[pj]
        diff   = Y_work[ri] - Y_work[rj]
        max_diff = np.abs(diff).max()
        if max_diff < 0.01:
            continue
        has_pos = (diff >  1e-6).any()
        has_neg = (diff < -1e-6).any()
        if has_pos and has_neg:
            crossing_pairs.append((ri, rj))

    n_crossing = len(crossing_pairs)
    print(f"  Crossing pairs found: {n_crossing:,}")

    if n_crossing == 0:
        return {t: None for t in query_times}

    results = {}
    for t_hours in query_times:
        t_idx = int(np.argmin(np.abs(time_vals - t_hours)))

        correct = 0
        total   = 0

        for ri, rj in crossing_pairs:
            true_i  = y_true[ri, t_idx]
            true_j  = y_true[rj, t_idx]

            if abs(true_i - true_j) < 1e-6:
                continue  # tied — skip

            # Determine true ranking
            true_i_better = true_i > true_j

            # Get predicted values at this t
            if y_pred_curves.shape[1] == 1:
                # single_timestep — only valid at t=8000h
                if t_hours != T_FIXED_HOURS:
                    total += 1
                    continue  # cannot evaluate at other times
                pred_i = float(y_pred_curves[ri, 0])
                pred_j = float(y_pred_curves[rj, 0])
            else:
                pred_i = float(y_pred_curves[ri, t_idx])
                pred_j = float(y_pred_curves[rj, t_idx])

            pred_i_better = pred_i > pred_j

            if true_i_better == pred_i_better:
                correct += 1
            total += 1

        results[t_hours] = float(correct / total) if total > 0 else None
        print(f"    t={t_hours:>6}h: {correct:,}/{total:,} = "
              f"{results[t_hours]*100:.1f}%" if results[t_hours] else
              f"    t={t_hours:>6}h: N/A")

    return results


# ── EVALUATE ONE EXPERIMENT ───────────────────────────────────────────────────
def evaluate_experiment(split, formulation, device, time_vals):
    run_name   = f"{split}_{formulation}"
    model_path = os.path.join(MODELS_DIR, f"{run_name}_best.pth")
    print(f"\n{'='*60}")
    print(f"  Evaluating: {run_name}")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  ERROR: model not found at {model_path}")
        return None

    # Load splits
    with open(SPLITS_PATH, 'r') as f:
        splits_data = json.load(f)
    test_indices = splits_data[split]['test']
    print(f"  Test samples: {len(test_indices):,}")

    # Load test data
    node_feats, edge_indices, y_curves, curve_hashes = load_test_data(
        H5_PATH, test_indices, time_vals
    )

    # Load model
    input_dim  = 6 if formulation == 'time_conditioned' else 5
    n_time     = len(time_vals)
    output_dim = n_time if formulation == 'full_curve' else 1

    model = GAT_LN_HEAD(
        input_dim=input_dim, hidden_dim=64, output_dim=output_dim,
        dropout_rate=0.3, num_heads=8
    ).to(device)
    model.load_state_dict(torch.load(model_path,
                                      map_location=device,
                                      weights_only=True))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Standard metrics — evaluate at t=8000h for all formulations
    t8k_idx  = int(np.argmin(np.abs(time_vals - T_FIXED_HOURS)))
    t_max    = float(time_vals[-1])
    t8k_norm = T_FIXED_HOURS / t_max

    print(f"  Computing standard metrics at t={T_FIXED_HOURS}h...")
    preds_8k = predict_at_time(
        model, node_feats, edge_indices, t8k_norm,
        formulation, time_vals, device
    )
    y_true_8k = y_curves[:, t8k_idx]

    mse, mae, r2 = compute_standard_metrics(y_true_8k, preds_8k)
    print(f"  MSE : {mse:.6f}")
    print(f"  MAE : {mae:.6f}")
    print(f"  R²  : {r2:.4f}")

    # Per-timestep MSE (for time_conditioned only — others would be too slow)
    per_timestep_mse = None
    if formulation == 'time_conditioned':
        print(f"  Computing per-timestep MSE across full curve...")
        pred_curves = predict_full_curve(
            model, node_feats, edge_indices, formulation,
            time_vals, device
        )
        per_timestep_mse = []
        for t_idx in range(len(time_vals)):
            ts_mse = float(np.mean(
                (y_curves[:, t_idx] - pred_curves[:, t_idx]) ** 2
            ))
            per_timestep_mse.append(ts_mse)

        # Per-timestep MSE plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time_vals, per_timestep_mse, color='steelblue', linewidth=1.5)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("MSE")
        ax.set_title(f"Per-Timestep MSE — {run_name}\n"
                     f"Overall MSE at t=8k: {mse:.6f}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(f"{OUT_DIR}/{run_name}_per_timestep.png", dpi=150)
        plt.close(fig)
        print(f"  -> Saved {run_name}_per_timestep.png")

        # Full curve for crossing analysis
        pred_curves_for_crossing = pred_curves

    elif formulation == 'full_curve':
        print(f"  Predicting full curves for crossing analysis...")
        pred_curves_for_crossing = predict_full_curve(
            model, node_feats, edge_indices, formulation,
            time_vals, device
        )
    else:
        # single_timestep — only predict at t=8000h
        pred_curves_for_crossing = preds_8k[:, np.newaxis]

    # Crossing accuracy
    print(f"  Computing crossing accuracy...")
    crossing_acc = compute_crossing_accuracy(
        y_curves, pred_curves_for_crossing, time_vals,
        CROSSING_QUERY_TIMES, curve_hashes
    )

    # Save metrics
    metrics = {
        "split"           : split,
        "formulation"     : formulation,
        "n_test"          : len(test_indices),
        "mse_at_8k"       : mse,
        "mae_at_8k"       : mae,
        "r2_at_8k"        : r2,
        "crossing_accuracy": {
            str(k): v for k, v in crossing_acc.items()
        },
        "per_timestep_mse": per_timestep_mse,
    }

    out_path = f"{OUT_DIR}/{run_name}_metrics.json"
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> Saved {run_name}_metrics.json")

    return metrics


# ── COMPARISON PLOTS AND REPORT ───────────────────────────────────────────────
def generate_comparison_report(all_metrics):
    """Generate cross-experiment comparison table and plots."""

    # Filter out None results
    results = [m for m in all_metrics if m is not None]

    # ── Comparison table ──────────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("  ADES-v2 EVALUATION RESULTS — ALL EXPERIMENTS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(
        f"  {'Experiment':35s}  {'MSE(8k)':>10}  {'MAE(8k)':>10}  {'R²':>8}"
    )
    report_lines.append(f"  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*8}")

    for m in results:
        name = f"{m['split']}_{m['formulation']}"
        report_lines.append(
            f"  {name:35s}  {m['mse_at_8k']:>10.6f}  "
            f"{m['mae_at_8k']:>10.6f}  {m['r2_at_8k']:>8.4f}"
        )

    report_lines.append("")
    report_lines.append("  CROSSING ACCURACY (fraction of crossing pairs correctly ranked)")
    report_lines.append("")

    # Header
    t_headers = "  " + f"{'Experiment':35s}"
    for t in CROSSING_QUERY_TIMES:
        t_headers += f"  {t//1000}k".rjust(7)
    report_lines.append(t_headers)
    report_lines.append(f"  {'-'*35}" + "  ------" * len(CROSSING_QUERY_TIMES))

    for m in results:
        name = f"{m['split']}_{m['formulation']}"
        row  = f"  {name:35s}"
        for t in CROSSING_QUERY_TIMES:
            acc = m['crossing_accuracy'].get(str(t))
            if acc is None:
                row += "     N/A"
            else:
                row += f"  {acc*100:5.1f}%"
        report_lines.append(row)

    report_lines.append("")
    report_lines.append("  KEY FINDINGS")
    report_lines.append("  " + "-" * 50)

    # Auto-generate key findings
    tc_curve = next((m for m in results
                     if m['split']=='curve_hash'
                     and m['formulation']=='time_conditioned'), None)
    fc_curve = next((m for m in results
                     if m['split']=='curve_hash'
                     and m['formulation']=='full_curve'), None)
    tc_alloc = next((m for m in results
                     if m['split']=='allocation'
                     and m['formulation']=='time_conditioned'), None)
    tc_md5   = next((m for m in results
                     if m['split']=='hw_md5'
                     and m['formulation']=='time_conditioned'), None)
    tc_wl    = next((m for m in results
                     if m['split']=='hw_wl'
                     and m['formulation']=='time_conditioned'), None)
    st_curve = next((m for m in results
                     if m['split']=='curve_hash'
                     and m['formulation']=='single_timestep'), None)

    if tc_curve and fc_curve:
        diff = abs(tc_curve['mse_at_8k'] - fc_curve['mse_at_8k'])
        report_lines.append(
            f"  Time-conditioned vs full curve (curve_hash split):"
        )
        report_lines.append(
            f"    MSE difference = {diff:.6f} — "
            + ("virtually identical" if diff < 0.0002
               else "meaningful difference")
        )

    if tc_curve and tc_alloc:
        pct_harder = (tc_alloc['mse_at_8k'] - tc_curve['mse_at_8k']) \
                     / tc_curve['mse_at_8k'] * 100
        report_lines.append(
            f"  Allocation split {pct_harder:+.1f}% harder than curve_hash split"
        )
        report_lines.append(
            f"  (model generalises less well to unseen task strategies)"
        )

    if tc_md5 and tc_wl:
        diff_pct = (tc_wl['mse_at_8k'] - tc_md5['mse_at_8k']) \
                   / tc_md5['mse_at_8k'] * 100
        report_lines.append(
            f"  WL topology split {diff_pct:+.1f}% vs MD5 wiring split"
        )

    report_txt = '\n'.join(report_lines)
    print("\n" + report_txt)

    with open(f"{OUT_DIR}/summary_report.txt", 'w') as f:
        f.write(report_txt)
    print(f"\n  -> Saved summary_report.txt")

    # ── MSE comparison bar chart ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    names   = [f"{m['split']}\n{m['formulation']}" for m in results]
    mses    = [m['mse_at_8k'] for m in results]
    colors  = []
    for m in results:
        if m['formulation'] == 'time_conditioned':
            colors.append('steelblue')
        elif m['formulation'] == 'full_curve':
            colors.append('darkorange')
        else:
            colors.append('mediumseagreen')

    bars = ax.bar(names, mses, color=colors, edgecolor='white')
    for bar, val in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(mses)*0.01,
                f'{val:.5f}', ha='center', va='bottom', fontsize=8)

    ax.set_title("Test MSE at t=8,000h — All Experiments\n"
                 "Blue=time_conditioned  Orange=full_curve  Green=single_timestep",
                 fontsize=12)
    ax.set_ylabel("MSE")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/mse_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved mse_comparison.png")

    # ── Crossing accuracy comparison ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap    = plt.cm.tab10

    for ci, m in enumerate(results):
        name = f"{m['split']}_{m['formulation']}"
        accs = []
        ts   = []
        for t in CROSSING_QUERY_TIMES:
            acc = m['crossing_accuracy'].get(str(t))
            if acc is not None:
                accs.append(acc * 100)
                ts.append(t)
        if accs:
            ax.plot(ts, accs, marker='o', linewidth=2,
                    color=cmap(ci / len(results)), label=name)

    ax.axhline(50, color='gray', linestyle='--', alpha=0.5,
               label='Random baseline (50%)')
    ax.set_xlabel("Query time (hours)")
    ax.set_ylabel("Crossing accuracy (%)")
    ax.set_title("Crossing Accuracy Across Query Times\n"
                 "(% of crossing pairs correctly ranked at each time)",
                 fontsize=12)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(alpha=0.3)
    ax.set_ylim(40, 105)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/crossing_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved crossing_accuracy.png")

    # ── Per-timestep MSE overlay (time_conditioned only) ──────────────────
    tc_experiments = [m for m in results
                      if m['formulation'] == 'time_conditioned'
                      and m['per_timestep_mse'] is not None]

    if tc_experiments:
        with h5py.File(H5_PATH, 'r') as h5:
            time_vals = h5.attrs['time_values']

        fig, ax = plt.subplots(figsize=(14, 6))
        for ci, m in enumerate(tc_experiments):
            label = f"{m['split']}_time_conditioned"
            ax.plot(time_vals, m['per_timestep_mse'],
                    color=cmap(ci / len(tc_experiments)),
                    linewidth=1.5, label=label)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("MSE")
        ax.set_title("Per-Timestep MSE — Time-Conditioned Experiments\n"
                     "(shows where in the curve the model struggles most)",
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(f"{OUT_DIR}/per_timestep_mse_overlay.png", dpi=150)
        plt.close(fig)
        print(f"  -> Saved per_timestep_mse_overlay.png")


# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="ADES-v2 Evaluation")
    parser.add_argument('--split',
        choices=['curve_hash', 'allocation', 'hw_md5', 'hw_wl'],
        help="Split to evaluate (required unless --all)")
    parser.add_argument('--formulation',
        choices=['time_conditioned', 'full_curve', 'single_timestep'],
        help="Formulation to evaluate (required unless --all)")
    parser.add_argument('--all', action='store_true',
        help="Evaluate all experiments and generate comparison report")
    return parser.parse_args()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with h5py.File(H5_PATH, 'r') as h5:
        time_vals = h5.attrs['time_values']

    if args.all:
        print(f"\nEvaluating all {len(ALL_EXPERIMENTS)} experiments...")
        all_metrics = []
        for split, formulation in ALL_EXPERIMENTS:
            metrics = evaluate_experiment(split, formulation, device, time_vals)
            all_metrics.append(metrics)

        print("\n\nGenerating comparison report...")
        generate_comparison_report(all_metrics)
        print(f"\n✅ All evaluations complete. Results in {OUT_DIR}/")

    else:
        if not args.split or not args.formulation:
            print("ERROR: provide --split and --formulation, or use --all")
            sys.exit(1)
        metrics = evaluate_experiment(
            args.split, args.formulation, device, time_vals
        )
        if metrics:
            print(f"\n✅ Evaluation complete. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()