"""
04_train.py
===========
STEP 4 of the ADES-v2 pipeline — GNN Training.

Trains one experiment at a time, controlled by command-line arguments.
Submit one Slurm job per experiment row in the experiment table.

Usage:
  python src/scripts/04_train.py \\
      --split        curve_hash \\
      --formulation  time_conditioned \\
      [--epochs      300] \\
      [--lr          0.0005] \\
      [--batch_size  64] \\
      [--patience    20] \\
      [--hidden_dim  64] \\
      [--heads       8] \\
      [--dropout     0.3] \\
      [--no_wandb]

Arguments:
  --split         : curve_hash | allocation | hw_md5 | hw_wl
  --formulation   : time_conditioned | full_curve | single_timestep
  --no_wandb      : disable wandb logging (useful for quick tests)

Formulations:
  time_conditioned  — f(graph, t) → R(t)
      Node features: [5-dim one-hot | t_norm]  (6-dim total)
      Output dim  : 1
      t is sampled randomly from 221 timesteps per batch
      This is the primary formulation

  full_curve        — f(graph) → [R(0), R(100), ..., R(22000)]
      Node features: 5-dim one-hot
      Output dim  : 221

  single_timestep   — f(graph) → R(8000h)
      Node features: 5-dim one-hot
      Output dim  : 1
      Simplest baseline

Model: GAT_LN_HEAD
  conv1: GATConv(input_dim, hidden_dim, heads) → LayerNorm → ReLU
  conv2: GATConv(hidden_dim*heads, hidden_dim//2, heads) → LayerNorm → Dropout
  pool:  global_mean_pool
  fc:    Linear(hidden_dim//2 * heads, output_dim)

Outputs:
  models/{split}_{formulation}_best.pth   — best val MSE checkpoint
  models/{split}_{formulation}_final.pth  — final epoch checkpoint
  logs/train_{split}_{formulation}.log    — full training log

Module: pytorch/2.4
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool

# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="ADES-v2 GNN Training")

    parser.add_argument('--split',
        choices=['curve_hash', 'allocation', 'hw_md5', 'hw_wl'],
        required=True,
        help="Split strategy to use")

    parser.add_argument('--formulation',
        choices=['time_conditioned', 'full_curve', 'single_timestep'],
        required=True,
        help="Prediction formulation")

    parser.add_argument('--epochs',      type=int,   default=300)
    parser.add_argument('--lr',          type=float, default=0.0005)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--patience',    type=int,   default=20,
        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument('--hidden_dim',  type=int,   default=64)
    parser.add_argument('--heads',       type=int,   default=8)
    parser.add_argument('--dropout',     type=float, default=0.3)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--no_wandb',    action='store_true',
        help="Disable wandb logging")

    # Fixed paths
    parser.add_argument('--h5_path',     default="data/dataset.h5")
    parser.add_argument('--splits_path', default="data/splits.json")
    parser.add_argument('--models_dir',  default="models")
    parser.add_argument('--logs_dir',    default="logs")

    return parser.parse_args()


# ── MODEL ─────────────────────────────────────────────────────────────────────
class GAT_LN_HEAD(torch.nn.Module):
    """
    Graph Attention Network with LayerNorm and multi-head attention.
    Carried forward from previous intake — best performing architecture.

    input_dim  : 5 for full_curve / single_timestep
                 6 for time_conditioned (5 + t_norm appended)
    output_dim : 1   for time_conditioned / single_timestep
                 221 for full_curve
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 dropout_rate, num_heads=8):
        super().__init__()
        self.conv1 = GATConv(input_dim,          hidden_dim,     heads=num_heads)
        self.ln1   = torch.nn.LayerNorm(hidden_dim * num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads,
                              hidden_dim // 2,   heads=num_heads)
        self.ln2   = torch.nn.LayerNorm((hidden_dim // 2) * num_heads)
        self.drop  = torch.nn.Dropout(p=dropout_rate)
        self.fc    = torch.nn.Linear((hidden_dim // 2) * num_heads, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.ln1(self.conv1(x, edge_index)))
        x = self.drop(F.relu(self.ln2(self.conv2(x, edge_index))))
        x = global_mean_pool(x, batch)
        return self.fc(x)


# ── DATASET ───────────────────────────────────────────────────────────────────
class ReliabilityDataset(Dataset):
    """
    Lazy HDF5 dataset. Reads one sample at a time from dataset.h5.

    For time_conditioned formulation:
      - samples a random timestep t per __getitem__ call
      - appends t_norm = t / 22000 to every node's feature vector
      - returns scalar target R(t)

    For full_curve formulation:
      - returns full 221-point curve as target

    For single_timestep formulation:
      - returns R at t=8000h as target
    """
    def __init__(self, h5_path, indices, formulation, time_vals,
                 t_fixed_idx=80):
        self.h5_path      = h5_path
        self.indices      = np.array(indices, dtype=np.int64)
        self.formulation  = formulation
        self.time_vals    = time_vals
        self.t_max        = float(time_vals[-1])
        self.n_timesteps  = len(time_vals)
        self.t_fixed_idx  = t_fixed_idx   # index of t=8000h

        # Open HDF5 handle — kept open for the lifetime of the dataset
        # Each worker in DataLoader gets its own copy via worker_init_fn
        self._h5 = None

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open()
        sample_idx = int(self.indices[idx])

        # Load node features and edge index
        node_feats = torch.tensor(
            self._h5['features/node_features'][sample_idx],
            dtype=torch.float32
        )  # [30, 5]

        e_start    = int(self._h5['edges/edge_ptr'][sample_idx])
        e_end      = int(self._h5['edges/edge_ptr'][sample_idx + 1])
        edge_index = torch.tensor(
            self._h5['edges/edge_index'][:, e_start:e_end],
            dtype=torch.long
        )  # [2, E]

        y_curve = self._h5['targets/y_curve'][sample_idx]  # [221]

        # Build target and optionally inject time feature
        if self.formulation == 'time_conditioned':
            # Sample random timestep
            t_idx   = np.random.randint(0, self.n_timesteps)
            t_val   = float(self.time_vals[t_idx])
            t_norm  = t_val / self.t_max

            # Append t_norm to every node's feature vector → [30, 6]
            t_feat     = torch.full((node_feats.shape[0], 1), t_norm)
            node_feats = torch.cat([node_feats, t_feat], dim=1)

            target = torch.tensor([y_curve[t_idx]], dtype=torch.float32)

        elif self.formulation == 'full_curve':
            target = torch.tensor(y_curve, dtype=torch.float32)  # [221]

        else:  # single_timestep
            target = torch.tensor(
                [y_curve[self.t_fixed_idx]], dtype=torch.float32
            )

        return Data(x=node_feats, edge_index=edge_index, y=target)

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def worker_init_fn(worker_id):
    """Each DataLoader worker needs its own HDF5 file handle."""
    pass   # HDF5 is opened lazily on first __getitem__ call per worker


def collate_fn(batch):
    """Batch a list of PyG Data objects."""
    return Batch.from_data_list(batch)


# ── TRAINING ──────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, formulation):
    """
    Evaluate model on a dataloader.
    Returns mean MSE and mean MAE across all samples.
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            pred   = model(batch)          # [B, output_dim]
            target = batch.y               # [B, output_dim] or [B*221] for full curve

            # Reshape target to match pred for full_curve
            if formulation == 'full_curve':
                target = target.view(pred.shape)

            target = target.view(pred.shape)
            mse = F.mse_loss(pred, target, reduction='sum').item()
            mae = F.l1_loss( pred, target, reduction='sum').item()
            n   = pred.numel()

            total_mse += mse
            total_mae += mae
            n_samples  += n

    return total_mse / n_samples, total_mae / n_samples


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    run_name = f"{args.split}_{args.formulation}"
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.logs_dir,   exist_ok=True)

    log_path   = os.path.join(args.logs_dir,
                               f"train_{run_name}.log")
    best_path  = os.path.join(args.models_dir,
                               f"{run_name}_best.pth")
    final_path = os.path.join(args.models_dir,
                               f"{run_name}_final.pth")

    log_lines = []
    def log(msg=""):
        print(msg, flush=True)
        log_lines.append(msg)

    def save_log():
        with open(log_path, 'w') as f:
            f.write('\n'.join(log_lines))

    log("=" * 60)
    log(f"  ADES-v2 Training: {run_name}")
    log("=" * 60)
    log(f"  Split       : {args.split}")
    log(f"  Formulation : {args.formulation}")
    log(f"  Epochs      : {args.epochs}")
    log(f"  LR          : {args.lr}")
    log(f"  Batch size  : {args.batch_size}")
    log(f"  Patience    : {args.patience}")
    log(f"  Hidden dim  : {args.hidden_dim}")
    log(f"  Heads       : {args.heads}")
    log(f"  Dropout     : {args.dropout}")
    log(f"  wandb       : {'disabled' if args.no_wandb else 'enabled'}")

    # ── wandb init ────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="ADES-v2",
                name=run_name,
                config={
                    "split"       : args.split,
                    "formulation" : args.formulation,
                    "epochs"      : args.epochs,
                    "lr"          : args.lr,
                    "batch_size"  : args.batch_size,
                    "patience"    : args.patience,
                    "hidden_dim"  : args.hidden_dim,
                    "heads"       : args.heads,
                    "dropout"     : args.dropout,
                }
            )
            log("  wandb: initialised successfully")
        except Exception as e:
            log(f"  wandb: failed to initialise ({e}) — continuing without")
            use_wandb = False

    # ── Load splits ───────────────────────────────────────────────────────
    log(f"\nLoading splits from {args.splits_path}...")
    with open(args.splits_path, 'r') as f:
        splits_data = json.load(f)

    split_info  = splits_data[args.split]
    train_idx   = split_info['train']
    val_idx     = split_info['val']
    test_idx    = split_info['test']

    log(f"  Train : {len(train_idx):,}")
    log(f"  Val   : {len(val_idx):,}")
    log(f"  Test  : {len(test_idx):,}")

    # ── Load time values ──────────────────────────────────────────────────
    with h5py.File(args.h5_path, 'r') as h5:
        time_vals = h5.attrs['time_values']
        n_time    = len(time_vals)

    t8k_idx = int(np.argmin(np.abs(time_vals - 8000)))
    log(f"  Time steps: {n_time}  t=8000h index: {t8k_idx}")

    # ── Model dimensions ──────────────────────────────────────────────────
    input_dim  = 6 if args.formulation == 'time_conditioned' else 5
    output_dim = n_time if args.formulation == 'full_curve' else 1

    log(f"  Input dim  : {input_dim}")
    log(f"  Output dim : {output_dim}")

    # ── Datasets and loaders ──────────────────────────────────────────────
    log("\nBuilding datasets...")

    train_ds = ReliabilityDataset(
        args.h5_path, train_idx, args.formulation,
        time_vals, t8k_idx
    )
    val_ds   = ReliabilityDataset(
        args.h5_path, val_idx, args.formulation,
        time_vals, t8k_idx
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    log(f"  Train batches : {len(train_loader)}")
    log(f"  Val batches   : {len(val_loader)}")

    # ── Model, optimiser, scheduler ───────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"\nDevice: {device}")

    model = GAT_LN_HEAD(
        input_dim   = input_dim,
        hidden_dim  = args.hidden_dim,
        output_dim  = output_dim,
        dropout_rate= args.dropout,
        num_heads   = args.heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Model params  : {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    if use_wandb:
        wandb.watch(model, log='gradients', log_freq=100)

    # ── Training loop ─────────────────────────────────────────────────────
    log(f"\nStarting training for {args.epochs} epochs "
        f"(patience={args.patience})...")

    best_val_mse    = float('inf')
    epochs_no_improv = 0
    training_start  = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        total_loss = 0.0
        n_train    = 0

        for batch in train_loader:
            batch  = batch.to(device)
            optimizer.zero_grad()
            pred   = model(batch)
            target = batch.y

            target = target.view(pred.shape)

            loss = F.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * pred.numel()
            n_train    += pred.numel()

        train_mse = total_loss / n_train

        # Validation
        val_mse, val_mae = evaluate(model, val_loader, device, args.formulation)

        # LR scheduler step
        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start

        log(f"Ep {epoch:03d}/{args.epochs}  "
            f"train_mse={train_mse:.6f}  "
            f"val_mse={val_mse:.6f}  "
            f"val_mae={val_mae:.6f}  "
            f"lr={current_lr:.2e}  "
            f"time={epoch_time:.0f}s")

        # wandb logging
        if use_wandb:
            wandb.log({
                "epoch"    : epoch,
                "train_mse": train_mse,
                "val_mse"  : val_mse,
                "val_mae"  : val_mae,
                "lr"       : current_lr,
                "epoch_time": epoch_time,
            })

        # Early stopping and best model saving
        if val_mse < best_val_mse:
            best_val_mse     = val_mse
            epochs_no_improv = 0
            torch.save(model.state_dict(), best_path)
            log(f"  -> New best val MSE: {best_val_mse:.6f}  "
                f"(saved to {best_path})")
            if use_wandb:
                wandb.run.summary['best_val_mse'] = best_val_mse
                wandb.run.summary['best_epoch']   = epoch
        else:
            epochs_no_improv += 1
            if epochs_no_improv >= args.patience:
                log(f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs)")
                break

    # Final save
    torch.save(model.state_dict(), final_path)
    total_time = time.time() - training_start
    log(f"\nTraining complete in {total_time/60:.1f} minutes")
    log(f"Best val MSE : {best_val_mse:.6f}")
    log(f"Best model   : {best_path}")
    log(f"Final model  : {final_path}")

    if use_wandb:
        wandb.run.summary['total_time_min'] = total_time / 60
        wandb.finish()

    save_log()
    log("✅ Done.")


if __name__ == "__main__":
    main()