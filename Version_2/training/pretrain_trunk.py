"""
training/pretrain_trunk.py -- supervised pre-training of the shared trunk.

P2's third claim: the trunk is worth keeping, the policy-gradient route to
discovering the signal is not. Three sessions of PPO runs have spent GPU-hours
asking a policy-gradient estimator to find, through a reward that mixes DSR,
step-PnL, a hold penalty, a diversity bonus and an overtrading penalty, a
relationship that a regression can be handed directly.

This script trains

        forward_features(obs, hidden) -> trunk -> edge_head -> edge_bps

against the TRADEABLE forward return, supervised, and writes a checkpoint that
`train.py --pretrained-trunk` loads before attaching the policy and critic.

WHAT "TRADEABLE" MEANS HERE, PRECISELY
--------------------------------------
The target is the same quantity `eval/xsec_book.py` sizes against, and it is
built the same way, because a trunk pre-trained on a target the env cannot
trade is worth less than no pre-training at all:

  * EXECUTION FRAME. The observation ends at bar t; the order it produces fills
    and marks at bar t+1's `execution_price_column` (P1). So the return is
    measured open(t+1) -> open(exit), never close-to-close. This is not a
    detail: on this panel the same features predict the close-to-close return
    about twice as strongly as the one the env can actually capture, and a
    trunk fit to the close frame would be learning that gap.
  * SESSION-CAPPED. `flatten_at_session_close` liquidates on the last bar of
    every session, so the hold is cut there. A decision whose fill lands on the
    closing bar has no tradeable return at all and is masked out of the loss.
  * CROSS-SECTIONALLY DEMEANED. Both sides. The book P2 builds is
    dollar-neutral, so the market factor is not available to it, and a trunk
    that spends capacity predicting the market factor has learned something the
    book cannot use. Predictions are demeaned across names at each step for the
    same reason -- otherwise the head can score by predicting the common move.
  * IN BPS. The head's output is an edge in basis points over the hold, which
    is what `(|edge| - lambda*cost)+` needs. A z-scored or rank output would
    make lambda meaningless. `target_sd_bps` is saved with the checkpoint so
    the scale survives the round trip.

THE COLD HIDDEN STATE IS DELIBERATE
-----------------------------------
Each training chunk starts from a zeroed LSTM hidden state rather than carrying
one across chunks. That matches how the env starts a rollout, and it forces the
trunk to be useful from a cold start rather than only deep into a sequence.
Chunks are contiguous in time and the hidden state is carried WITHIN a chunk,
so the recurrence still has `--bptt` steps to work with.

WHAT THIS DOES NOT DO
---------------------
It does not train the policy or the critic, and it makes no claim that a good
trunk produces a good policy. It removes one of the four things a PPO run
conflates -- "can the features predict the tradeable return at all" -- so that
a subsequent run's failure is attributable to the other three.

Usage
-----
    python training/pretrain_trunk.py                       # hold 24, 3 epochs
    python training/pretrain_trunk.py --hold 12 --epochs 5
    python training/pretrain_trunk.py --out checkpoints/trunk_h24.pt
    python train.py --pretrained-trunk checkpoints/trunk_pretrained.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import MultiTickerRolloutDataset  # noqa: E402
from env.vec_trading_env import (  # noqa: E402
    _to_execution_frame,
    load_aligned_price_frames,
)
from training.config import TrainingConfig  # noqa: E402
from training.ppo_hybrid import HybridActorCritic  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    # dataset.py prints emoji; a Windows cp1252 console raises on them before
    # anything here gets a chance to run. Same guard alpha_lab.py uses.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

Tensor = torch.Tensor

# The env appends these to every observation before the network sees it
# (`_augment_obs_with_portfolio_state`): position fraction and unrealized PnL
# fraction. There is no portfolio here, so both are zero -- the FLAT state,
# which is the state every decision to open is actually taken from.
N_PORTFOLIO_CHANNELS = 2


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

def build_target(aligned_dates: pd.DatetimeIndex, tickers, hold: int, column: str):
    """Cross-sectionally demeaned tradeable forward return, [T, N] float32 bps.

    Returns (target, mask, sd_bps). `mask` is False where no tradeable return
    exists -- a fill landing on a session's closing bar, or a missing price --
    and those entries must not enter the loss.
    """
    frames = load_aligned_price_frames(tickers, aligned_dates, columns=(column,))
    Px = _to_execution_frame(frames[column])          # [T, N], fills at t+1

    ny = aligned_dates.tz_convert("America/New_York")
    day_id = pd.factorize(pd.Series(ny.normalize()))[0].astype(np.int64)
    T = len(day_id)
    session_last = np.empty(T, dtype=np.int64)
    cur = T - 1
    for i in range(T - 1, -1, -1):
        if i == T - 1 or day_id[i] != day_id[i + 1]:
            cur = i
        session_last[i] = cur
    # Same transposition into the execution frame as eval/xsec_book.py: the
    # bar whose price the fill lands in is t+1, and the exec index that reads
    # that bar is one lower.
    sl_x = np.maximum(_to_execution_frame(session_last) - 1, 0)

    t = np.arange(T)
    exit_idx = np.minimum(np.minimum(t + hold, sl_x), T - 1)
    ok = exit_idx > t

    out = np.full_like(Px, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ok] = np.log(Px[exit_idx[ok]] / Px[t[ok]]) * 1e4

    m = np.nanmean(out, axis=1, keepdims=True)
    tgt = out - np.nan_to_num(m, nan=0.0)
    mask = np.isfinite(tgt)
    sd = float(np.nanstd(tgt[mask])) if mask.any() else 1.0
    return np.nan_to_num(tgt, nan=0.0).astype(np.float32), mask, max(sd, 1e-6)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TrunkEdgeModel(nn.Module):
    """The actor-critic's feature stack with a scalar edge head bolted on.

    The whole `HybridActorCritic` is instantiated rather than just its
    encoders, so the state_dict this writes keys 1:1 against what `train.py`
    loads. The policy and critic heads are constructed, never trained, and
    never saved -- attaching them freshly is the point.
    """

    def __init__(self, n_features: int, cfg: TrainingConfig) -> None:
        super().__init__()
        self.backbone = HybridActorCritic(n_features=n_features, cfg=cfg)
        # The head lives ON the backbone, not beside it, so the tensors this
        # writes key exactly as `train.py` will load them. Aliased here only
        # for readability.
        self.edge_head = self.backbone.edge_head
        nn.init.zeros_(self.edge_head.bias)

    def init_hidden(self, batch_size: int, device: torch.device):
        return self.backbone.init_hidden(batch_size, device)

    def forward(self, obs: Tensor, hidden):
        trunk, new_hidden = self.backbone.forward_features(obs, hidden)
        pred = self.edge_head(trunk).squeeze(-1)          # [n_envs]
        # Demean across names, matching the target. Without this the head can
        # lower its loss by predicting the common move, which the dollar-neutral
        # book it feeds cannot trade.
        return pred - pred.mean(), new_hidden

    def trunk_state_dict(self):
        """Only the shared feature stack -- no policy, no critic, no edge head."""
        keep = ("cnn.", "lstm.", "cross_attn.", "fusion.")
        return {k: v for k, v in self.backbone.state_dict().items()
                if k.startswith(keep)}


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def _obs_at(dataset, idx: int, device: torch.device) -> Tensor:
    """[n_envs, window, n_features] with the two portfolio channels zeroed in."""
    obs = dataset[idx].to(device)                       # [n_envs, window, F]
    pad = obs.new_zeros(obs.shape[0], obs.shape[1], N_PORTFOLIO_CHANNELS)
    return torch.cat((obs, pad), dim=-1)


@torch.no_grad()
def evaluate(model, dataset, target, mask, sd, device, stride: int, bptt: int):
    """Val loss and information coefficient, on chunks matching training's.

    IC is the mean per-step cross-sectional correlation between prediction and
    target -- the quantity the book's hurdle actually consumes, since both
    sides are demeaned per bar. Steps with fewer than three usable names are
    skipped rather than contributing a degenerate correlation.
    """
    model.eval()
    n = len(dataset)
    losses, ics = [], []
    for start in range(0, max(n - bptt, 1), stride):
        hidden = model.init_hidden(dataset.n_envs, device)
        for t in range(start, min(start + bptt, n)):
            if not mask[t].any():
                continue
            pred, hidden = model(_obs_at(dataset, t, device), hidden)
            y = torch.as_tensor(target[t] / sd, device=device)
            m = torch.as_tensor(mask[t], device=device)
            losses.append(float(nn.functional.huber_loss(pred[m], y[m])))
            if int(m.sum()) >= 3:
                p, q = pred[m].float(), y[m].float()
                p = p - p.mean()
                q = q - q.mean()
                den = p.norm() * q.norm()
                if float(den) > 0:
                    ics.append(float((p @ q) / den))
    model.train()
    return (float(np.mean(losses)) if losses else float("nan"),
            float(np.mean(ics)) if ics else float("nan"),
            len(ics))


@torch.no_grad()
def dump_edge(model, datasets, sd, device, bptt, out_path):
    """Write the model's edge, in bps, keyed by timestamp, for eval/xsec_book.py.

    This is the only honest way to answer the question P2 actually poses: does a
    LEARNED edge clear the cost hurdle where the ridge could not? The book needs
    edge[t, name] in bps on its own timeline, so the predictions are emitted
    with their timestamps rather than their row indices and the book reindexes
    them. Row indices would silently misalign the moment either side's panel
    changes width.

    Chunks are contiguous and stride by `bptt`, each starting from a cold hidden
    state -- the same regime the model was trained under, and the same one a
    rollout runs in.
    """
    model.eval()
    stamps, rows = [], []
    for ds in datasets:
        n = len(ds)
        # dataset[i] is the window ENDING at aligned_dates[i + window - 1]; that
        # bar is what the prediction is about.
        off = ds.window_size - 1
        for start in range(0, n, bptt):
            hidden = model.init_hidden(ds.n_envs, device)
            for t in range(start, min(start + bptt, n)):
                pred, hidden = model(_obs_at(ds, t, device), hidden)
                rows.append((pred * sd).float().cpu().numpy())
                stamps.append(ds.aligned_dates[t + off].value)
    model.train()

    edge = np.asarray(rows, dtype=np.float32)
    np.savez_compressed(out_path,
                        timestamps=np.asarray(stamps, dtype=np.int64),
                        tickers=np.asarray(datasets[0].tickers),
                        edge_bps=edge)
    print(f"[dump] {edge.shape[0]:,} bars x {edge.shape[1]} names -> {out_path}")
    print(f"[dump] |edge| p50 {np.percentile(np.abs(edge), 50):.3f}  "
          f"p90 {np.percentile(np.abs(edge), 90):.3f}  "
          f"p99.9 {np.percentile(np.abs(edge), 99.9):.3f} bps  "
          f"-- compare against the round trip in eval/xsec_book.py's [hurdle] line")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Supervised pre-training of the trunk.")
    ap.add_argument("--hold", type=int, default=24,
                    help="forward-return horizon in bars (default 24, P2's)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bptt", type=int, default=64,
                    help="steps per chunk; the hidden state is carried within "
                         "a chunk and reset between chunks")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-stride", type=int, default=512,
                    help="stride between val chunks; > bptt subsamples val")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="cap chunks per epoch (smoke tests)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--dump-edge", type=str, default=None,
                    help="after training, write the best checkpoint's edge (bps, "
                         "timestamped) here for `eval/xsec_book.py --edge npz`")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 92)
    print("TRUNK PRE-TRAINING -- supervised, against the tradeable forward return")
    print("=" * 92)

    train_ds = MultiTickerRolloutDataset(window_size=cfg.env.window_size,
                                         split="train", device=str(device))
    val_ds = MultiTickerRolloutDataset(window_size=cfg.env.window_size,
                                       split="val", device=str(device))

    col = cfg.env.execution_price_column
    tr_tgt, tr_mask, sd = build_target(train_ds.aligned_dates, train_ds.tickers,
                                       args.hold, col)
    va_tgt, va_mask, _ = build_target(val_ds.aligned_dates, val_ds.tickers,
                                      args.hold, col)
    print(f"[target] hold {args.hold} bars, execution frame on '{col}', "
          f"demeaned per bar; train sd {sd:.2f} bps, "
          f"{100 * tr_mask.mean():.1f}% of cells usable")

    # The dataset indexes WINDOWS: dataset[i] ends at bar i + window - 1, so the
    # target for that observation is the one at that bar, not at i. Getting this
    # wrong shifts every label by the window length and trains on a lookahead.
    off = cfg.env.window_size - 1
    n_features = len(train_ds.feature_names) + N_PORTFOLIO_CHANNELS
    print(f"[model] n_features {n_features} "
          f"({len(train_ds.feature_names)} panel + {N_PORTFOLIO_CHANNELS} portfolio, "
          f"zeroed), trunk_dim {cfg.model.trunk_dim}")

    model = TrunkEdgeModel(n_features=n_features, cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def aligned(tgt, mask, ds):
        """Target/mask reindexed onto window positions of `ds`."""
        n = len(ds)
        return tgt[off:off + n], mask[off:off + n]

    tr_t, tr_m = aligned(tr_tgt, tr_mask, train_ds)
    va_t, va_m = aligned(va_tgt, va_mask, val_ds)

    n_train = len(train_ds)
    starts = np.arange(0, max(n_train - args.bptt, 1), args.bptt)
    print(f"[data] {n_train:,} train windows -> {len(starts)} chunks of {args.bptt} "
          f"({train_ds.n_envs} names each); {len(val_ds):,} val windows")
    print()

    best_ic = -float("inf")
    out_path = args.out or os.path.join(cfg.run.checkpoint_dir, "trunk_pretrained.pt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        np.random.shuffle(starts)
        chunks = starts[:args.max_chunks] if args.max_chunks else starts
        run_loss, n_steps, t0 = 0.0, 0, time.time()

        for ci, start in enumerate(chunks):
            hidden = model.init_hidden(train_ds.n_envs, device)
            opt.zero_grad(set_to_none=True)
            chunk_loss, used = 0.0, 0

            for t in range(int(start), min(int(start) + args.bptt, n_train)):
                if not tr_m[t].any():
                    # Still step the recurrence: skipping the forward pass would
                    # hand the LSTM a hidden state from a bar it never saw.
                    with torch.no_grad():
                        _, hidden = model(_obs_at(train_ds, t, device), hidden)
                    continue
                pred, hidden = model(_obs_at(train_ds, t, device), hidden)
                y = torch.as_tensor(tr_t[t] / sd, device=device)
                m = torch.as_tensor(tr_m[t], device=device)
                chunk_loss = chunk_loss + nn.functional.huber_loss(pred[m], y[m])
                used += 1

            if used == 0:
                continue
            loss = chunk_loss / used
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            # The hidden state is carried within the chunk only; detaching is
            # unnecessary because the next chunk builds a fresh one.
            run_loss += float(loss)
            n_steps += 1
            if (ci + 1) % 25 == 0:
                print(f"  epoch {epoch} chunk {ci + 1}/{len(chunks)}  "
                      f"loss {run_loss / max(n_steps, 1):.5f}  "
                      f"({time.time() - t0:.0f}s)")

        vl, vic, nic = evaluate(model, val_ds, va_t, va_m, sd, device,
                                args.eval_stride, args.bptt)
        print(f"[epoch {epoch}] train loss {run_loss / max(n_steps, 1):.5f}   "
              f"val loss {vl:.5f}   val IC {vic:+.5f} over {nic:,} bars")

        # Selection on val IC, saved only when it improves. IC and not loss:
        # the book consumes the ORDERING and the scale separately, and a head
        # can lower Huber loss by shrinking toward zero without ordering
        # anything better.
        if np.isfinite(vic) and vic > best_ic:
            best_ic = vic
            torch.save({
                "trunk": model.trunk_state_dict(),
                "edge_head": model.edge_head.state_dict(),
                "target_sd_bps": sd,
                "hold": args.hold,
                "n_features": n_features,
                "window_size": cfg.env.window_size,
                "execution_price_column": col,
                "val_ic": vic,
                "epoch": epoch,
            }, out_path)
            print(f"           saved -> {out_path} (val IC {vic:+.5f})")

    if args.dump_edge:
        if os.path.exists(out_path):
            blob = torch.load(out_path, map_location=device, weights_only=True)
            model.backbone.load_state_dict(blob["trunk"], strict=False)
            model.edge_head.load_state_dict(blob["edge_head"])
            print(f"[dump] reloaded best checkpoint (val IC {blob.get('val_ic', float('nan')):+.5f})")
        os.makedirs(os.path.dirname(args.dump_edge) or ".", exist_ok=True)
        dump_edge(model, [train_ds, val_ds], sd, device, args.bptt, args.dump_edge)

    print()
    print("=" * 92)
    if not np.isfinite(best_ic) or best_ic <= 0:
        print("No epoch produced a positive validation IC. The trunk did not learn a")
        print("tradeable cross-sectional signal, and attaching a policy to it will not")
        print("create one. Do not start a PPO run on this checkpoint.")
    else:
        print(f"Best validation IC {best_ic:+.5f}. Feed this checkpoint to")
        print(f"  python train.py --pretrained-trunk {out_path}")
        print("and to eval/xsec_book.py before that, to see whether an edge this size")
        print("clears the cost hurdle at any lambda. An IC that does not clear the")
        print("hurdle is not tradeable however significant it is.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
