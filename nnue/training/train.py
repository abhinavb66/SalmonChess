#PyTorch training for the SalmonNNUE network, exporting weights to the plain
#.npz format that nnue/network.py loads (torch is a training-only dependency).
#
#Value target: sigmoid(stockfish_cp / 400) — the standard WDL-ish squash.
#Policy target: Stockfish best move as a from*64+to class (stm-mirrored).
#Loss = MSE(sigmoid(v), value_target) + 0.5 * CrossEntropy(policy, best_move).
#
#Run from the repo root:
#  python3 -m nnue.training.train --smoke          #1 tiny run, asserts learning
#  python3 -m nnue.training.train --epochs 20 --out nets/salmon_nnue_v4.npz
#
#Nets are versioned artifacts (nets/README.md): a candidate exports to the
#gitignored nets/salmon_nnue_dev.npz (the default when --out is omitted) and
#is only committed as the next nets/salmon_nnue_vN.npz once it measures
#better (gauntlet + h2h), updating _DEFAULT_NET in SalmonNNUE.py to match.

import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from nnue import network

PAD = network.FEATURES          #padding index 768 -> zero embedding row


class TorchNet(nn.Module):
    """Torch mirror of nnue.network.Net (same shapes, clamp(0,1) activations)."""

    def __init__(self):
        super().__init__()
        self.ft = nn.Embedding(network.FEATURES + 1, network.ACC_SIZE, padding_idx=PAD)
        self.ft_b = nn.Parameter(torch.zeros(network.ACC_SIZE))
        self.l1_w = nn.Parameter(torch.empty(2 * network.ACC_SIZE, network.L1_SIZE))
        self.l1_b = nn.Parameter(torch.zeros(network.L1_SIZE))
        self.v_w = nn.Parameter(torch.empty(network.L1_SIZE))
        self.v_b = nn.Parameter(torch.zeros(()))
        self.p_w = nn.Parameter(torch.empty(2 * network.ACC_SIZE, network.POLICY_SIZE))
        self.p_b = nn.Parameter(torch.zeros(network.POLICY_SIZE))
        nn.init.normal_(self.ft.weight, 0, 0.05)
        nn.init.normal_(self.l1_w, 0, 0.1)
        nn.init.normal_(self.v_w, 0, 0.1)
        nn.init.normal_(self.p_w, 0, 0.02)
        with torch.no_grad():
            self.ft.weight[PAD].zero_()

    #own_idx/opp_idx: int64 [B,32] feature indices (PAD-padded), stm-first.
    def forward(self, own_idx, opp_idx):
        acc_own = self.ft(own_idx).sum(dim=1) + self.ft_b
        acc_opp = self.ft(opp_idx).sum(dim=1) + self.ft_b
        x = torch.clamp(torch.cat([acc_own, acc_opp], dim=1), 0.0, 1.0)
        h = torch.clamp(x @ self.l1_w + self.l1_b, 0.0, 1.0)
        v_raw = h @ self.v_w + self.v_b
        logits = x @ self.p_w + self.p_b
        return v_raw, logits

    #Single-position forward from feature lists; value passed through sigmoid
    #to match Net.value_policy's contract. Used by the NumPy-parity test.
    def forward_features(self, own_feats, opp_feats):
        own = torch.full((1, 32), PAD, dtype=torch.long)
        opp = torch.full((1, 32), PAD, dtype=torch.long)
        own[0, :len(own_feats)] = torch.tensor(own_feats)
        opp[0, :len(opp_feats)] = torch.tensor(opp_feats)
        v_raw, logits = self.forward(own, opp)
        return torch.sigmoid(v_raw[0]), logits[0]

    @classmethod
    def from_net(cls, net):
        model = cls()
        with torch.no_grad():
            model.ft.weight[:PAD] = torch.from_numpy(net.ft_w)
            model.ft.weight[PAD].zero_()
            model.ft_b.copy_(torch.from_numpy(net.ft_b))
            model.l1_w.copy_(torch.from_numpy(net.l1_w))
            model.l1_b.copy_(torch.from_numpy(net.l1_b))
            model.v_w.copy_(torch.from_numpy(net.v_w))
            model.v_b.fill_(net.v_b)
            model.p_w.copy_(torch.from_numpy(net.p_w))
            model.p_b.copy_(torch.from_numpy(net.p_b))
        return model

    def to_net(self):
        with torch.no_grad():
            return network.Net(
                self.ft.weight[:PAD].cpu().numpy(), self.ft_b.cpu().numpy(),
                self.l1_w.cpu().numpy(), self.l1_b.cpu().numpy(),
                self.v_w.cpu().numpy(), float(self.v_b),
                self.p_w.cpu().numpy(), self.p_b.cpu().numpy())


#Vectorized feature/target construction from the shard arrays (see gen_data).
def build_tensors(sq, pc, stm, score, bf, bt):
    n = sq.shape[0]
    pad_mask = pc == 255
    piece = (pc >> 1).astype(np.int64)
    color = (pc & 1).astype(np.int64)
    sq64 = sq.astype(np.int64)

    def persp_idx(persp):
        rel = sq64 if persp == 0 else (sq64 ^ 56)
        side = (color != persp).astype(np.int64)
        idx = side * 384 + piece * 64 + rel
        idx[pad_mask] = PAD
        return idx

    idx_w, idx_b = persp_idx(0), persp_idx(1)
    stm_is_b = (stm == 1)[:, None]
    own = np.where(stm_is_b, idx_b, idx_w)
    opp = np.where(stm_is_b, idx_w, idx_b)

    bf64, bt64 = bf.astype(np.int64), bt.astype(np.int64)
    mirror = stm == 1
    bf64[mirror] ^= 56
    bt64[mirror] ^= 56
    policy_target = bf64 * 64 + bt64
    value_target = 1.0 / (1.0 + np.exp(-score.astype(np.float32) / 400.0))

    return (torch.from_numpy(own), torch.from_numpy(opp),
            torch.from_numpy(value_target), torch.from_numpy(policy_target))


def load_data(data_dir, limit=None):
    shards = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    if not shards:
        raise SystemExit(f"no shards in {data_dir} — run gen_data first")
    parts = {k: [] for k in ("sq", "pc", "stm", "score", "bf", "bt")}
    total = 0
    for path in shards:
        with np.load(path) as z:
            for k in parts:
                parts[k].append(z[k])
            total += z["sq"].shape[0]
        if limit and total >= limit:
            break
    arrs = {k: np.concatenate(v)[:limit] for k, v in parts.items()}
    print(f"loaded {arrs['sq'].shape[0]} positions from {len(parts['sq'])} shards")
    return build_tensors(**arrs)


def train(args):
    torch.manual_seed(0)
    #Train on GPU when available (batches are moved per step); the exported
    #.npz is identical in format either way — play-time inference stays NumPy.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    own, opp, value_t, policy_t = load_data(args.data_dir,
                                            20_000 if args.smoke else None)
    n = own.shape[0]
    model = TorchNet().to(device)
    epochs = 2 if args.smoke else args.epochs
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=args.lr / 10)
    ce = nn.CrossEntropyLoss()

    first_epoch_loss = last_epoch_loss = None
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total, batches = 0.0, 0
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            v_raw, logits = model(own[b].to(device), opp[b].to(device))
            loss = (nn.functional.mse_loss(torch.sigmoid(v_raw),
                                           value_t[b].to(device))
                    + 0.5 * ce(logits, policy_t[b].to(device)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        sched.step()
        avg = total / batches
        if first_epoch_loss is None:
            first_epoch_loss = avg
        last_epoch_loss = avg
        print(f"epoch {epoch + 1}/{epochs}  loss {avg:.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}", flush=True)

    if args.smoke:
        assert last_epoch_loss < first_epoch_loss, "smoke: loss did not decrease"
        print("smoke OK: loss decreased")

    out = args.out or (os.path.join(REPO, "nets", "smoke.npz") if args.smoke
                       else os.path.join(REPO, "nets", "salmon_nnue_dev.npz"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model.to_net().save(out)
    print(f"exported {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(REPO, "data"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    train(ap.parse_args())
