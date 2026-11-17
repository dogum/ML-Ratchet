"""Train one WakeNet rung on a simulated bank only (zero real wake labels).

Writes resumable checkpoints every 500 steps to work/ckpt/<name>.resume and final EMA weights
to work/ckpt/<name>.pt.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, HERE)

import _paths as _P  # noqa: E402
from model import EdgeWakeGNN, beta_nll, count_parameters  # noqa: E402


def compute_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    on = batch["online"]
    free = batch["is_free"]
    waked = on & (~free)

    eta_p = out["eta_pred"]
    lsig = out["log_sig_eta"]
    udef_p = out["u_def_pred"]
    eta_t = batch["eta_target"]
    udef_t = batch["udef_target"]

    nll_all = beta_nll(eta_p, eta_t, lsig, beta=0.5)
    loss_waked_nll = nll_all[waked].mean()
    loss_free_nll = nll_all[free].mean() if free.any() else torch.tensor(0.0, device=eta_p.device)

    loss_waked_l1 = F.smooth_l1_loss(eta_p[waked], eta_t[waked], beta=0.05)
    loss_udef_l1 = F.smooth_l1_loss(udef_p[waked], udef_t[waked], beta=0.04)

    loss_farm_nll = beta_nll(out["farm_wake_pred"], batch["farm_wake_target"], out["log_sig_farm"], beta=0.5).mean()
    loss_farm_l1 = F.smooth_l1_loss(out["farm_wake_pred"], batch["farm_wake_target"], beta=0.03)

    return (
        1.00 * loss_waked_nll
        + 0.45 * loss_waked_l1
        + 0.20 * loss_free_nll
        + 0.20 * loss_udef_l1
        + 0.45 * loss_farm_nll
        + 0.30 * loss_farm_l1
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bank", nargs="+", required=True, help="One or more simulated bank names in work/banks/")
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--agg", default="gated_sum", choices=["gated_sum", "softmax_attn"])
    ap.add_argument("--init-from", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("WAKE_THREADS", "2")))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Load one or more simulated banks (supports mixed banks when needed)
    banks = [np.load(_P.work("banks", f"{b}.npz")) for b in args.bank]
    data = {
        "node_x": torch.from_numpy(np.concatenate([b["node_x"].astype(np.float32) for b in banks], axis=0)),
        "edge_x": torch.from_numpy(np.concatenate([b["edge_x"].astype(np.float32) for b in banks], axis=0)),
        "edge_mask": torch.from_numpy(np.concatenate([b["edge_mask"] for b in banks], axis=0)),
        "online": torch.from_numpy(np.concatenate([b["online"] for b in banks], axis=0)),
        "is_free": torch.from_numpy(np.concatenate([b["is_free"] for b in banks], axis=0)),
        "eta_anchor": torch.from_numpy(np.concatenate([b["eta_anchor"].astype(np.float32) for b in banks], axis=0)),
        "eta_target": torch.from_numpy(np.concatenate([b["eta_target"].astype(np.float32) for b in banks], axis=0)),
        "udef_target": torch.from_numpy(np.concatenate([b["udef_target"].astype(np.float32) for b in banks], axis=0)),
        "farm_wake_target": torch.from_numpy(np.concatenate([b["farm_wake_target"].astype(np.float32) for b in banks], axis=0)),
    }
    n_total = len(data["node_x"])

    arch = {
        "hidden": args.hidden,
        "n_layers": args.layers,
        "head_hidden": args.head_hidden,
        "anchored": not args.no_anchor,
        "use_edges": not args.no_edges,
        "agg": args.agg,
    }
    net = EdgeWakeGNN(**arch)
    n_params = count_parameters(net)

    if args.init_from:
        prev = torch.load(_P.work("ckpt", f"{args.init_from}.pt"), map_location="cpu", weights_only=False)
        net.load_state_dict(prev["ema_state"], strict=False)
        print(f"[{args.name}] Initialised weights from {args.init_from}", flush=True)

    ema = copy.deepcopy(net)
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    ckpt_path = _P.work("ckpt", f"{args.name}.pt", mkdir=True)
    resume_path = _P.work("ckpt", f"{args.name}.resume", mkdir=True)

    start_step = 0
    if os.path.exists(resume_path):
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        net.load_state_dict(st["model_state"])
        ema.load_state_dict(st["ema_state"])
        opt.load_state_dict(st["opt_state"])
        start_step = int(st["step"])
        print(f"[{args.name}] Resumed from step {start_step}", flush=True)

    print(f"[{args.name}] Training EdgeWakeGNN ({n_params:,} params, arch={arch}) on {args.bank} ({n_total:,} farms) for {args.steps} steps...", flush=True)
    t0 = time.time()
    ema_decay = 0.995
    running_loss = 0.0

    for step in range(start_step + 1, args.steps + 1):
        # Cosine schedule with 150-step warmup
        if step <= 150:
            lr_now = args.lr * (step / 150.0)
        else:
            prog = (step - 150.0) / max(args.steps - 150.0, 1.0)
            lr_now = args.lr * (0.04 + 0.96 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        idx = rng.integers(0, n_total, size=args.batch_size)
        batch = {k: v[idx] for k, v in data.items()}

        opt.zero_grad(set_to_none=True)
        out = net(
            batch["node_x"], batch["edge_x"], batch["edge_mask"],
            batch["online"], batch["is_free"], batch["eta_anchor"],
        )
        loss = compute_loss(out, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()

        with torch.no_grad():
            d = min(ema_decay, (1.0 + step) / (10.0 + step))
            for p_e, p_m in zip(ema.parameters(), net.parameters()):
                p_e.mul_(d).add_(p_m, alpha=1.0 - d)

        running_loss += float(loss.item())
        if step % 500 == 0 or step == args.steps:
            avg_l = running_loss / 500.0 if step % 500 == 0 else running_loss / (step % 500)
            running_loss = 0.0
            torch.save(
                {
                    "step": step,
                    "arch": arch,
                    "n_params": n_params,
                    "banks": args.bank,
                    "model_state": net.state_dict(),
                    "ema_state": ema.state_dict(),
                    "opt_state": opt.state_dict(),
                },
                resume_path,
            )
            print(f"  [{args.name}] step {step:4d}/{args.steps}  loss={avg_l:.4f}  lr={lr_now:.2e}  ({time.time()-t0:.1f}s)", flush=True)

    payload = {
        "name": args.name,
        "steps": args.steps,
        "arch": arch,
        "n_params": n_params,
        "banks": args.bank,
        "ema_state": ema.state_dict(),
    }
    torch.save(payload, ckpt_path)
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print(f"[{args.name}] Saved final EMA checkpoint to {ckpt_path} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
