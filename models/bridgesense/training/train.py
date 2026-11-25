"""Train a BridgeSense Transformer rung strictly on simulated banks.

Zero real labelled examples ever enter training. Single-threaded CPU friendly
(OMP_NUM_THREADS=1, torch.set_num_threads(1)), resumable checkpoints every 500 steps.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from _paths import work
from model import BridgeSenseNet, beta_nll_loss, count_parameters


def load_bank(name: str) -> dict[str, np.ndarray]:
    path = work("banks", f"{name}.npz")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def load_banks(names: list[str]) -> dict[str, np.ndarray]:
    if len(names) == 1:
        return load_bank(names[0])
    parts = [load_bank(n) for n in names]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def score_on_bank(model: nn.Module, bank: dict[str, np.ndarray], batch_size: int = 256) -> dict[str, float]:
    model.eval()
    n = len(bank["y_changed"])
    probs, dk_p, dk_s, env_p, env_s = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            out = model(
                torch.from_numpy(bank["x_ref"][sl].astype(np.float32)),
                torch.from_numpy(bank["x_q"][sl].astype(np.float32)),
                torch.from_numpy(bank["x_mode"][sl].astype(np.float32)),
                torch.from_numpy(bank["m_mask"][sl].astype(np.float32)),
                torch.from_numpy(bank["g_feat"][sl].astype(np.float32)),
                torch.from_numpy(bank["anchor"][sl].astype(np.float32)),
                torch.from_numpy(bank["mode_anchor"][sl].astype(np.float32)),
            )
            probs.append(torch.sigmoid(out["logit_change"]).numpy())
            dk_p.append(out["dk_mu"].numpy())
            dk_s.append(torch.exp(out["dk_logsig"]).numpy())
            env_p.append(out["env_mu"].numpy())
            env_s.append(torch.exp(out["env_logsig"]).numpy())

    p = np.concatenate(probs)
    dk = np.concatenate(dk_p)
    dks = np.concatenate(dk_s)
    env = np.concatenate(env_p)
    envs = np.concatenate(env_s)

    y_chg = bank["y_changed"]
    y_dk = bank["y_dk"]
    y_env = bank["y_env"]
    anc_dk = bank["anchor"][:, 0]
    anc_env = bank["anchor"][:, 1]
    cold = bank["is_cold_extrap"] > 0

    auc = float(roc_auc_score(y_chg, p))
    brier = float(np.mean((p - y_chg) ** 2))
    mae_dk = float(np.mean(np.abs(dk - y_dk)))
    mae_dk_anc = float(np.mean(np.abs(anc_dk - y_dk)))
    dk_skill = float(np.clip(1.0 - mae_dk / max(mae_dk_anc, 1e-6), 0.0, 1.0))

    mae_env = float(np.mean(np.abs(env - y_env)))
    mae_env_anc = float(np.mean(np.abs(anc_env - y_env)))
    env_skill = float(np.clip(1.0 - mae_env / max(mae_env_anc, 1e-6), 0.0, 1.0))

    # Healthy cold-extrapolation specificity (1 - false-alarm rate at p >= 0.5)
    hc = (y_chg < 0.5) & cold
    cold_spec = float(1.0 - np.mean(p[hc] >= 0.5)) if np.any(hc) else float(1.0 - np.mean(p[y_chg < 0.5] >= 0.5))
    cov_dk = float(np.mean(np.abs(dk - y_dk) <= dks))
    cov_env = float(np.mean(np.abs(env - y_env) <= envs))

    # Composite frozen-set objective in [0, 1]
    sim_score = 0.35 * auc + 0.35 * dk_skill + 0.15 * env_skill + 0.15 * cold_spec
    return {
        "sim_score": round(sim_score, 5),
        "auc": round(auc, 5),
        "brier": round(brier, 5),
        "dk_mae": round(mae_dk, 4),
        "dk_anchor_mae": round(mae_dk_anc, 4),
        "dk_skill": round(dk_skill, 5),
        "env_mae": round(mae_env, 4),
        "env_skill": round(env_skill, 5),
        "cold_spec": round(cold_spec, 5),
        "cov_dk_68": round(cov_dk, 4),
        "cov_env_68": round(cov_env, 4),
    }


def train_rung(
    name: str,
    train_banks: list[str],
    val_bank: str,
    steps: int = 3000,
    batch_size: int = 128,
    lr: float = 1.5e-3,
    seed: int = 42,
    d_model: int = 64,
    n_heads: int = 4,
    d_ff: int = 128,
    n_t_layers: int = 2,
    n_m_layers: int = 2,
    dropout: float = 0.05,
    use_cross_attn: bool = True,
    use_anchor: bool = True,
) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    tr = load_banks(train_banks)
    va = load_bank(val_bank)
    n_tr = len(tr["y_changed"])

    cfg = {
        "d_model": d_model, "n_heads": n_heads, "d_ff": d_ff,
        "n_t_layers": n_t_layers, "n_m_layers": n_m_layers,
        "dropout": dropout, "use_cross_attn": use_cross_attn, "use_anchor": use_anchor,
    }
    model = BridgeSenseNet(**cfg)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.03)

    ckpt_path = work("checkpoints", f"{name}.pt", mkdir=True)
    best_score = -1.0
    best_state = None
    best_val = None
    t0 = time.time()

    print(f"[{name}] params={n_params:,} | train={train_banks} ({n_tr}) | val={val_bank} | steps={steps} | seed={seed}", flush=True)

    for step in range(1, steps + 1):
        model.train()
        idx = rng.integers(0, n_tr, size=batch_size)
        x_ref = torch.from_numpy(tr["x_ref"][idx].astype(np.float32))
        x_q = torch.from_numpy(tr["x_q"][idx].astype(np.float32))
        x_mode = torch.from_numpy(tr["x_mode"][idx].astype(np.float32))
        m_mask = torch.from_numpy(tr["m_mask"][idx].astype(np.float32))
        g_feat = torch.from_numpy(tr["g_feat"][idx].astype(np.float32))
        anchor = torch.from_numpy(tr["anchor"][idx].astype(np.float32))
        mode_anchor = torch.from_numpy(tr["mode_anchor"][idx].astype(np.float32))

        y_chg = torch.from_numpy(tr["y_changed"][idx].astype(np.float32))
        y_dk = torch.from_numpy(tr["y_dk"][idx].astype(np.float32))
        y_env = torch.from_numpy(tr["y_env"][idx].astype(np.float32))
        y_mdk = torch.from_numpy(tr["y_mode_dk"][idx].astype(np.float32))

        out = model(x_ref, x_q, x_mode, m_mask, g_feat, anchor, mode_anchor)
        loss_cls = F.binary_cross_entropy_with_logits(out["logit_change"], y_chg)
        loss_dk = beta_nll_loss(out["dk_mu"], out["dk_logsig"], y_dk, beta=0.5)
        loss_env = beta_nll_loss(out["env_mu"], out["env_logsig"], y_env, beta=0.5)
        # Masked Huber on per-mode implied stiffness shift
        m_err = F.smooth_l1_loss(out["mode_dk"], y_mdk, reduction="none", beta=2.0) * m_mask
        loss_mode = m_err.sum() / torch.clamp(m_mask.sum(), min=1.0)

        loss = loss_cls + 0.35 * loss_dk + 0.25 * loss_env + 0.12 * loss_mode

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == steps:
            val_m = score_on_bank(model, va)
            elapsed = time.time() - t0
            print(
                f"  step {step:4d}/{steps} ({elapsed:5.1f}s) | loss={loss.item():.4f} | "
                f"sim={val_m['sim_score']:.4f} (auc={val_m['auc']:.4f}, dk_mae={val_m['dk_mae']:.2f}%, "
                f"cold_spec={val_m['cold_spec']:.3f})",
                flush=True,
            )
            if val_m["sim_score"] > best_score:
                best_score = val_m["sim_score"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_val = val_m
                torch.save({"state_dict": best_state, "config": cfg, "name": name, "seed": seed,
                            "train_banks": train_banks, "val_bank": val_bank, "val_metrics": best_val}, ckpt_path)

    meta = {
        "name": name, "params": n_params, "config": cfg, "seed": seed,
        "train_banks": train_banks, "val_bank": val_bank,
        "best_val": best_val, "elapsed_s": round(time.time() - t0, 1),
    }
    with open(work("checkpoints", f"{name}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[{name}] DONE -> best sim={best_score:.4f} saved to {ckpt_path}", flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bank", required=True, help="Comma-separated bank names, e.g. train2 or train2,train3")
    ap.add_argument("--val", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--no-cross-attn", action="store_true")
    ap.add_argument("--no-anchor", action="store_true")
    args = ap.parse_args()
    train_rung(
        name=args.name,
        train_banks=[b.strip() for b in args.bank.split(",") if b.strip()],
        val_bank=args.val,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        d_model=args.d_model,
        use_cross_attn=not args.no_cross_attn,
        use_anchor=not args.no_anchor,
    )


if __name__ == "__main__":
    main()
