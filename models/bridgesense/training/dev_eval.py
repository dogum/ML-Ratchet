"""Evaluate checkpoints on frozen simulated validation sets and the KW51 real Dev split ONLY.

Strictly never touches Exam A (KW51 held-out), Exam B (Concrete Box-Girder), or Exam C (HBTA).
"""
from __future__ import annotations

import json
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from _paths import MODEL, work
from features import extract_features
from model import BridgeSenseNet
from realdata import build_kw51_blocks, load_kw51_hourly
from train import load_bank, score_on_bank


def load_model_ckpt(ckpt_names: list[str] | str) -> list[BridgeSenseNet]:
    if isinstance(ckpt_names, str):
        ckpt_names = [ckpt_names]
    models = []
    for name in ckpt_names:
        path = name if os.path.exists(name) else work("checkpoints", f"{name}.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = BridgeSenseNet(**ck["config"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append(m)
    return models


def predict_episode(models: list[BridgeSenseNet], ep: dict) -> dict[str, float | np.ndarray]:
    f = extract_features(ep["T_ref"], ep["T_q"], ep["df_ref"], ep["df_q"], ep["mask_ref"], ep["mask_q"])
    x_ref = torch.from_numpy(f["x_ref"][None])
    x_q = torch.from_numpy(f["x_q"][None])
    x_mode = torch.from_numpy(f["x_mode"][None])
    m_mask = torch.from_numpy(f["m_mask"][None])
    g_feat = torch.from_numpy(f["g_feat"][None])
    anchor = torch.from_numpy(f["anchor"][None])
    mode_anchor = torch.from_numpy(f["mode_anchor"][None])

    logits, dks, dk_vars, envs, env_vars, mdks = [], [], [], [], [], []
    with torch.no_grad():
        for m in models:
            o = m(x_ref, x_q, x_mode, m_mask, g_feat, anchor, mode_anchor)
            logits.append(float(o["logit_change"][0]))
            dks.append(float(o["dk_mu"][0]))
            dk_vars.append(float(torch.exp(2.0 * o["dk_logsig"][0])))
            envs.append(float(o["env_mu"][0]))
            env_vars.append(float(torch.exp(2.0 * o["env_logsig"][0])))
            mdks.append(o["mode_dk"][0].numpy())

    logit_mean = float(np.mean(logits))
    prob = float(1.0 / (1.0 + np.exp(-np.clip(logit_mean, -30.0, 30.0))))
    dk_mean = float(np.mean(dks))
    dk_sig = float(np.sqrt(np.mean(dk_vars) + np.var(dks)))
    env_mean = float(np.mean(envs))
    env_sig = float(np.sqrt(np.mean(env_vars) + np.var(envs)))
    mode_dk = np.mean(mdks, axis=0)
    return {
        "logit": logit_mean,
        "p_change": prob,
        "dk_pct": dk_mean,
        "dk_sig": dk_sig,
        "env_pct": env_mean,
        "env_sig": env_sig,
        "mode_dk": mode_dk,
        "anchor_dk_pct": float(f["anchor"][0]),
        "anchor_env_pct": float(f["anchor"][1]),
    }


def score_ensemble_on_bank(models: list[BridgeSenseNet], bank_name: str) -> dict[str, float]:
    if len(models) == 1:
        return score_on_bank(models[0], load_bank(bank_name))
    bank = load_bank(bank_name)
    n = len(bank["y_changed"])
    bs = 256
    logits_all, dk_all, dk_v_all, env_all, env_v_all = [], [], [], [], []
    with torch.no_grad():
        for m in models:
            lp, dp, dv, ep, ev = [], [], [], [], []
            for i in range(0, n, bs):
                sl = slice(i, min(i + bs, n))
                o = m(
                    torch.from_numpy(bank["x_ref"][sl].astype(np.float32)),
                    torch.from_numpy(bank["x_q"][sl].astype(np.float32)),
                    torch.from_numpy(bank["x_mode"][sl].astype(np.float32)),
                    torch.from_numpy(bank["m_mask"][sl].astype(np.float32)),
                    torch.from_numpy(bank["g_feat"][sl].astype(np.float32)),
                    torch.from_numpy(bank["anchor"][sl].astype(np.float32)),
                    torch.from_numpy(bank["mode_anchor"][sl].astype(np.float32)),
                )
                lp.append(o["logit_change"].numpy())
                dp.append(o["dk_mu"].numpy())
                dv.append(torch.exp(2.0 * o["dk_logsig"]).numpy())
                ep.append(o["env_mu"].numpy())
                ev.append(torch.exp(2.0 * o["env_logsig"]).numpy())
            logits_all.append(np.concatenate(lp))
            dk_all.append(np.concatenate(dp))
            dk_v_all.append(np.concatenate(dv))
            env_all.append(np.concatenate(ep))
            env_v_all.append(np.concatenate(ev))

    logit = np.mean(logits_all, axis=0)
    p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
    dk = np.mean(dk_all, axis=0)
    dks = np.sqrt(np.mean(dk_v_all, axis=0) + np.var(dk_all, axis=0))
    env = np.mean(env_all, axis=0)
    envs = np.sqrt(np.mean(env_v_all, axis=0) + np.var(env_all, axis=0))

    y_chg, y_dk, y_env = bank["y_changed"], bank["y_dk"], bank["y_env"]
    anc_dk, anc_env = bank["anchor"][:, 0], bank["anchor"][:, 1]
    cold = bank["is_cold_extrap"] > 0

    auc = float(roc_auc_score(y_chg, p))
    brier = float(np.mean((p - y_chg) ** 2))
    mae_dk = float(np.mean(np.abs(dk - y_dk)))
    mae_dk_anc = float(np.mean(np.abs(anc_dk - y_dk)))
    dk_skill = float(np.clip(1.0 - mae_dk / max(mae_dk_anc, 1e-6), 0.0, 1.0))
    mae_env = float(np.mean(np.abs(env - y_env)))
    mae_env_anc = float(np.mean(np.abs(anc_env - y_env)))
    env_skill = float(np.clip(1.0 - mae_env / max(mae_env_anc, 1e-6), 0.0, 1.0))
    hc = (y_chg < 0.5) & cold
    cold_spec = float(1.0 - np.mean(p[hc] >= 0.5)) if np.any(hc) else float(1.0 - np.mean(p[y_chg < 0.5] >= 0.5))
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
        "cov_dk_68": round(float(np.mean(np.abs(dk - y_dk) <= dks)), 4),
        "cov_env_68": round(float(np.mean(np.abs(env - y_env) <= envs)), 4),
    }


def load_kw51_dev_blocks() -> list[dict]:
    splits = json.load(open(os.path.join(MODEL, "splits.json")))
    dev_ids = {b["block_id"] for b in splits["kw51_dev"]}
    _, all_blocks = build_kw51_blocks(load_kw51_hourly(), block_days=7)
    return [b for b in all_blocks if b["block_id"] in dev_ids]


def eval_on_kw51_dev(ckpt_names: list[str] | str, dev_blocks: list[dict] | None = None) -> dict:
    from calibrate import estimate_kw51_post_retrofit_true_dk
    models = load_model_ckpt(ckpt_names)
    if dev_blocks is None:
        dev_blocks = load_kw51_dev_blocks()
    post_true_dk = estimate_kw51_post_retrofit_true_dk(dev_blocks)
    rows = []
    y_true, p_pred = [], []
    healthy_abs_dk = []
    all_abs_dk = []
    freeze_probs = []
    post_dk = []

    for b in dev_blocks:
        r = predict_episode(models, b)
        y_true.append(b["is_changed"])
        p_pred.append(r["p_change"])
        tdk = 0.0 if b["is_changed"] == 0 else post_true_dk
        all_abs_dk.append(abs(r["dk_pct"] - tdk))
        if b["is_changed"] == 0:
            healthy_abs_dk.append(abs(r["dk_pct"]))
            if b["is_freezing"]:
                freeze_probs.append(r["p_change"])
        else:
            post_dk.append(r["dk_pct"])
        rows.append({
            "block_id": b["block_id"],
            "stratum": b["stratum"],
            "T_min": b["T_min"],
            "is_changed": b["is_changed"],
            "p_change": round(r["p_change"], 4),
            "dk_pct": round(r["dk_pct"], 2),
            "dk_sig": round(r["dk_sig"], 2),
            "anchor_dk_pct": round(r["anchor_dk_pct"], 2),
        })

    y_arr = np.array(y_true, dtype=float)
    p_arr = np.array(p_pred, dtype=float)
    auc = float(roc_auc_score(y_arr, p_arr))
    brier = float(np.mean((p_arr - y_arr) ** 2))
    healthy_dk_mae = float(np.mean(healthy_abs_dk))
    all_dk_mae = float(np.mean(all_abs_dk))
    freeze_p_mean = float(np.mean(freeze_probs)) if freeze_probs else 0.0
    post_dk_mean = float(np.mean(post_dk)) if post_dk else 0.0
    return {
        "dev_auc": round(auc, 4),
        "dev_brier": round(brier, 4),
        "dev_dk_mae": round(all_dk_mae, 3),
        "dev_healthy_dk_mae": round(healthy_dk_mae, 3),
        "dev_freeze_p_mean": round(freeze_p_mean, 4),
        "dev_post_dk_mean": round(post_dk_mean, 3),
        "rows": rows,
    }
