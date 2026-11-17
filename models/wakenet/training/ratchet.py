"""Judge and re-score WakeNet rungs with mlratchet.Ratchet and per-half-year dev gating.

Usage:
    python ratchet.py judge r1 "Textbook 2-D Jensen physics; node-only MLP" val1
    python ratchet.py rescore val2
    python ratchet.py release r7
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(MODEL)))

import _paths as _P  # noqa: E402
from mlratchet import Ratchet, Rung, gates  # noqa: E402
from model import EdgeWakeGNN  # noqa: E402

LADDER = Path(_P.MODEL) / "ratchet.yaml"
GATE_KEYS = ["waked_mae_h1", "waked_mae_h2", "farm_mae"]


def load_ckpt_model(rung_id: str) -> EdgeWakeGNN:
    st = torch.load(_P.work("ckpt", f"{rung_id}.pt"), map_location="cpu", weights_only=False)
    net = EdgeWakeGNN(**st["arch"])
    net.load_state_dict(st["ema_state"])
    net.eval()
    net.anchor = st.get("anchor", "textbook")
    return net


def _bank_for(net: EdgeWakeGNN, bank_name: str) -> str:
    """A rung anchored on the dev-tuned physics (v1.1) reads the same farms built with that anchor:
    `<bank>_tuned` has the same seed, targets and labels as `<bank>`; only the anchor features differ."""
    return f"{bank_name}_tuned" if getattr(net, "anchor", "textbook") == "tuned" else bank_name


@torch.no_grad()
def predict_bank(net: EdgeWakeGNN, z: dict, bs: int = 256) -> tuple[np.ndarray, np.ndarray]:
    n = len(z["node_x"])
    eta_out = np.zeros(z["eta_anchor"].shape, dtype=np.float32)
    farm_out = np.zeros(n, dtype=np.float32)
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        out = net(
            torch.from_numpy(z["node_x"][sl].astype(np.float32)),
            torch.from_numpy(z["edge_x"][sl].astype(np.float32)),
            torch.from_numpy(z["edge_mask"][sl]),
            torch.from_numpy(z["online"][sl]),
            torch.from_numpy(z["is_free"][sl]),
            torch.from_numpy(z["eta_anchor"][sl].astype(np.float32)),
        )
        eta_out[sl] = out["eta_pred"].numpy()
        farm_out[sl] = out["farm_wake_pred"].numpy()
    return eta_out, farm_out


def score_frozen(rung_id: str, bank_name: str) -> float:
    net = load_ckpt_model(rung_id)
    z = np.load(_P.work("banks", f"{_bank_for(net, bank_name)}.npz"))
    eta_p, farm_p = predict_bank(net, z)

    on = z["online"]
    free = z["is_free"]
    waked = on & (~free)
    eta_t = z["eta_target"].astype(np.float32)
    farm_t = z["farm_wake_target"].astype(np.float32)

    mae_w = float(np.mean(np.abs(eta_p[waked] - eta_t[waked])))
    mae_w_dumb = float(np.mean(np.abs(1.0 - eta_t[waked])))

    mae_f = float(np.mean(np.abs(farm_p - farm_t)))
    mae_f_dumb = float(np.mean(np.abs(0.0 - farm_t)))

    skill_w = 1.0 - mae_w / max(mae_w_dumb, 1e-6)
    skill_f = 1.0 - mae_f / max(mae_f_dumb, 1e-6)
    return round(float(0.65 * skill_w + 0.35 * skill_f), 4)


def score_dev(rung_id: str) -> dict:
    net = load_ckpt_model(rung_id)
    z = np.load(_P.work("banks", f"{_bank_for(net, 'dev_2019_graphs')}.npz"))
    eta_p, farm_p = predict_bank(net, z)

    on = z["online"]
    free = z["is_free"]
    waked = on & (~free)
    eta_t = z["eta_true"]
    farm_t = z["farm_wake_true"]
    ts = z["ts"]

    h1 = np.array([str(s)[5:7] <= "06" for s in ts])
    h2 = ~h1
    w1 = waked & h1[:, None]
    w2 = waked & h2[:, None]

    return {
        "waked_mae": round(float(np.mean(np.abs(eta_p[waked] - eta_t[waked]))), 4),
        "waked_mae_h1": round(float(np.mean(np.abs(eta_p[w1] - eta_t[w1]))), 4),
        "waked_mae_h2": round(float(np.mean(np.abs(eta_p[w2] - eta_t[w2]))), 4),
        "farm_mae": round(float(np.mean(np.abs(farm_p - farm_t))), 4),
    }


def _load_ratchet(frozen_set: str) -> tuple[Ratchet, dict]:
    doc = yaml.safe_load(LADDER.read_text(encoding="utf-8"))
    lad = doc["ladders"][0]
    gate = gates.no_worse_than(1.05, keys=GATE_KEYS, lower_is_better=True)
    r = Ratchet(
        doc["model"],
        margin=float(lad.get("margin", 0.01)),
        dev_gate=gate,
        objective=lad["objective"],
        rule=lad["rule"],
        frozen_set=frozen_set,
        name=lad["name"],
    )
    r.rungs = [Rung(**{k: v for k, v in x.items() if k in Rung.__dataclass_fields__}) for x in lad.get("rungs", [])]
    return r, doc


def _save_ratchet(r: Ratchet, old_doc: dict):
    old_lad = old_doc["ladders"][0]
    fsets = list(old_lad.get("frozen_sets", []))
    if r.frozen_set and r.frozen_set not in fsets:
        fsets.append(r.frozen_set)
    d = r.to_dict()
    out_lad = {
        "name": d["name"],
        "objective": d["objective"],
        "rule": d["rule"],
        "margin": d["margin"],
        "dev_gate": old_lad.get("dev_gate", "Each of H1 waked MAE, H2 waked MAE, and farm wake-loss MAE on Kelmarsh 2019 dev may rise by at most 5%"),
        "frozen_sets": fsets,
        "scored_on": r.frozen_set,
    }
    for extra in ("note", "columns_source", "columns"):
        if old_lad.get(extra):
            out_lad[extra] = old_lad[extra]
    out_lad["rungs"] = d["rungs"]
    out_doc = {"model": r.model, "ladders": [out_lad]}
    LADDER.write_text(yaml.safe_dump(out_doc, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main():
    mode = sys.argv[1]
    if mode == "judge":
        rid, change, bank = sys.argv[2], sys.argv[3], sys.argv[4]
        r, doc = _load_ratchet(bank)
        sim = score_frozen(rid, bank)
        dev = score_dev(rid)
        cols_pct = [round(100.0 * dev["waked_mae_h1"], 2), round(100.0 * dev["waked_mae_h2"], 2), round(100.0 * dev["farm_mae"], 2)]
        rung = r.judge(rid, change, sim=sim, dev=dev, metrics=cols_pct)
        _save_ratchet(r, doc)
        print(f"[judge on {bank}] {rung.id}: sim={sim:.4f}, dev={dev} -> {rung.decision.upper()} ({rung.why})", flush=True)
    elif mode == "rescore":
        bank = sys.argv[2]
        r, doc = _load_ratchet(bank)
        scores = {rg.id: score_frozen(rg.id, bank) for rg in r.rungs}
        changed = r.rescore(scores, frozen_set=bank)
        _save_ratchet(r, doc)
        print(f"[rescore on {bank}] scores={scores}, changed={changed}", flush=True)
        log_path = _P.work("logs", f"rescore_{bank}.json", mkdir=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"frozen_set": bank, "scores": scores, "changed": changed}, f, indent=2)
    elif mode == "release":
        rid = sys.argv[2] if len(sys.argv) > 2 else None
        doc = yaml.safe_load(LADDER.read_text(encoding="utf-8"))
        bank = doc["ladders"][0]["scored_on"]
        r, doc = _load_ratchet(bank)
        rel = r.release(rid)
        _save_ratchet(r, doc)
        print(f"[release] Marked {rel.id} as released (sim={rel.sim:.4f})", flush=True)
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
