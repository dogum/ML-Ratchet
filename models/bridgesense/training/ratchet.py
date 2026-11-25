"""Run the BridgeSense ratchet ladder using mlratchet.Ratchet.

Reads work/rungs_registry.json (ordered list of trained rungs with checkpoint name(s) and change notes),
scores every rung on the latest frozen simulated validation set (replaying all prior decisions in order
whenever the physics version changes) and on the frozen KW51 real Dev split, and writes
models/bridgesense/ratchet.yaml and work/ratchet_log.json.

Never touches Exam A, Exam B, or Exam C.
"""
from __future__ import annotations

import argparse
import json
import os
import yaml

from mlratchet import Ratchet, gates
from _paths import MODEL, work
from dev_eval import eval_on_kw51_dev, load_kw51_dev_blocks, load_model_ckpt, score_ensemble_on_bank


DEFAULT_REGISTRY = [
    {"id": "r1", "ckpts": ["r1"], "change": "Textbook linear thermoelastic 2-D modal FEM + synthetic Gaussian OMA noise (physics p1, train1)"},
    {"id": "r2", "ckpts": ["r2"], "change": "Add 1-D Fourier deck heat diffusion, asphalt/ballast freezing phase transition, axial bearing restraint, warm-to-cold extrapolation (physics p2, train2)"},
    {"id": "r2_no_xattn", "ckpts": ["r2_no_xattn"], "change": "Architecture ablation on p2: remove baseline-to-query thermal cross-attention"},
    {"id": "r3", "ckpts": ["r3"], "change": "Inject real healthy KW51 autumn OMA noise + 44% correlated mode dropouts + dev-calibrated slow seasonal drift + mass-loaded retrofits (physics p3, train3)"},
    {"id": "r4_mix", "ckpts": ["r4_mix"], "change": "Mixed bank (p2 + p3, 24k episodes) with linear-OLS physics anchor"},
    {"id": "r5_no_anchor", "ckpts": ["r5_no_anchor"], "change": "Anchor ablation on p3: predict dK/K directly without the OLS output anchor"},
    {"id": "r6_mix_no_anchor", "ckpts": ["r6_mix_no_anchor"], "change": "Mixed bank (p2 + p3, 24k episodes) without the OLS output anchor"},
    {"id": "r7_ens3", "ckpts": ["r4_mix", "r6_s2", "r6_s3"], "change": "3-seed Transformer ensemble on mixed p2 + p3 bank with OLS physics anchor"},
]


def run_ratchet(frozen_set: str, release_id: str | None = None) -> dict:
    reg_path = work("rungs_registry.json")
    registry = json.load(open(reg_path)) if os.path.exists(reg_path) else DEFAULT_REGISTRY
    dev_blocks = load_kw51_dev_blocks()

    # Cache scores per (rung_id, frozen_set) and per rung_id on KW51 dev
    cache_path = work("ratchet_scores_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {"sim": {}, "dev": {}}

    for entry in registry:
        rid = entry["id"]
        ckpts = entry["ckpts"]
        sim_key = f"{rid}@{frozen_set}"
        if sim_key not in cache["sim"]:
            models = load_model_ckpt(ckpts)
            cache["sim"][sim_key] = score_ensemble_on_bank(models, frozen_set)
            print(f"  scored {rid} on {frozen_set}: sim={cache['sim'][sim_key]['sim_score']:.4f} "
                  f"(auc={cache['sim'][sim_key]['auc']:.4f}, dk_mae={cache['sim'][sim_key]['dk_mae']:.2f}%, "
                  f"cold_spec={cache['sim'][sim_key]['cold_spec']:.3f})", flush=True)
        if rid not in cache["dev"]:
            dres = eval_on_kw51_dev(ckpts, dev_blocks)
            cache["dev"][rid] = {k: v for k, v in dres.items() if k != "rows"}
            print(f"  scored {rid} on KW51 dev: auc={dres['dev_auc']:.4f}, brier={dres['dev_brier']:.4f}, "
                  f"healthy_dk_mae={dres['dev_healthy_dk_mae']:.2f}%, freeze_P={dres['dev_freeze_p_mean']:.3f}", flush=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    def _bridge_dev_gate(cand, inc):
        reasons = []
        ok = True
        if cand["dev_auc"] < inc["dev_auc"] - 0.02:
            ok = False
            reasons.append(f"dev_auc {cand['dev_auc']:.3f} vs {inc['dev_auc']:.3f}")
        brier_lim = max(inc["dev_brier"] * 1.15, inc["dev_brier"] + 0.015)
        if cand["dev_brier"] > brier_lim:
            ok = False
            reasons.append(f"dev_brier {cand['dev_brier']:.4f} > {brier_lim:.4f}")
        dk_lim = max(inc["dev_dk_mae"] * 1.15, inc["dev_dk_mae"] + 0.05)
        if cand["dev_dk_mae"] > dk_lim:
            ok = False
            reasons.append(f"dev_dk_mae {cand['dev_dk_mae']:.3f}% > {dk_lim:.3f}%")
        return ok, ("dev gate passed" if ok else "dev gate: " + "; ".join(reasons))

    rat = Ratchet(
        model="bridgesense",
        name="BridgeSense",
        margin=0.01,
        dev_gate=_bridge_dev_gate,
        higher_is_better=True,
        objective=(
            "Composite skill in [0, 1] on frozen simulated bridge episodes: "
            "0.35*AUC + 0.35*dK_skill + 0.15*env_skill + 0.15*cold_extrap_specificity"
        ),
        rule=(
            "Keep if frozen-set composite skill beats the incumbent by more than +0.01 AND on KW51 dev "
            "(10 blocks, seed 20261124) AUC drops <= 0.02, Brier rises <= max(15%, +0.015), and 10-block "
            "dK/K MAE rises <= max(15%, +0.05 pp)."
        ),
        frozen_set=frozen_set,
    )

    for entry in registry:
        rid = entry["id"]
        sim_m = cache["sim"][f"{rid}@{frozen_set}"]
        dev_m = cache["dev"][rid]
        rat.judge(
            rid,
            entry["change"],
            sim=sim_m["sim_score"],
            dev={
                "dev_auc": dev_m["dev_auc"],
                "dev_brier": dev_m["dev_brier"],
                "dev_dk_mae": dev_m["dev_dk_mae"],
            },
            metrics=[
                round(sim_m["auc"], 3),
                round(sim_m["dk_mae"], 2),
                round(sim_m["cold_spec"], 3),
                round(dev_m["dev_auc"], 3),
                round(dev_m["dev_brier"], 4),
                round(dev_m["dev_dk_mae"], 3),
            ],
        )

    if release_id is not None:
        rat.release(release_id)
    elif rat.incumbent is not None:
        rat.release(rat.incumbent.id)

    lad = rat.to_dict()
    lad["dev_gate"] = "KW51 dev AUC drops <= 0.02; Brier rises <= max(15%, +0.015); 10-block dK/K MAE rises <= max(15%, +0.05 pp)"
    lad["frozen_sets"] = sorted({k.split("@")[1] for k in cache["sim"]})
    lad["columns_source"] = f"Frozen set ({frozen_set}) and KW51 dev (10 blocks); real exams never used to choose"
    lad["columns"] = ["Sim AUC", "Sim dK MAE (%)", "Sim cold spec", "Dev AUC", "Dev Brier", "Dev dK MAE (%)"]

    doc = {"model": "bridgesense", "ladders": [lad]}
    yaml_path = os.path.join(MODEL, "ratchet.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)

    log_path = work("ratchet_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"doc": doc, "cache": cache}, f, indent=2)

    print(f"\n=== RATCHET LADDER (scored on {frozen_set}) ===")
    for r in rat.rungs:
        print(f"  {r.id:12s} | sim={r.sim:.4f} | {r.decision:8s} | {r.why}")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="val3")
    ap.add_argument("--release", default=None)
    args = ap.parse_args()
    run_ratchet(args.val, release_id=args.release)


if __name__ == "__main__":
    main()
