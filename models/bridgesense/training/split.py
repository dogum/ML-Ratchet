"""Create and freeze models/bridgesense/splits.json BEFORE any training.

Split rules (seed 20261124):
  - KW51 warm reference baseline: 2018-10-01 .. 2018-11-11 (6 weeks, T > 2 C, healthy only).
    Also supplies the unlabelled healthy OMA residual noise & missing-mode pool.
  - KW51 Dev (10 blocks): stratified seeded draw from KW51 pre-retrofit healthy (4 mild + 2 freezing)
    and post-retrofit (4 blocks). Used ONLY for the ratchet dev gate, twin-test calibration,
    dev-tuned baselines, and widen-only error-bar factors.
  - KW51 Exam A: all remaining KW51 blocks (held-out healthy mild, held-out healthy freezing,
    all during-retrofit scaffolding blocks, and remaining post-retrofit blocks).
  - Concrete Box-Girder Exam B: 100% unseen bridge (0 blocks in dev).
  - Hell Bridge Test Arena (HBTA) Exam C: 100% unseen bridge (0 states in dev).
"""
from __future__ import annotations

import json
import os
import numpy as np

from _paths import MODEL
from realdata import build_kw51_blocks, load_kw51_hourly


def main() -> None:
    rng = np.random.default_rng(20261124)
    kw = load_kw51_hourly()
    ref_pack, blocks = build_kw51_blocks(kw, block_days=7)

    by_stratum: dict[str, list[dict]] = {}
    for b in blocks:
        by_stratum.setdefault(b["stratum"], []).append(b)

    dev_ids = set()
    # Stratified small dev draw: 4 healthy_mild, 2 healthy_freezing, 4 post_retrofit
    dev_quotas = {"healthy_mild": 4, "healthy_freezing": 2, "post_retrofit": 4}
    for strat, q in dev_quotas.items():
        pool = [b["block_id"] for b in by_stratum.get(strat, [])]
        pick = rng.choice(pool, size=min(q, len(pool)), replace=False).tolist()
        dev_ids.update(pick)

    dev_blocks = [b for b in blocks if b["block_id"] in dev_ids]
    exam_blocks = [b for b in blocks if b["block_id"] not in dev_ids]

    splits = {
        "seed": 20261124,
        "kw51_reference_baseline": {
            "start": ref_pack["start"],
            "end": ref_pack["end"],
            "n_hours": int(len(ref_pack["idx"])),
            "T_min_C": round(float(np.min(ref_pack["T3"][:, 0])), 2),
            "T_max_C": round(float(np.max(ref_pack["T3"][:, 0])), 2),
        },
        "kw51_dev": [
            {k: b[k] for k in ("block_id", "start", "end", "stratum", "phase", "is_changed", "T_min", "T_mean")}
            for b in dev_blocks
        ],
        "kw51_exam": [
            {k: b[k] for k in ("block_id", "start", "end", "stratum", "phase", "is_changed", "T_min", "T_mean")}
            for b in exam_blocks
        ],
        "zero_shot_exams": {
            "boxgirder": "Zenodo 20522191 (CC BY 4.0) — 100% exam, 0 blocks in dev",
            "hbta": "Zenodo 14632942 (CC BY 4.0) — 100% exam, 0 states in dev",
        },
    }
    out_path = os.path.join(MODEL, "splits.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"  KW51 ref baseline: {ref_pack['start'][:10]} .. {ref_pack['end'][:10]} ({len(ref_pack['idx'])} h, T_min={splits['kw51_reference_baseline']['T_min_C']} C)")
    print(f"  KW51 dev:  {len(dev_blocks)} blocks ->", {s: sum(1 for b in dev_blocks if b['stratum']==s) for s in set(b['stratum'] for b in dev_blocks)})
    print(f"  KW51 exam: {len(exam_blocks)} blocks ->", {s: sum(1 for b in exam_blocks if b['stratum']==s) for s in set(b['stratum'] for b in exam_blocks)})


if __name__ == "__main__":
    main()
