"""Write site/tests/fixtures/trees_reference.json: a hand-built two-tree ensemble with its
mlratchet.gbm predictions and exact TreeSHAP values, which site/tests/trees.test.ts checks the
TypeScript runtime against (the site's CI job has no Python). tests/test_treetwin_core.py
fails if this file goes stale.

Run from the repository root:  python experiments/treetwin/export_trees_reference.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "site" / "tests" / "fixtures" / "trees_reference.json"


def build() -> dict:
    import json, numpy as np
    from mlratchet.gbm import TreeEnsemble

    # Two depth-3 trees with repeated feature splits on paths and NaN defaults
    t0_feat = [0, 1, 0, -1, -1, -1, -1]
    t0_thr  = [0.5, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0]
    t0_dl   = [1, 0, 1, 1, 1, 1, 1]
    t0_l    = [1, 2, 3, -1, -1, -1, -1]
    t0_r    = [6, 5, 4, -1, -1, -1, -1]
    t0_v    = [0.0, 0.0, 0.0, 1.0, 2.5, -0.8, 0.4]
    t0_c    = [100.0, 60.0, 40.0, 15.0, 25.0, 20.0, 40.0]

    t1_feat = [2, -1, 1, -1, -1]
    t1_thr  = [0.0, 0.0, -0.2, 0.0, 0.0]
    t1_dl   = [1, 1, 1, 1, 1]
    t1_l    = [1, -1, 3, -1, -1]
    t1_r    = [2, -1, 4, -1, -1]
    t1_v    = [0.0, -1.2, 0.0, 0.7, 1.9]
    t1_c    = [100.0, 35.0, 65.0, 30.0, 35.0]

    ens = TreeEnsemble(
        tree_offsets=np.array([0, len(t0_feat), len(t0_feat) + len(t1_feat)], dtype=np.int32),
        feat=np.array(t0_feat + t1_feat, dtype=np.int32),
        thresh=np.array(t0_thr + t1_thr, dtype=np.float64),
        default_left=np.array(t0_dl + t1_dl, dtype=np.uint8),
        left=np.array(t0_l + t1_l, dtype=np.int32),
        right=np.array(t0_r + t1_r, dtype=np.int32),
        value=np.array(t0_v + t1_v, dtype=np.float64),
        cover=np.array(t0_c + t1_c, dtype=np.float64),
        base_score=np.array([0.25], dtype=np.float64),
        tree_output=np.array([0, 0], dtype=np.int32),
        n_features=4, n_outputs=1, objective="regression",
    )
    X = np.array([
        [-1.0, 0.5, -0.5, 3.0],
        [0.0, 0.5, 0.5, -1.0],
        [0.2, 2.0, 0.1, 0.0],
        [1.5, -1.0, 2.0, 0.0],
        [np.nan, np.nan, 0.2, 1.0],
    ])
    preds = ens.predict(X)
    phi, ev = ens.treeshap(X)
    out = {
        "tree_offsets": ens.tree_offsets.tolist(),
        "feat": ens.feat.tolist(),
        "thresh": ens.thresh.tolist(),
        "default_left": ens.default_left.tolist(),
        "left": ens.left.tolist(),
        "right": ens.right.tolist(),
        "value": ens.value.tolist(),
        "cover": ens.cover.tolist(),
        "base_score": ens.base_score.tolist(),
        "tree_output": ens.tree_output.tolist(),
        "n_features": ens.n_features,
        "n_outputs": ens.n_outputs,
        "X": [[None if np.isnan(v) else float(v) for v in row] for row in X],
        "preds": preds.tolist(),
        "phi": phi.tolist(),
        "ev": float(ev),
    }
    return out


if __name__ == "__main__":
    import json

    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
