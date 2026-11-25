// Parity test: site/src/lib/nn/trees.ts (evalTrees + exact treeShap) against mlratchet.gbm.
// Run: npm test (in site/)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { evalTrees, evalTreesRaw, treeShap, type TreePack } from '../src/lib/nn/trees.ts';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

const pyScript = `
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
print(json.dumps(out))
`;

test('trees runtime: TypeScript evalTrees and exact treeShap match Python mlratchet.gbm', () => {
  const raw = execFileSync('python3', ['-c', pyScript], { cwd: root, encoding: 'utf-8' });
  const d = JSON.parse(raw);
  const pack: TreePack = {
    tree_offsets: new Int32Array(d.tree_offsets),
    feat: new Int32Array(d.feat),
    thresh: new Float64Array(d.thresh),
    default_left: new Uint8Array(d.default_left),
    left: new Int32Array(d.left),
    right: new Int32Array(d.right),
    value: new Float64Array(d.value),
    cover: new Float64Array(d.cover),
    base_score: new Float64Array(d.base_score),
    tree_output: new Int32Array(d.tree_output),
    n_features: d.n_features,
    n_outputs: d.n_outputs,
    objective: 'regression',
  };

  let worstPred = 0;
  let worstShap = 0;
  for (let i = 0; i < d.X.length; i++) {
    const x = d.X[i].map((v: number | null) => (v === null ? Number.NaN : v));
    const p = evalTrees(pack, x)[0];
    const pRaw = evalTreesRaw(pack, x)[0];
    worstPred = Math.max(worstPred, Math.abs(p - d.preds[i]), Math.abs(pRaw - d.preds[i]));
    const { phi, expectedValue } = treeShap(pack, x, 0);
    assert.ok(Math.abs(expectedValue - d.ev) < 1e-12, `ev diff ${Math.abs(expectedValue - d.ev)}`);
    let sum = expectedValue;
    for (let j = 0; j < d.n_features; j++) {
      worstShap = Math.max(worstShap, Math.abs(phi[j] - d.phi[i][j]));
      sum += phi[j];
    }
    assert.ok(Math.abs(sum - p) < 1e-12, `efficiency diff ${Math.abs(sum - p)}`);
  }
  assert.ok(worstPred < 1e-12, `worst pred diff ${worstPred}`);
  assert.ok(worstShap < 1e-12, `worst shap diff ${worstShap}`);
});
