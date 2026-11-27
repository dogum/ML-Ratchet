// Parity test: site/src/lib/nn/trees.ts (evalTrees + exact treeShap) against mlratchet.gbm.
// Run: npm test (in site/)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { evalTrees, evalTreesRaw, treeShap, type TreePack } from '../src/lib/nn/trees.ts';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

// Reference values written by experiments/treetwin/export_trees_reference.py (kept in sync by
// tests/test_treetwin_core.py), so this test needs no Python.
const refPath = join(root, 'site', 'tests', 'fixtures', 'trees_reference.json');

test('trees runtime: TypeScript evalTrees and exact treeShap match Python mlratchet.gbm', () => {
  const d = JSON.parse(readFileSync(refPath, 'utf-8'));
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
