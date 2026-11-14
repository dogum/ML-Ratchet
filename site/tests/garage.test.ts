// The garage (site/src/pages/garage.astro) runs EnergyNet, TripDetective and HybridPilot in the browser on RoadSim runs
// precomputed by shared/vehicle/garage/export_garage.py. Every reference case must match the Python outputs.
// Run: npm test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parsePack, type PackManifest } from '../src/lib/nn/pack.ts';
import { run, type Case } from '../src/garage/parity.ts';

function loadPack(dir: string, name: string) {
  const m = JSON.parse(readFileSync(join(dir, `${name}.json`), 'utf-8')) as PackManifest;
  const b = readFileSync(join(dir, `${name}.bin`));
  return parsePack<any>(m, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

const here = dirname(fileURLToPath(import.meta.url));
const pub = join(here, '..', 'public');
const ref = join(pub, 'garage', 'reference.json');

test('garage: browser networks match Python on the precomputed RoadSim runs', { skip: !existsSync(ref) }, async () => {
  const R = JSON.parse(readFileSync(ref, 'utf-8')) as { cases: Case[]; tolerance: number };
  const load = (dir: string, name: string) => loadPack(dir === 'garage' ? join(pub, 'garage') : join(pub, 'instruments', dir), name);
  const worst: Record<string, number> = { en: 0, td: 0, hp: 0 };
  const count: Record<string, number> = { en: 0, td: 0, hp: 0 };
  for (const c of R.cases) {
    const out = await run(c, load);
    assert.equal(out.length, c.outputs.length, `${c.kind} ${c.v} r${c.r} t${c.t} l${c.l}: output length`);
    for (let i = 0; i < out.length; i++) {
      const d = Math.abs(out[i] - c.outputs[i]) / Math.max(1, Math.abs(c.outputs[i]));
      worst[c.kind] = Math.max(worst[c.kind], d);
    }
    count[c.kind]++;
  }
  console.log(`  garage: EnergyNet ${count.en} cases, worst ${worst.en.toExponential(2)}; TripDetective ${count.td} cases, worst ${worst.td.toExponential(2)}; `
    + `HybridPilot ${count.hp} cases, worst ${worst.hp.toExponential(2)} (tolerance ${R.tolerance})`);
  for (const k of Object.keys(worst)) assert.ok(worst[k] <= R.tolerance, `garage ${k}: worst difference ${worst[k]} > ${R.tolerance}`);
});
