// Every live instrument must reproduce its model's Python outputs on the exported real examples.
// Run: npm test   (node --test with type stripping; no browser needed)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parsePack, type PackManifest } from '../src/lib/nn/pack.ts';
import type { Tensor } from '../src/lib/nn/tensor.ts';

const here = dirname(fileURLToPath(import.meta.url));
const pub = join(here, '..', 'public', 'instruments');

export function loadPack(dir: string, name: string) {
  const m = JSON.parse(readFileSync(join(dir, `${name}.json`), 'utf-8')) as PackManifest;
  const b = readFileSync(join(dir, `${name}.bin`));
  return parsePack<any>(m, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

const ids = existsSync(pub) ? readdirSync(pub).filter((d) => existsSync(join(pub, d, 'reference.json'))) : [];

for (const id of ids) {
  test(`${id}: browser network matches Python`, async () => {
    const dir = join(pub, id);
    const parity = join(here, '..', 'src', 'instruments', id, 'parity.ts');
    assert.ok(existsSync(parity), `src/instruments/${id}/parity.ts is missing`);
    const mod = await import(pathToFileURL(parity).href);
    const ref = JSON.parse(readFileSync(join(dir, 'reference.json'), 'utf-8'));
    let worst = 0;
    for (const c of ref.cases) {
      const out: ArrayLike<number> = await mod.run(c, (name: string) => loadPack(dir, name));
      assert.equal(out.length, c.outputs.length, `${id} case ${c.example}/${c.variant}: output length`);
      for (let i = 0; i < out.length; i++) {
        const d = Math.abs(out[i] - c.outputs[i]) / Math.max(1, Math.abs(c.outputs[i]));
        worst = Math.max(worst, d);
      }
    }
    console.log(`  ${id}: ${ref.cases.length} cases, worst relative difference ${worst.toExponential(2)}`);
    assert.ok(worst <= ref.tolerance, `${id}: worst difference ${worst} > tolerance ${ref.tolerance}`);
  });
}

export type { Tensor };
