// Copy each model's interactive page (models/<id>/lab/index.html) to public/models/<id>/lab/, so the site serves it
// at /models/<id>/lab/. Most labs are one self-contained file; a few load sibling files (scripts, JSON) that sit next
// to index.html in lab/. Every file in lab/ is copied except the src/ folder (the page's sources).
import { cpSync, existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const models = resolve(here, '../../models');
const out = resolve(here, '../public/models');

rmSync(out, { recursive: true, force: true });
let n = 0;
for (const id of readdirSync(models)) {
  const lab = join(models, id, 'lab');
  if (!existsSync(join(lab, 'index.html'))) continue;
  const dst = join(out, id, 'lab');
  mkdirSync(dst, { recursive: true });
  for (const f of readdirSync(lab)) {
    if (f === 'src') continue;
    cpSync(join(lab, f), join(dst, f), { recursive: true });
  }
  n++;
}
console.log(`copied ${n} labs to public/models/*/lab/`);
