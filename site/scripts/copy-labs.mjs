// Copy each model's self-contained interactive page (models/<id>/lab/index.html) to
// public/models/<id>/lab/index.html, so the site serves it at /models/<id>/lab/.
import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const models = resolve(here, '../../models');
const out = resolve(here, '../public/models');

rmSync(out, { recursive: true, force: true });
let n = 0;
for (const id of readdirSync(models)) {
  const src = join(models, id, 'lab', 'index.html');
  if (!existsSync(src)) continue;
  const dst = join(out, id, 'lab', 'index.html');
  mkdirSync(dirname(dst), { recursive: true });
  copyFileSync(src, dst);
  n++;
}
console.log(`copied ${n} labs to public/models/*/lab/`);
