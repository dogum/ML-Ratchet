// usage: node shot.js page.html outprefix [checks.json]
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [page_, out, checks] = process.argv.slice(2);
  const html = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>' + fs.readFileSync(page_, 'utf8') + '</body></html>';
  fs.writeFileSync(out + '_wrapped.html', html);
  const browser = await chromium.launch();
  for (const [w, h, tag, scheme] of [[1280, 900, 'desk', 'light'], [400, 860, 'phone', 'dark']]) {
    const pg = await browser.newPage({ viewport: { width: w, height: h }, colorScheme: scheme });
    const errs = []; pg.on('console', m => { if (m.type() === 'error') errs.push(m.text()); }); pg.on('pageerror', e => errs.push(String(e.stack || e)));
    await pg.goto('file://' + require('path').resolve(out + '_wrapped.html'));
    await pg.waitForTimeout(2500);
    await pg.screenshot({ path: `${out}_${tag}.png`, fullPage: false });
    const sw = await pg.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
    console.log(tag, 'errors', errs.slice(0, 5), 'scrollWidth', sw);
    if (checks && tag === 'desk') {
      const C = JSON.parse(fs.readFileSync(checks, 'utf8'));
      const res = await pg.evaluate(C => C.map(c => { const r = predict(c.p, c.Re, c.alpha); return [r.cl, r.cd, r.cm, r.cl_sd]; }), C);
      let md = 0; C.forEach((c, i) => { ['cl', 'cd', 'cm', 'cl_sd'].forEach((k, j) => c[k].forEach((v, q) => { md = Math.max(md, Math.abs(v - res[i][j][q]) / Math.max(1e-3, Math.abs(v))); })); });
      console.log('JS vs Python max rel diff', md);
    }
    await pg.close();
  }
  await browser.close();
})();
