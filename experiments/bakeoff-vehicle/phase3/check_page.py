"""Headless check of the page at desktop and phone widths, light and dark: horizontal overflow, console errors,
section screenshots. usage: python3 check_page.py <page.html> [shots_dir]"""
import asyncio, os, sys
from playwright.async_api import async_playwright
SRC = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(SRC), 'shots')
TMP = '/tmp/claude-0/706c0d77-bfd2-594c-b22e-e532edd10b67/scratchpad/vbo_wrapped.html'
async def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.dirname(TMP), exist_ok=True)
    html = open(SRC).read()
    open(TMP, 'w').write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><style>:root{color-scheme:light}body{margin:0;font:14px system-ui}[hidden]{display:none!important}img{max-width:100%}</style></head><body>' + html + '</body></html>')
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for name, w, h, scheme in [('desk_light', 1280, 900, 'light'), ('desk_dark', 1280, 900, 'dark'), ('phone_light', 390, 844, 'light'), ('phone_dark', 390, 844, 'dark')]:
            ctx = await b.new_context(viewport={'width': w, 'height': h}, color_scheme=scheme, device_scale_factor=1)
            pg = await ctx.new_page(); errs = []
            pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
            pg.on('pageerror', lambda e: errs.append('PAGEERROR ' + str(e)))
            await pg.goto('file://' + TMP, wait_until='load', timeout=60000)
            await pg.wait_for_timeout(600)
            sw = await pg.evaluate('document.documentElement.scrollWidth'); H = await pg.evaluate('document.documentElement.scrollHeight')
            wide = await pg.evaluate('''() => [...document.querySelectorAll('body *')].filter(e => { const r = e.getBoundingClientRect(); return r.right > innerWidth + 1 && !e.closest('.tframe') && getComputedStyle(e).position !== 'absolute'; }).slice(0, 5).map(e => e.tagName + '.' + e.className + '#' + e.id)''')
            print(name, 'scrollWidth', sw, 'viewport', w, 'height', H, 'errors', [e for e in errs if 'fonts.g' not in e][:5], 'wide', wide)
            for sec in ['.hero', '#showcase', '#boards', '#wins', '#families', '#compute', '#findings', '#river', '#limits']:
                el = await pg.query_selector(sec)
                if el and (name == 'desk_light' or (name == 'phone_light' and sec in ('.hero', '#showcase', '#boards', '#wins', '#compute')) or (name == 'desk_dark' and sec in ('#showcase', '#wins', '#compute')) or (name == 'phone_dark' and sec == '#showcase')):
                    await el.screenshot(path=f'{OUT}/{name}_{sec.strip("#.")}.png')
            await ctx.close()
        await b.close()
asyncio.run(main())
