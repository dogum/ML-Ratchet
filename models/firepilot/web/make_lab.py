"""Write lab/index.html: the published FirePilot page as a clean standalone HTML file (import, 21 Nov 2026).

    python web/make_lab.py [PAGE]

The published page is web/template.html with web/fp.js and one `const DATA = {...};` line filled in by
web/build_page.py, inside a host wrapper (a <!doctype>, <head> and <body> added by the page host, plus a host-only
<meta>). build_page.py cannot be re-run here: it needs the exam's stored per-fire inputs and masks
(results/exam/<fire>_ctx.npz, <fire>_<budget>_masks.npz), which were not in the release zip. So this script takes the
DATA line from PAGE (default: the current lab/index.html; on import it was the page as published), fills
template.html and fp.js exactly as build_page.py does, and wraps the result in a plain HTML document. The page body
is unchanged; the fonts still load from Google Fonts, with system fallbacks offline.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, '..', 'lab', 'index.html')
HOST_META = '<meta name="claude:viewport-fit-sides" content="cover">\n'


def data_line(page):
    i = page.index('const DATA = ')
    j = page.index(';\n', i)
    return page[i:j + 1]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else LAB
    page = open(src, encoding='utf-8').read()
    tpl = open(os.path.join(HERE, 'template.html'), encoding='utf-8').read()
    js = open(os.path.join(HERE, 'fp.js'), encoding='utf-8').read()
    body = tpl.replace('/*FPJS*/', js).replace('/*DATA*/', data_line(page))   # as build_page.py does
    assert body.count(HOST_META) == 1
    body = body.replace(HOST_META, '')
    k = body.index('</style>\n') + len('</style>\n')                          # title, fonts and styles go in <head>
    html = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
            '<meta name="description" content="FirePilot: a learned containment-line planner, graded in the FireSim '
            'simulator on 48 real exam fires. The value network plans live in your browser.">\n'
            + body[:k] + '</head>\n<body>\n' + body[k:].lstrip('\n').rstrip('\n') + '\n</body>\n</html>\n')
    with open(LAB, 'w', encoding='utf-8', newline='\n') as f:
        f.write(html)
    print(f'{os.path.normpath(LAB)}: {len(html.encode()) / 1024:.0f} KB from {src}')


if __name__ == '__main__':
    main()
