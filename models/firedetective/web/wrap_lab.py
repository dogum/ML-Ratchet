"""Write lab/index.html: the released FireDetective page as a plain standalone HTML file.

    python web/wrap_lab.py [PAGE]     # PAGE: training/build_page.py's output (default WORK/web/firedetective.html)

The page is training/template.html with the data (__DATA__: the 3 members' weights rounded to 5 significant figures,
dev calibration, all 944 exam fires' input features, predictions and truths, the ratchet log, the twin table, the
exam tables) and training/fdnet.js (__FDNET__) filled in. build_page.py needs the exam input records under WORK
(data/real/exam/*.pkl from exam.py build), so it cannot run without the fire data rebuild. lab/index.html (21 Nov
2026) was made from the published copy of the page instead: that copy is byte for byte the template, the data and
fdnet.js (checked on import), inside a host wrapper. This script removes a host wrapper if there is one and gives the
page its own doctype, head and body; nothing inside the page changes.
"""
import re
import sys
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL / 'training'))
from _paths import WORK  # noqa: E402

DESCRIPTION = ('FireDetective reads a wildfire\'s satellite record and infers what drove it: crown fire, wind and live '
               'fuel moisture, with error bars. Trained only on simulated fires; the network runs in your browser on '
               'all 944 real exam fires.')


def inner_page(html: str) -> str:
    """The template part: drop a host wrapper (<!doctype ...><head>...</head><body> ... </body></html>) if present."""
    if html.lstrip().lower().startswith('<!doctype'):
        i = html.index('<body>') + len('<body>')
        html = html[i:]
        j = html.rindex('</body>')
        html = html[:j]
    return html.strip('\n')


def wrap(page: str) -> str:
    m = re.match(r'(?s)\s*(<title>.*?</title>)\s*(<meta name="claude:viewport-fit-sides"[^>]*>)?\s*(.*?</style>)\n(.*)$', page)
    if not m:
        raise SystemExit('unexpected page layout: expected <title>, links and one <style> before the body')
    title, _host_meta, head_rest, body = m.groups()
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f'<meta name="description" content="{DESCRIPTION}">\n{title}\n{head_rest}\n</head>\n<body>\n'
            f'{body.rstrip()}\n</body>\n</html>\n')


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(WORK) / 'web' / 'firedetective.html'
    page = inner_page(src.read_text(encoding='utf-8'))
    for token in ('__DATA__', '__FDNET__'):
        if token in page:
            raise SystemExit(f'{token} is not filled in: run training/build_page.py first')
    out = MODEL / 'lab' / 'index.html'
    html = wrap(page)
    out.write_text(html, encoding='utf-8', newline='\n')
    print(f'wrote {out} ({len(html.encode("utf-8")) / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
