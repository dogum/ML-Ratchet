"""Add the fifth law (road, 22 Nov 2026) to the shipped release without rebuilding the other four.

The other laws' banks and exam point files were never shipped, so build_release.py / webdata.py cannot rebuild them
from this folder. This script:
  1. build_release.main(['road']): the road entry merged into the shipped lawfinder_laws.json, examples/road.csv
     (WORK/release/);
  2. the lab page: the shipped lab/index.html is template.html with one data blob; the blob is read back, the road block
     (webdata.law_block + build_page.process) appended, limits refreshed, and the page re-assembled from the current
     template -> WORK/web/lawfinder.html;
  3. copies those into models/lawfinder/ (lawfinder_laws.json, examples/road.csv, lab/index.html) and the road records
     into training/results/.
Chooses nothing; reads the exam as reported (results/exam_road_reported.json, run 1 with its errata).
"""
import json, os, shutil
from _paths import ROOT, MODEL, SRC
import build_release, build_page, webdata


def main():
    build_release.main(['road'])
    page = open(f'{MODEL}/lab/index.html', encoding='utf-8').read()
    a = page.index('const DATA = ') + len('const DATA = ')
    b = page.index('\n', a)
    D = json.loads(page[a:b].rstrip().rstrip(';'))
    L = build_page.process(webdata.law_block('road'))
    D['laws'] = [x for x in D['laws'] if x['id'] != 'road'] + [L]
    D['limits'] = json.load(open(f'{ROOT}/web/limits.json', encoding='utf-8'))
    tpl = open(f'{ROOT}/web/template.html', encoding='utf-8').read()
    out = tpl.replace('/*DATA*/null', json.dumps(D, separators=(',', ':'), default=float))
    open(f'{ROOT}/web/lawfinder.html', 'w', encoding='utf-8', newline='\n').write(out)
    print('page KB', len(out.encode()) / 1e3, 'laws', [x['id'] for x in D['laws']])
    shutil.copy(f'{ROOT}/release/lawfinder_laws.json', f'{MODEL}/lawfinder_laws.json')
    shutil.copy(f'{ROOT}/release/examples/road.csv', f'{MODEL}/examples/road.csv')
    shutil.copy(f'{ROOT}/web/lawfinder.html', f'{MODEL}/lab/index.html')
    for f in ('road_ratchet.json', 'road_search_p1.json', 'road_search_p2.json', 'exam_road.json', 'exam_road_reported.json'):
        shutil.copy(f'{ROOT}/results/{f}', f'{SRC}/results/{f}')
    print('copied into', MODEL)


if __name__ == '__main__':
    main()
