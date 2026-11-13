"""CellSense NMC811: fetch the processed diagnostic tests of Pozzato, Allam & Onori 2022 (LG INR21700-M50T, ten cells aged
with real driving profiles) from the authors' OSF project, the repository named in their data article:

    Pozzato G., Allam A., Onori S. (2022) Lithium-ion battery aging dataset based on electric vehicle real-driving profiles.
    Data in Brief 41, 107995. doi:10.1016/j.dib.2022.107995. Data: https://osf.io/qsabn/ (view-only link printed in the
    article: https://osf.io/qsabn/?view_only=2a03b6c78ef14922a3e244f3d549de78). Licence: CC BY 4.0 (stated on the OSF node).

The files sit on the project's Dropbox storage add-on; this script uses OSF's public, documented v2 API
(https://developer.osf.io/) with the article's view-only key, lists the folders and downloads only the processed .mat
files of the diagnostic tests (capacity, HPPC, EIS) (the raw xlsx are 249 GB and are not needed). Everything goes to
work/data_m50t/ (git-ignored).

    python nmc_fetch.py list            # write the file tree to work/data_m50t/tree.json
    python nmc_fetch.py get [pattern]   # download processed .mat files whose path contains the pattern(s)
"""
import _paths as _P
import os, sys, json, time, urllib.request, urllib.parse

VO = '2a03b6c78ef14922a3e244f3d549de78'
API = 'https://api.osf.io/v2/nodes/qsabn/files/dropbox'
OUT = os.path.join(_P.WORK, 'data_m50t')


def get_json(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'ml-ratchet-cellsense'}), timeout=120) as r:
                return json.load(r)
        except Exception as e:  # transient network errors: back off and retry, never change hosts
            print('retry', k, e, flush=True); time.sleep(5 * (k + 1))
    raise RuntimeError(url)


def listing(path='/', depth=0, max_depth=6):
    url = API + urllib.parse.quote(path) + '?view_only=' + VO + '&page[size]=100'; out = []
    while url:
        d = get_json(url)
        for x in d.get('data', []):
            a = x['attributes']; rec = {'kind': a['kind'], 'path': a['path'], 'name': a['name'], 'size': a.get('size'),
                                        'download': x.get('links', {}).get('download')}
            out.append(rec)
            if a['kind'] == 'folder' and depth < max_depth: out += listing(a['path'], depth + 1, max_depth)
        url = d.get('links', {}).get('next')
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == 'list':
        T = listing('/diagnostic_tests/'); json.dump(T, open(os.path.join(OUT, 'tree.json'), 'w'), indent=1)
        for t in T: print(t['kind'][0], t['size'], t['path'])
    elif sys.argv[1] == 'get':
        pats = sys.argv[2:] or ['_processed_mat']
        T = json.load(open(os.path.join(OUT, 'tree.json')))
        for t in T:
            if t['kind'] != 'file' or not any(p in t['path'] for p in pats) or not t['path'].endswith(('.mat', '.m')): continue
            dst = os.path.join(OUT, t['path'].strip('/').replace('/', os.sep))
            if os.path.exists(dst) and (t['size'] is None or os.path.getsize(dst) == t['size']): continue
            os.makedirs(os.path.dirname(dst), exist_ok=True); url = t['download']
            url += ('&' if '?' in url else '?') + 'view_only=' + VO
            print('get', t['path'], t['size'], flush=True)
            urllib.request.urlretrieve(url, dst + '.part'); os.replace(dst + '.part', dst)
