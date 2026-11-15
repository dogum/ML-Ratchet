"""Which real stars TransitNet will read: confirmed transiting planets (NASA Exoplanet Archive, pscomppars), TESS Objects of
Interest with their follow-up dispositions (ExoFOP), and the TESS 2-minute SPOC light curves available for each at MAST."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, csv, time, numpy as np
from astroquery.mast import Observations
D = _P.WORK + '/data/'

def tic(s):
    s = str(s).replace('TIC', '').strip(); return s.split('.')[0] if s else ''

def products(tics, chunk=400):
    """SPOC 2-min light-curve URIs per TIC: {tic: [(sector, uri), ...]}"""
    out = {}
    for i in range(0, len(tics), chunk):
        for attempt in range(3):
            try:
                obs = Observations.query_criteria(obs_collection='TESS', dataproduct_type='timeseries', provenance_name='SPOC', t_exptime=[100, 150], target_name=list(tics[i:i + chunk])); break
            except Exception as e:
                print('retry', e, flush=True); time.sleep(5)
        for r in obs:
            u = str(r['dataURL'])
            if u.endswith('_lc.fits'): out.setdefault(str(r['target_name']), []).append((int(r['sequence_number']), u))
        print(i + chunk, len(out), flush=True)
    return {k: sorted(set(v)) for k, v in out.items()}

BULK = 'https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{}_lc.sh'

def bulk_sector(s):
    """every SPOC 2-minute light curve of one sector, from MAST's bulk download script: [(tic, sector, uri)].
    Added 15 Nov 2026: the portal query above timed out on 400-star chunks from the rebuild machine; the bulk scripts list
    the same products (SPOC, 120 s) and are a documented MAST endpoint."""
    import os, re, subprocess
    fn = D + f'bulk/tesscurl_sector_{s}_lc.sh'; os.makedirs(D + 'bulk', exist_ok=True)
    if not os.path.exists(fn):
        r = subprocess.run(['curl', '-sS', '-f', '-L', '-m', '600', '-o', fn + '.part', BULK.format(s)], capture_output=True, text=True)
        if r.returncode != 0: return []
        os.replace(fn + '.part', fn)
    out = []
    for m in re.finditer(r'uri=(mast:TESS/product/(tess\d+-s(\d{4})-(\d{16})-\d{4}-s_lc\.fits))', open(fn).read()):
        out.append((str(int(m.group(4))), int(m.group(3)), m.group(1)))
    return out

def products_bulk(tics, sectors=range(1, 200)):
    want = set(tics); out = {}
    for s in sectors:
        rows = bulk_sector(s)
        if not rows and s > 100: break
        for t, sec, u in rows:
            if t in want: out.setdefault(t, []).append((sec, u))
        print('sector', s, len(rows), 'stars with products', len(out), flush=True)
    return {k: sorted(set(v)) for k, v in out.items()}

if __name__ == '__main__':
    P = list(csv.DictReader(open(D + 'pscomppars.csv'))); T = list(csv.DictReader(open(D + 'toi.csv')))
    tics = sorted({tic(r['tic_id']) for r in P} | {tic(r['TIC ID']) for r in T} - {''})
    print(len(tics), 'unique TIC IDs', flush=True)
    prod = products_bulk(tics); json.dump(prod, open(D + 'lc_products.json', 'w'))
    print('with 2-min SPOC light curves:', len(prod))
