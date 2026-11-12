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

if __name__ == '__main__':
    P = list(csv.DictReader(open(D + 'pscomppars.csv'))); T = list(csv.DictReader(open(D + 'toi.csv')))
    tics = sorted({tic(r['tic_id']) for r in P} | {tic(r['TIC ID']) for r in T} - {''})
    print(len(tics), 'unique TIC IDs', flush=True)
    prod = products(tics); json.dump(prod, open(D + 'lc_products.json', 'w'))
    print('with 2-min SPOC light curves:', len(prod))
