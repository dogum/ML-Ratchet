"""Pick the real stars and sectors: planet hosts (exam and dev), false-positive TOIs, false alarms, and a pool of ordinary
2-minute targets with no known signal, whose light curves supply the real noise that simulated transits are injected into."""
import _paths as _P  # portable paths (see _paths.py)
import csv, json, random, numpy as np
from astroquery.mast import Observations
D = _P.WORK + '/data/'
tic = lambda s: str(s).replace('TIC', '').strip().split('.')[0]
f = lambda x: float(x) if x not in ('', None) else np.nan

def spread(lst, k):
    if len(lst) <= k: return lst
    idx = np.round(np.linspace(0, len(lst) - 1, k)).astype(int); return [lst[i] for i in idx]

if __name__ == '__main__':
    prod = json.load(open(D + 'lc_products.json')); P = list(csv.DictReader(open(D + 'pscomppars.csv'))); T = list(csv.DictReader(open(D + 'toi.csv')))
    toi_tics = {tic(r['TIC ID']) for r in T}; pl_tics = {tic(r['tic_id']) for r in P}
    pl = [r for r in P if tic(r['tic_id']) in prod and 0.3 <= f(r['pl_orbper']) <= 25 and np.isfinite(f(r['pl_tranmid']))
          and f(r['sy_tmag']) <= 13.5 and (f(r['pl_ratror']) ** 2 * 1e6 >= 300 or f(r['sy_tmag']) <= 10.5)]
    fp = [r for r in T if r['TFOPWG Disposition'] == 'FP' and tic(r['TIC ID']) in prod and 0.3 <= f(r['Period (days)']) <= 25]
    fa = [r for r in T if r['TFOPWG Disposition'] == 'FA' and tic(r['TIC ID']) in prod and 0.3 <= f(r['Period (days)']) <= 25]
    jobs = {}
    for r in pl: k = tic(r['tic_id']); jobs[k] = spread(prod[k], 3)
    for r in fp: k = tic(r['TIC ID']); jobs.setdefault(k, spread(prod[k], 2))
    for r in fa: k = tic(r['TIC ID']); jobs.setdefault(k, spread(prod[k], 1))
    # noise pool: ordinary 2-minute targets, no TOI, no known planet, a few sectors spread over the mission
    rng = random.Random(7); pool = []
    for s in (3, 9, 15, 22, 28, 36, 44, 50, 57, 64):
        obs = Observations.query_criteria(obs_collection='TESS', dataproduct_type='timeseries', provenance_name='SPOC', t_exptime=[100, 150], sequence_number=s)
        cand = sorted({(str(r['target_name']), int(r['sequence_number']), str(r['dataURL'])) for r in obs if str(r['dataURL']).endswith('_lc.fits')})
        cand = [c for c in cand if c[0] not in toi_tics and c[0] not in pl_tics and c[0] not in jobs]; pick = rng.sample(cand, min(220, len(cand))); pool += pick
        print('sector', s, len(cand), 'candidates, picked', len(pick), flush=True)
    meta = dict(planets=[{k: r[k] for k in r if not k.endswith('reflink')} | {'tic': tic(r['tic_id'])} for r in pl],
                fps=[{k: r[k] for k in ('TIC ID', 'TOI', 'TFOPWG Disposition', 'TESS Disposition', 'Period (days)', 'Epoch (BJD)', 'Duration (hours)', 'Depth (ppm)', 'TESS Mag', 'Stellar Eff Temp (K)', 'Stellar log(g) (cm/s^2)', 'Stellar Radius (R_Sun)', 'Comments')} | {'tic': tic(r['TIC ID'])} for r in fp + fa],
                pool=[c[0] for c in pool])
    json.dump(meta, open(D + 'targets.json', 'w'))
    J = [(k, s, u) for k, v in jobs.items() for s, u in v] + [list(c) for c in pool]
    json.dump(J, open(D + 'jobs.json', 'w'))
    print('planets', len(pl), 'hosts', len({tic(r['tic_id']) for r in pl}), '| FP', len(fp), 'FA', len(fa), '| pool', len(pool), '| files', len(J))
