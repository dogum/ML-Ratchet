"""The real stars, read once by the pipeline: every confirmed planet in targets.json and every TOI that follow-up found to
be a false positive (FP) or false alarm (FA), each from its catalogue ephemeris and duration, with the star's other known
transiting bodies (planets, other TOIs) cut out first. Stored with the published answers."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, csv, time, numpy as np
sys.path.insert(0, _P.SRC); import pipeline as PL
D = _P.WORK + '/data/'; f = lambda x: float(x) if x not in ('', None) else np.nan
tic = lambda s: str(s).replace('TIC', '').strip().split('.')[0]

def main():
    M = json.load(open(D + 'targets.json')); P_all = list(csv.DictReader(open(D + 'pscomppars.csv'))); T_all = list(csv.DictReader(open(D + 'toi.csv')))
    sys_pl = {}; sys_toi = {}
    for r in P_all: sys_pl.setdefault(tic(r['tic_id']), []).append(r)
    for r in T_all: sys_toi.setdefault(tic(r['TIC ID']), []).append(r)
    def others_of(k, P_self):
        o = []
        for r in sys_pl.get(k, []):
            Pp, Tp, dp = f(r['pl_orbper']), f(r['pl_tranmid']), f(r['pl_trandur'])
            if np.isfinite([Pp, Tp, dp]).all() and abs(Pp / P_self - 1) > 0.01: o.append((Pp, Tp - 2457000, dp / 24))
        for r in sys_toi.get(k, []):
            Pp, Tp, dp = f(r['Period (days)']), f(r['Epoch (BJD)']), f(r['Duration (hours)'])
            if np.isfinite([Pp, Tp, dp]).all() and Pp > 0 and abs(Pp / P_self - 1) > 0.01 and all(abs(Pp / x[0] - 1) > 0.01 for x in o): o.append((Pp, Tp - 2457000, dp / 24))
        return o
    items = [('planet', r, r['tic'], f(r['pl_orbper']), f(r['pl_tranmid']) - 2457000, f(r['pl_trandur']) / 24, f(r['pl_orbpererr1'])) for r in M['planets']]
    items += [(r['TFOPWG Disposition'], r, r['tic'], f(r['Period (days)']), f(r['Epoch (BJD)']) - 2457000, f(r['Duration (hours)']) / 24, np.nan) for r in M['fps']]
    R = {k: [] for k in ('local', 'secondary', 'glob', 'oddeven', 'cent', 's', 'sc')}; meta = []; t0 = time.time()
    for n, (kind, r, k, P, T0, dur, sigP) in enumerate(items):
        if not np.isfinite([P, T0, dur]).all() or dur <= 0: continue
        secs = PL.load(k)
        if not secs: continue
        try:
            V, info = PL.read(secs, P, T0, dur, sigP=sigP, others=others_of(k, P))
        except Exception as e:
            print('fail', k, e); continue
        if info['ntr'] < 1: continue
        star = PL.star_scalars(secs[0]['hdr'])
        for a in ('local', 'secondary', 'glob', 'oddeven', 'cent'): R[a].append(V[a].astype(np.float32))
        R['sc'].append(np.float32(np.log10(max(info['cent_noise'], 1e-4))))
        R['s'].append(PL.scalars(info['P'], dur, info, star))
        m = dict(kind=kind, tic=k, P=P, dur=dur, info={a: info[a] for a in ('depth', 'scale', 'noise_bin', 'ntr', 'snr', 'n_aligned', 'P', 'T0', 'cent_noise')}, star=star, nsec=len(secs))
        if kind == 'planet':
            m.update(name=r['pl_name'], host=r['hostname'], k=f(r['pl_ratror']), k_err=f(r['pl_ratrorerr1']), b=f(r['pl_imppar']), b_err=f(r['pl_impparerr1']), aR=f(r['pl_ratdor']), aR_err=f(r['pl_ratdorerr1']),
                     T14_h=f(r['pl_trandur']), rade=f(r['pl_rade']), st_rad=f(r['st_rad']), facility=r['disc_facility'], ecc=f(r['pl_orbeccen']))
        else:
            m.update(name='TOI-' + r['TOI'], comments=r['Comments'], tess_disp=r['TESS Disposition'])
        meta.append(m)
        if n % 200 == 0: print(n, len(items), len(meta), '%.0fs' % (time.time() - t0), flush=True)
    np.savez_compressed(D + 'real.npz', **{a: np.stack(v) for a, v in R.items()}); json.dump(meta, open(D + 'real_meta.json', 'w'))
    print('real readings', len(meta), {k: sum(m['kind'] == k for m in meta) for k in ('planet', 'FP', 'FA')})

if __name__ == '__main__':
    main()
