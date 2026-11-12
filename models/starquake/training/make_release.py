"""Assemble release/: pipeline.py (= src/sq.py + a Kepler loader), starquake.py, weights, demo.py with three real exam
stars, training/ (all scripts), MODEL_CARD.md, README.md; zip to <work>/dist/starquake-v1.0.zip."""
import _paths as _P  # portable paths (see _paths.py)
import os, sys, json, shutil, glob, numpy as np
S = _P.WORK + '/'; R = S + 'release/'
LOADER = '''

def load_kepler(src):
    """Kepler long-cadence light curve -> ppm on the cadence grid (NaN in gaps).
    src: list of *_llc.fits paths (needs astropy), or a dict/npz with arrays cad, flux (PDCSAP), quarter, q (SAP_QUALITY)."""
    if isinstance(src, (list, tuple)):
        from astropy.io import fits
        C = {a: [] for a in ('cad', 'flux', 'quarter', 'q')}
        for p in src:
            with fits.open(p) as h:
                d = h[1].data; C['cad'].append(np.asarray(d['CADENCENO'])); C['flux'].append(np.asarray(d['PDCSAP_FLUX'], float))
                C['q'].append(np.asarray(d['SAP_QUALITY'])); C['quarter'].append(np.full(len(d), h[0].header['QUARTER']))
        src = {a: np.concatenate(v) for a, v in C.items()}
    return grid_lc(np.asarray(src['cad']).astype(np.int64), np.asarray(src['flux'], np.float64), np.asarray(src['quarter']), np.asarray(src['q']).astype(np.int64))
'''
DEMO = '''"""StarQuake demo: three real Kepler red giants from the exam (a fixed random draw), read from the whole light curve
and from a 90-day piece. Runs in a few seconds with numpy and scipy."""
import os, json, time, numpy as np
from starquake import StarQuake
H = os.path.dirname(os.path.abspath(__file__)); sq = StarQuake(); truth = json.load(open(os.path.join(H, 'examples', 'truth.json')))
for f in sorted(os.listdir(os.path.join(H, 'examples'))):
    if not f.endswith('.npz'): continue
    kic = f[:-4]; z = np.load(os.path.join(H, 'examples', f)); t = truth[kic]; t0 = time.time()
    import pipeline as PL; y = PL.load_kepler(z)
    for name, seg in (('whole light curve', y), ('90 days', y[20000:20000 + int(90 * 86400 / PL.DT)])):
        r = sq.read_ppm(seg, teff=t['teff'], e_teff=t['e_teff'])
        print(f"KIC {kic} ({name}, {r['days']:.0f} d): numax {r['numax']:.2f} +- {r['numax_err']:.2f} uHz (published {t['numax']:.2f}), "
              f"dnu {r['dnu']:.3f} +- {r['dnu_err']:.3f} uHz (published {t['dnu']:.3f}), M {r['mass']:.2f} +- {r['mass_err']:.2f} (published {t['M']:.2f}), R {r['radius']:.2f} +- {r['radius_err']:.2f} (published {t['R']:.2f})")
    print(f"   {time.time() - t0:.1f} s")
'''

def main(card, readme, ex_kics):
    os.makedirs(R + 'examples', exist_ok=True); os.makedirs(R + 'training', exist_ok=True)
    open(R + 'pipeline.py', 'w').write(open(S + 'src/sq.py').read() + LOADER)
    open(R + 'demo.py', 'w').write(DEMO); open(R + 'MODEL_CARD.md', 'w').write(card); open(R + 'README.md', 'w').write(readme)
    Y = json.load(open(S + 'data/yu2018.json')); truth = {}
    for k in ex_kics:
        z = np.load(f'{S}data/lc/{k}.npz'); np.savez_compressed(f'{R}examples/{k}.npz', cad=z['cad'], flux=z['flux'], quarter=z['quarter'], q=z['q'])
        truth[str(k)] = {a: Y[str(k)][a] for a in ('numax', 'e_numax', 'dnu', 'e_dnu', 'teff', 'e_teff', 'M', 'R', 'kp', 'phase')}
    json.dump(truth, open(R + 'examples/truth.json', 'w'), indent=1)
    for p in glob.glob(S + 'src/*.py'): shutil.copy(p, R + 'training/')
    for p in ('ratchet.json', 'physics_check.json', 'sigma_calibration.json', 'baselines.json'): shutil.copy(S + p, R + 'training/')
    shutil.copy(S + 'web/sqnet.js', R + 'training/')
    os.makedirs(_P.OUT, exist_ok=True); out = _P.OUT + '/starquake-v1.0'
    tmp = _P.WORK + '/zipstage/starquake-v1.0'; shutil.rmtree(_P.WORK + '/zipstage', ignore_errors=True); shutil.copytree(R, tmp, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.make_archive(out, 'zip', _P.WORK + '/zipstage', 'starquake-v1.0'); print('zip', out + '.zip', os.path.getsize(out + '.zip') / 1e6, 'MB')

if __name__ == '__main__':
    T = json.load(open(S + 'web/texts.json')); main(open(S + 'release_card.md').read(), open(S + 'release_readme.md').read(), T['demo_kics'])
