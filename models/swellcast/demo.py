"""SwellCast demo (numpy only, runs in seconds).
Part 1: raw NOAA NDBC files for buoy 51004 (southeast of Hawaii), 2018 -> pipeline -> window around Hurricane Olivia's swell
        -> SwellCast. Truth from the IBTrACS best track.
Part 2: five pre-cut real exam windows (examples/*.npz, a fixed random draw from the exam, not picked by result)."""
import os, glob, json, time, numpy as np
import pipeline as PL, swellcast as SC
HERE = os.path.dirname(os.path.abspath(__file__)); t_start = time.time()
model = SC.SwellCast()

def show(r, truth=None, centre=None):
    lo, hi = r['D_km'] * 10 ** -r['logD_sig'], r['D_km'] * 10 ** r['logD_sig']; hw = min(180, 57.3 / np.sqrt(r['kappa']))
    print(f"  swell detected: {'yes' if r['detected'] else 'no'} (score {r['p_swell']:.2f})")
    print(f"  distance  {r['D_km']:7.0f} km  (68%: {lo:.0f}-{hi:.0f})" + (f"   truth {truth['D']:7.0f} km" if truth else ''))
    tt = (lambda t: PL.utc(centre + t)) if centre is not None else (lambda t: f'{t:+.0f} h')
    print(f"  storm peak {tt(r['t0'])} +- {r['t0_sig']:.0f} h" + (f"   truth {tt(truth['t0'])}" if truth else ''))
    print(f"  swell from {r['dir']:5.0f} deg +- {hw:.0f}" + (f"            truth {truth['dir']:5.0f} deg" if truth else ''))
    print(f"  storm Hs   {r['hs']:5.1f} m (x/÷ {10 ** r['loghs_sig']:.1f})")

if __name__ == '__main__':
    print('Part 1: raw NDBC files, buoy 51004, Hurricane Olivia (2018)')
    d = os.path.join(HERE, 'examples/raw')
    rec = PL.read_ndbc(os.path.join(d, '51004w2018.txt.gz'), os.path.join(d, '51004d2018.txt.gz'), os.path.join(d, '51004j2018.txt.gz'))
    meta = json.load(open(os.path.join(d, 'olivia.json')))
    win, tc = PL.window(rec, meta['window_centre_hour']); print('  window centre', PL.utc(tc)); show(model.read(win), meta['truth'], tc)
    print('\nPart 2: pre-cut exam windows')
    for f in sorted(glob.glob(os.path.join(HERE, 'examples/*.npz'))):
        z = np.load(f); t = json.loads(str(z['truth'])); print(f"{t['name']} -> buoy {t['buoy']} ({t['basin']} {t['year']})")
        show(model.read(z['x']), t, int(z['centre_hour']))
    print(f'\ndone in {time.time() - t_start:.1f} s')
