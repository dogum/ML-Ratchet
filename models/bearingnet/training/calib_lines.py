"""Dev-only calibration 2: how sharp and how many are the envelope-spectrum lines of real faults vs their simulated twins?
For each CWRU dev fault reading (DE/FE channels) and Paderborn dev fault readings: in the kurtogram-band squared-envelope
spectrum, for harmonics k = 1..10 of the defect frequency (true speed), line prominence = peak within +-1% over the local
median (+-12%), in dB; and the fitted line position (slip, % below theory). Twins with physics VERSION.
Usage: calib_lines.py VERSION [VERSION ...]"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import bearsim as BS, features as FE, realdata as RD
F = np.fft.rfftfreq(FE.NWIN, 1 / FE.FS)

def lines(x, f_def, K=10):
    x = np.asarray(x, float); ses, _ = FE.envelope_spectrum(np.fft.rfft(x - x.mean()), *FE.sk_band(x)); out, pos = [], []
    for k in range(1, K + 1):
        m = (F > k * f_def * 0.975) & (F < k * f_def * 1.005); w = (F > k * f_def * 0.88) & (F < k * f_def * 1.12)
        j = np.argmax(np.where(m, ses, 0)); out.append(10 * np.log10(ses[j] / np.median(ses[w]))); pos.append(F[j] / (k * f_def) - 1)
    return np.array(out), np.array(pos)

def fdef(r):
    o = FE.orders(**r['geom']); return {'OR': o['BPFO'], 'IR': o['BPFI'], 'B': 2 * o['BSF']}[r['cls']] * r['f_true']

if __name__ == '__main__':
    S = RD.splits(); rng = np.random.default_rng(11); _, ref, f98 = RD.cwru_rec(98); pbref = {}
    for b in S['pb_noise']:
        z = np.load(RD.D + 'pb/%s.npz' % b); meta = json.loads(str(z['meta']))
        for x, m in zip(z['x'], meta): pbref.setdefault(m['cond'], []).append((x.astype(float), m['rpm'] / 60))
    R = [r for r in RD.cwru_readings([i for i in S['cwru_dev'] if i != 99]) if r['kind'] == 'main' and r['chan'] in ('DE', 'FE')]
    R += [r for r in RD.pb_readings([b for b in S['pb_dev'] if not b.startswith('K0')], per_cond=2)]
    res = {}
    for r in R:
        x = FE.pad_short(r['x'])[:FE.NWIN]; fd = fdef(r); L, P = lines(x, fd)
        bg, fbg = (ref[r['chan']], f98) if r['src'] == 'CWRU' else pbref[r['cond']][0]
        ex = 10 * np.log10(BS.band_power(x) / np.mean([BS.band_power(bg[s:s + FE.NWIN]) for s in range(0, len(bg) - FE.NWIN, FE.NWIN)])); snr = 10 * np.log10(max(10 ** (ex / 10) - 1, 10 ** -2.5))
        key = '%s %s %s' % (r['src'], r['cls'], r['loc']); d = res.setdefault(key, dict(real=[], pos=[]))
        d['real'].append(L); d['pos'].append(P)
        for v in sys.argv[1:]:
            Lt = []
            for rep in range(3):
                Pp = BS.draw_params(rng, r['cls'], v); Pp['snr_db'] = snr
                xt, _ = BS.simulate(rng, bg, fbg, cls=r['cls'], geom=r['geom'], size_mm=max(r['size_mm'], 0.1), rho=r['f_true'] / fbg, P=Pp, distract=False, aperiodic=False, version=v)
                Lt.append(lines(xt, fd)[0])
            d.setdefault(v, []).append(np.mean(Lt, 0))
    out = {}
    for k, d in sorted(res.items()):
        o = dict(n=len(d['real']), real_db_k1_10=np.round(np.median(d['real'], 0), 1).tolist(), real_mean_db=round(float(np.mean(d['real'])), 2),
                 slip_pct_median=round(float(-100 * np.median(np.array(d['pos'])[:, :3])), 2))
        for v in sys.argv[1:]: o[v + '_db_k1_10'] = np.round(np.median(d[v], 0), 1).tolist(); o[v + '_mean_db'] = round(float(np.mean(d[v])), 2)
        out[k] = o; print(k, json.dumps(o))
    json.dump(out, open(_P.WORK + '/checks/calib_lines_%s.json' % '_'.join(sys.argv[1:]), 'w'), indent=1)
