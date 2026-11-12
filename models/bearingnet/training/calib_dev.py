"""Calibration on DEV only + twin test. For every real dev fault reading:
  * excess band power (500-5700 Hz) over a healthy reference of the same machine and sensor (CWRU: noise file 98, same
    channel; Paderborn: K001-K003 at the same operating condition) -> the fault's real 'SNR';
  * envelope ring time: e-folding time of the squared-envelope autocorrelation in the kurtogram band, x2 = amplitude decay time.
Then simulate each reading's twin (same bearing, speed, class, size, background machine, measured SNR) with physics v1 and v2
and compare ring time and the network's reading of real vs twin. Writes checks/calib_dev.json.
Usage: calib_dev.py CKPT"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import bearsim as BS, features as FE, realdata as RD, bn_simbank as SB, bnet as BN

def ring_ms(x):
    lo, hi = FE.sk_band(x); _, _ = FE.envelope_spectrum(np.fft.rfft(x - x.mean()), lo, hi)
    g = FE._gain(lo, hi); X = np.fft.rfft(x - x.mean()); Z = np.zeros(FE.NWIN, complex); Z[:len(X)] = 2 * X * g; e = np.abs(np.fft.ifft(Z)) ** 2; e = e - e.mean()
    E = np.fft.rfft(e, 2 * len(e)); r = np.fft.irfft(np.abs(E) ** 2)[:240]; r = r / r[0]; k = np.argmax(r < np.exp(-1))
    return 2 * 1000 * k / FE.FS

def main(ckpt):
    S = RD.splits(); net = BN.load_net(_P.WORK + '/ckpt/' + ckpt); rng = np.random.default_rng(7)
    _, ref, f98 = RD.cwru_rec(98); pbref = {}
    for b in S['pb_noise']:
        z = np.load(RD.D + 'pb/%s.npz' % b); meta = json.loads(str(z['meta']))
        for x, m in zip(z['x'], meta): pbref.setdefault(m['cond'], []).append((x.astype(float), m['rpm'] / 60))
    R = [r for r in RD.cwru_readings([i for i in S['cwru_dev'] if i != 99]) if r['kind'] == 'main' and r['chan'] in ('DE', 'FE')]
    R += [r for r in RD.pb_readings([b for b in S['pb_dev'] if not b.startswith('K0')], per_cond=2)]
    rows = []
    for r in R:
        x = FE.pad_short(r['x'])[:FE.NWIN]
        if r['src'] == 'CWRU': bg, fbg = ref[r['chan']], f98
        else: bg, fbg = pbref[r['cond']][0]
        ex = 10 * np.log10(BS.band_power(x) / np.mean([BS.band_power(bg[s:s + FE.NWIN]) for s in range(0, len(bg) - FE.NWIN, FE.NWIN)]))
        snr = 10 * np.log10(max(10 ** (ex / 10) - 1, 10 ** -2.5))
        row = dict(src=r['src'], rec=r['rec'], cls=r['cls'], loc=r['loc'], excess_db=round(float(ex), 1), snr_db=round(float(snr), 1), ring_ms_real=round(ring_ms(x), 2))
        F, sc, _ = FE.window_features(x, r['geom'], r['f_nom']); o = BN.decode(BN.raw(net, F[None], sc[None]))
        row['net_real'] = BN.CLS[int(o['logp'][0].argmax())]
        for v in ('v1', 'v2'):
            P = BS.draw_params(rng, r['cls'], v); P['snr_db'] = snr
            xt, _ = BS.simulate(rng, bg, fbg, cls=r['cls'], geom=r['geom'], size_mm=max(r['size_mm'], 0.1), rho=r['f_true'] / fbg, P=P, distract=False, aperiodic=False, version=v)
            F, sc, _ = FE.window_features(xt, r['geom'], r['f_nom']); o = BN.decode(BN.raw(net, F[None], sc[None]))
            row['ring_ms_twin_' + v] = round(ring_ms(xt.astype(float)), 2); row['net_twin_' + v] = BN.CLS[int(o['logp'][0].argmax())]
        rows.append(row); print(row, flush=True)
    summ = {}
    for grp, sel in (('CWRU near', lambda w: w['src'] == 'CWRU' and w['loc'] == 'near'), ('CWRU far', lambda w: w['src'] == 'CWRU' and w['loc'] == 'far'), ('PB', lambda w: w['src'] == 'PB')):
        W = [w for w in rows if sel(w)]
        if not W: continue
        summ[grp] = dict(n=len(W), excess_db_range=[min(w['excess_db'] for w in W), max(w['excess_db'] for w in W)], excess_db_median=float(np.median([w['excess_db'] for w in W])),
                         ring_ms_real_median=float(np.median([w['ring_ms_real'] for w in W])), ring_ms_twin_v1_median=float(np.median([w['ring_ms_twin_v1'] for w in W])),
                         ring_ms_twin_v2_median=float(np.median([w['ring_ms_twin_v2'] for w in W])),
                         net_right_real=float(np.mean([w['net_real'] == w['cls'] for w in W])), net_right_twin_v1=float(np.mean([w['net_twin_v1'] == w['cls'] for w in W])),
                         net_right_twin_v2=float(np.mean([w['net_twin_v2'] == w['cls'] for w in W])))
    print(json.dumps(summ, indent=1)); json.dump(dict(ckpt=ckpt, summary=summ, rows=rows), open(_P.WORK + '/checks/calib_dev_%s.json' % ckpt.replace('.pt', ''), 'w'), indent=1)

if __name__ == '__main__':
    main(sys.argv[1])
