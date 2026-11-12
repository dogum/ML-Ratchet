"""Everything the showpiece page needs, computed with the released numpy model (not PyTorch)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, base64, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.WORK + '/release')
import gwphys as G, chirpnet
from realeval import prepare
from viz import cwt_energy

GWTC1 = {   # GWTC-1 Table III (source frame, 90% CI) and Table I network SNR (GstLAL)
    'GW150914': dict(label='GW150914', date='14 Sep 2015', mc=28.6, mc_p=1.7, mc_m=1.5, m1=35.6, m2=30.6, z=0.09, dl=440, snr=24.4, note='The first gravitational waves ever detected'),
    'LVT151012': dict(label='GW151012', date='12 Oct 2015', mc=15.2, mc_p=2.1, mc_m=1.2, m1=23.2, m2=13.6, z=0.21, dl=1080, snr=10.0, note='Faint enough that LIGO first called it only a candidate'),
    'GW151226': dict(label='GW151226', date='26 Dec 2015', mc=8.9, mc_p=0.3, mc_m=0.3, m1=13.7, m2=7.7, z=0.09, dl=450, snr=13.1, note='The Boxing Day event: lighter black holes, a longer chirp'),
    'GW170104': dict(label='GW170104', date='4 Jan 2017', mc=21.4, mc_p=2.2, mc_m=1.8, m1=30.8, m2=20.0, z=0.20, dl=990, snr=13.0, note='Three billion light-years away'),
}

def f16(a): return base64.b64encode(np.asarray(a, '<f2').tobytes()).decode()
def u8(a): return base64.b64encode(np.asarray(a, np.uint8).tobytes()).decode()

def main(npz, out):
    net = chirpnet.ChirpNet(npz); D = prepare(); meta = net.meta
    freqs = np.geomspace(20, 512, 96)
    ev_out = []
    for name, d in D.items():
        tev = d['tev'] - d['t0']; a = int(round((tev - 6) * G.FS)); n = 8 * G.FS
        H = d['H1'][a:a + n]; L = d['L1'][a:a + n]
        hits = net.scan(d['H1'], d['L1'], gps_start=d['t0'], whitened=True, stride=1 / 16)
        hit = min(hits, key=lambda h: abs(h['gps'] - d['tev'])) if hits else None
        starts = np.arange(0, n - G.WIN + 1, 128)
        x = np.stack([np.stack([H[s:s + G.WIN], L[s:s + G.WIN]]) for s in starts]).astype(np.float32)
        o, _c = net.statistic(x)
        spec = []
        for det in (H, L):
            E = cwt_energy(det, G.FS, freqs, Q=6).reshape(96, 512, -1).mean(-1)
            spec.append(u8(np.clip(E * 10, 0, 255)))
        # band-passed audio around the merger (30-350 Hz), from the whitened data
        seg = d['H1'][a + int(3 * G.FS): a + int(6.6 * G.FS)].astype(np.float64)
        F = np.fft.rfft(seg); f = np.fft.rfftfreq(len(seg), 1 / G.FS); F[(f < 30) | (f > 350)] = 0
        aud = np.fft.irfft(F, n=len(seg)); aud /= np.max(np.abs(aud)) + 1e-9
        ref = GWTC1[name]
        ev_out.append({'id': name, **ref, 'mc_det_ref': ref['mc'] * (1 + ref['z']),
                       'mc_det_lo': (ref['mc'] - ref['mc_m']) * (1 + ref['z']), 'mc_det_hi': (ref['mc'] + ref['mc_p']) * (1 + ref['z']),
                       'gps': d['tev'], 'H': f16(H), 'L': f16(L), 'specH': spec[0], 'specL': spec[1], 'audio': f16(aud),
                       'trace_t': (starts / G.FS + 1.25 - 6).round(4).tolist(), 'trace': o[:, 0].round(3).tolist(),
                       'hit': hit})
    # injection-lab templates: whitened, projected onto H1/L1, merger at 1.6 s in a 2 s window, unit network SNR
    import lal, lalsimulation as ls
    ps = G.training_psds(); f = np.fft.rfftfreq(16 * 4096, 1 / 4096)
    SH = np.interp(f, *ps['H1'][0]); SL = np.interp(f, *ps['L1'][0])
    Hd = lal.CachedDetectors[lal.LHO_4K_DETECTOR]; Ld = lal.CachedDetectors[lal.LLO_4K_DETECTOR]
    ra, dec, psi, gmst = 1.95, -1.27, 0.82, 0.0      # a sky position near GW150914's
    fpH, fxH = lal.ComputeDetAMResponse(Hd.response, ra, dec, psi, gmst); fpL, fxL = lal.ComputeDetAMResponse(Ld.response, ra, dec, psi, gmst)
    gps = lal.LIGOTimeGPS(1126259462); g0 = lal.GreenwichMeanSiderealTime(gps)
    dH = lal.TimeDelayFromEarthCenter(Hd.location, ra - gmst + g0, dec, gps); dL = lal.TimeDelayFromEarthCenter(Ld.location, ra - gmst + g0, dec, gps)
    MCS = [6, 7, 8, 9, 10, 12, 14, 16, 18, 21, 24, 27, 30, 34, 38, 43]; QS = [1.0, 0.6, 0.3]
    tpl = []; info = []
    for q in QS:
        for mc in MCS:
            m1, m2 = G.masses_from_mc_q(mc, q)
            hp, hc = ls.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, 0.0, 0, 0, 0.0, 400e6 * lal.PC_SI, 0.5, 0.0, 0, 0, 0, 1 / 16, 20.0, 2048.0, 20.0, None, ls.IMRPhenomD)
            Hp = hp.data.data[: len(f)] * 4096; Hc = hc.data.data[: len(f)] * 4096; sh = np.exp(-2j * np.pi * f * 8)
            sig = []
            for (fp, fx, S, dt) in ((fpH, fxH, SH, dH), (fpL, fxL, SL, dL)):
                X = (fp * Hp + fx * Hc) * sh * np.exp(-2j * np.pi * f * dt)
                sig.append(G.whiten_dft(X, 16 * 4096, 4096, S))
            env = sig[0] ** 2 + sig[1] ** 2; pk = int(np.argmax(env))
            s = np.stack([z[pk - int(1.6 * G.FS): pk - int(1.6 * G.FS) + G.WIN] for z in sig])
            s /= np.sqrt(G.PHI * np.sum(s ** 2))
            tpl.append(s.astype(np.float32)); info.append({'mc': mc, 'q': q, 'm1': round(m1, 1), 'm2': round(m2, 1)})
    tpl = np.stack(tpl)
    # ratchet log
    R = json.load(open(_P.WORK + '/ratchet.json'))
    rungs = [{'ckpt': r['ckpt'], 'note': r['note'], 'rejudged': bool(r.get('rejudged')), 'tta': bool(r.get('tta')), 'accepted': r['accepted'], 'why': r['why'], 'sens': r['val']['sens'], 'eff': r['val']['eff'],
              'mc_err': r['val']['mc_relerr_med'], 'cov68': r['val']['mc_cov68'], 't_err_ms': r['val']['t_abserr_ms'], 'q_err': r['val']['q_abserr_med'],
              'sens_glitch': r['sens_glitch'], 'real': {k: {'mc': v['mc'], 'lo': v['mc_lo'], 'hi': v['mc_hi'], 'score': v['score']} for k, v in r['real'].items()},
              'bg_max': r['bg']['max'], 'single_fa': r['val'].get('single_fa'), 'hard_bg': r.get('hard_bg')} for r in R]
    bg = net.bg
    data = {'events': ev_out, 'fs': G.FS, 'freqs': freqs.round(2).tolist(), 'seg_t0': -6,
            'tpl': f16(tpl), 'tpl_info': info, 'tpl_t_merge': 1.6, 'phi': G.PHI,
            'threshold': meta['threshold'], 'bg_hist': np.histogram(bg, bins=50, range=(-25, 5))[0].tolist(), 'bg_n': int(len(bg)), 'bg_max': float(bg.max()),
            'bg_live_s': meta['background_livetime_s'], 'rungs': rungs, 'release': meta.get('release', {})}
    json.dump(data, open(out, 'w'))
    print('web data MB %.2f' % (len(json.dumps(data)) / 1e6))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
