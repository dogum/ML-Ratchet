"""Physics checks: (1) kinematic orders vs the CWRU published table (kin.py); (2) the simulator's squared-envelope
spectrum puts its lines where the kinematics say: outer race at k*BPFO, inner race at k*BPFI with shaft-rate sidebands,
ball at 2*BSF with cage-rate sidebands (slip 0, jitter 1%, SNR +10 dB in real CWRU-normal background)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import bearsim as BS, features as FE, realdata as RD
c, ch, fr0 = RD.cwru_rec(RD.splits()['cwru_noise'][0]); bg = ch['DE']; g = RD.G['DE']; o = FE.orders(**g)
rng = np.random.default_rng(5); out = {}; F = np.fft.rfftfreq(FE.NWIN, 1 / FE.FS)
def peak(ses, f, tol=0.03):
    m = (F > f * (1 - tol)) & (F < f * (1 + tol)); i = np.argmax(np.where(m, ses, 0)); return F[i], ses[i] / np.median(ses[m | ((F > f * 0.7) & (F < f * 1.3))])
for cls, key, sb in (('OR', 'BPFO', None), ('IR', 'BPFI', 'shaft'), ('B', 'BSF', 'cage')):
    P = BS.draw_params(rng, cls); P.update(slip=0.0, jitter=0.01, snr_db=10.0, gate=False, eps=0.5, amin=0.05)
    x, tr = BS.simulate(rng, bg, fr0, cls=cls, geom=g, P=P, rho=1.0, distract=False, aperiodic=False); fr = tr['f_r']
    X = np.fft.rfft(x - x.mean()); ses, _ = FE.envelope_spectrum(X, 500, 5700)
    fdef = (2 * o['BSF'] if cls == 'B' else o[key]) * fr; rec = {}
    for k in (1, 2, 3):
        fp, pr = peak(ses, k * fdef, 0.02); rec['h%d' % k] = dict(theory_hz=round(k * fdef, 3), found_hz=round(float(fp), 3), rel_err=round(float(fp / (k * fdef) - 1), 5), prominence=round(float(pr), 1))
    if sb:
        fs_ = fr if sb == 'shaft' else o['FTF'] * fr
        for s in (-1, 1):
            fp, pr = peak(ses, fdef + s * fs_, 0.01); rec['sideband%+d' % s] = dict(theory_hz=round(fdef + s * fs_, 3), found_hz=round(float(fp), 3), prominence=round(float(pr), 1))
    out[cls] = rec; print(cls, json.dumps(rec))
out['kinematics_vs_CWRU_table'] = json.load(open(_P.WORK + '/checks/kinematics_check.json'))
out['bin_width_hz'] = FE.DF; json.dump(out, open(_P.WORK + '/checks/physics_check.json', 'w'), indent=1)
