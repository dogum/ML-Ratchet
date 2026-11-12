"""Data for the page's early-life figure: each calibration cell's real change between cycles 10 and 100 (dQ_100-10, smoothed
over 24 mV) and the simulator's fit with slow graphite (early_calibration.json)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import realdata as RD, calibrate as CA, agesim as AS, simbank as SB, fit_life as FL, dq_early as DQ, calib_early as CE

def main():
    cells, V = RD.load(); F = json.load(open(_P.WORK + '/aging_calibration.json')); E = json.load(open(_P.WORK + '/early_calibration.json')); out = []
    for k in CA.CAL:
        f, e = F[k], E[k]; P0 = CA.params(np.array(FL.CA_FITS[k]['theta']), 30.0); P0['Q_Li'] = P0['Q_Li'] * SB.Q_LI_TRIM
        R = FL.rates_from(np.array(f['z'])[None], f['c']); tr = AS.trajectory(P0, R, n_max=110, kappa=np.array([f['kappa']]))
        Pa = AS.state_at(P0, R, tr, np.array([1])); Pa['f_s'][:] = 0.0; Q10 = CE.fill(SB.curves(Pa)[0])
        Pb = {a: v.copy() for a, v in Pa.items()}; Pb['f_s'][:] = e['f_s']; Pb['R_s'][:] = np.exp(e['log_R_s']); Pb['R_e'] = Pb['R_e'] * np.exp(e['log_m_e']); Pb['Q_Li'] = Pb['Q_Li'] * (1 - e['dlli'])
        sim = CE.sm(CE.fill(SB.curves(Pb)[0]) - Q10)[0]; real = CE.sm(DQ.real_dq(cells[k]))
        out.append(dict(cell=k, life=int(f['life']), f_s=round(e['f_s'], 4), real=np.round(real, 5).tolist(), sim=np.round(sim, 5).tolist(),
                        rms_real=float(np.sqrt(np.mean(real ** 2))), rms_resid=float(np.sqrt(np.mean((sim - real) ** 2)))))
        print(k, out[-1]['life'], 'f_s %.3f  explained %.0f%%' % (e['f_s'], 100 * (1 - (out[-1]['rms_resid'] / out[-1]['rms_real']) ** 2)), flush=True)
    json.dump(out, open(_P.LAB + '/early.json', 'w'))

if __name__ == '__main__':
    main()
