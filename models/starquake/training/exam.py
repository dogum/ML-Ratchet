"""Final exam of the released rung on the 420 exam stars (never used for any decision), at five observing lengths,
with the classical and dumb baselines, coverage of the dev-calibrated error bars (published error added in
quadrature), results by numax stratum, and mass/radius against Yu et al. 2018 and Gaia DR3 FLAME radii."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import snet as SN
D = _P.WORK + '/data/'; STRATA = [(8, 15), (15, 25), (25, 40), (40, 70), (70, 130), (130, 250)]
NS, DS, TS = 3090.0, 135.1, 5777.0

def run(ckpt, widen):
    net = SN.load_net(_P.WORK + '/ckpt/' + ckpt); _, P, mt = SN.real(net, 'exam'); G = json.load(open(D + 'gaia.json')); BL = json.load(open(_P.WORK + '/baselines.json'))
    nm, dn = np.array([m['numax'] for m in mt]), np.array([m['dnu'] for m in mt]); L = np.array([m['L'] for m in mt])
    e1, e2 = P['ln_numax'] - np.log(nm), P['ln_dnu'] - np.log(dn); s1, s2 = P['sig_numax'] * widen[0], P['sig_dnu'] * widen[1]
    p1, p2 = np.array([m['e_numax'] for m in mt]) / nm, np.array([m['e_dnu'] for m in mt]) / dn
    c1 = np.nan_to_num(np.log(np.array([m['cl_numax'] for m in mt]) / nm), nan=9.0); c2 = np.nan_to_num(np.log(np.array([m['cl_dnu'] for m in mt]) / dn), nan=9.0)
    OUT = dict(ckpt=ckpt, widen=widen, lengths={}, rows=[])
    for l in (0, 365, 90, 54, 27):
        m = L == l; r = dict(n=int(m.sum()))
        for nmz, e, c, s, p, hit in (('numax', e1, c1, s1, p1, 0.05), ('dnu', e2, c2, s2, p2, 0.02)):
            r[nmz] = dict(med=float(np.median(np.abs(e[m]))), hit=float(np.mean(np.abs(e[m]) < hit)), out=float(np.mean(np.abs(e[m]) > (0.15 if nmz == 'numax' else 0.05))), bias=float(np.median(e[m])),
                          cov68=float(np.mean(np.abs(e[m]) < np.sqrt(s[m] ** 2 + p[m] ** 2))), cov95=float(np.mean(np.abs(e[m]) < 2 * np.sqrt(s[m] ** 2 + p[m] ** 2))), sig_med=float(np.median(s[m])),
                          cl_med=float(np.median(np.abs(c[m]))), cl_hit=float(np.mean(np.abs(c[m]) < hit)), cl_out=float(np.mean(np.abs(c[m]) > (0.15 if nmz == 'numax' else 0.05))),
                          dumb_med=BL[f'exam_{l}']['dumb'][f'{nmz}_med'])
            r[nmz]['strata'] = [dict(lo=lo, hi=hi, n=int((m & (nm >= lo) & (nm < hi)).sum()), med=float(np.median(np.abs(e[m & (nm >= lo) & (nm < hi)]))), cl_med=float(np.median(np.abs(c[m & (nm >= lo) & (nm < hi)])))) for lo, hi in STRATA]
        OUT['lengths'][str(l)] = r
    # mass and radius from the whole light curve, with Yu's Teff, vs Yu's no-correction solutions and Gaia FLAME
    m = L == 0; t = np.array([x['teff'] for x in mt])[m]; et = np.array([x['e_teff'] for x in mt])[m]
    a, b, tt = np.exp(P['ln_numax'][m]) / NS, np.exp(P['ln_dnu'][m]) / DS, t / TS
    M = a ** 3 * b ** -4 * tt ** 1.5; R = a * b ** -2 * tt ** 0.5
    eM = M * np.sqrt((3 * s1[m]) ** 2 + (4 * s2[m]) ** 2 + (1.5 * et / t) ** 2); eR = R * np.sqrt(s1[m] ** 2 + (2 * s2[m]) ** 2 + (0.5 * et / t) ** 2)
    yM, yR = np.array([x['M'] for x in mt])[m], np.array([x['R'] for x in mt])[m]; ok = np.isfinite(yM)
    kic = [x['kic'] for x in np.array(mt)[m]]; gR = np.array([G.get(str(k), {}).get('R_flame') or np.nan for k in kic], float)
    gl = np.array([G.get(str(k), {}).get('R_flame_lo') or np.nan for k in kic], float); gh = np.array([G.get(str(k), {}).get('R_flame_hi') or np.nan for k in kic], float); g = np.isfinite(gR)
    cR = (np.array([x['cl_numax'] for x in np.array(mt)[m]]) / NS) * (np.array([x['cl_dnu'] for x in np.array(mt)[m]]) / DS) ** -2 * tt ** 0.5
    OUT['mass_radius'] = dict(n=int(ok.sum()), M_med_frac=float(np.median(np.abs(M[ok] / yM[ok] - 1))), R_med_frac=float(np.median(np.abs(R[ok] / yR[ok] - 1))), M_bias=float(np.median(M[ok] / yM[ok] - 1)), R_bias=float(np.median(R[ok] / yR[ok] - 1)),
                              M_sig_med=float(np.median(eM / M)), R_sig_med=float(np.median(eR / R)),
                              gaia_n=int(g.sum()), gaia_R_ratio_med=float(np.median(R[g] / gR[g])), gaia_R_scatter=float(1.4826 * np.median(np.abs(R[g] / gR[g] - np.median(R[g] / gR[g])))),
                              gaia_yuR_ratio_med=float(np.median(yR[g] / gR[g])), gaia_yuR_scatter=float(1.4826 * np.median(np.abs(yR[g] / gR[g] - np.median(yR[g] / gR[g])))),
                              gaia_clR_ratio_med=float(np.nanmedian(cR[g] / gR[g])), gaia_clR_scatter=float(1.4826 * np.nanmedian(np.abs(cR[g] / gR[g] - np.nanmedian(cR[g] / gR[g])))))
    for i, x in enumerate(mt):
        OUT['rows'].append([x['kic'], x['L'], round(x['numax'], 3), round(x['dnu'], 4), round(float(np.exp(P['ln_numax'][i])), 3), round(float(np.exp(P['ln_dnu'][i])), 4), round(float(s1[i]), 4), round(float(s2[i]), 4),
                            round(x['cl_numax'], 3) if np.isfinite(x['cl_numax']) else None, round(x['cl_dnu'], 4) if np.isfinite(x['cl_dnu']) else None])
    OUT['mr_rows'] = [[int(k), round(float(r_), 3), round(float(y_), 3), None if not np.isfinite(gg) else round(float(gg), 3), round(float(mm), 3), round(float(ym), 3)] for k, r_, y_, gg, mm, ym in zip(kic, R, yR, gR, M, yM)]
    OUT['detection'] = detection(net)
    json.dump(OUT, open(_P.WORK + '/exam.json', 'w')); return OUT

def comb_stat(x2):
    import classical as C
    return np.array([C.collapsed(x.astype(float)).max() for x in x2])

def detection(net):
    """Detection on the exam: the 420 giants (should say yes) and 120 unscreened quiet dwarfs (should say no), per length.
    Network: p > 0.5 (fixed, not tuned). Classical: comb strength above a per-length threshold chosen on dev only
    (18 dev giants vs 60 dev dwarfs, best balanced accuracy)."""
    z = np.load(D + 'real.npz'); m = json.load(open(D + 'real_meta.json')); zn = np.load(D + 'realneg.npz'); mn = json.load(open(D + 'realneg_meta.json'))
    cp, cn = comb_stat(z['x2']), comb_stat(zn['x2']); out = {}
    for L in (0, 365, 90, 54, 27):
        sel = lambda M, sp: np.array([i for i, x in enumerate(M) if x['split'] == sp and x['L'] == L])
        dp, dn_, ep, en = sel(m, 'dev'), sel(mn, 'dev_neg'), sel(m, 'exam'), sel(mn, 'exam_neg')
        th = np.unique(np.concatenate([cp[dp], cn[dn_]])); ba = [0.5 * (np.mean(cp[dp] > t) + np.mean(cn[dn_] <= t)) for t in th]; t = float(th[int(np.argmax(ba))])
        pp = SN.real(net, 'exam', L)[1]['p_det']; pn = SN.real(net, 'exam_neg', L)[1]['p_det']
        out[str(L)] = dict(n_pos=int(len(ep)), n_neg=int(len(en)), tpr=float(np.mean(pp > 0.5)), fpr=float(np.mean(pn > 0.5)), cl_threshold=t,
                           cl_tpr=float(np.mean(cp[ep] > t)), cl_fpr=float(np.mean(cn[en] > t)))
    return out

if __name__ == '__main__':
    w = json.load(open(_P.WORK + '/sigma_calibration.json'))
    O = run(w['ckpt'], w['widen'])
    for l, r in O['lengths'].items():
        print(f"L={l:>4} n={r['n']}: numax med {r['numax']['med']:.4f} hit {r['numax']['hit']:.3f} out {r['numax']['out']:.3f} cov68 {r['numax']['cov68']:.2f} | classical med {r['numax']['cl_med']:.4f} hit {r['numax']['cl_hit']:.3f} | dumb {r['numax']['dumb_med']:.3f}")
        print(f"            dnu   med {r['dnu']['med']:.4f} hit {r['dnu']['hit']:.3f} out {r['dnu']['out']:.3f} cov68 {r['dnu']['cov68']:.2f} | classical med {r['dnu']['cl_med']:.4f} hit {r['dnu']['cl_hit']:.3f} | dumb {r['dnu']['dumb_med']:.3f}")
    print(json.dumps(O['mass_radius'], indent=1)); print(json.dumps(O['detection'], indent=1))
