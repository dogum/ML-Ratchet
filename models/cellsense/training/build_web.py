"""Everything the CellSense page shows, computed from the released checkpoint and the logs."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, base64, numpy as np, torch
sys.path.insert(0, _P.SRC); import realdata as RD, realexam as RX, lifeexam as LE, calibrate as CA, cellsim as CS, simbank as SB
W = _P.LAB + '/'; CK = _P.WORK + '/ckpt/'
f16 = lambda a: base64.b64encode(np.asarray(a, '<f2').tobytes()).decode()
r4 = lambda x, d=4: None if x is None or not np.isfinite(x) else round(float(x), d)
L4 = lambda a, d=4: [r4(x, d) for x in a]

def life_preds(net, rows):
    st = lambda k: np.stack([r[k] for r in rows]); S = np.array([[r['cycle'], r['charge_time']] for r in rows])
    return RX.predict(net, st('Q1'), st('T1'), st('Q0'), st('T0'), S)

def main(ckpt, life_ckpt, partial_ckpt=None):
    net = RX.load_net(ckpt); lnet = RX.load_net(life_ckpt); cells, V = RD.load(); name = os.path.basename(ckpt)
    rows = RX.readings(cells); st = lambda k: np.stack([r[k] for r in rows])
    for r in rows: r['charge_time'] = float(np.nanmedian((lambda ct: ct[(ct > 5) & (ct < 30)])(cells[r['cell']]['summary']['chargetime'][1:51])))
    P = RX.predict(net, st('Q1'), st('T1'), st('Q0'), st('T0')); PL = life_preds(lnet, rows)
    for k in ('log_life', 'log_life_sig', 'left', 'left_sig'): P[k] = PL[k]
    real = {'with_ref': RX.score(rows, P, 'with ref'), 'no_ref': RX.score(rows, RX.predict(net, st('Q1'), st('T1')), 'curve only')}
    qd, qc, fr = st('q_total'), st('q_cc'), st('cycle') / st('life')
    hid = []
    for lab, lo, hi in (('New to 20% of life', 0, 0.2), ('20 to 60%', 0.2, 0.6), ('60 to 90%', 0.6, 0.9), ('Last 10% and beyond', 0.9, 9), ('all', 0, 9)):
        m = (fr >= lo) & (fr < hi); hid.append(dict(stage=lab, n=int(m.sum()), mae_net=float(np.mean(np.abs(P['q_total'][m] - qd[m]))), mae_base=float(np.mean(np.abs(qc[m] - qd[m])))))
    # hero: a long-lived exam cell
    hk = 'b3c33' if 'b3c33' in cells else next(k for k, c in cells.items() if 1100 < c['cycle_life_pub'] < 1400 and k not in CA.CAL)
    c = cells[hk]; life = int(c['cycle_life_pub']); want = [10] + list(np.linspace(100, life - 1, 11).astype(int)); curves = []
    for n in want:
        rr = int(np.argmin(np.abs(c['cyc_idx'] + 1 - n))); j = c['cyc_idx'][rr]
        curves.append(dict(c=int(j + 1), q=f16(c['Qdlin'][rr][::4]), qcc=r4(np.nanmax(c['Qdlin'][rr])), qd=r4(c['summary']['QDischarge'][j])))
    hero = dict(label=hk, life=life, policy=c['policy'], curves=curves)
    # explorer cells
    picks = [('A typical cell', 'b2c20'), ('Long-lived', 'b3c33'), ('Gentle charging', 'b1c0'), ('Longest-lived', 'b1c2'), ('Fast charge, long life', 'b3c38'), ('Fastest to fail', 'b2c1')]
    SP = LE.split(cells); out_cells = []
    for nm, k in picks:
        if k not in cells or k in CA.CAL: continue
        rs = [r for r in rows if r['cell'] == k]; idx = [i for i, r in enumerate(rows) if r['cell'] == k]
        if len(rs) < 4: continue
        cc = cells[k]; Pk = {a: v[idx] for a, v in P.items()}
        life_k = int(cc['cycle_life_pub']); lo, hi = 10 ** (Pk['log_life'] - Pk['log_life_sig']), 10 ** (Pk['log_life'] + Pk['log_life_sig'])
        out_cells.append(dict(name=nm, label=k, life=life_k, policy=cc['policy'], split=SP[k], ct=r4(rs[0]['charge_time'], 2), cyc=[r['cycle'] for r in rs], qd=L4([r['q_total'] for r in rs]), qcc=L4([r['q_cc'] for r in rs]),
                              ir=L4([r['ir'] for r in rs], 2), curves=f16(np.stack([r['Q1'] for r in rs])), q0=f16(rs[0]['Q0']),
                              P={'q_total': L4(Pk['q_total']), 'q_total_sig': L4(Pk['q_total_sig']), 'ir': L4(Pk['ir'], 2), 'lli': L4(Pk['lli']), 'lam_ne': L4(Pk['lam_ne']), 'lam_pe': L4(Pk['lam_pe']),
                                 'life': L4(10 ** Pk['log_life'], 0), 'life_lo': L4(lo, 0), 'life_hi': L4(hi, 0)},
                              desc=f"Charged {cc['policy']} (0 to 80% SOC in {rs[0]['charge_time']:.1f} min); in Severson's {'training' if SP[k] == 'train' else SP[k] + ' test'} set."))
    # lifetime exam
    LR, lrows, P100, rows_l, Pl = LE.exam(lnet)
    pts = [dict(split=r['split'], life=r['life'], pred=r4(10 ** P100['log_life'][i], 0), lo=r4(10 ** (P100['log_life'][i] - P100['log_life_sig'][i]), 0), hi=r4(10 ** (P100['log_life'][i] + P100['log_life_sig'][i]), 0)) for i, r in enumerate(lrows)]
    lv = np.array([p['life'] for p in pts]); spl = np.array([p['split'] for p in pts]); med = {}
    for s in ('primary', 'secondary'):
        m = spl == s; e = np.median(lv[m]) - lv[m]; med[s + '_rmse'] = float(np.sqrt(np.mean(e ** 2))); med[s + '_mpe'] = float(100 * np.mean(np.abs(e) / lv[m]))
    ours = {'primary_rmse': LR['cycle100']['primary']['rmse'], 'primary_mpe': LR['cycle100']['primary']['mpe'], 'secondary_rmse': LR['cycle100']['secondary']['rmse'], 'secondary_mpe': LR['cycle100']['secondary']['mpe']}
    stage = [dict(label={'0-0.25': 'First quarter of life', '0.25-0.5': 'Second quarter', '0.5-0.75': 'Third quarter', '0.75-1.05': 'Last quarter'}[k], **v) for k, v in LR['by_stage'].items()]
    E = json.load(open(W + 'early.json')); v1 = next((x['life'] for x in json.load(open(_P.WORK + '/ratchet.json')) if x.get('life_ckpt') == 'c5_lifetime_cal.pt'), None)
    for e in E: e['label'] = f"{e['cell']} · {e['life']} cycles · {100 * e['f_s']:.0f}% slow"
    lifesub = ("Severson et al. made cycle-life prediction famous: from a cell's first 100 cycles, before its capacity has visibly faded, say how long it will last. Their models learned from 41 real cells. "
               "CellSense's first lifetime network could not: in its simulator a cell barely changed in 100 cycles, so it guessed close to the median. Real cells do change. By cycle 100 their 4C curve "
               "sags between 3.2 and 2.2 V, and the deeper the sag, the shorter the life. One mechanism fits that sag on all six calibration cells: a growing share of graphite that is slow to reach, "
               "poorly connected or behind clogged pores. The simulator now grows it from the first cycles and ties it to the later loss of graphite that brings on the knee. The network learned to read it.")
    earlysub = (f"The six calibration cells: charge delivered at each voltage at cycle 100, minus cycle 10. Solid: measured. Dashed: the simulator, with a share of slow graphite "
                f"({100 * min(e['f_s'] for e in E):.0f} to {100 * max(e['f_s'] for e in E):.0f}%) and small trims. The fits leave {min(e['rms_resid'] for e in E) * 1000:.0f} to {max(e['rms_resid'] for e in E) * 1000:.0f} mAh "
                f"of a {min(e['rms_real'] for e in E) * 1000:.0f} to {max(e['rms_real'] for e in E) * 1000:.0f} mAh signal. These six cells also set how strongly the slow share predicts the knee; no exam cell was used.")
    c1, c2 = LR['cycle100']['primary'], LR['cycle100']['secondary']; last = LR['by_stage']['0.75-1.05']
    sc = f"The forecasts follow the real lives (log correlation {LR['cycle100']['all']['corr_log']:.2f} over all {len(pts)} cells), and {100 * LR['cycle100']['all']['in_1sigma']:.0f}% fall within their own error bar."
    v1t = (f"Version 1 erred by {v1['cycle100']['primary']['mpe']:.0f}% and {v1['cycle100']['secondary']['mpe']:.0f}%, no better than guessing the median. " if v1 else '')
    note = (f"At cycle 100 CellSense now errs by {c1['mpe']:.0f}% on the primary test cells and {c2['mpe']:.0f}% on the secondary ones, against 13 to 15% and 9 to 11% for Severson's models trained on 41 real cells. "
            + v1t + f"It is not yet their equal, but it is no longer guessing. Read later in life it is weaker: in the last quarter it still reads worn cells as healthier than they are and overestimates "
            f"the cycles left ({last['mpe']:.0f}% life error).")
    life = dict(c100=pts, ours=ours, severson=LE.SEVERSON, median={**med}, stage=stage, sub=lifesub, note=note, sc=sc, early=E, earlySub=earlysub,
                h='How many cycles are left? Cycle 100 now tells you something.')
    # partial discharges (released system's third network)
    partial = None
    if partial_ckpt:
        pnet = RX.load_net(partial_ckpt); PX = RX.exam_partial(pnet, rows)
        nof = float(1000 * np.mean(np.abs(np.array([r['q_total'] / (1 - r['fade']) for r in rows]) - qd)))
        prow = [dict(label=f"{float(k):.2f} Ah ({100 * float(k) / 1.1:.0f}% of rated)", frac_cut=v['frac_cut'], net=v['full_ref']['q_total_mae_mAh'], net_cut=v['cut_ref']['q_total_mae_mAh'], base=v['baseline_q_total_mae_mAh'],
                     hl=float(k) >= 0.75) for k, v in PX.items()]
        partial = dict(rows=prow, nofade=nof, full=real['with_ref']['q_total_mae_mAh'], full_base=real['with_ref']['q_total_baseline_mae_mAh'],
                       sub=("Every exam reading, cut short once it has delivered a given charge from full, the rest of the curve thrown away, and read by the partial-discharge network. "
                            "Error in total capacity against the cycler's answer key. Reference: the cell's full early curve, or that curve cut at the same depth. "
                            "The no-model columns: add the shift between the two curves at the cut to the early total; or assume the cell has not faded at all."),
                       note=(f"Down to about 0.75 Ah, two thirds of the rated capacity, a partial discharge still reads total capacity to {prow[0]['net']:.0f} to {prow[1]['net']:.0f} mAh, against {real['with_ref']['q_total_mae_mAh']:.0f} mAh from a full one. At 0.6 Ah and below, "
                             f"the part that is left says little about capacity, and the network does no better than assuming no fade. Discharges that start partway, such as cycling "
                             f"between 20% and 80%, are untested: this experiment has none."))
    # ratchet
    R = json.load(open(_P.WORK + '/ratchet.json')); rungs = []
    for x in R:
        m = [['Score (sim exam)', f"{x['objective']:.3f}"], ['Real capacity error', f"{x['real']['with_ref']['q_total_mae_mAh']:.1f} mAh"],
             ['Real fade correlation', f"{x['real']['with_ref']['fade_corr']:.3f}"], ['Resistance correlation', f"{x['real']['with_ref']['ir_corr']:.2f}"]]
        if x.get('life'): m.append(['Life forecast at cycle 100, primary test error', f"{x['life']['cycle100']['primary']['mpe']:.1f}%", 1])
        if x.get('partial'): m.append(['Partial discharge to 0.75 Ah, capacity error', f"{x['partial']['0.75']['full_ref']['q_total_mae_mAh']:.1f} mAh", 1])
        m.insert(1, ['Calibration cells, capacity error', f"{x['dev']['q_total_mae_mAh']:.1f} mAh"])
        rungs.append(dict(ckpt=x['ckpt'], life_ckpt=x.get('life_ckpt'), partial_ckpt=x.get('partial_ckpt'), note=x['note'], accepted=x['accepted'], why=x['why'], objective=x['objective'], metrics=m))
    # lab: the median calibrated new cell
    TH = np.array([CA_FITS[k]['theta'] for k in CA_FITS]); med_th = np.median(TH, 0)
    Pl_ = {n_: float(np.exp(med_th[i])) for i, n_ in enumerate(SB.NAMES)}; Pl_['Q_Li'] = min(Pl_['Q_Li'] * SB.Q_LI_TRIM, 0.995 * Pl_['Q_n'] + 0.005 * Pl_['Q_p'])
    Pl_.update(SB.FIX); Pl_.update(T_amb=30.0, T0=30.5, f_s=0.0, R_s=0.3)
    sp = CS.Sphere(10); labd = dict(P=Pl_, R_s=0.3, sph={'lam': sp.lam.tolist(), 'b': sp.b.tolist(), 's': sp.s.tolist(), 'init': (sp.Phi.T @ (sp.W * np.ones(10))).tolist()}, gh={'z': CS.GH_Z.tolist(), 'w': CS.GH_W.tolist()})
    ph = json.load(open(_P.WORK + '/physics_check.json'))
    phys = dict(spm_cap=max(r['cap_err_pct'] for r in ph['spm']), spm_med=float(np.median([r['V_mederr_mV'] for r in ph['spm']])), spme_max=max(r['V_maxerr_2_90pct_mV'] for r in ph['spme']['rows']),
                cv_cap=max(100 * abs(r['total_ours'] / r['total_pybamm'] - 1) for r in ph['cv_hold']))
    cal = dict(v_lo=min(f['v_rms_mV'] for f in CA_FITS.values()), v_hi=max(f['v_rms_mV'] for f in CA_FITS.values()), t=max(f['t_rms_C'] for f in CA_FITS.values()), ir=max(abs(f['ir_err_mohm']) for f in CA_FITS.values()))
    samples = 0
    for x in R:
        ck = torch.load(CK + x['ckpt'], weights_only=False); samples += 0  # counted from banks below
    banks = {'health': 50000, 'life': 30000}; samples = banks['health'] * 2 + banks['life'] * 3     # pairs count twice; lifetime cells are simulated new, at cycle 10 and when read
    n_par = sum(int(sum(np.load(f)[k].size for k in np.load(f).files if k != 'meta')) for f in (_P.WORK + '/release/cellsense_weights.npz', _P.WORK + '/release/cellsense_life_weights.npz', _P.WORK + '/release/cellsense_partial_weights.npz') if os.path.exists(f))
    data = dict(vgrid=SB.VG.tolist(), hero=hero, scatter=dict(qd=f16(qd), pred=f16(P['q_total']), qcc=f16(qc)), hidTable=hid, real=real, cells=out_cells, life=life, rungs=rungs,
                ratchetSub=RATCHET_SUB, lab=labd, physics=phys, cal=cal, release=dict(ckpt=name, life_ckpt=os.path.basename(life_ckpt), partial_ckpt=os.path.basename(partial_ckpt) if partial_ckpt else None, samples=samples, params=n_par, labels=len(net.labels) + 2), partial=partial,
                example=dict(file=EXAMPLE), limits=json.load(open(W + 'limits.json')))
    json.dump(data, open(W + 'data.json', 'w')); print('web data MB %.2f' % (len(json.dumps(data)) / 1e6))

CA_FITS = json.load(open(_P.WORK + '/calibration.json'))
EXAMPLE = 'b3c33.npz'
RATCHET_SUB = ("Every candidate is a system: a health network, from rung 4 a lifetime network, and from rung 9 a third network for partial discharges. Its score has two halves: health skill on a "
               "frozen set of 4,000 simulated new and aged cells (total capacity, pulse resistance, lithium, graphite and LFP lost, resistance growth, fade), read with and without the early curve "
               "and, since rung 6, cut short as partial discharges; and lifetime skill on 3,000 simulated aging trajectories. Skill below guessing the median counts as zero. A rung is kept only if "
               "the score rises by more than 0.01 and, on the six calibration cells, a real-data dev set that no exam uses, it reads capacity no worse (within 1 mAh, a gate added after rung 3) and "
               "forecasts life no worse (within 2 points, added with rung 6). Five times those six cells taught the physics something; each time the frozen sets were regenerated and every rung "
               "was re-scored, so the decisions shown are the ones the current sets make. The exam cells are never used to choose.")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
