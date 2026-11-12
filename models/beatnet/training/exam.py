"""Final exam (run once, after every choice is frozen): BeatNet vs baselines on MIT-BIH DS2 (inter-patient exam), the
rest of DS1 and SVDB (extra held-out sets); QRS width on the QT database; prior choice and QRS calibration on dev only.
Writes exam.json."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); import bnet as BN, baselines as BL, train as TR, common as C
OUT = _P.WORK + '/exam.json'
PUBLISHED = {   # inter-patient DS2 results as reported (trained on real DS1 labels)
    'de Chazal et al. 2004 (linear discriminant, IEEE TBME 51:1196)': dict(se=[0.869, 0.759, 0.777, 0.894], ppv=[0.992, 0.385, 0.819, 0.086]),
    'Zhang et al. 2014 (feature selection + SVM, Comput Biol Med 48:1)': dict(se=[0.889, 0.791, 0.855, 0.938], ppv=[0.990, 0.360, 0.928, 0.138])}

def dev_prior(net, train_prev):
    """one scalar lam chosen on dev: logits += lam * log(prev_dev / prev_train), F prevalence floored at 0.5%"""
    _, idx, o = BN.real(net, 'dev'); y = BN.real_bank('mitdb')['y'][idx]; prev = np.bincount(y[y < 4], minlength=4) / (y < 4).sum(); prev = np.maximum(prev, 0.005); prev /= prev.sum()
    base = np.log(prev / train_prev); res = {}
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = BN.scores(y, BN.probs(o, lam * base).argmax(1)); res[lam] = s['sv_f1']
    lam = max(res, key=lambda k: (round(res[k], 3), -k)); return lam, (lam * base).tolist(), res, prev.tolist()

def qrs_cal(net):
    """QRS width: linear map from the simulated (vector) definition to the cardiologists' calipers, and a widen-only
    sigma factor, both fitted on the 8 QT-database dev records"""
    z = np.load(_P.WORK + '/data/qt_beats.npz'); meta = json.load(open(_P.WORK + '/data/qt_meta.json')); rec2split = {m['idx']: m['split'] for m in meta}; lead = {m['idx']: m['lead'] for m in meta}
    Z = {k: z[k] for k in z.files}; o = BN.predict(net, Z); q = np.exp(o[:, 4] * TR.QSD + TR.QMU); s = np.exp(np.clip(o[:, 5], -4, 2)) * TR.QSD * q
    dev = np.array([rec2split[r] == 'dev' for r in Z['rec']]); A = np.stack([np.ones(dev.sum()), q[dev]], 1); a, b = np.linalg.lstsq(A, Z['qrs'][dev], rcond=None)[0]
    pred = a + b * q; sig = s * abs(b); r = np.abs(pred[dev] - Z['qrs'][dev]) / sig[dev]; fac = max(1.0, float(np.percentile(r, 68.27)))
    ex = ~dev; ml = np.array([lead[r] == 'MLII' for r in Z['rec']]); cw = BL.qrs_width(Z['w'].astype(np.float32))
    ac, bc = np.linalg.lstsq(np.stack([np.ones(dev.sum()), cw[dev]], 1), Z['qrs'][dev], rcond=None)[0]; cls = ac + bc * cw
    def st(m): return dict(n=int(m.sum()), records=int(len(np.unique(Z['rec'][m]))), beatnet_mae=float(np.mean(np.abs(pred[m] - Z['qrs'][m]))), classical_mae=float(np.mean(np.abs(cls[m] - Z['qrs'][m]))),
                         median_guess_mae=float(np.mean(np.abs(np.median(Z['qrs'][dev]) - Z['qrs'][m]))), coverage_68=float(np.mean(np.abs(pred[m] - Z['qrs'][m]) < fac * sig[m])), truth_median=float(np.median(Z['qrs'][m])))
    return dict(a=float(a), b=float(b), sig_factor=fac, classical_a=float(ac), classical_b=float(bc), exam_all=st(ex), exam_MLII=st(ex & ml), exam_other_leads=st(ex & ~ml), dev=st(dev))

def det_scores(y, pred, missed):
    """y: class of the matched reference beat (0-3; 4 = Q, left out; 5 = false detection); missed: reference beats per class
    the detector never found. Confusion rows N S V F + 'no beat'; columns N S V F + 'missed'."""
    m = y != 4; y, pred = y[m], pred[m]; cm = np.zeros((5, 5), int); yy = np.where(y == 5, 4, y); np.add.at(cm, (yy, pred), 1); cm[:4, 4] = np.asarray(missed)[:4]
    se = np.diag(cm)[:4] / np.maximum(cm[:4].sum(1), 1); ppv = np.diag(cm)[:4] / np.maximum(cm[:, :4].sum(0), 1); f1 = 2 * se * ppv / np.maximum(se + ppv, 1e-9)
    return dict(cm=cm.tolist(), se=se.tolist(), ppv=ppv.tolist(), f1=f1.tolist(), macro_f1=float(np.mean(f1[cm[:4].sum(1) > 0])), sv_f1=float(np.mean(f1[1:3])))

def ece(p, y, nb=10):
    m = y < 4; p, y = p[m], y[m]; c = p.max(1); k = p.argmax(1) == y; e = 0; bins = []
    for lo in np.linspace(0, 1, nb + 1)[:-1]:
        s = (c >= lo) & (c < lo + 1 / nb)
        if s.any(): e += s.mean() * abs(k[s].mean() - c[s].mean()); bins.append([float(lo), int(s.sum()), float(c[s].mean()), float(k[s].mean())])
    return float(e), bins

if __name__ == '__main__':
    ck = sys.argv[1]; net = BN.load_net(ck); bank = TR.load(sys.argv[2]); tp = np.bincount(bank['y'], minlength=4) / len(bank['y'])
    lam, lp, lamres, prev = dev_prior(net, tp); R = dict(ckpt=ck, prior=dict(lam=lam, logprior=lp, dev_sv_f1_by_lam={str(k): v for k, v in lamres.items()}, dev_prevalence=prev, train_prevalence=tp.tolist()))
    ld = BL.fit_ld(bank)
    # classical AF baseline: RR irregularity (median |successive log-RR difference| over +-10 beats) above a threshold fitted on simulations
    irr = bank['rr'][:, 4]; isaf = bank['info'][:, 1] == 1; ths = np.quantile(irr, np.linspace(0.05, 0.95, 91))
    th_af = float(max(ths, key=lambda t: 0.5 * ((irr[isaf] > t).mean() + (irr[~isaf] <= t).mean()))); R['af_rule_threshold'] = th_af
    for split in ('ds2', 'ds1rest', 'svdb', 'dev'):
        db, idx = BN.split_idx(split); Z = BN.real_bank(db); y = Z['y'][idx]; o = BN.predict(net, Z, idx); p = BN.probs(o, lp)
        r = dict(beatnet=BN.scores(y, p.argmax(1)), beatnet_raw=BN.scores(y, o[:, :4].argmax(1)), majority=BN.scores(y, np.zeros(len(y), int)), rules=BN.scores(y, BL.rules(Z, idx)),
                 ld_sim=BN.scores(y, ld.predict(BL.ld_features(Z, idx))), counts=np.bincount(y, minlength=5).tolist())
        if o.shape[1] > 6 and db == 'mitdb':
            rh = BN.rhythm(db)[idx]; r['af'] = dict(beatnet=BN.af_scores(rh, o[:, 6]), rr_rule=BN.af_scores(rh, Z['rr'][idx, 4] - th_af), never=BN.af_scores(rh, -np.ones(len(rh))))
            afrec = {}
            for rec in np.unique(Z['rec'][idx]):
                m = Z['rec'][idx] == rec; afrec[int(rec)] = [float((rh[m] == 1).mean()), float((o[m, 6] > 0).mean()), float((Z['rr'][idx][m, 4] > th_af).mean())]
            r['af']['burden_per_record'] = afrec      # [reference AF share, BeatNet AF share, rule AF share]
            print(split, 'AF beatnet', {k: round(v, 3) for k, v in r['af']['beatnet'].items()}, 'rule', {k: round(v, 3) for k, v in r['af']['rr_rule'].items()}, flush=True)
        r['ece'], r['reliability'] = ece(p, y); per = {}
        for rec in np.unique(Z['rec'][idx]):
            m = Z['rec'][idx] == rec; per[int(rec)] = dict(counts=np.bincount(y[m], minlength=5).tolist(), cm=BN.scores(y[m], p[m].argmax(1))['cm'])
        r['per_record'] = per; R[split] = r
        print(split, 'BeatNet Se', np.round(r['beatnet']['se'], 3), '+P', np.round(r['beatnet']['ppv'], 3), '| rules S/V F1', np.round(r['rules']['f1'][1:3], 3), '| LD-sim S/V F1', np.round(r['ld_sim']['f1'][1:3], 3), flush=True)
        if split == 'ds2': np.save(_P.WORK + '/ds2_probs.npy', p.astype(np.float32))
    # same exam with beats located by our own QRS detector (detbank.py): false detections count against PPV, missed beats against Se
    zd = np.load(BN.D + 'mitdb_det_beats.npz'); Zd = {k: zd[k] for k in zd.files}; dmeta = {m['rec']: m for m in json.load(open(BN.D + 'mitdb_det_beats_meta.json'))}
    for split, recs in (('ds2_detector', C.DS2), ('dev_detector', C.dev_records())):
        idx = np.nonzero(np.isin(Zd['rec'], recs))[0]; y = Zd['y'][idx]; o = BN.predict(net, Zd, idx); p = BN.probs(o, lp); missed = np.sum([dmeta[r]['missed'] for r in recs], 0)
        R[split] = dict(beatnet=det_scores(y, p.argmax(1), missed), rules=det_scores(y, BL.rules(Zd, idx), missed), majority=det_scores(y, np.zeros(len(y), int), missed),
                        detector=dict(n_ref=int(np.sum([dmeta[r]['ref_counts'][:4] for r in recs])), missed=missed.tolist(), false_det=int((y == 5).sum())))
        print(split, 'BeatNet Se', np.round(R[split]['beatnet']['se'], 3), '+P', np.round(R[split]['beatnet']['ppv'], 3), 'missed', missed.tolist(), 'false', int((y == 5).sum()), flush=True)
    R['qrs'] = qrs_cal(net); R['published'] = PUBLISHED; print('QRS', json.dumps(R['qrs'], indent=0)); json.dump(R, open(OUT, 'w'), indent=1)
