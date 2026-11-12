"""Everything the TransitNet page shows, computed from the released checkpoint, the exam and the ratchet log."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, base64, numpy as np
sys.path.insert(0, _P.SRC); import pipeline as PL, tnet as TN, simbank as SB, model as M
D = _P.WORK + '/data/'; W = _P.LAB + '/'
f16 = lambda a: base64.b64encode(np.asarray(a, '<f2').tobytes()).decode()
f32 = lambda a: base64.b64encode(np.asarray(a, '<f4').tobytes()).decode()
r4 = lambda x, d=4: None if x is None or not np.isfinite(x) else round(float(x), d)

def pred_dict(P, j):
    return {k: r4(v[j], 5) for k, v in P.items()}

def raw_series(tic, P, T0, max_days=27.5, binm=30):
    """the star's first sector: PDCSAP flux, 30-minute bins, ppm; transit times for tick marks"""
    secs = PL.load(tic); s = secs[0]; t, f = s['t'], s['f']; b = np.floor((t - t[0]) / (binm / 1440)).astype(int); n = b.max() + 1
    cnt = np.bincount(b, minlength=n); sm = np.bincount(b, f, minlength=n); ok = cnt > 3; tb = (np.arange(n) + 0.5) * binm / 1440
    tt, ff = tb[ok], (sm[ok] / cnt[ok] - 1) * 1e6; marks = [float(m - t[0]) for m in np.arange(T0 - np.ceil((T0 - t[0]) / P) * P, t[-1], P) if m >= t[0]]
    return dict(t=f16(tt), f=f16(ff)), marks, s['sector'], len(secs)

def star_info(m):
    st = m['star']; return dict(rad=st['rad'] if st['have'] else None, teff=st['teff'] if st['have'] else None)

def main(ckpt):
    net = TN.load_net(ckpt); Mt = json.load(open(D + 'real_meta.json')); S = json.load(open(D + 'split.json')); Z = np.load(D + 'real.npz')
    allidx = list(range(len(Mt))); P = TN.predict(net, TN.real_arrays(allidx))
    ex, eidx, PE = TN.exam(net); R = json.load(open(_P.WORK + '/ratchet.json')); rel = [r for r in R if r['accepted']][-1]
    BL = json.load(open(D + 'baseline_fits.json')); bfit = dict(zip(BL['idx'], BL['fits']))
    name_idx = {Mt[i]['name']: i for i in S['exam']}
    def star_card(i, label, tag, desc):
        m = Mt[i]; raw, marks, sec, nsec = raw_series(m['tic'], m['info']['P'], m['info']['T0'])
        pub = dict(k=r4(m.get('k')), b=r4(m.get('b')), aR=r4(m.get('aR')), T14_h=r4(m.get('T14_h'))) if m['kind'] == 'planet' else None
        return dict(name=m['name'] + (f" · {m['host']}" if m.get('host') and m['host'] not in m['name'] else ''), label=label, tag=tag, desc=desc, kind=m['kind'], P=m['info']['P'], dur=m['dur'], scale=m['info']['scale'],
                    local=f32(Z['local'][i].ravel()), secondary=f32(Z['secondary'][i].ravel()), oddeven=f32(Z['oddeven'][i].ravel()), cent=f32(Z['cent'][i].ravel()) if net.use_cent else None,
                    raw=raw, marks=marks, rawLabel=f"TESS sector {sec}{f' (1 of {nsec} read)' if nsec > 1 else ''}, 30-minute bins; ticks mark the transits", pred=pred_dict(P, i), pub=pub, star=star_info(m))
    picks = json.load(open(W + 'picks.json'))
    stars = [star_card(name_idx[p['name']], p['label'], p['tag'], p['desc']) for p in picks['stars'] if p['name'] in name_idx]
    h = picks['hero']; hi = name_idx[h['name']]; hero = star_card(hi, h['name'], '', ''); hero.update(title=h['title'], note=h['note'])
    # lab stars: ordinary stars from the noise pool, detrended, one sector, 2-minute cadence
    labst = []
    for L in picks['lab']:
        s = PL.load(L['tic'])[0]; intr = np.zeros(len(s['t']), bool); fd = PL.detrend(s['t'], s['f'], intr, 0.75)
        c1, c2 = (np.nan_to_num(PL.detrend_c(s['t'], s[c], intr, 0.75)) for c in ('c1', 'c2')); st = PL.star_scalars(s['hdr'])
        labst.append(dict(name=L['name'], desc=L['desc'], sector=s['sector'], t=f32(s['t']), f=f16(fd - 1), c1=f16(c1), c2=f16(c2), tmag=st['tmag'], teff=st['teff'], logg=st['logg'], rho=st['rho'] if st['have'] else 1.4, have=st['have'], rad=st['rad'] if st['have'] else 1.0))
    # exam figures
    em = [Mt[i] for i in eidx]; kind = np.array([m['kind'] for m in em]); isp = kind == 'planet'; isf = kind == 'FP'
    pm = [m for m in em if m['kind'] == 'planet']; Pp = {k: v[isp] for k, v in PE.items()}; snr = np.array([m['info']['snr'] for m in pm])
    tags = lambda m: 'NEB' if ('NEB' in m['comments'] or 'centroid' in m['comments'].lower() or 'nearby' in m['comments'].lower()) else ('EB' if ('EB' in m['comments'] or 'SB' in m['comments'] or m['tess_disp'] == 'EB') else 'other')
    fpm = [m for m in em if m['kind'] == 'FP']; pf = PE['p_planet'][isf]; tg = np.array([tags(m) for m in fpm])
    cls = [dict(label='Confirmed planets', n=int(isp.sum()), called=float(np.mean(PE['p_planet'][isp] > 0.5)), hl=True),
           dict(label='False positives, all', n=int(isf.sum()), called=float(np.mean(pf > 0.5))),
           dict(label='  ...flagged by a centroid offset or a nearby star', n=int((tg == 'NEB').sum()), called=float(np.mean(pf[tg == 'NEB'] > 0.5))),
           dict(label='  ...eclipsing or spectroscopic binaries', n=int((tg == 'EB').sum()), called=float(np.mean(pf[tg == 'EB'] > 0.5))),
           dict(label='  ...other reasons', n=int((tg == 'other').sum()), called=float(np.mean(pf[tg == 'other'] > 0.5))),
           dict(label='False alarms (no real signal)', n=int((kind == 'FA').sum()), called=float(np.mean(PE['p_planet'][kind == 'FA'] > 0.5)))]
    kt, bt, at = (np.array([m.get(x, np.nan) for m in pm], float) for x in ('k', 'b', 'aR'))
    kp, bp, ap = 10 ** Pp['log_k'], Pp['b'], 10 ** Pp['log_aR']; kn = np.sqrt(np.clip([m['info']['depth'] for m in pm], 0, None))
    fit = [bfit.get(i) for i in [eidx[j] for j in np.where(isp)[0]]]; kf, bf, af = (np.array([f[x] if f else np.nan for f in fit], float) for x in ('k', 'b', 'aR'))
    med = lambda e, m: np.nanmedian(e[m]) if np.isfinite(e[m]).any() else np.nan
    par = []
    for lab, lo, hi in (('7 to 20', 7, 20), ('20 to 100', 20, 100), ('100 and up', 100, 1e9), ('all', 7, 1e9)):
        m = (snr >= lo) & (snr < hi)
        for who, kk, bb, aa in (('TransitNet', kp, bp, ap), ('Least-squares fit', kf, bf, af)):
            par.append(dict(label=f'{lab}: {who}', n=int(m.sum()), k=f"{100 * med(np.abs(kk / kt - 1), m):.1f}%", b=f"{med(np.abs(bb - bt), m):.2f}", a=f"{100 * med(np.abs(aa / at - 1), m):.1f}%", hl=who == 'TransitNet'))
        par.append(dict(label=f'{lab}: square root of the depth', n=int(m.sum()), k=f"{100 * med(np.abs(kn / kt - 1), m):.1f}%", b='—', a='—'))
    ex_all = ex['params_all']
    exam = dict(h=f"{ex['n_planets']:,} real planets, {ex['n_fp']:,} real impostors, published answers.", p_planet=[round(float(v), 4) for v in PE['p_planet'][isp]], p_fp=[round(float(v), 4) for v in pf],
                cls=cls, par=par, k=dict(pub=f16(kt), pred=f16(kp), snr=f16(np.minimum(snr, 6e4))), b=dict(pub=f16(bt), pred=f16(bp), snr=f16(np.minimum(snr, 6e4))), a=dict(pub=f16(at), pred=f16(ap), snr=f16(np.minimum(snr, 6e4))),
                auc=ex['auc'], coverage=dict(k=ex_all['k_in_1sigma'], b=ex_all['b_in_1sigma'], aR=ex_all['aR_in_1sigma']))
    W_ = json.load(open(W + 'texts.json'))
    data = dict(hero=hero, stars=stars, lab=dict(stars=labst), exam=exam, rungs=[], nStars=len(eidx), version=W_['version'])
    for i, r in enumerate(R):
        m = [['Score (sim exam)', f"{r['objective']:.3f}"], ['Dev log-loss', f"{r['dev']['logloss']:.3f}"], ['Dev Rp/R* error', f"{100 * r['dev']['k_frac_med']:.1f}%"],
             ['Exam AUC', f"{r['exam']['auc']:.3f}"], ['Exam Rp/R* error', f"{100 * r['exam']['params_all']['k_frac']['median']:.1f}%"], ['Exam b error', f"{r['exam']['params_all']['b_abs']['median']:.3f}"]]
        data['rungs'].append(dict(note=r['note'], accepted=r['accepted'], why=r['why'], released=r is rel, metrics=m))
    fmt = dict(ex=ex, exam=exam, rel=rel, bl=BL, n=len(eidx))
    for k, v in W_.items():
        if k == 'exam': data['exam'].update(v)
        elif k != 'version': data[k] = v
    json.dump(data, open(W + 'data.json', 'w')); print('web data MB %.2f' % (len(json.dumps(data)) / 1e6))

if __name__ == '__main__':
    main(sys.argv[1])
