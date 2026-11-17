"""Final exam on the 217 real/experimental exam runs (never used for choosing). Usage: python3 exam.py <model>
Calibration: one temperature T >= 1 on the detection log-odds, fitted on the 20 dev runs (widen-only).
Writes exam/exam.json and exam/exam_runs.parquet."""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, CLASSES, CI, EXAM2CLASS, load_real_runs, season_of, severity_bucket)  # noqa: E402
import evaluate as ev  # noqa: E402
from ratchet import load_model  # noqa: E402

SOURCE = {'ers_rp1312': 'Iowa ERS (lab)', 'lbnl_mzvav2_ers_exp': 'Iowa ERS (lab)',
          'lbnl_szcav_flexlab': 'LBNL FLEXLAB (lab)', 'lbnl_szvav_flexlab': 'LBNL FLEXLAB (lab)',
          'lbnl_rtu_ornl': 'ORNL RTU (lab)', 'nesbitt': 'Nesbitt Hall (real building)'}


def fit_temperature(d, t):
    """Widen-only temperature on detection log-odds (dev): minimise log loss over T in [1, 8]."""
    best = (1e9, 1.0)
    for T in np.linspace(1, 8, 141):
        p = np.clip(1 / (1 + np.exp(-d / T)), 1e-6, 1 - 1e-6)
        ll = -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))
        best = min(best, (ll, T))
    return float(best[1])


def ece(p, t, nb=5):
    bins = np.linspace(0, 1, nb + 1)
    e, rel = 0.0, []
    for a, b in zip(bins[:-1], bins[1:]):
        s = (p >= a) & (p < b if b < 1 else p <= b)
        if s.any():
            e += s.mean() * abs(p[s].mean() - t[s].mean())
            rel.append(dict(bin=f'{a:.1f}-{b:.1f}', n=int(s.sum()), mean_p=float(p[s].mean()), frac_fault=float(t[s].mean())))
    return float(e), rel


def boot_auc(t, s, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in ug}
    v = []
    for _ in range(n):
        idx = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
        a = ev.auc(t[idx], s[idx])
        if a == a:
            v.append(a)
    return [float(np.percentile(v, 5)), float(np.percentile(v, 95))]


def metrics(df, thr=0.5):
    t = df.is_fault.to_numpy()
    s = df.d.to_numpy()
    p = df.p.to_numpy()
    both = bool(t.any() and (~t).any())
    tpr = float((p[t] > thr).mean()) if t.any() else None
    fpr = float((p[~t] > thr).mean()) if (~t).any() else None
    o = dict(n=len(df), n_fault=int(t.sum()), auc=ev.auc(t, s), accuracy=float(((p > thr) == t).mean()),
             TPR=tpr, FPR=fpr, balanced_accuracy=0.5 * (tpr + 1 - fpr) if both else None)
    for fa in (0.02, 0.05, 0.10):
        o[f'recall_at_{int(fa * 100)}pct_FA'] = ev.recall_at_fpr(t, s, fa) if both else None
    return o


def ers_energy_proxy(run_ids):
    """Air-side coil + fan energy proxy (kWh) of ERS runs from the real trend: supply flow x rho cp x
    |T_sa - T_ma - 0.8 K fan heat| + fan power, while the fan runs."""
    import buildingdata as bd
    d = bd.fault_data('ers_rp1312', list(run_ids))
    out = {}
    for rid, g in d.groupby('run_id'):
        on = g.fan_sa_status.to_numpy() > 0.5
        q = g.flow_sa_m3s.to_numpy() * 1.2 * 1.006 * np.abs(g.T_sa.to_numpy() - g.T_ma.to_numpy() - 0.8)
        fan = g.fan_sa_power_kW.to_numpy() + g.fan_ra_power_kW.to_numpy()
        out[rid] = float(np.nansum((q + fan)[on]) / 60.0)
    return out


def main(name, split='exam'):
    os.makedirs(f'{ROOT}/exam', exist_ok=True)
    model = load_model(name)
    dev = load_real_runs('dev')
    lo_d, _, _ = ev.run_predict(model, dev)
    t_dev = np.array([r.fault_type != 'none' for r, _ in dev], float)
    T = fit_temperature(ev.det_logit(lo_d), t_dev)
    runs = load_real_runs(split)
    lo, pa, wa = ev.run_predict(model, runs)
    d = ev.det_logit(lo)
    p = 1 / (1 + np.exp(-d / T))
    pr = ev.softmax(lo[:, 1:], T)          # same widen-only dev temperature on the diagnosis logits
    rows = []
    for i, (r, x) in enumerate(runs):
        cls = EXAM2CLASS.get(r.fault_type)
        top = np.argsort(-pr[i])[:3] + 1
        rows.append(dict(run_id=r.run_id, facility=r.facility, dataset=r.dataset, source=SOURCE[r.dataset], unit=r.unit, date=str(r.date),
                         season=season_of(r.date), fault_type=r.fault_type, severity=str(r.severity),
                         sev_bucket=severity_bucket(r.fault_type, r.severity), is_fault=r.fault_type != 'none',
                         diag_class=cls if cls is not None else ('other' if r.fault_type != 'none' else 'none'),
                         d=float(d[i]), p=float(p[i]), pred1=CLASSES[top[0]], pred3=[CLASSES[k] for k in top],
                         par_pred=float(pa[i, top[0]]), conf1=float(pr[i, top[0] - 1]), probs=[round(float(q), 4) for q in pr[i]], waste_pred=float(wa[i]), group=r.group,
                         rtu=r.unit == 'RTU'))
    df = pd.DataFrame(rows)
    df.to_parquet(f'{ROOT}/exam/{split}_runs.parquet', index=False)
    t = df.is_fault.to_numpy()
    res = dict(model=name, temperature_dev=T, n=len(df))
    res['all'] = metrics(df)
    res['all']['auc_90ci'] = boot_auc(t, df.d.to_numpy(), df.group.to_numpy())
    res['ahu'] = metrics(df[~df.rtu])
    res['ahu']['auc_90ci'] = boot_auc(df[~df.rtu].is_fault.to_numpy(), df[~df.rtu].d.to_numpy(), df[~df.rtu].group.to_numpy())
    res['rtu'] = metrics(df[df.rtu])
    res['by_source'] = {k: metrics(g) for k, g in df.groupby('source')}
    res['by_dataset'] = {k: metrics(g) for k, g in df.groupby('dataset')}
    res['by_season'] = {k: metrics(g) for k, g in df.groupby('season')}
    # severity: detection rate of faulty runs per bucket (normals give the false-alarm side)
    res['by_severity'] = {k: dict(n=len(g), detected_at_0p5=float((g.p > 0.5).mean()),
                                  auc_vs_all_normals=ev.auc(np.r_[np.ones(len(g)), np.zeros((~df.is_fault).sum())].astype(bool),
                                                            np.r_[g.d, df[~df.is_fault].d]))
                          for k, g in df[df.is_fault].groupby('sev_bucket')}
    res['detection_by_fault_type'] = {k: dict(n=len(g), detected_at_0p5=float((g.p > 0.5).mean()), mean_p=float(g.p.mean()))
                                      for k, g in df[df.is_fault].groupby('fault_type')}
    # diagnosis
    dg = df[df.is_fault & ~df.diag_class.isin(['other'])]
    res['diagnosis'] = dict(n=len(dg), top1=float((dg.pred1 == dg.diag_class).mean()),
                            top3=float(np.mean([c in p3 for c, p3 in zip(dg.diag_class, dg.pred3)])),
                            chance_majority=float(dg.diag_class.value_counts(normalize=True).iloc[0]),
                            majority_class=str(dg.diag_class.value_counts().index[0]),
                            by_class={k: dict(n=len(g), top1=float((g.pred1 == k).mean()),
                                              top3=float(np.mean([k in p3 for p3 in g.pred3])),
                                              predicted=g.pred1.value_counts().to_dict()) for k, g in dg.groupby('diag_class')},
                            by_source={k: dict(n=len(g), top1=float((g.pred1 == g.diag_class).mean())) for k, g in dg.groupby('source')})
    res['n_other_excluded_from_diagnosis'] = int((df.is_fault & (df.diag_class == 'other')).sum())
    nes = df[df.dataset == 'nesbitt']
    res['nesbitt'] = dict(detection=metrics(nes),
                          by_fault_type={k: dict(n=len(g), mean_p=float(g.p.mean()), detected_at_0p5=float((g.p > 0.5).mean()))
                                         for k, g in nes.groupby('fault_type')})
    # diagnosis calibration: top-1 class probability (given a fault) vs whether it was right
    if len(dg):
        conf = dg.conf1.to_numpy()
        ok = (dg.pred1 == dg.diag_class).to_numpy().astype(float)
        e2, rel2 = ece(conf, ok)
        res['diagnosis']['ece_top1'] = e2
        res['diagnosis']['reliability_top1'] = rel2
    # calibration
    e, rel = ece(df.p.to_numpy(), t.astype(float))
    res['calibration'] = dict(ece_5bin=e, ece_ahu=ece(df[~df.rtu].p.to_numpy(), df[~df.rtu].is_fault.to_numpy().astype(float))[0],
                              ece_rtu=ece(df[df.rtu].p.to_numpy(), df[df.rtu].is_fault.to_numpy().astype(float))[0], brier=float(np.mean((df.p - t) ** 2)), reliability=rel,
                              brier_uncalibrated=float(np.mean((1 / (1 + np.exp(-df.d)) - t) ** 2)))
    # baselines
    import buildingdata as bd
    bs_ = bd.baselines('faults_summary') if False else json.load(open(os.path.join(__import__('common').BLDG, 'data', 'proc', 'baselines', 'faults_summary.json')))
    ap = bd.baselines('faults_apar')
    ap = ap[ap.split == split]
    m = df.merge(ap[['run_id', 'score']], on='run_id', how='left')
    res['baseline_apar'] = dict(ahu_auc=bs_['ahu_all']['exam_auc'], rtu_auc=bs_['rtu']['exam_auc'],
                                ahu_by_dataset=bs_['ahu_all']['exam_auc_by_dataset'],
                                all_auc_same_runs=ev.auc(m.is_fault, m.score.fillna(0)),
                                recall_at_2pct_FA_ahu=ev.recall_at_fpr(m[~m.rtu].is_fault.to_numpy(), m[~m.rtu].score.fillna(0).to_numpy()),
                                by_source={k: ev.auc(g.is_fault, g.score.fillna(0)) for k, g in m.groupby('source')},
                                rule_ahu=bs_['ahu_all']['rule'], rule_rtu=bs_['rtu']['rule'])
    thr_ap = np.where(m.rtu, bs_['rtu']['threshold_dev'], bs_['ahu_all']['threshold_dev'])
    mb = m.assign(d=m.score.fillna(0), p=(m.score.fillna(0) > thr_ap).astype(float))
    res['baseline_apar']['metrics'] = dict(all=metrics(mb), ahu=metrics(mb[~mb.rtu]), rtu=metrics(mb[mb.rtu]),
                                           by_source={k: metrics(g) for k, g in mb.groupby('source')})
    res['baseline_majority'] = dict(ahu=bs_['ahu_all']['majority'], rtu=bs_['rtu']['majority'],
                                    all_accuracy=float(max(t.mean(), 1 - t.mean())))
    # energy waste on ERS paired twins (fault AHU-A vs healthy AHU-B same day)
    ers = df[df.dataset == 'ers_rp1312']
    if len(ers):
        E = ers_energy_proxy(ers.run_id)
        pairs = []
        for date, g in ers.groupby('date'):
            a, b = g[g.unit == 'AHU-A'], g[g.unit == 'AHU-B']
            if len(a) and len(b):
                ea, eb = E[a.run_id.iloc[0]], E[b.run_id.iloc[0]]
                pairs.append(dict(date=date, fault=a.fault_type.iloc[0], real_extra=(ea - eb) / max(eb, 1e-6),
                                  pred_A=a.waste_pred.iloc[0], pred_B=b.waste_pred.iloc[0]))
        pp = pd.DataFrame(pairs)
        from scipy.stats import spearmanr
        res['ers_energy_pairs'] = dict(n=len(pp), spearman_pred_vs_real=float(spearmanr(pp.pred_A - pp.pred_B, pp.real_extra)[0]),
                                       median_real_extra=float(pp.real_extra.median()),
                                       median_pred=float((pp.pred_A - pp.pred_B).median()), pairs=pairs)
    json.dump(res, open(f'{ROOT}/exam/{split}.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k in ('all', 'ahu', 'rtu', 'by_source', 'diagnosis', 'calibration', 'temperature_dev')},
                     indent=1, default=float)[:6000])
    return res


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'exam')
