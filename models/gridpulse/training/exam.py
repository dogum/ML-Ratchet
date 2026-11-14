"""FINAL EXAM (run once, after every choice is frozen). Never used for choosing anything.
GB: 412 exam events from NESO 1 s data -> inertia vs NESO half-hourly outturn, MW lost by label tier, nadir/settling
forecasts from the first 5/10 s; baselines. ERCOT: summary tests from the operations reports (no traces).
Usage: python3 exam.py <rung> <ercot_head>  (widening factors from widen.json)"""
from common import *
import evaluate as EV, ercot_head as H, measure as M, gridpulse_np as GPN
import griddata as gd

W = json.load(open(f'{ROOT}/widen.json'))


def gb_inputs(ev):
    gm = gd.gb_genmix().set_index('time_utc')['GENERATION']
    rows, wins = [], []
    for _, r in ev.iterrows():
        try:
            m = EV.real_gb(r.event_id)
        except Exception as e:
            print('skip', r.event_id, e); continue
        t30 = pd.Timestamp(r.t0_utc).floor('30min')
        dem = float(gm.get(t30, np.nan))
        wins.append(m['w'][0]); rows.append(dict(event_id=r.event_id, year=int(r.year), demand=dem, nadir_meas=float(m['nadir'][0]),
                                                 settle_meas=float(m['settle'][0]), f0=float(m['f0'][0])))
    return pd.DataFrame(rows), np.array(wins)


def mdape(p, y):
    return float(np.median(np.abs(p / y - 1)) * 100)


def gb(rung):
    ev = gd.events('GB', split='exam').reset_index(drop=True)
    df, w = gb_inputs(ev)
    ev = ev.set_index('event_id').loc[df.event_id].reset_index()
    fac = np.array(W['gb_factors'])
    out = {}
    for N in (5, 10, 60):
        mu, sg = EV.predict(rung, w, np.full(len(w), N), np.ones(len(w)), np.full(len(w), 50.0), df.demand.values)
        sg = sg * fac
        df[f'loss_{N}'] = np.exp(mu[:, 0]); df[f'loss_sig_{N}'] = sg[:, 0]
        df[f'E_{N}'] = np.exp(mu[:, 1]) / 1e3; df[f'E_sig_{N}'] = sg[:, 1]
        df[f'beta_{N}'] = np.exp(mu[:, 2]); df[f'head_{N}'] = np.exp(mu[:, 3]); df[f'ffr_{N}'] = np.exp(mu[:, 4]) - 100
        df[f'nadir_{N}'] = mu[:, 5] / 10; df[f'nadir_sig_{N}'] = sg[:, 5] / 10
        df[f'settle_{N}'] = mu[:, 6] / 10; df[f'settle_sig_{N}'] = sg[:, 6] / 10
        mu2, sg2 = EV.predict(rung, w, np.full(len(w), N), np.ones(len(w)), np.full(len(w), 50.0), np.full(len(w), np.nan))
        df[f'E_trace_only_{N}'] = np.exp(mu2[:, 1]) / 1e3; df[f'loss_trace_only_{N}'] = np.exp(mu2[:, 0])
        # forecast baselines: running minimum so far; settling = mean of the last 3 s seen
        df[f'nadir_minsofar_{N}'] = w[:, M.PRE:M.PRE + N + 1].min(1)
        df[f'settle_persist_{N}'] = w[:, M.PRE + N - 2:M.PRE + N + 1].mean(1)
    for c in ['loss_MW', 'loss_confidence', 'inertia_outturn_GVAs', 'rocof_1s_Hz_s', 't_nadir_s', 'f0_Hz']:
        df[c] = ev[c].values
    E_dev = 230.0
    # demand-scaling baseline: E = k x demand, k = median over the 4 GB dev events of NESO E / demand
    k_dem = float(np.median([gd.events('GB', split='dev').set_index('event_id').loc[e].inertia_outturn_GVAs / EV.DEV_DEMAND[e]
                             for e in GB_DEV]))
    # ---- inertia
    s = df[df.inertia_outturn_GVAs.notna()]; y = s.inertia_outturn_GVAs.values
    inert = dict(n=int(len(s)))
    for N in (10, 60):
        z = np.abs(np.log(s[f'E_{N}'] / y)) / s[f'E_sig_{N}']
        inert[f'gridpulse_N{N}'] = dict(MdAPE=mdape(s[f'E_{N}'].values, y), bias_med_pct=float(np.median(s[f'E_{N}'] / y - 1) * 100),
                                        cov1=float((z <= 1).mean()), cov2=float((z <= 2).mean()))
        inert[f'trace_only_N{N}'] = dict(MdAPE=mdape(s[f'E_trace_only_{N}'].values, y),
                                         bias_med_pct=float(np.median(s[f'E_trace_only_{N}'] / y - 1) * 100))
    inert['baseline_constant_230'] = dict(MdAPE=mdape(np.full(len(y), E_dev), y), bias_med_pct=float(np.median(E_dev / y - 1) * 100))
    pd_ = k_dem * s.demand.values
    ok = np.isfinite(pd_)
    inert['baseline_k_x_demand'] = dict(k_s=k_dem * 1e3, n=int(ok.sum()), MdAPE=mdape(pd_[ok], y[ok]))
    lab = s[s.loss_MW.notna() & (s.rocof_1s_Hz_s > 0)]
    Er = lab.loss_MW / 1e3 * lab.f0_Hz / (2 * lab.rocof_1s_Hz_s)
    inert['baseline_rocof_with_label_loss'] = dict(n=int(len(lab)), MdAPE=mdape(Er.values, lab.inertia_outturn_GVAs.values),
                                                   gridpulse_same_events=mdape(lab.E_60.values, lab.inertia_outturn_GVAs.values))
    inert['by_year'] = {int(yr): dict(n=int(len(g)), gridpulse=mdape(g.E_60.values, g.inertia_outturn_GVAs.values),
                                      constant=mdape(np.full(len(g), E_dev), g.inertia_outturn_GVAs.values),
                                      median_true=float(g.inertia_outturn_GVAs.median()), median_pred=float(g.E_60.median()))
                        for yr, g in s.groupby('year')}
    s2 = s.assign(drop=-s.nadir_meas)
    bins = [(0, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 5)]
    inert['by_drop_Hz'] = {f'{a}-{b}': dict(n=int(((s2['drop'] >= a) & (s2['drop'] < b)).sum()),
                                             gridpulse=mdape(s2[(s2['drop'] >= a) & (s2['drop'] < b)].E_60.values,
                                                             s2[(s2['drop'] >= a) & (s2['drop'] < b)].inertia_outturn_GVAs.values))
                           for a, b in bins}
    out['inertia'] = inert
    # ---- MW lost by tier
    loss = {}
    for tier in ('medium', 'low'):
        g = df[(df.loss_confidence == tier) & (df.rocof_1s_Hz_s > 0)]
        if not len(g):
            continue
        y = g.loss_MW.values; gi = g[g.inertia_outturn_GVAs.notna()]
        z = np.abs(np.log(g.loss_60 / y)) / g.loss_sig_60
        loss[tier] = dict(n=int(len(g)), gridpulse_N60=mdape(g.loss_60.values, y), gridpulse_N10=mdape(g.loss_10.values, y),
                          trace_only_N60=mdape(g.loss_trace_only_60.values, y),
                          cov1=float((z <= 1).mean()), cov2=float((z <= 2).mean()),
                          bias_med_pct=float(np.median(g.loss_60 / y - 1) * 100),
                          rocof_assumed_E230=mdape(2 * E_dev * 1e3 * g.rocof_1s_Hz_s.values / g.f0_Hz.values, y),
                          rocof_true_E=mdape(2 * gi.inertia_outturn_GVAs.values * 1e3 * gi.rocof_1s_Hz_s.values / gi.f0_Hz.values, gi.loss_MW.values),
                          dumb_dev_median=mdape(np.full(len(y), float(gd.events('GB', split='dev').loss_MW.median())), y))
    out['loss'] = loss
    # ---- nadir / settling forecasts
    fc = {}
    for N in (5, 10):
        late = df.t_nadir_s > N
        for nm, sub in (('all', df), ('nadir_after_N', df[late])):
            zn = np.abs(sub[f'nadir_{N}'] - sub.nadir_meas) / sub[f'nadir_sig_{N}']
            zs = np.abs(sub[f'settle_{N}'] - sub.settle_meas) / sub[f'settle_sig_{N}']
            fc[f'N{N}_{nm}'] = dict(n=int(len(sub)),
                                    nadir_MAE_mHz=float(np.abs(sub[f'nadir_{N}'] - sub.nadir_meas).mean() * 1e3),
                                    nadir_baseline_minsofar_MAE_mHz=float(np.abs(sub[f'nadir_minsofar_{N}'] - sub.nadir_meas).mean() * 1e3),
                                    nadir_cov1=float((zn <= 1).mean()), nadir_cov2=float((zn <= 2).mean()),
                                    settle_MAE_mHz=float(np.abs(sub[f'settle_{N}'] - sub.settle_meas).mean() * 1e3),
                                    settle_baseline_persist_MAE_mHz=float(np.abs(sub[f'settle_persist_{N}'] - sub.settle_meas).mean() * 1e3),
                                    settle_cov1=float((zs <= 1).mean()), settle_cov2=float((zs <= 2).mean()))
    out['forecast'] = fc
    df.to_parquet(f'{ROOT}/logs/exam_gb_predictions.parquet', index=False)
    np.save(f'{ROOT}/logs/exam_gb_windows.npy', w.astype(np.float32))
    return out


def ercot(name):
    Z = np.load(f'{CKPT}/ercot_head_{name}.npz'); fac = W['ercot_factors']
    d = gd.events('ERCO', split='dev'); e = gd.events('ERCO', split='exam')
    e = e[(e.mw_loss > 0) & (e.delta_f_Hz > 0) & e.inertia_MWs.notna() & e.load_MW.notna()].reset_index(drop=True)
    e['renew_share_pct'] = e.renew_share_pct.fillna(d.renew_share_pct.median())
    T = H.features_from_events(e)
    k = float((d.mw_loss / d.delta_f_Hz).median())
    res = dict(n=int(len(e)), k_MW_per_Hz_dev=k)
    pred = {}
    for kk, (xs, y) in H.SPECS.items():
        Wk = {a.split('__')[1]: Z[a] for a in Z if a.startswith(kk + '__')}
        mu, sg = H.np_pred(Wk, T[xs].values); sg = sg * fac[kk]
        z = np.abs(mu - T[y].values) / sg; pred[kk] = np.exp(mu)
        res[kk] = dict(MdAPE=mdape(np.exp(mu), np.exp(T[y].values)), cov1=float((z <= 1).mean()), cov2=float((z <= 2).mean()),
                       bias_med_pct=float(np.median(np.exp(mu - T[y].values) - 1) * 100))
    res['loss']['baseline_k_delta_f'] = mdape(k * e.delta_f_Hz.values, e.mw_loss.values)
    res['loss']['baseline_dev_median'] = mdape(np.full(len(e), d.mw_loss.median()), e.mw_loss.values)
    res['dfnad']['baseline_loss_over_k'] = mdape(e.mw_loss.values / k, e.delta_f_Hz.values)
    res['E']['baseline_dev_median'] = mdape(np.full(len(e), d.inertia_MWs.median()), e.inertia_MWs.values)
    kd = float((d.inertia_MWs / d.load_MW).median())
    res['E']['baseline_k_x_load'] = mdape(kd * e.load_MW.values, e.inertia_MWs.values)
    res['by_year_loss'] = {int(yr): dict(n=int(len(g)), gridpulse=mdape(pred['loss'][g.index], g.mw_loss.values),
                                         k_delta_f=mdape(k * g.delta_f_Hz.values, g.mw_loss.values))
                           for yr, g in e.groupby(e.time_utc.dt.year)}
    e.assign(pred_loss=pred['loss'], pred_df=pred['dfnad'], pred_E=pred['E']).to_parquet(f'{ROOT}/logs/exam_ercot_predictions.parquet', index=False)
    return res


if __name__ == '__main__':
    rung, head = sys.argv[1], sys.argv[2]
    R = dict(rung=rung, ercot_head=head, widen=W, GB=gb(rung), ERCOT=ercot(head))
    json.dump(R, open(f'{ROOT}/exam.json', 'w'), indent=1, default=float)
    print(json.dumps(R, indent=1, default=float))
