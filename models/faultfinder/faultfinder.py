"""FaultFinder v1.1: air-handler / rooftop-unit fault detection and diagnosis from one day of BMS trend log, optionally
compared with the same unit's last 1-3 logged days. numpy only. Trained only on BuildingSim simulations (MODEL_CARD.md).

    from faultfinder import FaultFinder
    ff = FaultFinder()
    r = ff.read_day(df)                              # one local day, 1- to 15-min trend log, contract column names
    r = ff.read_day(df, ref_days=[df_jul9, df_jul10]) # with the same unit's earlier days (at most 7 days before; last 3 used)
    r['p_fault'], r['diagnosis'][:3]

Reference days are optional. Without them the network runs in single-day mode and gives v1.0's log-odds exactly (v1.1
= v1.0 + 47 input rows that are zero without references). Pass only days of the same unit, before the day being read,
within 7 days; their own fault status is not needed (a reference day may itself be faulty, as in operation).

Input channels (contract names, any subset; missing ones are allowed):
  T_oa T_ra T_ma T_sa (degC), T_sa_sp (or T_sa_sp_cool / T_sa_sp_heat), T_zone, damper_oa valve_cc valve_hc
  fan_sa_speed (0..1), fan_sa_power_kW (or any fan power/current proxy), fan_sa_status (0/1), occ (0/1),
  comp1_status / comp2_status (RTU compressor stages, used as the cooling signal when present).
Index: local time (pandas DatetimeIndex, or a 'time_local' column).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAWC = ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_sa_sp', 'T_zone', 'damper_oa', 'valve_cc', 'valve_hc', 'fan_speed',
        'fan_power', 'fan_on', 'occ']
R = {c: i for i, c in enumerate(RAWC)}
NT = 96
NF = 35
NF7 = 82          # v1.1 input: today (35) + reference median (35) + today - median (11) + reference count (1)
MAX_REF = 3
CLASSES = ['none', 'oa_damper_stuck', 'oa_damper_leak', 'economizer', 'cc_valve_stuck', 'cc_valve_leak',
           'hc_valve_stuck', 'hc_valve_leak', 'reheat_valve_leak', 'cooling_capacity', 'low_airflow',
           'sensor_bias_T_sa', 'sensor_bias_T_oa', 'sensor_bias_T_ma', 'sensor_bias_T_ra', 'zone_setpoint',
           'schedule', 'vav_box_stuck', 'plant_efficiency']
LABELS = {'none': 'no fault', 'oa_damper_stuck': 'outdoor-air damper stuck', 'oa_damper_leak': 'outdoor-air damper leaking',
          'economizer': 'economizer not working', 'cc_valve_stuck': 'cooling valve stuck',
          'cc_valve_leak': 'cooling valve leaking', 'hc_valve_stuck': 'heating valve stuck',
          'hc_valve_leak': 'heating valve leaking', 'reheat_valve_leak': 'VAV reheat valve leaking',
          'cooling_capacity': 'low cooling capacity (fouled coil / condenser, refrigerant charge)',
          'low_airflow': 'low supply airflow (filter / fan)', 'sensor_bias_T_sa': 'supply-air temperature sensor or setpoint offset',
          'sensor_bias_T_oa': 'outdoor-air temperature sensor offset', 'sensor_bias_T_ma': 'mixed-air temperature sensor offset',
          'sensor_bias_T_ra': 'return-air temperature sensor offset', 'zone_setpoint': 'zone setpoint / zone sensor offset',
          'schedule': 'operating schedule wrong', 'vav_box_stuck': 'VAV box damper stuck',
          'plant_efficiency': 'chiller / boiler efficiency loss'}


# ------------------------------------------------------------------ trend log -> raw day [13, 96]
def raw_day(df, day=None):
    """Contract trend log (pandas DataFrame, local time) -> raw day array [13, 96] (NaN = missing), 15-min means."""
    import pandas as pd
    d = df.copy()
    if 'time_local' in d:
        d.index = pd.DatetimeIndex(pd.to_datetime(d['time_local'].to_numpy()))
    n = len(d)
    g = lambda c: d[c].to_numpy(float) if c in d and d[c].notna().any() else np.full(n, np.nan)
    o = {}
    for c in ('T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_zone'):
        o[c] = g(c)
    sp = g('T_sa_sp')
    if not np.isfinite(sp).any():
        spc, sph, hc = g('T_sa_sp_cool'), g('T_sa_sp_heat'), np.nan_to_num(g('valve_hc'))
        sp = np.where((hc > 0.02) & np.isfinite(sph), sph, spc)
    o['T_sa_sp'] = np.where(sp < 1.0, np.nan, sp)
    o['damper_oa'] = g('damper_oa')
    if 'comp1_status' in d and d['comp1_status'].notna().any():
        o['valve_cc'] = 0.5 * (np.nan_to_num(g('comp1_status')) + np.nan_to_num(g('comp2_status')))
    else:
        o['valve_cc'] = g('valve_cc')
    o['valve_hc'] = g('valve_hc')
    o['fan_speed'] = g('fan_sa_speed')
    fp = g('fan_sa_power_kW')
    if not np.isfinite(fp).any():
        fp = g('x_fan_sa_current_A')
    o['fan_power'] = fp
    fs = g('fan_sa_status')
    if not np.isfinite(fs).any():
        spd = g('fan_sa_speed')
        fs = np.where(np.isfinite(spd), (spd > 0.05).astype(float), np.nan)
    o['fan_on'] = fs
    o['occ'] = g('occ')
    for c in ('T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_zone', 'T_sa_sp'):
        x = np.where((o[c] < -45) | (o[c] > 70), np.nan, o[c])
        v = x[np.isfinite(x)]
        if c != 'T_sa_sp' and len(v) > 10 and v.max() - v.min() < 1e-3:     # flatlined sensor = missing
            x = x * np.nan
        o[c] = x
    for c in ('damper_oa', 'valve_cc', 'valve_hc', 'fan_speed'):
        o[c] = np.clip(o[c], -0.05, 1.05)
    frame = pd.DataFrame(o, index=d.index)[RAWC]
    day = frame.index[0].normalize() if day is None else pd.Timestamp(day)
    sel = frame[(frame.index >= day) & (frame.index < day + pd.Timedelta(days=1))]
    x = np.full((len(RAWC), NT), np.nan, np.float32)
    if len(sel):
        b = ((sel.index - day).total_seconds() // 900).astype(int)
        gm = sel.groupby(b).mean()
        gm = gm[(gm.index >= 0) & (gm.index < NT)]
        x[:, gm.index.to_numpy()] = gm.to_numpy(np.float32).T
    return x


# ------------------------------------------------------------------ features (same as training/features.py)
_TT = np.arange(NT) / NT * 2 * np.pi
SIN, COS = np.sin(_TT), np.cos(_TT)


def _fill(v):
    m = np.isfinite(v)
    if m.all():
        return v
    if not m.any():
        return np.zeros_like(v)
    i = np.arange(len(v))
    return np.interp(i, i[m], v[m])


def featurize(x):
    x = np.asarray(x, np.float64)
    M = np.isfinite(x)
    avail = M.mean(axis=1) > 0
    f = np.zeros((NF, NT), np.float32)
    X = np.stack([_fill(x[i]) for i in range(len(RAWC))])
    fp = x[R['fan_power']]
    if np.isfinite(fp).any():
        p = np.nanpercentile(fp, 95)
        X[R['fan_power']] = X[R['fan_power']] / p if p > 1e-9 else 0.0
    k = 0
    for c in ('T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_sa_sp', 'T_zone'):
        f[k] = (X[R[c]] - 20.0) / 10.0 if avail[R[c]] else 0.0
        k += 1
    for a, b, s in (('T_ma', 'T_ra', 5.0), ('T_sa', 'T_ma', 5.0), ('T_sa', 'T_sa_sp', 3.0), ('T_oa', 'T_ra', 10.0),
                    ('T_zone', 'T_ra', 3.0)):
        f[k] = np.clip((X[R[a]] - X[R[b]]) / s, -5, 5) if (avail[R[a]] and avail[R[b]]) else 0.0
        k += 1
    if avail[R['T_oa']] and avail[R['T_ma']] and avail[R['T_ra']]:
        den = X[R['T_oa']] - X[R['T_ra']]
        ok = np.abs(den) > 3.0
        f[k] = np.clip(np.where(ok, (X[R['T_ma']] - X[R['T_ra']]) / np.where(ok, den, 1.0), 0.0), -0.5, 1.5)
        frm = ok.astype(np.float32)
    else:
        frm = np.zeros(NT, np.float32)
    k += 1
    for c in ('damper_oa', 'valve_cc', 'valve_hc', 'fan_speed', 'fan_power', 'fan_on', 'occ'):
        f[k] = np.clip(X[R[c]], -0.2, 2.0) if avail[R[c]] else 0.0
        k += 1
    f[k], f[k + 1] = SIN, COS
    k += 2
    f[k:k + 13] = M.astype(np.float32)
    f[k + 13] = frm
    return f


# ------------------------------------------------------------------ reference days (same as training/context.py)
def context(f_today, f_refs=None):
    """f_today [35, 96]; f_refs [n, 35, 96] (n = 0..3) or None -> [82, 96]: today's rows, the per-bin median of the
    reference days' rows, today minus that median for the 6 temperature and 5 physics rows, n / 3. No reference: zeros."""
    f_today = np.asarray(f_today, np.float32)
    out = np.zeros((NF7,) + f_today.shape[1:], np.float32)
    out[:NF] = f_today
    if f_refs is not None and len(f_refs):
        m = np.median(np.asarray(f_refs, np.float32), axis=0)
        out[NF:2 * NF] = m
        out[2 * NF:2 * NF + 11] = f_today[:11] - m[:11]
        out[-1] = len(f_refs) / 3.0
    return out


# ------------------------------------------------------------------ network (BatchNorm folded)
def _conv(x, W, b):
    K = W.shape[2]
    p = K // 2
    L = x.shape[2]
    xp = np.pad(x, ((0, 0), (0, 0), (p, p)))
    cols = np.stack([xp[:, :, k:k + L] for k in range(K)], axis=-1)
    return np.einsum('nclk,ock->nol', cols, W, optimize=True) + b[None, :, None]


def _pool(x):
    L = x.shape[2] // 2 * 2
    return x[:, :, :L].reshape(x.shape[0], x.shape[1], -1, 2).max(-1)


def forward(W, F):
    r = lambda z: np.maximum(z, 0)
    h = r(_conv(F, W['c1_W'], W['c1_b']))
    h = _pool(r(_conv(h, W['c2_W'], W['c2_b'])))
    h = _pool(r(_conv(h, W['c3_W'], W['c3_b'])))
    h = r(_conv(h, W['c4_W'], W['c4_b']))
    h = np.concatenate([h.mean(-1), h.max(-1)], 1)
    h = r(h @ W['fc_W'].T + W['fc_b'])
    return h @ W['cls_W'].T + W['cls_b'], h @ W['par_W'].T + W['par_b'], (h @ W['waste_W'].T + W['waste_b'])[:, 0]


class FaultFinder:
    def __init__(self, weights=os.path.join(HERE, 'faultfinder_weights.npz')):
        self.W = {k: v for k, v in np.load(weights).items()}
        self.T = float(self.W['temperature'])
        self.nf = int(self.W['c1_W'].shape[1])          # 82 (v1.1, reference days) or 35 (v1.0)

    def inputs(self, xs, refs=None):
        """xs [N, 13, 96] raw days; refs: None or one list per day of 0-3 raw reference days -> network input."""
        F = []
        for i, x in enumerate(xs):
            f = featurize(x)
            if self.nf > NF:
                rr = list(refs[i]) if refs is not None and refs[i] is not None else []
                f = context(f, [featurize(z) for z in rr[-MAX_REF:]])
            F.append(f)
        return np.stack(F).astype(np.float32)

    def predict_raw(self, xs, refs=None):
        """xs: [N, 13, 96] raw days; refs: optional, one list per day of the unit's earlier raw days (oldest first;
        the last 3 are used) -> list of result dicts."""
        lo, pa, wa = forward(self.W, self.inputs(xs, refs))
        f = lo[:, 1:]
        m = f.max(1, keepdims=True)
        d = (m[:, 0] + np.log(np.exp(f - m).sum(1))) - lo[:, 0]          # log-odds of any fault
        p = 1 / (1 + np.exp(-d / self.T))
        z = f / self.T
        e = np.exp(z - z.max(1, keepdims=True))
        pr = e / e.sum(1, keepdims=True)
        out = []
        for i in range(len(xs)):
            order = np.argsort(-pr[i])
            nref = min(len(refs[i]), MAX_REF) if (refs is not None and refs[i] is not None and self.nf > NF) else 0
            out.append(dict(p_fault=float(p[i]), log_odds=float(d[i]), n_reference_days=nref,
                            diagnosis=[(CLASSES[k + 1], LABELS[CLASSES[k + 1]], float(pr[i, k])) for k in order],
                            parameter=float(pa[i, order[0] + 1]), extra_hvac_energy_frac=float(wa[i])))
        return out

    def read_day(self, df, day=None, ref_days=None):
        """df: one day's trend log. ref_days: optional list of the same unit's earlier days, oldest first, each a
        DataFrame (its first local day is read) or a (DataFrame, day) pair."""
        refs = None
        if ref_days:
            refs = [[raw_day(*r) if isinstance(r, tuple) else raw_day(r) for r in ref_days]]
        return self.predict_raw(raw_day(df, day)[None], refs)[0]
