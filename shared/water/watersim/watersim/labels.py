"""Labels for Phase 2 (BasinDetective, SnowSense, FlowCast): the true parameters of a simulated basin plus the
hidden states and signatures computed from the simulation. Signatures that real data also allow (BFI by filter,
recessions, runoff ratio, ...) are computed from the simulated outlet flow exactly as from a gauge record."""
import numpy as np
import pandas as pd

from .models import CAPACITY_LABEL
from . import signatures as sg


def _snow_stats(ts):
    swe = ts['SWE']
    wy = swe.index.year + (swe.index.month >= 10)
    peaks, melt_out, peak_doy = [], [], []
    for y, g in swe.groupby(wy):
        if len(g) < 330:
            continue
        peaks.append(g.max())
        if g.max() > 5:
            i = int(np.argmax(g.values))
            peak_doy.append(i)
            after = np.where(g.values[i:] < 1.0)[0]
            melt_out.append(i + int(after[0]) if len(after) else np.nan)
    return {
        'swe_peak_mean_mm': float(np.mean(peaks)) if peaks else 0.0,
        'swe_peak_wy_day_mean': float(np.nanmean(peak_doy)) if peak_doy else np.nan,
        'melt_out_wy_day_mean': float(np.nanmean(melt_out)) if np.isfinite(melt_out).any() else np.nan,
        'snow_days_frac': float((swe > 1.0).mean()),
    }


def compute_labels(out):
    ts = out['ts']
    if out.get('dt_days', 1.0) < 1.0 - 1e-9:   # signatures are defined on daily data
        num = ts.select_dtypes('number')
        state = [c for c in ('SWE', 'snow_liquid', 'SCA', 'soil', 'groundwater', 'routing_store', 'channel_storage',
                             'irrigation_store', 'res_storage_hm3', 'T_mean') if c in num]
        ts = num.drop(columns=state).resample('D').sum().join(num[state].resample('D').mean())
    model = out['model']
    pm = out['params']['model']
    sp = out['spec']
    P = ts['P'].values
    Q = ts['Q'].values
    psum = P.sum()
    lab = {
        'model': model, 'snow_model': out['snow'], 'routing': out['routing'],
        'params_model': dict(pm), 'params_snow': dict(out['params']['snow']),
        'params_routing': dict(out['params']['routing']),
        'soil_capacity_mm': float(CAPACITY_LABEL[model](pm)) if model in CAPACITY_LABEL else np.nan,
        'p_mean': float(P.mean()),
        'pet_mean': float(ts['PET'].mean()),
        'aridity': float(ts['PET'].sum() / psum) if psum > 0 else np.nan,
        'frac_snow': float(ts['snowfall'].sum() / psum) if psum > 0 else 0.0,
        # CAMELS/Caravan definition: share of precipitation on days with mean air temperature below 0 C
        'frac_snow_t0': float(ts['P_forcing'][ts['T_mean'] < 0].sum() / max(ts['P_forcing'].sum(), 1e-12)),
        'et_mean': float(ts['ET'].mean()),
        'evaporative_index': float(ts['ET'].sum() / psum) if psum > 0 else np.nan,
        'runoff_ratio': float(Q.sum() / psum) if psum > 0 else np.nan,
        'q_mean': float(Q.mean()),
        'soil_moisture_mean_mm': float(ts['soil'].mean()),
        'groundwater_mean_mm': float(ts['groundwater'].mean()),
        'recharge_mean': float(ts['recharge'].mean()),
        # model truth: share of outlet flow from the slow (groundwater/baseflow) component
        'baseflow_frac_model': float(ts['Q_base_routed'].sum() / max(ts['Q_natural'].sum(), 1e-12)),
        'quickflow_frac_model': float(ts['Q_quick_routed'].sum() / max(ts['Q_natural'].sum(), 1e-12)),
        'gw_exchange_mean': float(ts['exchange'].mean()),
        'bfi_lyne_hollick': sg.bfi(Q, 'lyne_hollick'),
        'bfi_eckhardt': sg.bfi(Q, 'eckhardt'),
        'flashiness_rb': sg.richards_baker(Q),
        'withdrawal_frac': float((ts['withdrawal'].sum() + ts['res_withdrawal_mm'].sum()) / max(ts['Q_natural'].sum(), 1e-12)),
    }
    rec = sg.recession(Q)
    lab.update({f'recession_{k}': v for k, v in rec.items()})
    lab.update(_snow_stats(ts))
    lab.update({f'sig_{k}': v for k, v in sg.camels_signatures(Q, P, ts.index).items()})
    # model-truth recession constants (per day)
    if model == 'hbv':
        lab['k_slow_true'] = pm['K2']
        lab['k_fast_true'] = pm['K1']
    elif model == 'sacsma':
        lab['k_slow_true'] = pm['LZPK']
        lab['k_fast_true'] = pm['UZK']
    elif model == 'gr4j':
        # GR4J's routing store is nonlinear; its linearised recession rate at the mean storage
        R = max(ts['groundwater'].mean(), 1e-6)
        X3 = pm['X3']
        lab['k_slow_true'] = float(1 - (1 + (R / X3) ** 4) ** -0.25 + R * (R / X3) ** 4 / R * (1 + (R / X3) ** 4) ** -1.25)
        lab['k_fast_true'] = np.nan
    elif model == 'vic_lite':
        lab['k_slow_true'] = pm['DS'] * pm['DSMAX'] / (pm['WS'] * pm['W2M'])
        lab['k_fast_true'] = np.nan
    reg = out.get('reservoirs') or []
    if reg:
        cap = sum(r['reservoir'].C for r in reg)
        vol = ts['Q_natural'].mean() * 365.25 * float(sp['area_km2']) / 1000.0  # hm3/yr
        lab['regulation_degree'] = float(cap / max(vol, 1e-9))
        lab['n_reservoirs'] = len(reg)
    else:
        lab['regulation_degree'] = 0.0
        lab['n_reservoirs'] = 0
    for k in ('aridity', 'p_mean', 'frac_snow'):
        if sp.get(k) is not None:
            lab[f'spec_{k}'] = sp[k]
    return lab
