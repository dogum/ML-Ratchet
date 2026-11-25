"""Reference rows that are not arms (computed by the harness for the dev gate and the exam). Each gives a per-record
trip energy prediction in MJ (whole trip, before the VED valid-share scaling that score.real_units applies), NaN where
the reference has no prediction; all are point predictions (q05..q95 = pred).

  ephys        EnergyNet's generic physics estimate E_phys at the record's tier (the table track's input log E_phys)
  label        VED: fueleconomy.gov window-sticker label x distance (RoadData baseline, fitted on dev)
  moves_like   VED: VSP operating-mode rates per vehicle type (MOVES-like; RoadData baseline, fitted on dev)
  class_median VED: dev median energy per km per (VED type, body class) (RoadData baseline)
  regression   EPA: log-linear regression on dev test cars, log y ~ log test mass + log road-load force at 50 mph
               (+ log displacement + log rated hp for fuel); y = gal/mi or wall kWh/mi (RoadData / EnergyNet epa_fix)
  dev_median   EPA: dev median per architecture and test; Mersin: dev median fuel per trip (same route) per fuel;
               ICCT: the dev truck's value on the same cycle; ZTBus: pooled dev kWh/km x distance (the 294 dev
               missions of the full set, from their metadata: energy and distance only)
The dumb baseline (climatology, sim-train median energy per km per architecture x class) and EnergyNet v2.1 as
released are registered reference ARMS in arms_ref/ (they run through the same pipeline as every arm).
"""
import functools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GAL = 3.785411784
REF_ROWS = {
    'ephys': 'EnergyNet physics estimate E_phys alone (generic road-load model, no network)',
    'label': 'VED: window-sticker label x distance (RoadData, dev-fitted mix)',
    'moves_like': 'VED: MOVES-like VSP operating-mode model (RoadData, dev-fitted)',
    'class_median': 'VED: dev median energy per km per (type, class) (RoadData)',
    'regression': 'EPA: dev-fitted log-linear regression on the test car spec',
    'dev_median': 'EPA / Mersin / ICCT / ZTBus: dev median (EPA, Mersin), dev truck same cycle (ICCT), dev kWh/km x distance (ZTBus)',
}
EPA_REG = [('conv', 'FTP', 'FTP_mpg_fix'), ('conv', 'HWY', 'HWY_mpg_fix'), ('conv', 'US06', 'US06_mpg_fix'),
           ('conv', 'FTPcold', 'FTP_cold_mpg'), ('conv', 'SC03', 'SC03_mpg'),
           ('hev', 'FTP', 'FTP_mpg_fix'), ('hev', 'HWY', 'HWY_mpg_fix'), ('hev', 'US06', 'US06_mpg_fix'),
           ('hev', 'FTPcold', 'FTP_cold_mpg'), ('hev', 'SC03', 'SC03_mpg'),
           ('bev', 'FTP', 'CD_UDDS_kWh_per_mi'), ('bev', 'HWY', 'CD_HWY_kWh_per_mi')]


@functools.lru_cache(maxsize=1)
def _epa_table():
    import realdata as rd
    from build_real import epa_fixed
    e = rd.epa_testcars().join(epa_fixed())
    return e.assign(F50=e.A_N + e.B_N_per_mps * 22.352 + e.C_N_per_mps2 * 22.352 ** 2)


@functools.lru_cache(maxsize=1)
def epa_models():
    """dev fits: {(arch, test): dict(beta, features, median)} (y = gal/mi for fuel, kWh/mi for BEV)."""
    e = _epa_table(); out = {}
    for arch, test, col in EPA_REG:
        d = e[(e.architecture == arch) & e[col].notna() & (e[col] > 0) & e.F50.gt(0) & e.test_mass_kg.gt(0)].copy()
        if arch != 'bev':
            d = d[d.displacement_L.gt(0) & d.rated_hp.gt(0) & d.fuel.eq('gasoline')]
        d['y'] = d[col] if arch == 'bev' else 1 / d[col]
        dv = d[d.split == 'dev']
        if len(dv) < 10:
            continue
        Xc = ['test_mass_kg', 'F50'] + ([] if arch == 'bev' else ['displacement_L', 'rated_hp'])
        X = np.c_[np.ones(len(dv)), np.log(dv[Xc].values.astype(float))]
        beta, *_ = np.linalg.lstsq(X, np.log(dv.y.values), rcond=None)
        out[(arch, test)] = dict(beta=beta.tolist(), features=Xc, median=float(dv.y.median()), n_dev=int(len(dv)))
    return out


def _epa_unit_value(vids, arch, test, kind):
    e = _epa_table(); m = epa_models().get(({'hev_powersplit': 'hev'}.get(arch, arch), test))
    if m is None:
        return np.full(len(vids), np.nan)
    if kind == 'dev_median':
        return np.full(len(vids), m['median'])
    r = e.reindex(vids)
    X = r[m['features']].values.astype(float)
    ok = np.all(X > 0, 1) & np.isfinite(X).all(1)
    p = np.full(len(vids), np.nan)
    p[ok] = np.exp(np.c_[np.ones(ok.sum()), np.log(X[ok])] @ np.asarray(m['beta']))
    return p


@functools.lru_cache(maxsize=1)
def ztbus_dev_kpk():
    """pooled kWh/km of the 294 dev missions of the full ZTBus set (split file from the repo, metadata table local)."""
    sp = json.load(open(C.ZTB_FULL_SPLITS))['missions']
    M = pd.read_parquet(os.path.join(C.ROAD_DATA, 'proc', 'hd', 'ztbus_missions.parquet'))
    d = M[M.name.map(sp) == 'dev']
    return float(d.kWh.sum() / (d.drivenDistance.sum() / 1000)), int(len(d))


@functools.lru_cache(maxsize=1)
def _hd_dev():
    import realdata as rd
    h = rd.hd_data()
    m = h['mersin_trips']; m = m[m.q_label_ok & (m.split == 'dev')]
    t = h['icct_trucks']; t = t[t.cycle_available & (t.split == 'dev')]
    return {f: float(g.fuel_g.median()) for f, g in m.groupby('fuel')}, t.set_index('cycle').g_per_km.to_dict()


def reference_preds(Fr):
    """Fr: bake.real_frame(...) -> {ref name: per-record MJ array (NaN = no prediction)}."""
    n = len(Fr); out = {k: np.full(n, np.nan) for k in REF_ROWS}
    out['ephys'] = Fr.E_phys.values.astype(float)
    v = (Fr.recset == 'ved').values
    if v.any():
        vr = Fr.vr.values[v]
        for k, col in (('label', 'b_label'), ('moves_like', 'b_vsp'), ('class_median', 'b_class')):
            out[k][v] = Fr[col].values[v] / vr          # RoadData's baselines cover valid seconds already
    e = (Fr.recset == 'epa').values
    if e.any():
        E = Fr[e]
        miles = E.dist_m.values / C.MI
        for kind in ('regression', 'dev_median'):
            per = np.full(len(E), np.nan)
            for (arch, test), idx in E.groupby(['arch', 'test']).indices.items():
                per[idx] = _epa_unit_value(E.vehicle_id.values[idx], arch, test, kind)
            mj = np.where(E.arch.values == 'bev', per * 3.6 * 0.88, per * C.LHV * 745.0 * GAL) * miles
            out[kind][e] = mj
    h = (Fr.recset == 'hd').values
    if h.any():
        H = Fr[h]; med, icct = _hd_dev(); kpk, _ = ztbus_dev_kpk()
        p = np.full(len(H), np.nan)
        z = (H.set == 'ZTBus_leg').values
        p[z] = kpk * H.dist_km.values[z] * 3.6
        for fuel in ('diesel', 'cng'):
            m = ((H.set == 'Mersin') & (H.fuel == fuel)).values
            p[m] = med[fuel] * (C.CNG_LHV if fuel == 'cng' else C.LHV)
        t = (H.set == 'ICCT').values
        p[t] = H.cycle.map(icct).values[t].astype(float) * H.dist_km.values[t] * C.LHV
        out['dev_median'][h] = p
    return out
