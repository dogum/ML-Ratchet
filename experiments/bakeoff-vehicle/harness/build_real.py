"""Real records (dev gate now; exam only from exam.py) in EnergyNet v2.1's own record format and tiers.

python3 build_real.py dev          -> real/dev/{ved,epa,hd}.npz + _meta.parquet (+ hd_traces.npz for Mersin)
build('exam', guard_token)         -> real/exam/...   (exam.py only; refused otherwise)

This is models/energynet/training/realfeat.py (vendored, used through its own helpers) with four differences, all
needed to keep the exam out of Phase 1 and all checked to give the same records:
  * one split at a time (realfeat.hd() builds dev and exam together);
  * Mersin: the synthetic route of every trip comes from one generator (default_rng(33)) in table order, dev and exam
    interleaved. For the other split's trips the generator's draws are consumed exactly as mersin_route() consumes
    them (dirichlet, uniform, integers) without building the profile, so each trip gets EnergyNet's route;
  * EPA cycle labels come from the long test table by procedure name (training/epa_fix.py's FTP / HWY / US06 rule);
    the dev-fitted log-linear baseline is fitted here on dev only (refs.py);
  * EnergyNet v2.1's two public-spec inputs are written into the records: EPA test-car hybrids get their hybrid system
    from the model line (training/epa_hybkind.py, checks/hybrid_specs.py) and the ZTBus legs the trolleybus's 19 t
    curb mass as a reported mass (realfeat.ZTB_CURB_KG; v2.1 reads both, rungs r8/r9/r17). The original values are
    kept in the meta (arch0, m_level0) so EnergyNet's own D2 terms can be recomputed exactly.
Traces: VED and ZTBus legs are read from RoadData's files at batch time (meta 'trace_src'); EPA and ICCT rows are the
dynamometer schedule at the test temperature; Mersin synthetic speed profiles are stored in hd_traces.npz.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import config as C
import common as CM
import realfeat as RF
import features as F
import routes
import realdata as rd
from bank import TR_KEYS, _fill

LHV = C.LHV


def _out(split):
    if C.MOCK and split == 'exam':
        raise PermissionError('mock exam: exam records are never built')
    d = os.path.join(C.REAL, split)
    os.makedirs(d, exist_ok=True)
    return d


class Acc(RF.Acc):
    def save_to(self, path):
        D = {k: np.stack([np.asarray(r[k], dtype=np.float32) for r in self.rows]) for k in self.rows[0]}
        np.savez_compressed(path + '.npz', **D)
        pd.DataFrame(self.meta).to_parquet(path + '_meta.parquet')
        print(os.path.basename(path), len(self.rows), flush=True)


# ------------------------------------------------------------------ VED
def ved(split):
    A = Acc(); veh = rd.ved_vehicles()
    base = rd.baselines('ved_exam' if split == 'exam' else 'ved_dev').set_index('trip_id')
    for meta, arr in rd.load_trips(split, label_ok=True):
        vid = meta['vehicle_id']; r = veh.loc[vid]; arch = meta['architecture']; vt = meta['ved_type']
        spec = rd.load_vehicle(vid)
        P = F.prep({k: arr[k] for k in TR_KEYS})
        ok = arr['valid'] == 1
        y = float(CM.real_target(arr, vt).sum())
        m_rep = float(spec['mass_kg']) if spec.get('mass_kg') else None
        cls = spec['class']
        dens = 832.0 if r.fuel == 'diesel' else 745.0
        if arch == 'bev':
            lab = (np.nan, np.nan, RF.kwh100mi_to_mjkm(r.label_city_kWh_per_100mi), RF.kwh100mi_to_mjkm(r.label_hwy_kWh_per_100mi))
        else:
            lab = (RF.mpg_to_mjkm(r.label_city_mpg, dens), RF.mpg_to_mjkm(r.label_hwy_mpg, dens),
                   RF.kwh100mi_to_mjkm(r.label_city_kWh_per_100mi) if arch.startswith('phev') else np.nan,
                   RF.kwh100mi_to_mjkm(r.label_hwy_kWh_per_100mi) if arch.startswith('phev') else np.nan)
        s = arr['soc'][np.isfinite(arr['soc'])]
        soc = float(s[0]) if (len(s) and arch != 'conv') else np.nan
        warm = meta['v_start'] > 1.0
        cap = (spec.get('battery') or {}).get('capacity_kWh') if arch == 'bev' else None
        tr = (spec.get('transmission') or {}).get('type')
        tr = 'AMT' if tr == 'manual' else tr
        my = RF.ved_model_year(r)
        rec = dict(arch=F.ARCHS.index(arch), cls=F.CLASSES.index(cls), disp=r.displacement_L if np.isfinite(r.displacement_L) else np.nan,
                   diesel=float(r.fuel == 'diesel'), trans=F.TRANS.index(tr) if tr in F.TRANS else -1, cap=cap if cap else np.nan,
                   soc=soc, warm=float(warm), cold=0.0, dyno=0.0, y=y, vr=RF.valid_ratio(P, ok, m_rep or F.CLS[cls][0], cls), my=my)
        tier = dict(m_level=1 if m_rep else 0, disp=bool(np.isfinite(r.displacement_L)), trans=tr in F.TRANS, cap=bool(cap),
                    labf=bool(np.isfinite(lab[0]) and np.isfinite(lab[1])), labe=bool(np.isfinite(lab[2]) and np.isfinite(lab[3])),
                    soc=bool(np.isfinite(soc)), thermal=bool(warm),
                    maf=(arch != 'bev') and str(meta.get('fuel_source', '')).startswith(('maf', 'absload')),
                    my=bool(np.isfinite(my)))
        b = base.loc[meta['trip_id']] if meta['trip_id'] in base.index else None
        conv = {'g': LHV, 'kWh': 3.6, 'MJ': 1.0}[CM.unit_of(arch)]
        M = dict(set='VED', split=split, trip_id=meta['trip_id'], vehicle_id=vid, group=vt, arch=arch, cls=cls, T=meta['T_amb_mean'],
                 v_mean=meta['v_mean'], model_year=my, v_start=meta['v_start'], dist_km=meta['distance_m'] / 1000,
                 dur_s=meta['duration_s'], idle=meta['idle_frac'], climb=meta['elev_gain_m'], fuel_source=meta.get('fuel_source'),
                 match=r.match_confidence, unit=CM.unit_of(arch), conv=conv,
                 b_y=float(b.y) * conv if b is not None else np.nan, b_label=float(b.pred_label) * conv if b is not None else np.nan,
                 b_vsp=float(b.pred_vsp) * conv if b is not None else np.nan, b_class=float(b.pred_class) * conv if b is not None else np.nan,
                 trace_src=meta['trip_id'], T_test=np.nan, arch0=arch, m_level0=1 if m_rep else 0)
        A.add(P, cls, rec, tier, lab, M, m_rep=m_rep)
        if len(A.rows) % 2000 == 0: print(len(A.rows), flush=True)
    A.save_to(os.path.join(_out(split), 'ved'))


# ------------------------------------------------------------------ EPA
FTP_PROCS = {'Federal fuel 2-day exhaust (w/can load)', 'Federal fuel 3-day exhaust', 'CVS 75 and later (w/o can. load)'}


def epa_fixed():
    """training/epa_fix.py's label rebuild (FTP / HWY / US06 mpg from the long table by procedure name). All splits are
    rebuilt together because it is a per-test-car median of that car's own tests; nothing is fitted."""
    t = rd.epa_testcars(long=True)
    t = t[(t.fe_mpg > 3) & (t.fe_mpg < 300)]
    cyc = np.where(t.test_proc.isin(FTP_PROCS), 'FTP', np.where(t.test_proc == 'HWFE', 'HWY', np.where(t.test_proc == 'US06', 'US06', None)))
    t = t.assign(cyc=cyc); t = t[t.cyc.notna() & (t.fuel == 'gasoline')]
    w = t.groupby(['epa_vehicle_id', 'cyc']).fe_mpg.median().unstack()
    w.columns = [f'{c}_mpg_fix' for c in w.columns]
    return w


def hybrid_kinds(e):
    """training/epa_hybkind.py: EPA test-car hybrid -> (architecture, transmission) from the model line."""
    sys.path.insert(0, os.path.join(C.VEHICLE, 'checks'))
    import hybrid_specs as H
    from realdata import _TRANS
    KIND = {'p2': 'hev_parallel_p2', 'series': 'hev_series', 'ps': 'hev_powersplit', 'fps': 'hev_powersplit', 'ops': 'hev_powersplit'}
    out = {}
    for vid, r in e[e.architecture == 'hev'].iterrows():
        k = H.lookup(r.make, r.model, int(r.model_year), r.displacement_L)[0]
        a = KIND.get(k, 'hev_powersplit')
        out[vid] = (a, 'eCVT' if a == 'hev_powersplit' else ('single' if a == 'hev_series' else (_TRANS.get(str(r.trans_code)) or 'AT')))
    return out


def epa(split):
    A = Acc()
    e = rd.epa_testcars(split=split).join(epa_fixed())
    hk = hybrid_kinds(e)
    cyc = rd.cycles()
    Ps = {}
    for n, T in (('UDDS', 24.0), ('HWFET', 24.0), ('US06', 24.0), ('UDDS', -6.7), ('SC03', 35.0)):
        tr = _fill({k: cyc[n][1][k] for k in TR_KEYS if k in cyc[n][1]}); N = len(tr['v'])
        tr.update(T_amb=np.full(N, T, np.float32), wind_speed=np.zeros(N, np.float32), grade=np.zeros(N, np.float32))
        tr['rho_air'] = routes.rho_air(tr['T_amb'], np.zeros(N)).astype(np.float32)
        Ps[(n, T)] = F.prep(tr)
    from realdata import _TRANS
    for vid, r in e.iterrows():
        if r.architecture not in ('conv', 'hev', 'bev'): continue
        if not (r.test_mass_kg > 0 and r.A_N > 0 and r.C_N_per_mps2 > 0): continue
        arch0 = {'conv': 'conv', 'hev': 'hev_powersplit', 'bev': 'bev'}[r.architecture]
        cls = 'car' if r.veh_type == 'Car' else ('pickup' if 'PICKUP' in str(r.model).upper() else 'suv')
        m = float(r.test_mass_kg)
        cda = float(2 * r.C_N_per_mps2 / 1.1767); crr = float(r.A_N / (m * 9.81)); cv = float(np.clip(r.B_N_per_mps / r.A_N, -0.01, 0.03))
        tr0 = 'single' if arch0 == 'bev' else ('eCVT' if arch0 == 'hev_powersplit' else _TRANS.get(str(r.trans_code)))
        arch, tr = (hk[vid] if arch0 == 'hev_powersplit' and vid in hk else (arch0, tr0))
        disp = float(r.displacement_L) if (arch0 != 'bev' and pd.notna(r.displacement_L) and 0 < r.displacement_L < 10) else np.nan
        pw = float(r.rated_hp) * 0.7457 if (pd.notna(r.rated_hp) and r.rated_hp > 0 and arch0 != 'bev') else np.nan
        ng = float(r.gears) if pd.notna(r.gears) and r.gears > 0 else np.nan
        for c, legs in RF.EPA_CYC.items():
            col = (RF.EPA_COL[c] if arch0 != 'bev' else {'FTP': 'CD_UDDS_kWh_per_mi', 'HWY': 'CD_HWY_kWh_per_mi'}.get(c))
            if col is None or col not in r or not (pd.notna(r[col]) and r[col] > 0): continue
            if arch0 != 'bev' and r.fuel != 'gasoline': continue
            for (cn, cold) in legs:
                if arch0 == 'bev' and cold: continue
                T = RF.EPA_T.get(c, 24.0)
                rec = dict(arch=F.ARCHS.index(arch), cls=F.CLASSES.index(cls), disp=disp, power=pw, diesel=0.0,
                           trans=F.TRANS.index(tr) if tr in F.TRANS else -1, ngears=ng, cold=float(cold), warm=float(not cold),
                           dyno=1.0, y=float(r[col]), vr=1.0, soc=np.nan, my=float(r.model_year))
                tier = dict(m_level=2, rl=True, disp=bool(np.isfinite(disp)), power=bool(np.isfinite(pw)), trans=tr in F.TRANS,
                            ngears=bool(np.isfinite(ng)) and tr in F.TRANS, thermal=True, my=True)
                M = dict(set='EPA', split=split, vehicle_id=vid, family=r.family, arch=arch0, arch_in=arch, cls=cls, test=c, cycle=cn,
                         cold=cold, y=float(r[col]), ycol=col, model_year=int(r.model_year), make=r.make, model=r.model,
                         trace_src=cn, T_test=T, arch0=arch0, trans0=tr0, m_level0=2)
                A.add(Ps[(cn, T)], cls, rec, tier, meta=M, m_true=m, cda=cda, crr=crr, cv=cv)
        if len(A.rows) % 5000 == 0: print(len(A.rows), flush=True)
    A.save_to(os.path.join(_out(split), 'epa'))


# ------------------------------------------------------------------ heavy duty
def _skip_mersin_draws(rng, n_stops):
    """consume rng exactly as realfeat.mersin_route does, without building the profile."""
    rng.dirichlet(np.ones(n_stops + 1) * 3); rng.uniform(10, 30, n_stops + 1); rng.integers(1 << 30)


def hd(split):
    A = Acc(); h = rd.hd_data(); traces = {}
    # ZTBus legs (the 10 traced missions of RoadData v1; their legs are bit-identical to ZTBus_full's)
    for meta, arr in rd.load_trips(split, source='ZTBus'):
        if not meta.get('q_label_ok', True): continue
        P = F.prep({k: arr[k] for k in TR_KEYS})
        rec = dict(arch=6, cls=4, trans=F.TRANS.index('single'), dyno=0.0, warm=1.0, cold=0.0, y=meta['elec_kWh'] * 3.6, vr=1.0)
        # v2.1's public-spec input (rung r9): the bus's published 19 t curb mass as a reported mass
        A.add(P, 'bus', rec, dict(m_level=1, trans=True, thermal=True), m_rep=RF.ZTB_CURB_KG,
              meta=dict(set='ZTBus_leg', split=split, trip_id=meta['trip_id'], mission=meta['mission'], vehicle_id=meta['vehicle_id'],
                        dist_km=meta['distance_m'] / 1000, dur_s=float(len(P['v'])), T=meta['T_amb_mean'], y=meta['elec_kWh'] * 3.6,
                        trace_src=meta['trip_id'], T_test=np.nan, arch0='bev', m_level0=0, group='BUS', arch='bev', cls='bus'))
    # Mersin (synthetic route per trip, 2 draws; the generator's sequence covers both splits)
    m = h['mersin_trips']; m = m[m.q_label_ok]
    rng = np.random.default_rng(33)
    for r in m.itertuples():
        T = RF.MERSIN_T.get(pd.Timestamp(r.date).month, 16.0)
        for k in range(2):
            ns = max(int(r.Stops_served), 1)
            if r.split != split:
                _skip_mersin_draws(rng, ns); continue
            v = RF.mersin_route(rng, 23400.0, ns, float(r.duration_s))
            N = len(v)
            tr = _fill({'v': v.astype(np.float32)}); tr['T_amb'] = np.full(N, T, np.float32)
            tr['rho_air'] = routes.rho_air(tr['T_amb'], np.zeros(N)).astype(np.float32)
            P = F.prep(tr)
            key = f'{r.trip_id}_d{k}'; traces[key] = v.astype(np.float32)
            rec = dict(arch=0, cls=4, diesel=1.0, trans=F.TRANS.index('AT'), dyno=0.0, warm=1.0, cold=0.0, y=float(r.fuel_g) * LHV, vr=1.0)
            A.add(P, 'bus', rec, dict(m_level=0, trans=True, thermal=True),
                  meta=dict(set='Mersin', split=r.split, trip_id=r.trip_id, fuel=r.fuel, draw=k, fuel_g=float(r.fuel_g), T=T,
                            dur_s=float(r.duration_s), stops=int(r.Stops_served), ac=bool(r.ac_on), y=float(r.fuel_g) * LHV,
                            trace_src='mersin:' + key, T_test=T, arch0='conv', m_level0=0, group='BUS', arch='conv', cls='bus'))
    # ICCT trucks (published chassis-dynamometer results on cycles with traces)
    t = h['icct_trucks']; t = t[t.cycle_available & (t.split == split)]
    cyc = rd.cycles()
    for r in t.itertuples():
        tr = _fill({k: cyc[r.cycle_id][1][k] for k in TR_KEYS if k in cyc[r.cycle_id][1]}); N = len(tr['v'])
        tr['T_amb'] = np.full(N, 22.0, np.float32); tr['wind_speed'] = np.zeros(N, np.float32)
        tr['rho_air'] = routes.rho_air(tr['T_amb'], np.zeros(N)).astype(np.float32)
        P = F.prep(tr)
        disp = 14.5 if '14' in str(r.engine_L) else 12.5
        dist = P['v'].sum() / 1000
        rec = dict(arch=0, cls=6, disp=disp, power=315.0, diesel=1.0, trans=F.TRANS.index('AMT'), ngears=12.0, dyno=1.0, warm=1.0, cold=0.0,
                   y=float(r.g_per_km) * dist * LHV, vr=1.0)
        A.add(P, 'truck_hd', rec, dict(m_level=2, disp=True, power=True, trans=True, ngears=True, thermal=True),
              meta=dict(set='ICCT', split=r.split, truck=r.truck, cycle=r.cycle, L_per_100km=r.L_per_100km, g_per_km=r.g_per_km,
                        dist_km=dist, y=float(r.g_per_km) * dist * LHV, trip_id=f'{r.truck}_{r.cycle}', vehicle_id=r.truck,
                        trace_src='cycle:' + r.cycle_id, T_test=22.0, arch0='conv', m_level0=2, group='TRUCK', arch='conv', cls='truck_hd'),
              m_true=float(r.test_mass_kg))
    o = os.path.join(_out(split), 'hd')
    A.save_to(o)
    np.savez_compressed(o + '_traces.npz', **traces)


def build(split, token=None):
    if split == 'exam':
        import bake
        bake._require_exam(token)
    for f in (ved, epa, hd):
        f(split)
    json.dump({'split': split, 'built': pd.Timestamp.utcnow().isoformat()}, open(os.path.join(_out(split), 'BUILT.json'), 'w'))


if __name__ == '__main__':
    if sys.argv[1] != 'dev':
        raise SystemExit('only dev from the command line; the exam records are built by exam.py')
    which = sys.argv[2:] or ['ved', 'epa', 'hd']
    for w in which:
        {'ved': ved, 'epa': epa, 'hd': hd}[w]('dev')
    if all(os.path.exists(os.path.join(C.REAL, 'dev', f'{w}.npz')) for w in ('ved', 'epa', 'hd')):
        json.dump({'split': 'dev', 'built': pd.Timestamp.utcnow().isoformat()}, open(os.path.join(_out('dev'), 'BUILT.json'), 'w'))
