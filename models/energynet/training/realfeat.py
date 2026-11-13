"""Real-data records (same layout as the simulated bank) for dev gates and the exam.
python realfeat.py ved_dev | ved_exam | epa_dev | epa_exam | hd       -> data/real_<name>.npz + _meta.parquet
Each record carries a fixed tier: what the real data actually provides."""
import sys, json
from common import *
import numpy as np, pandas as pd
import realdata as rd
import features as F, routes
from bank import TR_KEYS, _fill

MI = 1609.344
FIELDS = ['arch', 'cls', 'm_true', 'm_rep', 'cda', 'crr', 'cv', 'disp', 'power', 'diesel', 'trans', 'ngears', 'cap', 'soc', 'cold', 'warm',
          'dyno', 'dur', 'dist', 'y', 'vr']
TF = dict(m_level=0, rl=False, disp=False, power=False, trans=False, ngears=False, cap=False, labf=False, labe=False, soc=False,
          thermal=False, maf=False, lab_noise=0.0)


def mpg_to_mjkm(mpg, dens=745.0):
    return dens * 3.785411784 * LHV / (mpg * MI / 1000) if mpg and np.isfinite(mpg) and mpg > 0 else np.nan


def kwh100mi_to_mjkm(k):
    return k * 3.6 / (MI / 10) if k and np.isfinite(k) and k > 0 else np.nan


def valid_ratio(P, valid, m, cls):
    mc, Cc, Rc, cvc, rot = F.CLS[cls]
    pa, pr, pg, pi = F.wheel_power(P, m, Cc * (m / mc) ** 0.5, Rc, cvc, rot, cls in F.HD)
    e = np.clip(pa + pr + pg + pi, 0, None) / 0.25 + 5000.0 * (m / 1500.0) ** 0.7
    return float(e[valid].sum() / e.sum())


class Acc:
    def __init__(self):
        self.rows = []; self.meta = []
    def add(self, P, cls, rec, tier, lab=(np.nan,) * 4, meta=None, m_true=None, m_rep=None, cda=None, crr=None, cv=None):
        mc, Cc, Rc, cvc, rot = F.CLS[cls]
        mt = m_true if m_true else mc; mr = m_rep if m_rep else mc
        PH = F.variants(P, cls, mt, cda if cda else Cc * (mt / mc) ** 0.5, crr if crr else Rc, cv if cv is not None else cvc, mr)
        r = dict(R=F.route_vec(P).astype(np.float32), PH=PH.astype(np.float32), lab=np.array(lab, np.float32))
        r.update({k: np.nan for k in FIELDS}); r.update(rec)
        r['m_true'] = m_true if m_true else np.nan; r['m_rep'] = m_rep if m_rep else np.nan
        r['cda'] = cda if cda else np.nan; r['crr'] = crr if crr else np.nan; r['cv'] = cv if cv is not None else np.nan
        r['dur'] = float(P['N']); r['dist'] = float(P['v'].sum())
        t = dict(TF); t.update(tier)
        for k, v in t.items(): r['T_' + k] = v
        self.rows.append(r); self.meta.append(meta or {})
    def save(self, name):
        D = {k: np.stack([np.asarray(r[k], dtype=np.float32 if k not in ('R', 'PH', 'lab') else np.float32) for r in self.rows]) for k in self.rows[0]}
        np.savez_compressed(f'{EN}/data/real_{name}.npz', **D)
        pd.DataFrame(self.meta).to_parquet(f'{EN}/data/real_{name}_meta.parquet')
        print(name, len(self.rows))


def ved(split):
    A = Acc(); veh = rd.ved_vehicles()
    base = rd.baselines('ved_exam' if split == 'exam' else 'ved_dev').set_index('trip_id')
    for meta, arr in rd.load_trips(split, label_ok=True):
        vid = meta['vehicle_id']; r = veh.loc[vid]; arch = meta['architecture']; vt = meta['ved_type']
        spec = rd.load_vehicle(vid)
        obs = {k: arr[k] for k in TR_KEYS}
        P = F.prep(obs)
        ok = arr['valid'] == 1
        y = float(real_target(arr, vt).sum())
        m_rep = float(spec['mass_kg']) if spec.get('mass_kg') else None
        cls = spec['class']
        dens = 832.0 if r.fuel == 'diesel' else 745.0
        if arch == 'bev':
            lab = (np.nan, np.nan, kwh100mi_to_mjkm(r.label_city_kWh_per_100mi), kwh100mi_to_mjkm(r.label_hwy_kWh_per_100mi))
        else:
            lab = (mpg_to_mjkm(r.label_city_mpg, dens), mpg_to_mjkm(r.label_hwy_mpg, dens),
                   kwh100mi_to_mjkm(r.label_city_kWh_per_100mi) if arch.startswith('phev') else np.nan,
                   kwh100mi_to_mjkm(r.label_hwy_kWh_per_100mi) if arch.startswith('phev') else np.nan)
        s = arr['soc'][np.isfinite(arr['soc'])]
        soc = float(s[0]) if (len(s) and arch != 'conv') else np.nan
        warm = meta['v_start'] > 1.0
        cap = (spec.get('battery') or {}).get('capacity_kWh') if arch == 'bev' else None
        tr = (spec.get('transmission') or {}).get('type')
        tr = 'AMT' if tr == 'manual' else tr
        rec = dict(arch=F.ARCHS.index(arch), cls=F.CLASSES.index(cls), disp=r.displacement_L if np.isfinite(r.displacement_L) else np.nan,
                   diesel=float(r.fuel == 'diesel'), trans=F.TRANS.index(tr) if tr in F.TRANS else -1, cap=cap if cap else np.nan,
                   soc=soc, warm=float(warm), cold=0.0, dyno=0.0, y=y, vr=valid_ratio(P, ok, m_rep or F.CLS[cls][0], cls))
        tier = dict(m_level=1 if m_rep else 0, disp=bool(np.isfinite(r.displacement_L)), trans=tr in F.TRANS, cap=bool(cap),
                    labf=bool(np.isfinite(lab[0]) and np.isfinite(lab[1])), labe=bool(np.isfinite(lab[2]) and np.isfinite(lab[3])),
                    soc=bool(np.isfinite(soc)), thermal=bool(warm),
                    maf=(arch != 'bev') and str(meta.get('fuel_source', '')).startswith(('maf', 'absload')))
        b = base.loc[meta['trip_id']] if meta['trip_id'] in base.index else None
        conv = {'g': LHV, 'kWh': 3.6, 'MJ': 1.0}[unit_of(arch)]
        M = dict(trip_id=meta['trip_id'], vehicle_id=vid, group=vt, arch=arch, cls=cls, T=meta['T_amb_mean'], v_mean=meta['v_mean'],
                 dist_km=meta['distance_m'] / 1000, dur_s=meta['duration_s'], idle=meta['idle_frac'], climb=meta['elev_gain_m'],
                 fuel_source=meta.get('fuel_source'), match=r.match_confidence, unit=unit_of(arch), conv=conv,
                 b_y=float(b.y) * conv if b is not None else np.nan,
                 b_label=float(b.pred_label) * conv if b is not None else np.nan,
                 b_vsp=float(b.pred_vsp) * conv if b is not None else np.nan,
                 b_class=float(b.pred_class) * conv if b is not None else np.nan)
        A.add(P, cls, rec, tier, lab, M, m_rep=m_rep)
        if len(A.rows) % 2000 == 0: print(len(A.rows), flush=True)
    A.save(f'ved_{split}')


EPA_CYC = {'FTP': [('UDDS', True), ('UDDS', False)], 'HWY': [('HWFET', False)], 'US06': [('US06', False)]}


def epa(split):
    A = Acc(); e = rd.epa_testcars(split=split).join(pd.read_parquet(f'{EN}/data/epa_fixed.parquet'))   # corrected cycle labels (epa_fix.py)
    cyc = rd.cycles()
    Ps = {}
    for n in ('UDDS', 'HWFET', 'US06'):
        tr = _fill({k: cyc[n][1][k] for k in TR_KEYS if k in cyc[n][1]}); N = len(tr['v'])
        tr.update(T_amb=np.full(N, 24.0, np.float32), wind_speed=np.zeros(N, np.float32), grade=np.zeros(N, np.float32))
        tr['rho_air'] = routes.rho_air(tr['T_amb'], np.zeros(N)).astype(np.float32)
        Ps[n] = F.prep(tr)
    from realdata import _TRANS
    for vid, r in e.iterrows():
        if r.architecture not in ('conv', 'hev', 'bev'): continue
        if not (r.test_mass_kg > 0 and r.A_N > 0 and r.C_N_per_mps2 > 0): continue
        arch = {'conv': 'conv', 'hev': 'hev_powersplit', 'bev': 'bev'}[r.architecture]
        cls = 'car' if r.veh_type == 'Car' else ('pickup' if 'PICKUP' in str(r.model).upper() else 'suv')
        m = float(r.test_mass_kg)
        cda = float(2 * r.C_N_per_mps2 / 1.1767); crr = float(r.A_N / (m * 9.81)); cv = float(np.clip(r.B_N_per_mps / r.A_N, -0.01, 0.03))
        tr = 'single' if arch == 'bev' else ('eCVT' if arch == 'hev_powersplit' else _TRANS.get(str(r.trans_code)))
        disp = float(r.displacement_L) if (arch != 'bev' and pd.notna(r.displacement_L) and 0 < r.displacement_L < 10) else np.nan
        pw = float(r.rated_hp) * 0.7457 if (pd.notna(r.rated_hp) and r.rated_hp > 0 and arch != 'bev') else np.nan
        ng = float(r.gears) if pd.notna(r.gears) and r.gears > 0 else np.nan
        for c, legs in EPA_CYC.items():
            col = (f'{c}_mpg_fix' if arch != 'bev' else {'FTP': 'CD_UDDS_kWh_per_mi', 'HWY': 'CD_HWY_kWh_per_mi', 'US06': None}[c])
            if col is None or col not in r or not (pd.notna(r[col]) and r[col] > 0): continue
            if arch != 'bev' and r.fuel != 'gasoline': continue
            for (cn, cold) in legs:
                if arch == 'bev' and cold: continue
                rec = dict(arch=F.ARCHS.index(arch), cls=F.CLASSES.index(cls), disp=disp, power=pw, diesel=0.0,
                           trans=F.TRANS.index(tr) if tr in F.TRANS else -1, ngears=ng, cold=float(cold), warm=float(not cold),
                           dyno=1.0, y=float(r[col]), vr=1.0, soc=np.nan)
                tier = dict(m_level=2, rl=True, disp=bool(np.isfinite(disp)), power=bool(np.isfinite(pw)), trans=tr in F.TRANS,
                            ngears=bool(np.isfinite(ng)) and tr in F.TRANS, thermal=True)
                M = dict(vehicle_id=vid, family=r.family, arch=arch, cls=cls, test=c, cycle=cn, cold=cold, y=float(r[col]),
                         ycol=col, model_year=int(r.model_year), make=r.make, model=r.model)
                A.add(Ps[cn], cls, rec, tier, meta=M, m_true=m, cda=cda, crr=crr, cv=cv)
        if len(A.rows) % 5000 == 0: print(len(A.rows), flush=True)
    A.save(f'epa_{split}')


def mersin_route(rng, L, n_stops, D):
    """synthetic 33M trip: n stops over L metres, total duration D s (bisection on the cruise speed; extra signal
    waiting time if even 18 km/h is too fast)."""
    x = rng.dirichlet(np.ones(n_stops + 1) * 3) * L
    dw = rng.uniform(10, 30, n_stops + 1)
    seed = int(rng.integers(1 << 30))
    def prof(vc, extra=0.0):
        segs = [(vc / 3.6, x[k], True, dw[k] + extra) for k in range(n_stops + 1)]
        old = routes._segments
        routes._segments = lambda rng_, kind, hd: segs
        try:
            return routes.speed_profile(np.random.default_rng(seed), 'bus', True, max_s=40000)
        finally:
            routes._segments = old
    lo, hi = 18.0, 60.0
    v = prof(lo)
    if len(v) < D:
        return prof(lo, (D - len(v)) / (n_stops + 1))
    for _ in range(9):
        mid = 0.5 * (lo + hi); v = prof(mid)
        if len(v) > D: lo = mid
        else: hi = mid
    return prof(0.5 * (lo + hi))


MERSIN_T = {2: 11.0, 3: 14.0, 4: 18.0, 5: 22.0}   # Mersin monthly mean temperature (climatology, approx.)


def hd():
    A = Acc(); h = rd.hd_data()
    # ZTBus legs (dev + exam)
    for split in ('dev', 'exam'):
        for meta, arr in rd.load_trips(split, source='ZTBus'):
            if not meta.get('q_label_ok', True): continue
            obs = {k: arr[k] for k in TR_KEYS}; P = F.prep(obs)
            rec = dict(arch=6, cls=4, trans=F.TRANS.index('single'), dyno=0.0, warm=1.0, cold=0.0, y=meta['elec_kWh'] * 3.6, vr=1.0)
            A.add(P, 'bus', rec, dict(m_level=0, trans=True, thermal=True),
                  meta=dict(set='ZTBus_leg', split=split, trip_id=meta['trip_id'], mission=meta['mission'], vehicle_id=meta['vehicle_id'],
                            dist_km=meta['distance_m'] / 1000, T=meta['T_amb_mean'], y=meta['elec_kWh'] * 3.6))
    # Mersin (synthetic route per trip, 2 draws)
    m = h['mersin_trips']; m = m[m.q_label_ok]
    rng = np.random.default_rng(33)
    for r in m.itertuples():
        T = MERSIN_T.get(pd.Timestamp(r.date).month, 16.0)
        for k in range(2):
            v = mersin_route(rng, 23400.0, max(int(r.Stops_served), 1), float(r.duration_s))
            N = len(v)
            tr = _fill({'v': v.astype(np.float32)}); tr['T_amb'] = np.full(N, T, np.float32)
            tr['rho_air'] = routes.rho_air(tr['T_amb'], np.zeros(N)).astype(np.float32)
            P = F.prep(tr)
            rec = dict(arch=0, cls=4, diesel=1.0, trans=F.TRANS.index('AT'), dyno=0.0, warm=1.0, cold=0.0, y=float(r.fuel_g) * LHV, vr=1.0)
            A.add(P, 'bus', rec, dict(m_level=0, trans=True, thermal=True),
                  meta=dict(set='Mersin', split=r.split, trip_id=r.trip_id, fuel=r.fuel, draw=k, fuel_g=float(r.fuel_g), T=T,
                            dur_s=float(r.duration_s), stops=int(r.Stops_served), ac=bool(r.ac_on), y=float(r.fuel_g) * LHV))
    # ICCT trucks
    t = h['icct_trucks']; t = t[t.cycle_available]
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
                        dist_km=dist, y=float(r.g_per_km) * dist * LHV), m_true=float(r.test_mass_kg))
    A.save('hd')


if __name__ == '__main__':
    w = sys.argv[1]
    if w.startswith('ved_'): ved(w[4:])
    elif w.startswith('epa_'): epa(w[4:])
    elif w == 'hd': hd()
