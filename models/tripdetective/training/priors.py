"""Wrappers around RoadSim's vehicle generator and the log emulator, versioned. RoadSim itself is never edited.
v1: the generator as published (modern EPA test-car donors, default strategy parameters).
v2: dev-twin calibrated priors (calibrations measured on the 16 dev cars only, twin.py / data/twin_*.parquet):
    - VED-era gasoline engines are less efficient than the modern ALPHA maps: eff_scale ~ N(0.88, 0.06) (conv),
      N(0.85, 0.06) (hybrids); twin fuel/km was 0.82-0.87 x real for conventional cars, 0.77-0.88 x for hybrids;
    - idle speed: conventional idle 650-820 rpm (<3 L) / 540-700 rpm (>=3 L); twin idle rpm was 0.67-0.94 x real;
    - hybrid supervisory controllers run the engine more: ECMS/A-ECMS equivalence factor s ~ U(2.4, 3.6),
      rule thresholds lower (power-split v_ev_max 6-18 m/s, P_on 3-15 % of engine power); twin engine-off-while-moving
      was 0.39-0.52 vs 0.19-0.32 real;
    - cabin heating/cooling: real drivers use far less than the simulator's PTC at full set-point (Leaf at stops in
      the cold: 1.4 kW real vs 7.6 kW twin): heat_max 1.5-6 kW, set-points 16-22 C, heating below 5-16 C; A/C share;
    - strategy parameters drawn widely for every hybrid (also widens the 'control behaviour' targets)."""
from common import *

VERSION = {'v1': 'RoadSim generator as is; emulator v1',
           'v2': 'dev-twin calibrated: engine efficiency/idle, hybrid strategy parameters, HVAC use',
           'v3': 'v2 + fleet mass prior matched to VED-era cars (dev + train_noise static weights): rejection sampling '
                 'to 85 % lognormal(median 1,750 kg total, sigma 0.20) + 15 % generator fleet; class mix car 0.60 / suv 0.30 / pickup 0.05 / van 0.05'}

# v3 mass prior. Source: published curb weights (+136 kg RoadData convention) of the 16 dev + 17 train_noise VED cars:
# median 1,724 kg, 5-95 % 1,270-2,177 kg (log-sd ~0.17). Payload in the generator adds ~0-250 kg, so the target total
# median is set a little higher (1,750 kg) and wider (0.20) than the static weights.
V3_MED, V3_SIG, V3_MIX = 1750.0, 0.20, 0.85
V3_CLS = {'car': 0.60, 'suv': 0.30, 'pickup': 0.05, 'van': 0.05}


def vehicle_prior(spec, rng, prior):
    if prior == 'v1':
        return spec
    arch = spec['architecture']; e = spec.get('engine')
    if e:
        hyb = arch != 'conv'
        e['eff_scale'] = round(float(np.clip(rng.normal(0.85 if hyb else 0.88, 0.06), 0.7, 1.05)), 3)
        if not hyb:
            d = e.get('displacement_L') or 2.0
            e['idle_rpm'] = round(float(rng.uniform(650, 820) if d < 3 else rng.uniform(540, 700)))
    tr = spec.get('transmission') or {}
    if arch == 'conv' and tr.get('type') in ('AT', 'DCT') and rng.random() < 0.5:
        # VED-era automatics: 4-5 speeds, engine ~1900-2500 rpm at 62 mph in top gear (twin rpm at 50-70 km/h was 0.89x real)
        g = [2.84, 1.57, 1.0, 0.69] if rng.random() < 0.5 else [3.55, 2.06, 1.4, 1.0, 0.71]
        spec['transmission'] = dict(type='AT', n_gears=len(g), gears=g)
        spec['final_drive'] = round(float(rng.uniform(1900, 2500)) * np.pi / 30 * spec['wheel_radius_m'] / (27.7 * g[-1]), 3)
    b = spec.get('battery')
    if b is not None and arch != 'conv':
        th = b.setdefault('thermal', {})
        th['heater_W'] = round(float(rng.choice([0, 1500, 3000]))); th['T_heat_on'] = round(float(rng.uniform(-15, 0)), 1)
    st = spec['control']['strategy']; p = {}
    if st in ('ecms', 'aecms', 'phev_cdcs', 'phev_blended'):
        p['s'] = round(float(rng.uniform(2.4, 3.6)), 3)
        if st == 'aecms': p['Kp'] = round(float(rng.uniform(6, 20)), 2)
    if st == 'powersplit_rule':
        p['v_ev_max'] = round(float(rng.uniform(6, 18)), 2); p['P_on_frac'] = round(float(rng.uniform(0.03, 0.15)), 3)
        p['P_off_frac'] = round(p['P_on_frac'] * float(rng.uniform(0.4, 0.7)), 3)
    if st == 'parallel_rule':
        p['v_ev_max'] = round(float(rng.uniform(5, 16)), 2); p['P_ev_frac'] = round(float(rng.uniform(0.04, 0.15)), 3)
    if st == 'series_power_follower':
        p['P_on_frac'] = round(float(rng.uniform(0.04, 0.2)), 3); p['P_off_frac'] = round(p['P_on_frac'] * 0.5, 3)
    spec['control']['params'] = p
    h = spec.setdefault('aux', {}).setdefault('HVAC', {})
    h['heat_max_W'] = round(float(rng.uniform(1500, 6000)))
    h['T_heat_set'] = round(float(rng.uniform(16, 22)), 1)
    h['heat_on_below_C'] = round(float(rng.uniform(5, 16)), 1)
    h['cool_on_above_C'] = round(float(rng.uniform(22, 28)), 1)
    h['enabled'] = bool(rng.random() < 0.9)
    return spec


def logger_tweak(cfg, spec, rng, prior):
    return cfg


_DENS = None


def _bank_density():
    """log-mass histogram of the generator fleet under the v3 class mix (20,000 draws, no simulation)"""
    global _DENS
    if _DENS is None:
        import fleet
        rng = np.random.default_rng(12345); ms = []
        for _ in range(20000):
            s = fleet.draw_vehicle(rng, cls_w=V3_CLS, quick=True); ms.append(s)
        h, e = np.histogram(np.log(ms), bins=40, range=(np.log(800), np.log(5000)), density=True)
        _DENS = (np.maximum(h, 1e-3), e)
    return _DENS


def accept_v3(mass_total, rng):
    h, e = _bank_density()
    x = np.log(mass_total); i = int(np.clip(np.searchsorted(e, x) - 1, 0, len(h) - 1))
    tgt = V3_MIX * np.exp(-0.5 * ((x - np.log(V3_MED)) / V3_SIG) ** 2) / (V3_SIG * np.sqrt(2 * np.pi)) + (1 - V3_MIX) * h[i]
    r = tgt / h[i]
    return rng.random() < min(1.0, r / 6.0)        # envelope constant 6 (max ratio near the light tail)
