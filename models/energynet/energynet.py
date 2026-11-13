"""EnergyNet: trip energy before the trip, with an error bar (numpy only).

    from energynet import EnergyNet
    net = EnergyNet()                                   # loads the 5-network ensemble in weights/ next to this file
    r = net.predict(trip, vehicle)
    r['value'], r['unit'], r['lo68'], r['hi68'], r['components_MJ']

trip    : dict of 1 Hz arrays (RoadSim/RoadData trip contract): v [m/s] required; grade, T_amb [C], wind_speed [m/s],
          wind_dir [deg FROM], heading [deg], rho_air [kg/m3], elev [m], surface [0 dry,1 concrete,2 wet,3 gravel,4 snow]
vehicle : dict, every field optional except class and architecture:
          class (car|suv|pickup|van|bus|truck_md|truck_hd), architecture (conv|hev_parallel_p2|hev_series|hev_powersplit|
          phev_powersplit|phev_series|bev), mass_kg + mass_kind ('exact' test/actual mass, or 'reported' e.g. curb rounded
          to 500 lb + 136 kg), road_load_ABC {A_N, B_N_per_mps, C_N_per_mps2} (needs exact mass), displacement_L,
          max_power_kW, fuel ('gasoline'|'diesel'), transmission (AT|DCT|CVT|AMT|single|eCVT), n_gears, battery_kWh,
          label_city / label_hwy (window-sticker mpg; for a BEV kWh/100 mi), label_city_kWh_per_100mi / label_hwy_... (PHEV),
          soc_start (displayed usable SOC 0..1), thermal ('warm' | 'cold' | None), obd_maf (True: predict what an OBD
          MAF-based fuel estimate reports), dyno (True: dynamometer test, no on-road accessories)
Output energy: fuel chemical energy (conv/HEV, fuel g x 0.042 MJ/g), battery DC net discharge (BEV), sum (PHEV)."""
import os, json
import numpy as np
import features as F
import glob
from model_np import ENet, Ensemble

HERE = os.path.dirname(os.path.abspath(__file__))
MI = 1609.344
LHV = 0.042


def _mjkm_from_mpg(mpg, diesel=False):
    return (832.0 if diesel else 745.0) * 3.785411784 * LHV / (mpg * MI / 1000) if mpg else None


def _mjkm_from_kwh100mi(k):
    return k * 3.6 / (MI / 10) if k else None


class EnergyNet:
    def __init__(self, weights=None, calib=None):
        ps = sorted(glob.glob(os.path.join(weights or os.path.join(HERE, 'weights'), '*.npz')))
        self.net = Ensemble(ps)
        F.PHYS_K.update(self.net.meta.get('phys_k', {}))
        cp = calib or os.path.join(HERE, 'sigma_factors.json')
        self.k = json.load(open(cp)) if os.path.exists(cp) else {}

    def features(self, trip, veh):
        arch, cls = veh['architecture'], veh['class']
        hd = cls in F.HD
        mc, Cc, Rc, cvc, rot = F.CLS[cls]
        P = F.prep(trip)
        kind = veh.get('mass_kind', 'exact')
        m = veh.get('mass_kg')
        ml = 0 if not m else (2 if kind == 'exact' else 1)
        m = float(m) if m else mc
        rl = veh.get('road_load_ABC') if ml == 2 else None
        if rl:
            cda = 2 * rl['C_N_per_mps2'] / 1.1767; crr = rl['A_N'] / (m * 9.81); cv = float(np.clip(rl['B_N_per_mps'] / rl['A_N'], -0.01, 0.03))
        else:
            cda, crr, cv = Cc * (m / mc) ** 0.5, Rc, cvc
        ph = F.phys_ints(P, m, cda, crr, cv, rot, hd)
        soc = veh.get('soc_start'); cap = veh.get('battery_kWh')
        dur = float(P['N']); dist = float(P['v'].sum())
        E, Eh, Eb, f = F.e_phys(ph, arch, m, dur, soc, cap, hd)
        diesel = veh.get('fuel') == 'diesel'
        if arch == 'bev':
            lf = None; le = (_mjkm_from_kwh100mi(veh.get('label_city')), _mjkm_from_kwh100mi(veh.get('label_hwy'))) if veh.get('label_city') else None
        else:
            lf = (_mjkm_from_mpg(veh.get('label_city'), diesel), _mjkm_from_mpg(veh.get('label_hwy'), diesel)) if veh.get('label_city') else None
            le = ((_mjkm_from_kwh100mi(veh.get('label_city_kWh_per_100mi')), _mjkm_from_kwh100mi(veh.get('label_hwy_kWh_per_100mi')))
                  if veh.get('label_city_kWh_per_100mi') else None)
        th = veh.get('thermal')
        V = F.vehicle_vec(arch, cls, m, ml, cda, crr, bool(rl), veh.get('displacement_L'), veh.get('max_power_kW'), diesel,
                          veh.get('transmission'), veh.get('n_gears'), cap, lf, le, soc, th == 'warm', th == 'cold',
                          bool(veh.get('obd_maf')), bool(veh.get('dyno')), (E, dist / 1000), dist / max(dur, 1))
        return np.concatenate([F.route_vec(P), F.phys_vec(ph, E, Eh, Eb, f, m, dist), V]), E, dist, dur

    def predict(self, trip, veh, group=None):
        x, E, dist, dur = self.features(trip, veh)
        p, s, c = self.net.predict(x[None], np.array([E]))
        arch = veh['architecture']
        g = group or ('ICE' if arch == 'conv' else 'EV' if arch == 'bev' else 'PHEV' if arch.startswith('phev') else 'HEV')
        cls = veh['class']
        if cls == 'bus': k = self.k.get('HD_ZTBus_leg' if arch == 'bev' else 'HD_Mersin', 1.0)
        elif cls in F.HD: k = self.k.get('HD_ICCT', 1.0)
        else: k = self.k.get(f'VED_{g}', 1.0)
        S = float(s[0]) * k
        e = float(p[0])
        unit, f = ('kWh', 1 / 3.6) if arch == 'bev' else (('MJ', 1.0) if arch.startswith('phev') else ('g fuel', 1 / LHV))
        return dict(energy_MJ=e, value=e * f, unit=unit, sigma_log=S, lo68=e * f * np.exp(-S), hi68=e * f * np.exp(S),
                    lo95=e * f * np.exp(-1.96 * S), hi95=e * f * np.exp(1.96 * S), physics_estimate_MJ=float(E),
                    components_MJ=dict(zip(['aero', 'rolling', 'friction_brakes', 'aux_hvac', 'powertrain_losses'], map(float, c[0]))),
                    distance_km=dist / 1000, duration_s=dur)
