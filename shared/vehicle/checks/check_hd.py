"""Check: heavy-duty vehicles against published ranges.

Class 8 high-roof sleeper tractor-trailer (diesel; and a battery-electric version): GEM test cycles = 40 CFR 1037
Appendix A transient cycle + 55 and 65 mi/hr cruise over the Appendix D grade profile, composite weighted 5/9/86 %
(sleeper cabs, 40 CFR 1037.510 table 1), payload 19 tons (standard payload, 40 CFR 1037.801), CO2 at 10,180 g/gal
diesel. Reference: CO2 standards for Class 8 high-roof sleeper cabs, 40 CFR 1037.106 table 1 (eCFR version of
2024-01-01): 75.7 (MY2021-23), 70.7 (MY2024-26), 64.3 (MY2027+) g/ton-mile; Phase 1 (MY2017-20) 72 g/ton-mile under
the older GEM. GEM credits idle-reduction/speed-limiter/etc. technologies that this reference truck does not have.

Class 6 box truck (diesel): vocational Multi-Purpose weighting 54/29/17 % (40 CFR 1037.510), payload 5.6 tons;
driving cycles only (the regulatory composite also adds drive/parked idle). Reference: CI Medium HDV multi-purpose
standards 265 / 246 / 235 g/ton-mile (MY2021-23 / 2024-26 / 2027+, 40 CFR 1037.105, 2024 eCFR).

40 ft transit buses (diesel, series diesel-hybrid, battery-electric) on the CBD14 cycle (SAE J1376 construction, 11.8
mph average incl. stops) and the HD transient cycle (15.3 mph), averaged over ambient temperatures 10/18/25/32 C with
HVAC on (a southern-California-like spread). Reference (measured in service, NREL/TP-5400-80022, Foothill Transit
Battery Electric Bus Evaluation, 2021, via OSTI): BEB 2.10-2.15 kWh/mi on a 10.6 mph route, 1.82-1.90 kWh/mi at
18-21 mph; CNG baseline 3.74 mpgge = 4.29 mpdge at 17.6 mph. A diesel bus on the same duty is expected near or somewhat
above the CNG diesel-equivalent value; hybrids are usually reported 10-40 % better than diesel (literature range,
not verified from a downloaded source).
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'roadsim'))
from roadsim import simulate, load_vehicle, make_trip
from roadsim.constants import MILE, GAL, FUELS, KWH

CO2_PER_GAL = 10180.0
WARM = {'engine': 88.0, 'tire_warm': 1.0, 'driveline': 80.0}


def run(vid, cyc, T=20.0, hvac=False, soc=None):
    spec = load_vehicle(vid)
    spec['aux'].setdefault('HVAC', {})['enabled'] = hvac
    o = simulate(spec, make_trip(cyc, T_amb=T, solar=400.0 if hvac else 0.0), init_temps=dict(WARM, cabin=21.0, battery=T), init_soc=soc)
    S = o['summary']
    mi = S['distance_m'] / MILE
    r = dict(miles=round(mi, 3), closure=o['ledger']['closure_rel'], trace_rms_mps=round(S['trace_rms_mps'], 3),
             trace_max_mps=round(S['trace_max_mps'], 3))
    if S['fuel_g'] > 0:
        gal = S['fuel_L'] / GAL
        Eb = o['ledger']['batt_chem_out'] * 1e3          # hybrid: correct net battery energy at 0.40 x 0.90
        gal += Eb / (0.40 * 0.90) / FUELS['diesel']['lhv'] / (FUELS['diesel']['density'] / 1e3) / 1e3 / GAL
        r['mpg'] = round(mi / gal, 3); r['gal_per_mi'] = gal / mi
    else:
        r['kWh_per_mi_dc'] = round(o['ledger']['batt_chem_out'] * 1e3 / KWH / mi, 3)
    return r


def composite(vid, w, payload_t):
    res = {c: run(vid, c) for c in ('hd_transient', 'gem_55', 'gem_65')}
    out = dict(cycles=res)
    if 'gal_per_mi' in res['gem_65']:
        gpm = sum(w[c] * res[c]['gal_per_mi'] for c in res)
        out['composite_mpg'] = round(1 / gpm, 3)
        out['composite_g_co2_per_ton_mile'] = round(gpm * CO2_PER_GAL / payload_t, 1)
    else:
        out['composite_kWh_per_mi_dc'] = round(sum(w[c] * res[c]['kWh_per_mi_dc'] for c in res), 3)
    for c in res:
        res[c].pop('gal_per_mi', None)
    return out


def main():
    out = dict(check='heavy-duty vs published ranges', method=__doc__.strip())
    ws = {'hd_transient': 0.05, 'gem_55': 0.09, 'gem_65': 0.86}
    t8 = composite('ref_class8_sleeper_diesel', ws, 19.0)
    t8['reference_g_per_ton_mile'] = {'MY2021-23 std': 75.7, 'MY2024-26 std': 70.7, 'MY2027+ std': 64.3}
    t8['note'] = 'reference truck has no Phase 2 technology packages (aero bin III-ish CdA 5.9, Crr 0.0063, no idle/AES credits); expected at or above the MY2021 standard'
    out['class8_sleeper_diesel'] = t8
    out['class8_bev'] = composite('ref_class8_bev', ws, 19.0)
    out['class8_bev']['note'] = 'no regulatory reference; NACFE Run on Less Electric reports ~1.5-2.2 kWh/mi for loaded Class 8 BEVs (literature, not verified)'
    wv = {'hd_transient': 0.54, 'gem_55': 0.29, 'gem_65': 0.17}
    md = composite('ref_class6_box_diesel', wv, 5.6)
    md['reference_g_per_ton_mile'] = {'MY2021-23 std': 265, 'MY2024-26 std': 246, 'MY2027+ std': 235}
    md['note'] = 'driving cycles only (no idle terms), so expected somewhat below the regulatory composite of a comparable vehicle'
    out['class6_box_diesel'] = md
    temps = [10.0, 18.0, 25.0, 32.0]
    bus = {}
    for vid in ('ref_bus40_diesel', 'ref_bus40_hybrid', 'ref_bus40_bev'):
        bus[vid] = {}
        for cyc in ('cbd14', 'hd_transient'):
            rr = [run(vid, cyc, T=T, hvac=True) for T in temps]
            k = 'mpg' if 'mpg' in rr[0] else 'kWh_per_mi_dc'
            if k == 'mpg':
                val = 1.0 / np.mean([1.0 / r[k] for r in rr])
            else:
                val = float(np.mean([r[k] for r in rr]))
            bus[vid][cyc] = {k: round(val, 3), 'by_temperature_C': {str(T): r[k] for T, r in zip(temps, rr)},
                             'worst_closure': max(r['closure'] for r in rr)}
    for cyc in ('cbd14', 'hd_transient'):
        bus['ref_bus40_hybrid'][cyc]['improvement_vs_diesel_pct'] = round(100 * (bus['ref_bus40_hybrid'][cyc]['mpg'] / bus['ref_bus40_diesel'][cyc]['mpg'] - 1), 1)
        bus['ref_bus40_bev'][cyc]['mpdge'] = round(37.64 / bus['ref_bus40_bev'][cyc]['kWh_per_mi_dc'], 2)
    bus['reference'] = dict(bev_kWh_per_mi=[1.82, 2.15], cng_mpdge_at_17_6mph=4.29, hybrid_improvement_pct_literature=[10, 40],
                            source='NREL/TP-5400-80022 (Foothill Transit BEB evaluation, 2021), OSTI 1799885')
    out['transit_bus_40ft'] = bus
    json.dump(out, open(os.path.join(HERE, 'heavy_duty.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'method'}, indent=1))


if __name__ == '__main__':
    main()
