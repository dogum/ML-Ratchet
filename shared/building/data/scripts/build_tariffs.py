"""Real retail tariffs from the OpenEI U.S. Utility Rate Database (URDB, bulk CSV) -> proc/tariffs/tariffs.json.

Selected: residential TOU (ThermoPilot) and commercial TOU + demand-charge rates in different climates, mostly utilities
that serve BDG2 sites. Parsed into a plain structure; buildingdata.tariff_prices()/bill() turn it into hourly prices and bills.
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import json, os
import numpy as np, pandas as pd

D = f'{BLDG_ROOT}/data'; O = f'{D}/proc/tariffs'; os.makedirs(O, exist_ok=True)
SEL = [  # key, URDB label, climate zone, BDG2 site(s) in the territory, note
    ('pge_e_tou_c', '69de73d20a9d4957880ffa6d', '3C', 'Bear, Swan', 'PG&E E-TOU-C (4-9 pm peak), baseline region W (all-electric)'),
    ('pge_b19_sec', '69de8cd7f6f3fc67370d4e1d', '3C', 'Bear, Swan', 'PG&E B-19 medium general demand TOU, secondary'),
    ('sce_tou_d_prime', '66cdf20a0f78ba14bd0796fe', '3B', '', 'SCE TOU-D-PRIME (heat pump / EV households)'),
    ('sce_tou_d_5_8', '66cdef520d535cf5c3058521', '3B', '', 'SCE TOU-D-5-8PM'),
    ('sce_tou_gs3_b', '675078fecb87bf5ac20350bf', '3B', '', 'SCE TOU-GS-3 option B, demand metered'),
    ('aps_r3', '69a718961822c9da260daf1b', '2B', 'Fox', 'APS R-3: TOU 4-7 pm weekdays with demand charge'),
    ('aps_e32_l', '69a71159f0ddf5680f0728db', '2B', 'Fox', 'APS E-32 L large general service TOU (demand)'),
    ('srp_e26', '69a1d205eb4d4f797b07dfec', '2B', 'Fox', 'SRP E-26 residential TOU'),
    ('austin_res_tou', '68b7ac90f9265971c2050c50', '2A', 'Bull', 'Austin Energy residential TOU pilot'),
    ('ouc_gsd_tou', '6998d74d23f4d2111a02549d', '2A', 'Panther, Gator', 'OUC general service demand secondary, option B TOU'),
    ('duke_fl_rt', '6998c33ad3c1f5b8140d5ab2', '2A', '', 'Duke Energy RT residential TOU'),
    ('xcel_co_re_tou', '69bd927af5cd25efec0e9aad', '5B', '', 'Xcel Energy Colorado RE-TOU'),
    ('rmp_ut_sched1_tou', '69c30f2893a800cd5603da5c', '5B', 'Bobcat', 'Rocky Mountain Power (Utah) Schedule 1 TOU'),
    ('xcel_mn_a04_tod', '67a3b13d539f2d3f2c056972', '6A', 'Hog', 'Xcel Energy Minnesota residential TOD (A04)'),
    ('xcel_mn_a15_tod', '67a4ee86f7f9d21dc80bbeb5', '6A', 'Hog', 'Xcel Energy Minnesota general service time-of-day (A15), demand'),
    ('nyseg_sc12_tou', '67bf95ae1a06c5ae710b8793', '6A', 'Cockatoo', 'NYSEG SC-12 residential TOU'),
    ('pseg_rs_tou_3p', '69e24a7a4db539998e061b9c', '5A', 'Eagle, Peacock', 'PSE&G RS-TOU-3P residential 3-period TOU'),
    ('coned_sc1_rate3', '67b4ee9d4beb07f0d40cb9eb', '4A', '', 'Con Edison SC-1 Rate III voluntary TOD (NYC)'),
    ('pepco_gtlv', '68b3acd4aa9285ab440c5b1d', '4A', 'Rat', 'Pepco DC GT low-voltage time-metered general service (demand)'),
]


def num(x):
    try:
        v = float(x); return None if np.isnan(v) else v
    except Exception:
        return None


def structure(r, prefix):
    per = []
    for i in range(12):
        tiers = []
        for j in range(12):
            rate = num(r.get(f'{prefix}/period{i}/tier{j}rate'))
            if rate is None: break
            tiers.append(dict(rate=rate, max=num(r.get(f'{prefix}/period{i}/tier{j}max')), adj=num(r.get(f'{prefix}/period{i}/tier{j}adj')) or 0.0))
        if not tiers: break
        per.append(tiers)
    return per


def sched(s):
    try: return json.loads(s)
    except Exception: return None


def main():
    u = pd.read_csv(f'{D}/raw/tariffs/usurdb.csv.gz', low_memory=False)
    u = u.set_index('label')
    out = {}
    for key, lab, cz, sites, note in SEL:
        r = u.loc[lab].to_dict()
        t = dict(key=key, urdb_label=lab, uri=f'https://apps.openei.org/USURDB/rate/view/{lab}', utility=r['utility'], name=r['name'],
                 sector=r['sector'], startdate=str(pd.to_datetime(r['startdate'], unit='s').date()) if num(r['startdate']) else None,
                 climate_zone=cz, bdg2_sites=sites, note=note, source=r.get('source'),
                 energy=dict(unit=r.get('energyrateunit'), periods=structure(r, 'energyratestructure'),
                             weekday=sched(r.get('energyweekdayschedule')), weekend=sched(r.get('energyweekendschedule'))),
                 demand_tou=dict(unit=r.get('demandrateunit'), periods=structure(r, 'demandratestructure'),
                                 weekday=sched(r.get('demandweekdayschedule')), weekend=sched(r.get('demandweekendschedule'))),
                 demand_flat=dict(unit=r.get('flatdemandunit'), periods=structure(r, 'flatdemandstructure'),
                                  month_period=[num(r.get(f'flatDemandMonth_{m}')) for m in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]),
                 fixed=dict(first_meter=num(r.get('fixedchargefirstmeter')), units=r.get('fixedchargeunits'),
                            min_charge=num(r.get('mincharge')), min_charge_units=r.get('minchargeunits')),
                 description=str(r.get('description'))[:1500])
        out[key] = t
        e = t['energy']['periods']
        rates = [p[0]['rate'] + p[0]['adj'] for p in e]
        print(f"{key:20s} {t['sector']:11s} {cz} periods={len(e)} energy $/kWh {min(rates):.3f}-{max(rates):.3f} "
              f"demandTOU={len(t['demand_tou']['periods'])} flat={len(t['demand_flat']['periods'])} fixed={t['fixed']['first_meter']} {t['fixed']['units']}")
    json.dump(out, open(f'{O}/tariffs.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
