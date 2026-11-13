"""fueleconomy.gov vehicles.csv -> clean table, plus label matches for each VED vehicle.
VED hides make/model/year, so a match is a candidate SET (same fuel, cylinders, displacement, powertrain type,
boost, transmission type if known, body family) over model years 2005-2018; label = candidate median, with spread."""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, _VEH)
from realdata.vedspec import ved_vehicle_table
ROOT = _VEH + '/data'

fe = pd.read_csv(f'{ROOT}/raw/fe/vehicles.csv', low_memory=False)
cols = ['id', 'year', 'make', 'model', 'baseModel', 'VClass', 'drive', 'trany', 'displ', 'cylinders', 'tCharger', 'sCharger',
        'atvType', 'fuelType', 'fuelType1', 'city08', 'highway08', 'comb08', 'UCity', 'UHighway', 'cityE', 'highwayE', 'combE',
        'cityCD', 'highwayCD', 'combinedCD', 'cityUF', 'highwayUF', 'combinedUF', 'range', 'rangeCity', 'rangeHwy', 'evMotor',
        'eng_dscr', 'startStop', 'co2TailpipeGpm', 'phevBlended']
fe = fe[cols].copy()
fe['trans_type'] = np.where(fe.trany.str.contains('variable|AV', case=False, na=False), 'CVT',
                            np.where(fe.trany.str.startswith('Manual', na=False), 'manual', 'AT'))
fe['boost'] = fe.tCharger.notna() | (fe.sCharger.fillna('') == 'S')
vc = fe.VClass.fillna('')
fe['body'] = np.where(vc.str.contains('Pickup'), 'pickup', np.where(vc.str.contains('Sport Utility|Special Purpose|Minivan|Van'), 'suv', 'car'))
fe['pt'] = np.select([fe.atvType == 'EV', fe.atvType == 'Plug-in Hybrid', fe.atvType == 'Hybrid',
                      fe.fuelType1.fillna('').str.contains('Diesel')], ['bev', 'phev', 'hev', 'diesel'], 'gas')
# Unadjusted (lab) city/highway MPG UCity/UHighway are closer to dyno physics; label MPG (city08/highway08) is the
# 'window sticker' value with the 5-cycle adjustment. We keep both.
fe.to_parquet(f'{ROOT}/proc/fe/fe_vehicles.parquet', index=False)

vt = ved_vehicle_table()
pool = fe[(fe.year >= 2005) & (fe.year <= 2018)]
rows = []
for vid, r in vt.iterrows():
    c = pool
    pin = None
    if r.ved_type == 'EV':
        c = c[(c.make == 'Nissan') & c.model.str.contains('Leaf') & (c.year == 2013)]; pin = '2013 Nissan Leaf (VED README)'
    elif r.ved_type == 'PHEV':
        if r.displacement_L == 1.4: c = c[(c.make == 'Chevrolet') & (c.model == 'Volt') & (c.year <= 2015)]; pin = 'Chevrolet Volt gen 1'
        elif r.displacement_L == 1.5: c = c[(c.make == 'Chevrolet') & (c.model == 'Volt') & (c.year >= 2016)]; pin = 'Chevrolet Volt gen 2'
        elif r.displacement_L == 2.0: c = c[(c.make == 'Ford') & c.model.str.contains('Energi')]; pin = 'Ford C-Max/Fusion Energi'
        elif r.displacement_L == 1.8: c = c[(c.make == 'Toyota') & c.model.str.contains('Prius') & (c.pt == 'phev')]; pin = 'Toyota Prius Plug-in/Prime'
    else:
        want_pt = 'hev' if r.ved_type == 'HEV' else ('diesel' if r.fuel == 'diesel' else 'gas')
        c = c[(c.pt == want_pt) & (np.abs(c.displ - r.displacement_L) < 0.051)]
        if np.isfinite(r.cylinders): c = c[c.cylinders == r.cylinders]
        if r.ved_type == 'ICE':
            c = c[c.boost == bool(r.turbo)]
            if r.transmission in ('CVT', 'manual', 'AT'): c = c[c.trans_type == r.transmission]
            fam = {'car': ['car'], 'suv': ['suv'], 'pickup': ['pickup', 'suv']}[r.cls]
            c2 = c[c.body.isin(fam)]
            if len(c2) >= 3: c = c2
    n = len(c)
    rec = dict(vehicle_id=vid, n_candidates=n, pinned_model=pin)
    if n:
        city = c.city08.astype(float); hwy = c.highway08.astype(float)
        rec.update(label_city_mpg=city.median(), label_hwy_mpg=hwy.median(), label_comb_mpg=c.comb08.astype(float).median(),
                   unadj_city_mpg=c.UCity.median(), unadj_hwy_mpg=c.UHighway.median(),
                   city_iqr_rel=float((city.quantile(.75) - city.quantile(.25)) / city.median()) if city.median() > 0 else np.nan,
                   label_city_kWh_per_100mi=c.cityE.replace(0, np.nan).median(), label_hwy_kWh_per_100mi=c.highwayE.replace(0, np.nan).median(),
                   label_ev_range_mi=c['range'].replace(0, np.nan).median(),
                   example_models='; '.join((c.year.astype(str) + ' ' + c.make + ' ' + c.model).value_counts().index[:6]),
                   n_makes=c.make.nunique())
        iq = rec['city_iqr_rel']
        rec['match_confidence'] = ('high' if pin or (n <= 6 and c.make.nunique() == 1) else
                                   'medium' if (iq is not None and iq <= 0.12 and n >= 3) else 'low')
    else:
        rec['match_confidence'] = 'none'
    rows.append(rec)
m = pd.DataFrame(rows).set_index('vehicle_id')
m.to_parquet(f'{ROOT}/proc/fe/ved_label_match.parquet')
print(m.match_confidence.value_counts()); print(m.n_candidates.describe())
print(m.join(vt[['ved_type']]).groupby(['ved_type', 'match_confidence']).size())
print(m.sample(8, random_state=1)[['n_candidates', 'label_city_mpg', 'label_hwy_mpg', 'city_iqr_rel', 'match_confidence', 'example_models']].to_string())
