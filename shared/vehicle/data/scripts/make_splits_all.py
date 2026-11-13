"""Write the frozen road/data/proc/splits.json: VED (by vehicle), EPA test cars + fueleconomy.gov (by model family,
shared between both tables), HD sources. Seed 20261112. Never re-run with changes after Phase 2 starts."""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import json, re, hashlib, numpy as np, pandas as pd
ROOT = _VEH + '/data'
SEED = 20261112
GENERIC = {'MODEL', 'GRAND', 'NEW', 'SANTA', 'RANGE', 'LAND', 'TOWN', 'E', 'F', 'C', 'S', 'X', 'GT', 'THE', 'MINI', 'SPORT',
           'RAM', 'SIERRA', 'SILVERADO', 'ID.', 'ALL', 'MERCEDES-BENZ', 'BMW', 'CLASS'}
DROP = r'\b(AWD|FWD|RWD|2WD|4WD|4X4|4X2|PICKUP|CAB|CHASSIS|HYBRID|PLUG-IN|PLUGIN|PHEV|EV|ELECTRIC|ENERGI|PRIME|E-TRON|TDI|DIESEL|FFV|CNG|SPORTBACK|WAGON|COUPE|CONVERTIBLE|SEDAN|HATCHBACK)\b'
MAKEFIX = {'CHRYSLER LLC': 'CHRYSLER', 'GENERAL MOTORS': 'CHEVROLET', 'FORD MOTOR COMPANY': 'FORD', 'TOYOTA': 'TOYOTA',
           'MERCEDES-BENZ': 'MERCEDES-BENZ', 'MERCEDES BENZ': 'MERCEDES-BENZ'}


def family(make, model):
    mk = str(make).upper().strip(); mk = MAKEFIX.get(mk, mk)
    m = re.sub(r'\(.*?\)', ' ', str(model).upper())
    m = re.sub(DROP, ' ', m)
    toks = [t for t in re.split(r'[\s/]+', m) if t]
    if not toks: return f'{mk}|?'
    k = toks[0] if (toks[0] not in GENERIC or len(toks) == 1) else toks[0] + ' ' + toks[1]
    return f'{mk}|{k}'


RANK = {'conv': 0, 'hev': 1, 'phev': 2, 'fcev': 2, 'bev': 3}

ved = json.load(open(f'{ROOT}/proc/splits_ved_part.json'))['VED']
epa = pd.read_parquet(f'{ROOT}/proc/epa/testcars_vehicles.parquet')
fe = pd.read_parquet(f'{ROOT}/proc/fe/fe_vehicles.parquet')
epa['family'] = [family(a, b) for a, b in zip(epa.make, epa.model)]
fe['family'] = [family(a, b) for a, b in zip(fe.make, fe.model)]
fe['architecture'] = fe.pt.map({'bev': 'bev', 'phev': 'phev', 'hev': 'hev', 'gas': 'conv', 'diesel': 'conv'})
fam = pd.concat([epa[['family', 'architecture']], fe[['family', 'architecture']]])
fam['r'] = fam.architecture.map(RANK).fillna(0)
farch = fam.groupby('family').r.max().map({v: k for k, v in RANK.items() if k != 'fcev'})
rng = np.random.default_rng(SEED)
dev_fams = []
for arch, g in farch.groupby(farch):
    ids = sorted(g.index)
    k = max(1, int(round(0.10 * len(ids))))
    dev_fams += list(rng.choice(ids, size=k, replace=False))
dev_fams = set(dev_fams)
epa['split'] = np.where(epa.family.isin(dev_fams), 'dev', 'exam')
fe['split'] = np.where(fe.family.isin(dev_fams), 'dev', 'exam')
hd = json.load(open(f'{ROOT}/proc/hd/hd_dev.json'))
out = {
    'version': 1, 'frozen': '2026-11-12', 'seed': SEED,
    'principle': 'Split by vehicle (VED) or model family (EPA/fueleconomy), never by trip/test. dev = small, may be looked at for calibration and gates. exam = never used for any choice. train_noise = real route/weather inputs whose labels are never used.',
    'VED': ved,
    'EPA_testcars_and_fueleconomy': {
        'unit': 'model family = make + base model name (drive/body/powertrain words removed), shared by both tables',
        'rule': '10% of families per architecture stratum (conv/hev/phev/bev; family stratum = most electrified variant) drawn with the seed',
        'dev_families': sorted(dev_fams),
        'counts': {'epa_vehicles_dev': int((epa.split == 'dev').sum()), 'epa_vehicles_exam': int((epa.split == 'exam').sum()),
                   'fe_rows_dev': int((fe.split == 'dev').sum()), 'fe_rows_exam': int((fe.split == 'exam').sum())}},
    'cycles': {'rule': 'all dynamometer schedules are train_noise inputs (no labels).'},
    'HD': {'ZTBus': {'unit': 'mission (only 2 physical buses exist; a by-vehicle split would leave one bus per side)',
                     'dev_missions': hd['ZTBus_dev_missions'], 'rule': 'one seeded traced mission per bus = dev; all other missions (traced legs and the 1409-mission table) = exam'},
           'ICCT_2018_trucks': {'unit': 'truck', 'dev': [hd['ICCT_dev_truck']], 'rule': 'one of three trucks drawn with the seed = dev'},
           'Mersin_33M': {'unit': 'time block (vehicles anonymised)', 'dev': hd['Mersin_dev'], 'rule': 'the 7-day short-term subsample is dev; the rest is exam'}},
}
s = json.dumps(out, indent=1, sort_keys=False)
out['sha256_of_content_without_this_field'] = hashlib.sha256(s.encode()).hexdigest()
json.dump(out, open(f'{ROOT}/proc/splits.json', 'w'), indent=1)
epa[['family', 'split']].to_parquet(f'{ROOT}/proc/epa/testcars_split.parquet')
fe[['id', 'family', 'split']].to_parquet(f'{ROOT}/proc/fe/fe_split.parquet', index=False)
print(out['EPA_testcars_and_fueleconomy']['counts'])
print(epa.groupby(['architecture', 'split']).size().unstack())
print(len(dev_fams), sorted(dev_fams)[:30])
