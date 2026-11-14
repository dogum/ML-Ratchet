"""Frozen dev/exam split of the Argonne D3 vehicles -> data/proc/d3/splits.json (never touches data/proc/splits.json).

By vehicle, stratified by category (conventional, hybrid, plugin, electric, alternative) and data era (legacy = the
"All Data" zip only; extended = Comprehensive/Extended 10 Hz datasets), seed 20261113, about half dev and half exam.
Decided from the file listing and the design rules below only; no test result was looked at.

Rules
1. Same-generation siblings form one unit and always land on the same side (2012/2013 Volt, 2012/2013 Prius Plug-in,
   2012 Leaf/2013 Leaf SV, 2017/2019 Bolt, C-Max Energi/Fusion Energi share a powertrain and battery): putting one
   in dev and its twin in exam would leak calibration into the exam.
2. A few units are pinned by design (reasons in PINS): the calibration targets named in shared/vehicle/V2_PLAN.md
   that have no second D3 example go to dev; a held-out modern example of each pinned architecture goes to exam.
3. The remaining units of each stratum are shuffled (seed 20261113) and each goes to the side with fewer vehicles
   in that stratum (ties: a coin from the same generator).
Calibration may fit only on dev vehicles. Exam vehicles are for the final report only.

usage: python d3_splits.py           (refuses to overwrite a different existing split unless --force)
"""
import os, sys, json, hashlib, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d3_fetch import VEHICLES, RAW

_VEH = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(_VEH, 'data', 'proc', 'd3', 'splits.json')
SEED = 20261113
SKIP = {'2020 Tesla Model 3'}          # not downloaded (90 GB); not in the split

UNITS = [
    ['2012 Chevrolet Volt', '2013 Chevrolet Volt'],
    ['2012 Toyota Prius Plug-in', '2013 Toyota Prius Plug-in'],
    ['2012 Nissan Leaf', '2013 Nissan Leaf SV'],
    ['2017 Chevrolet Bolt', '2019 Chevrolet Bolt'],
    ['2013 Ford Cmax Energi', '2013 Ford Fusion Energi'],
]

PINS = {
    '2021 Hyundai Sonata Hybrid': ('dev', 'the only modern P2 hybrid with full 10 Hz channels; V2_PLAN calibrates P2 (torque converter, engine starts) on it'),
    '2019 Acura MDX Sport Hybrid': ('exam', 'held-out modern P2-type hybrid (7-speed DCT with motor) for the P2 fix'),
    '2015 Honda Accord Hybrid': ('dev', 'the only i-MMD (e:HEV) in D3; V2_PLAN calibrates the direct-drive clutch on it; e:HEV held-out checks stay on the EPA list'),
    '2010 Toyota Prius': ('dev', 'Toyota power-split reference; V2_PLAN calibrates the Toyota HEV engine-on rules'),
    '2017 Toyota Prius Prime': ('dev', 'the PHEV with the richest channels (CD/CS, cabin, heater); V2_PLAN calibrates PHEV cabin and CD rules on it'),
    '2021 Toyota RAV4 Prime': ('exam', 'held-out modern PHEV (the other extended-data plug-in)'),
    '2018 Toyota Camry XLE': ('dev', 'conventional calibration anchor: its engine is the RoadSim A25A map'),
    '2017 Ford F150': ('exam', 'held-out modern conventional (turbo truck, 10-speed)'),
}


def era_of(listing_entry):
    paths = [f['path'] for f in listing_entry['files']]
    ext = any(p.endswith(('.tdms', 'Test Data.txt')) and ('Extended' in p or 'Comprehensive' in p) for p in paths)
    return 'extended' if ext else 'legacy'


def build():
    L = {v['name']: v for v in json.load(open(os.path.join(RAW, 'listing.json'), encoding='utf-8'))}
    cat = {name: c for c, name, *_ in VEHICLES if name not in SKIP}
    era = {n: era_of(L[n]) for n in cat}
    unit_of = {}
    for u in UNITS:
        for n in u:
            unit_of[n] = tuple(u)
    units = sorted({unit_of.get(n, (n,)) for n in cat})
    side = {}
    for u in units:
        p = {PINS[n][0] for n in u if n in PINS}
        assert len(p) <= 1, u
        if p:
            side[u] = p.pop()
    rng = np.random.default_rng(SEED)
    strata = sorted({(cat[u[0]], era[u[0]]) for u in units})
    for st in strata:
        U = [u for u in units if (cat[u[0]], era[u[0]]) == st]
        n = {'dev': sum(len(u) for u in U if side.get(u) == 'dev'), 'exam': sum(len(u) for u in U if side.get(u) == 'exam')}
        free = [u for u in U if u not in side]
        order = rng.permutation(len(free))
        for i in order:
            u = free[i]
            if n['dev'] < n['exam']: s = 'dev'
            elif n['exam'] < n['dev']: s = 'exam'
            else: s = 'dev' if rng.random() < 0.5 else 'exam'
            side[u] = s; n[s] += len(u)
    veh = {}
    for u in units:
        for name in u:
            vid = 'D3_' + __import__('re').sub(r'[^\w.-]+', '_', name)
            veh[vid] = dict(name=name, category=cat[name], era=era[name], split=side[u],
                            unit=list(u) if len(u) > 1 else None, pinned=PINS[name][1] if name in PINS else None)
    body = json.dumps(veh, sort_keys=True)
    out = dict(source='Argonne D3 (Downloadable Dynamometer Database)', seed=SEED, created='2026-11-13',
               rule=__doc__.split('Rules')[1].split('usage:')[0].strip(),
               excluded={'2020 Tesla Model 3': 'not downloaded (905 files, 90 GB)'},
               counts={s: {c: sum(1 for v in veh.values() if v['split'] == s and v['category'] == c) for c in sorted(set(cat.values()))}
                       for s in ('dev', 'exam')},
               vehicles=veh, sha256=hashlib.sha256(body.encode()).hexdigest(),
               sha256_of='json.dumps(vehicles, sort_keys=True)')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    out = build()
    if os.path.exists(OUT) and not a.force:
        old = json.load(open(OUT))
        if old['sha256'] != out['sha256']:
            sys.exit(f'splits.json exists with a different sha256 ({old["sha256"][:12]} vs {out["sha256"][:12]}); it is frozen. Use --force only if you mean to change it.')
        print('splits.json unchanged', out['sha256'][:12]); return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    for s in ('dev', 'exam'):
        print(s, out['counts'][s])
        for vid, v in out['vehicles'].items():
            if v['split'] == s:
                print(f"   {v['category']:12s} {v['era']:9s} {v['name']}{'  (pinned)' if v['pinned'] else ''}")
    print('sha256', out['sha256'])


if __name__ == '__main__':
    main()
