"""Convert MATPOWER case files (MATPOWER GitHub, data/*.m) to a simple JSON network format + parquet tables.

proc/networks/<name>.json : {name, baseMVA, source, licence, citation, bus:[...], gen:[...], branch:[...], gencost:[...]}
  bus:    id, type (1 PQ, 2 PV, 3 ref, 4 isolated), Pd_MW, Qd_MVAr, Gs_MW, Bs_MVAr, area, Vm_pu, Va_deg, baseKV, zone,
          Vmax_pu, Vmin_pu, name (if given), lat, lon (approximate, ACTIVSg Texas only, see DATA.md)
  gen:    bus, Pg_MW, Qg_MVAr, Qmax, Qmin, Vg_pu, mBase_MVA, status, Pmax_MW, Pmin_MW, type, fuel (if given)
  branch: from, to, r_pu, x_pu, b_pu, rateA_MVA, rateB_MVA, rateC_MVA, ratio, angle_deg, status, angmin, angmax
  gencost: model, startup, shutdown, n, coeffs (MATPOWER gencost rows, $/h polynomial or piecewise)
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, re, json, glob
import numpy as np, pandas as pd
RAW = f'{GRID_ROOT}/data/raw/matpower'; OUT = f'{GRID_ROOT}/data/proc/networks'
os.makedirs(OUT, exist_ok=True)
BUS = ['id', 'type', 'Pd_MW', 'Qd_MVAr', 'Gs_MW', 'Bs_MVAr', 'area', 'Vm_pu', 'Va_deg', 'baseKV', 'zone', 'Vmax_pu', 'Vmin_pu']
GEN = ['bus', 'Pg_MW', 'Qg_MVAr', 'Qmax', 'Qmin', 'Vg_pu', 'mBase_MVA', 'status', 'Pmax_MW', 'Pmin_MW']
BR = ['from', 'to', 'r_pu', 'x_pu', 'b_pu', 'rateA_MVA', 'rateB_MVA', 'rateC_MVA', 'ratio', 'angle_deg', 'status', 'angmin', 'angmax']


def block(txt, name):
    m = re.search(r'mpc\.' + name + r'\s*=\s*([\[{])(.*?)[\]}]\s*;', txt, re.S)
    if not m: return None
    body = m.group(2)
    rows = []
    for line in body.split('\n'):
        line = line.split('%')[0].strip().rstrip(';').strip()
        if not line: continue
        if m.group(1) == '{':
            rows.append(line.strip().strip("'").strip())
        else:
            rows.append([float(x) for x in re.split(r'[\s,]+', line) if x])
    return rows


def load_m(path):
    txt = open(path).read()
    base = float(re.search(r'mpc\.baseMVA\s*=\s*([\d.]+)', txt).group(1))
    bus = block(txt, 'bus'); gen = block(txt, 'gen'); br = block(txt, 'branch'); gc = block(txt, 'gencost')
    names = block(txt, 'bus_name'); gtype = block(txt, 'gentype'); gfuel = block(txt, 'genfuel')
    head = '\n'.join(l[1:].strip() for l in txt.split('\n')[1:40] if l.startswith('%'))
    B = pd.DataFrame([r[:13] for r in bus], columns=BUS)
    G = pd.DataFrame([r[:10] for r in gen], columns=GEN)
    R = pd.DataFrame([(r + [0, 0, -360, 360])[:13] if len(r) < 13 else r[:13] for r in br], columns=BR)
    if names and len(names) == len(B): B['name'] = names
    if gtype and len(gtype) == len(G): G['type'] = gtype
    if gfuel and len(gfuel) == len(G): G['fuel'] = gfuel
    for c in ['id', 'type', 'area', 'zone']: B[c] = B[c].astype(int)
    for c in ['bus', 'status']: G[c] = G[c].astype(int)
    for c in ['from', 'to', 'status']: R[c] = R[c].astype(int)
    GC = [dict(model=int(r[0]), startup=r[1], shutdown=r[2], n=int(r[3]), coeffs=r[4:]) for r in gc] if gc else []
    return base, B, G, R, GC, head


LIC = {'case9': 'MATPOWER case data (see case file header; IEEE/WSCC 9-bus, Anderson & Fouad)',
       'case14': 'IEEE 14-bus, public (U. Washington Power Systems Test Case Archive)',
       'case30': 'IEEE 30-bus, public (U. Washington archive)', 'case39': 'IEEE 39-bus New England (Athay et al. 1979)',
       'case57': 'IEEE 57-bus, public (U. Washington archive)', 'case118': 'IEEE 118-bus, public (U. Washington archive)',
       'case300': 'IEEE 300-bus, public (U. Washington archive)',
       'case_ACTIVSg200': 'CC-BY-4.0, Birchfield et al. 2017 (Texas A&M synthetic, Illinois footprint)',
       'case_ACTIVSg500': 'CC-BY-4.0, Birchfield et al. 2017 (Texas A&M synthetic, South Carolina footprint)',
       'case_ACTIVSg2000': 'CC-BY-4.0, Birchfield et al. 2017 (Texas A&M synthetic, Texas footprint)',
       'case_RTS_GMLC': 'RTS-GMLC (Barrows et al. 2020), see case header'}

# approximate coordinates for ACTIVSg2000 buses: bus names are Texas place names; centroid of the US Census
# Gazetteer place of that name (if the gazetteer file is available locally)
GEOCASE = {'case_ACTIVSg2000': 'tx', 'case_ACTIVSg200': 'il', 'case_ACTIVSg500': 'sc'}
def gazetteer(st):
    f = f'{GRID_ROOT}/data/raw/census/gaz_places_{st}.txt'
    if not os.path.exists(f): return None
    g = pd.read_csv(f, sep='\t', dtype=str); g.columns = [c.strip() for c in g.columns]
    g['key'] = g.NAME.str.upper().str.replace(r'\s+(CITY|TOWN|CDP|VILLAGE)$', '', regex=True).str.replace(r'[^A-Z ]', '', regex=True).str.replace(' ', '')
    return g.drop_duplicates('key').set_index('key')[['INTPTLAT', 'INTPTLONG']].astype(float)

index = {}
for path in sorted(glob.glob(f'{RAW}/*.m')):
    name = os.path.basename(path)[:-2]
    base, B, G, R, GC, head = load_m(path)
    geo = None
    gaz = gazetteer(GEOCASE[name]) if name in GEOCASE else None
    if gaz is not None and 'name' in B:
        key = B.name.str.replace(r'\s+\d+\s+\d+$', '', regex=True).str.upper().str.replace(r'[^A-Z ]', '', regex=True).str.replace(' ', '')
        B['lat'] = key.map(gaz.INTPTLAT); B['lon'] = key.map(gaz.INTPTLONG)
        geo = float(B.lat.notna().mean())
    net = dict(name=name, baseMVA=base, source=f'MATPOWER GitHub master, data/{name}.m', licence=LIC.get(name, ''),
               header=head, n_bus=len(B), n_gen=len(G), n_branch=len(R),
               bus=B.replace({np.nan: None}).to_dict('records'), gen=G.to_dict('records'), branch=R.to_dict('records'), gencost=GC)
    json.dump(net, open(f'{OUT}/{name}.json', 'w'))
    B.to_parquet(f'{OUT}/{name}_bus.parquet', index=False); G.to_parquet(f'{OUT}/{name}_gen.parquet', index=False)
    R.to_parquet(f'{OUT}/{name}_branch.parquet', index=False)
    index[name] = dict(n_bus=len(B), n_gen=len(G), n_branch=len(R), load_MW=float(B.Pd_MW.sum()), gen_Pmax_MW=float(G.Pmax_MW.sum()),
                       licence=LIC.get(name, ''), geo_coverage=geo)
    print(name, index[name])
json.dump(index, open(f'{OUT}/index.json', 'w'), indent=1)
