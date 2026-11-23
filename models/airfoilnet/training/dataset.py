"""Real data units for AirfoilNet: UIUC LSAT clean runs (vols 1-5) matched to as-tested coordinates.

unit = one airfoil design (all its wind-tunnel models together); family = designer series.
Writes data/units.json  {airfoil: {family, coords_src, coords:[[x,y],...], runs:[{vol,label,Re,kind,data}]}}
"""
import os, re, json, glob, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from uiuc_lsat import parse_all, _entries_concat  # noqa: E402
import xsolver as xs  # noqa: E402  (geometry helpers only)

RAW = os.path.join(HERE, '..', 'data', 'raw')
DB = os.path.join(RAW, 'coord_seligFmt', 'coord_seligFmt')

EXCLUDE_WORDS = ('trip', 'flap', 'gurney', 'tape', 'covering', 'serration', 'cubes', 'holes', 'square wave',
                 'open bay', 'u.s.t', 'l.s.t', 'baseline leading edge', 'plain trip', 'zig')

# label base -> canonical airfoil id (only where the automatic rule fails)
OVERRIDE = {'clark-y': 'clarky', 'pt-40': 'pt40', 'goe 417a': 'goe417a', 'falcon 56 mk iii': 'falcon',
            'cg ultimate': 'ultimate', 'ultra-sport 1000': 'us1000', 'trainer 60': 'trainer60', 'usnps-4': 'usnps4',
            'bw-3': 'bw3', 'fx 63-137': 'fx63137', 'naca 43012a': 'naca43012a', 'm06-13-128': 'm06',
            'ch 10-48-13': 'ch10', 'fx 74-cl5-140 mod': 'fx74mod', 'naca 0009': 'n0009', 'naca 6409': 'n6409',
            'naca 64a010': 'n64a', 'naca 2414': 'n2414', 'naca 2415': 'n2415', 'du 86-084/18': 'du86',
            'davis 3r': 'davis3r', 'cr-001': 'cr001', 'ld-79': 'ld79', 'm6 (65%)': 'm665', 'm6 (85%)': 'm685',
            'esa': 'esa', 'avistar': 'avistar', 'dh4009': 'dh4009', 'mb253515': 'mb25', 's1223 rtl': 's1223rtl',
            'ag35-r': 'ag35', 'gemini': 'gemini', 'wasp': 'wasp'}

KIT = {'avistar', 'ultimate', 'esa', 'falcon', 'pt40', 'trainer60', 'us1000', 'gemini', 'wasp', 'clarky', 'davis3r',
       'bw3', 'dh4009'}


def label_key(label):
    """'SD7037 (B)' -> ('sd7037', 'b'); 'E387 (C)' -> ('e387','c')"""
    lab = label.strip()
    m = re.match(r'^(.*?)\s*\(([A-Ea-e])\)\s*$', lab)
    model = None
    if m:
        lab, model = m.group(1), m.group(2).lower()
    base = lab.lower().strip()
    if base in OVERRIDE:
        return OVERRIDE[base], model
    return re.sub(r'[\s\-]', '', base), model


def family_of(aid):
    if aid in KIT:
        return 'kit:' + aid
    if aid.startswith('naca') or re.match(r'^n\d', aid):
        return 'NACA'
    m = re.match(r'^([a-z]+)', aid)
    p = m.group(1) if m else aid
    fam = {'sd': 'SD (Selig-Donovan)', 'sa': 'SA (Selig-Ananda)', 'sg': 'SG (Selig-Giguere)', 'sh': 'SH (Selig-Hamid)',
           's': 'S (Selig)', 'e': 'E (Eppler)', 'fx': 'FX (Wortmann)', 'ag': 'AG (Drela)', 'cal': 'CAL',
           'ma': 'MA', 'rg': 'RG (Girsberger)', 'a': 'A (Archer)', 'goe': 'Goe (Goettingen)', 'lrn': 'LRN',
           'gm': 'GM', 'm': 'M (Mueller)', 'mh': 'MH (Hepperle)', 'k': 'K', 'j': 'J', 'r': 'R', 'be': 'BE',
           'ch': 'CH', 'mb': 'MB', 'cr': 'CR', 'du': 'DU (Delft)', 'ld': 'LD', 'usnps': 'USNPS'}.get(p, p.upper())
    return fam


def load_coord_files():
    """filename -> (n,2) from the COORD0x.TXT concatenations (as-tested .dap and design .dat)."""
    out = {}
    for v in (1, 2, 3):
        for name, body in _entries_concat(os.path.join(RAW, f'volume0{v}', f'volume0{v}', f'COORD0{v}.TXT')):
            pts = []
            for line in body.splitlines():
                p = line.split()
                if len(p) >= 2:
                    try:
                        pts.append((float(p[0]), float(p[1])))
                    except ValueError:
                        pass
            if len(pts) > 10:
                out[name.lower()] = np.array(pts)
    return out


def find_coords(aid, model, cf):
    cands = []
    if model:
        cands += [f'{aid}{model}.dap']
    cands += [f'{aid}.dap', f'{aid}sm.dat', f'{aid}.dat']
    for c in cands:
        if c in cf:
            return c, cf[c]
    for c in (f'{aid}.dat', f'{aid}sm.dat'):
        p = os.path.join(DB, c)
        if os.path.exists(p):
            return 'db:' + c, xs.read_dat(p)
    return None, None


def build_units():
    recs = parse_all()
    cf = load_coord_files()
    units = {}
    skipped = {}
    for r in recs:
        com = (r['comment'] or '').lower()
        if 'clean' not in com or any(w in com for w in EXCLUDE_WORDS):
            if not (('flap = 0' in com) and False):
                skipped.setdefault(r['airfoil'], set()).add(r['comment']); continue
        if r['airfoil'] is None or 'flat plate' in r['airfoil'].lower():
            continue
        aid, model = label_key(r['airfoil'])
        if r['vol'] == 1 and r['kind'] == 'lift':
            for row in r['data']:
                row[2] = None          # volume 1 Cm is not valid
        u = units.setdefault(aid, dict(family=family_of(aid), runs=[], models={}))
        src, X = find_coords(aid, model, cf)
        u['models'][model or '-'] = src
        if X is not None:
            try:
                Y = xs.normalize(X)
                r2 = dict(vol=r['vol'], label=r['airfoil'], model=model or '-', Re=r['Re'], kind=r['kind'],
                          data=r['data'], coords_src=src)
                u['runs'].append(r2)
                u.setdefault('coords', {})[model or '-'] = Y.round(6).tolist()
            except Exception as e:
                print('coord fail', aid, e)
        else:
            u.setdefault('missing', []).append(r['airfoil'])
    units = {k: v for k, v in units.items() if v['runs']}
    return units, skipped


if __name__ == '__main__':
    units, skipped = build_units()
    json.dump(units, open(os.path.join(RAW, '..', 'units.json'), 'w'))
    for k, u in sorted(units.items(), key=lambda kv: kv[1]['family']):
        nl = sum(r['kind'] == 'lift' for r in u['runs']); nd = sum(r['kind'] == 'drag' for r in u['runs'])
        print(f"{u['family']:22s} {k:12s} models={u['models']} lift={nl} drag={nd} missing={u.get('missing')}")
    print(len(units), 'airfoils;', len(set(u['family'] for u in units.values())), 'families')
