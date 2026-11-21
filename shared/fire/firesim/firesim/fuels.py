"""Standard fire behaviour fuel models: Anderson (1982) 13 and Scott & Burgan (2005) 40, plus non-burnable codes.

Values are transcribed from Andrews (2018) RMRS-GTR-371, Table 8a-8e (identical to Scott & Burgan 2005, Table 7).
Loads are tons/acre, SAV ft2/ft3, depth ft, dead moisture of extinction percent, heat content Btu/lb, as published.
Constants for every model: 10-h SAV 109, 100-h SAV 30 ft2/ft3, particle density 32 lb/ft3, total mineral 0.0555,
effective mineral 0.010, heat content 8000 Btu/lb.

The particle order used everywhere in firesim is
    0: dead 1-h, 1: dead 10-h, 2: dead 100-h, 3: dead herbaceous (from curing), 4: live herbaceous, 5: live woody.
"""
import numpy as np

TONS_AC_TO_LB_FT2 = 2000.0 / 43560.0          # 0.0459137
LB_FT2_TO_KG_M2 = 4.88242764
FT_TO_M = 0.3048

# code: (number, name, type, w1, w10, w100, wherb, wwoody, sav1, savherb, savwoody, depth, mx)
_TABLE = {
    # Anderson (1982) 13
    'FM1': (1, 'Short grass', 'S', 0.74, 0, 0, 0, 0, 3500, None, None, 1.0, 12),
    'FM2': (2, 'Timber grass and understory', 'S', 2.0, 1.0, 0.50, 0.50, 0, 3000, 1500, None, 1.0, 15),
    'FM3': (3, 'Tall grass', 'S', 3.0, 0, 0, 0, 0, 1500, None, None, 2.5, 25),
    'FM4': (4, 'Chaparral', 'S', 5.0, 4.0, 2.0, 0, 5.0, 2000, None, 1500, 6.0, 20),
    'FM5': (5, 'Brush', 'S', 1.0, 0.50, 0, 0, 2.0, 2000, None, 1500, 2.0, 20),
    'FM6': (6, 'Dormant brush, hardwood slash', 'S', 1.5, 2.5, 2.0, 0, 0, 1750, None, None, 2.5, 25),
    'FM7': (7, 'Southern rough', 'S', 1.13, 1.87, 1.5, 0, 0.37, 1750, None, 1500, 2.5, 40),
    'FM8': (8, 'Short needle litter', 'S', 1.5, 1.0, 2.5, 0, 0, 2000, None, None, 0.2, 30),
    'FM9': (9, 'Long needle or hardwood litter', 'S', 2.92, 0.41, 0.15, 0, 0, 2500, None, None, 0.2, 25),
    'FM10': (10, 'Timber litter and understory', 'S', 3.0, 2.0, 5.0, 0, 2.0, 2000, None, 1500, 1.0, 25),
    'FM11': (11, 'Light logging slash', 'S', 1.5, 4.5, 5.5, 0, 0, 1500, None, None, 1.0, 15),
    'FM12': (12, 'Medium logging slash', 'S', 4.0, 14.0, 16.5, 0, 0, 1500, None, None, 2.3, 20),
    'FM13': (13, 'Heavy logging slash', 'S', 7.0, 23.0, 28.0, 0, 0, 1500, None, None, 3.0, 25),
    # Scott & Burgan (2005) 40
    'GR1': (101, 'Short, sparse, dry climate grass', 'D', 0.1, 0, 0, 0.3, 0, 2200, 2000, None, 0.4, 15),
    'GR2': (102, 'Low load, dry climate grass', 'D', 0.1, 0, 0, 1.0, 0, 2000, 1800, None, 1.0, 15),
    'GR3': (103, 'Low load, very coarse, humid climate grass', 'D', 0.1, 0.4, 0, 1.5, 0, 1500, 1300, None, 2.0, 30),
    'GR4': (104, 'Moderate load, dry climate grass', 'D', 0.25, 0, 0, 1.9, 0, 2000, 1800, None, 2.0, 15),
    'GR5': (105, 'Low load, humid climate grass', 'D', 0.4, 0, 0, 2.5, 0, 1800, 1600, None, 1.5, 40),
    'GR6': (106, 'Moderate load, humid climate grass', 'D', 0.1, 0, 0, 3.4, 0, 2200, 2000, None, 1.5, 40),
    'GR7': (107, 'High load, dry climate grass', 'D', 1.0, 0, 0, 5.4, 0, 2000, 1800, None, 3.0, 15),
    'GR8': (108, 'High load, very coarse, humid climate grass', 'D', 0.5, 1.0, 0, 7.3, 0, 1500, 1300, None, 4.0, 30),
    'GR9': (109, 'Very high load, humid climate grass', 'D', 1.0, 1.0, 0, 9.0, 0, 1800, 1600, None, 5.0, 40),
    'GS1': (121, 'Low load, dry climate grass-shrub', 'D', 0.2, 0, 0, 0.5, 0.65, 2000, 1800, 1800, 0.9, 15),
    'GS2': (122, 'Moderate load, dry climate grass-shrub', 'D', 0.5, 0.5, 0, 0.6, 1.0, 2000, 1800, 1800, 1.5, 15),
    'GS3': (123, 'Moderate load, humid climate grass-shrub', 'D', 0.3, 0.25, 0, 1.45, 1.25, 1800, 1600, 1600, 1.8, 40),
    'GS4': (124, 'High load, humid climate grass-shrub', 'D', 1.9, 0.3, 0.1, 3.4, 7.1, 1800, 1600, 1600, 2.1, 40),
    'SH1': (141, 'Low load, dry climate shrub', 'D', 0.25, 0.25, 0, 0.15, 1.3, 2000, 1800, 1600, 1.0, 15),
    'SH2': (142, 'Moderate load, dry climate shrub', 'S', 1.35, 2.4, 0.75, 0, 3.85, 2000, None, 1600, 1.0, 15),
    'SH3': (143, 'Moderate load, humid climate shrub', 'S', 0.45, 3.0, 0, 0, 6.2, 1600, None, 1400, 2.4, 40),
    'SH4': (144, 'Low load, humid climate timber-shrub', 'S', 0.85, 1.15, 0.2, 0, 2.55, 2000, 1800, 1600, 3.0, 30),
    'SH5': (145, 'High load, dry climate shrub', 'S', 3.6, 2.1, 0, 0, 2.9, 750, None, 1600, 6.0, 15),
    'SH6': (146, 'Low load, humid climate shrub', 'S', 2.9, 1.45, 0, 0, 1.4, 750, None, 1600, 2.0, 30),
    'SH7': (147, 'Very high load, dry climate shrub', 'S', 3.5, 5.3, 2.2, 0, 3.4, 750, None, 1600, 6.0, 15),
    'SH8': (148, 'High load, humid climate shrub', 'S', 2.05, 3.4, 0.85, 0, 4.35, 750, None, 1600, 3.0, 40),
    'SH9': (149, 'Very high load, humid climate shrub', 'D', 4.5, 2.45, 0, 1.55, 7.0, 750, 1800, 1500, 4.4, 40),
    'TU1': (161, 'Light load, dry climate timber-grass-shrub', 'D', 0.2, 0.9, 1.5, 0.2, 0.9, 2000, 1800, 1600, 0.6, 20),
    'TU2': (162, 'Moderate load, humid climate timber-shrub', 'S', 0.95, 1.8, 1.25, 0, 0.2, 2000, None, 1600, 1.0, 30),
    'TU3': (163, 'Moderate load, humid climate timber-grass-shrub', 'D', 1.1, 0.15, 0.25, 0.65, 1.1, 1800, 1600, 1400, 1.3, 30),
    'TU4': (164, 'Dwarf conifer understory', 'S', 4.5, 0, 0, 0, 2.0, 2300, None, 2000, 0.5, 12),
    'TU5': (165, 'Very high load, dry climate timber-shrub', 'S', 4.0, 4.0, 3.0, 0, 3.0, 1500, None, 750, 1.0, 25),
    'TL1': (181, 'Low load, compact conifer litter', 'S', 1.0, 2.2, 3.6, 0, 0, 2000, None, None, 0.2, 30),
    'TL2': (182, 'Low broadleaf litter', 'S', 1.4, 2.3, 2.2, 0, 0, 2000, None, None, 0.2, 25),
    'TL3': (183, 'Moderate load conifer litter', 'S', 0.5, 2.2, 2.8, 0, 0, 2000, None, None, 0.3, 20),
    'TL4': (184, 'Small downed logs', 'S', 0.5, 1.5, 4.2, 0, 0, 2000, None, None, 0.4, 25),
    'TL5': (185, 'High load conifer litter', 'S', 1.15, 2.5, 4.4, 0, 0, 2000, None, 1600, 0.6, 25),
    'TL6': (186, 'Moderate load broadleaf litter', 'S', 2.4, 1.2, 1.2, 0, 0, 2000, None, None, 0.3, 25),
    'TL7': (187, 'Large downed logs', 'S', 0.3, 1.4, 8.1, 0, 0, 2000, None, None, 0.4, 25),
    'TL8': (188, 'Long-needle litter', 'S', 5.8, 1.4, 1.1, 0, 0, 1800, None, None, 0.3, 35),
    'TL9': (189, 'Very high load broadleaf litter', 'S', 6.65, 3.3, 4.15, 0, 0, 1800, None, 1600, 0.6, 35),
    'SB1': (201, 'Low load activity fuel', 'S', 1.5, 3.0, 11.0, 0, 0, 2000, None, None, 1.0, 25),
    'SB2': (202, 'Moderate load activity fuel or low load blowdown', 'S', 4.5, 4.25, 4.0, 0, 0, 2000, None, None, 1.0, 25),
    'SB3': (203, 'High load activity fuel or moderate load blowdown', 'S', 5.5, 2.75, 3.0, 0, 0, 2000, None, None, 1.2, 25),
    'SB4': (204, 'High load blowdown', 'S', 5.25, 3.5, 5.25, 0, 0, 2000, None, None, 2.7, 25),
}
# Anderson (1982) values where GTR-371 Table 8 prints rounded inputs (its derived columns match these): FM7 loads
# 1.13/1.87 t/ac (live SAV 1500 as printed; pyretechnics uses 1550); FM9 1-h load 2.92 t/ac.
# Notes on the transcription: SH4 lists a herb SAV (1800) but no herb load; TL5 and TL9 list a woody SAV but no
# woody load. These SAVs have no effect. Fuel model 2 is static even though it has live herbaceous load.

# Scott & Burgan (2005) Table 7 lists 9000 Btu/lb for GR6 (GTR-371 Table 8 footnote says 8000 for all models);
# we follow the original source.
HEAT_OVERRIDE = {'GR6': 9000.0}

NONBURNABLE = {91: 'NB1 Urban/developed', 92: 'NB2 Snow/ice', 93: 'NB3 Agricultural', 98: 'NB8 Open water',
               99: 'NB9 Bare ground'}

SAV10, SAV100 = 109.0, 30.0
RHO_P = 32.0
S_T = 0.0555
S_E = 0.010
HEAT = 8000.0

NUM_TO_CODE = {v[0]: k for k, v in _TABLE.items()}
CODE_TO_NUM = {k: v[0] for k, v in _TABLE.items()}

# dense lookup array: index = LANDFIRE number (0..255) -> row in the arrays below (or -1 non-burnable)
MAXCODE = 256


def _build_arrays(extra=None):
    """Arrays indexed by row: w0 (lb/ft2, 6 particles), sav (6), depth (ft), mx (fraction), heat (Btu/lb, 6),
    dynamic (bool), rho_p, st, se (per particle). extra: dict of custom models {number: dict(...)}."""
    rows = []
    for code, v in _TABLE.items():
        num, name, typ, w1, w10, w100, wh, ww, s1, sh, sw, depth, mx = v
        h = HEAT_OVERRIDE.get(code, HEAT)
        rows.append(dict(num=num, code=code, name=name, dynamic=(typ == 'D'), w1=w1, w10=w10, w100=w100, wherb=wh,
                         wwoody=ww, sav1=s1, savherb=sh or 1500.0, savwoody=sw or 1500.0, depth=depth, mx=mx,
                         heat_dead=h, heat_live=h))
    if extra:
        for num, d in extra.items():
            r = dict(num=int(num), code=d.get('code', f'C{num}'), name=d.get('name', 'custom'),
                     dynamic=bool(d.get('dynamic', False)), heat_dead=d.get('heat_dead', HEAT),
                     heat_live=d.get('heat_live', HEAT))
            for k in ('w1', 'w10', 'w100', 'wherb', 'wwoody', 'sav1', 'savherb', 'savwoody', 'depth', 'mx'):
                r[k] = float(d.get(k, 0.0 if k.startswith('w') else 1500.0))
            rows = [x for x in rows if x['num'] != r['num']] + [r]
    n = len(rows)
    w0 = np.zeros((n, 6))
    sav = np.zeros((n, 6))
    heat = np.zeros((n, 6))
    depth = np.zeros(n)
    mx = np.zeros(n)
    dyn = np.zeros(n, np.bool_)
    nums = np.zeros(n, np.int64)
    for i, r in enumerate(rows):
        w0[i] = [r['w1'], r['w10'], r['w100'], 0.0, r['wherb'], r['wwoody']]
        sav[i] = [r['sav1'], SAV10, SAV100, r['savherb'], r['savherb'], r['savwoody']]
        heat[i] = [r['heat_dead']] * 4 + [r['heat_live']] * 2
        depth[i] = r['depth']
        mx[i] = r['mx'] / 100.0
        dyn[i] = r['dynamic']
        nums[i] = r['num']
    w0 *= TONS_AC_TO_LB_FT2
    lut = -np.ones(MAXCODE, np.int64)
    for i, num in enumerate(nums):
        lut[num] = i
    return dict(w0=w0, sav=sav, heat=heat, depth=depth, mx=mx, dynamic=dyn, nums=nums, lut=lut,
                codes=[r['code'] for r in rows], names=[r['name'] for r in rows])


FM = _build_arrays()


def fuel_models(as_frame=True):
    """The 53 standard fuel models (+ non-burnable codes) as a pandas DataFrame (published units) or dict."""
    recs = []
    for code, v in _TABLE.items():
        num, name, typ, w1, w10, w100, wh, ww, s1, sh, sw, depth, mx = v
        recs.append(dict(number=num, code=code, name=name, type='dynamic' if typ == 'D' else 'static',
                         load_1h_tac=w1, load_10h_tac=w10, load_100h_tac=w100, load_herb_tac=wh, load_woody_tac=ww,
                         sav_1h=s1, sav_herb=sh, sav_woody=sw, depth_ft=depth, mx_dead_pct=mx,
                         heat_btu_lb=HEAT_OVERRIDE.get(code, HEAT),
                         burnable=True))
    for num, name in NONBURNABLE.items():
        recs.append(dict(number=num, code=name.split()[0], name=name, type='nonburnable', burnable=False))
    if not as_frame:
        return recs
    import pandas as pd
    return pd.DataFrame(recs).set_index('number')


def lookup_rows(fuel, fm=None):
    """Map LANDFIRE fuel numbers (any int array) to table rows; -1 for non-burnable / unknown / nodata."""
    fm = fm or FM
    f = np.asarray(fuel)
    out = -np.ones(f.shape, np.int64)
    ok = (f >= 0) & (f < MAXCODE)
    out[ok] = fm['lut'][f[ok].astype(np.int64)]
    return out


def code_of(num):
    return NUM_TO_CODE.get(int(num), NONBURNABLE.get(int(num), 'NB?').split()[0])


def number_of(code):
    if isinstance(code, (int, np.integer)):
        return int(code)
    c = str(code).upper()
    if c.isdigit():
        return int(c)
    if c in CODE_TO_NUM:
        return CODE_TO_NUM[c]
    if 'FM' + c in CODE_TO_NUM:
        return CODE_TO_NUM['FM' + c]
    for k, v in NONBURNABLE.items():
        if v.split()[0] == c:
            return k
    raise KeyError(code)
