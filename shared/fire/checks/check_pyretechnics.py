"""Cross-check FireSim against pyretechnics (SIG's independent open Python implementation, PyPI 2026.10.20,
built in the isolated venv firesim/.refenv; values written by ref/pyre_run.py).

Two comparisons for the surface model:
  (i)  same fuel table: FireSim run with pyretechnics' own fuel parameters, so only the equations differ;
  (ii) as shipped: each package with its own fuel table (shows the table differences found).
Plus crown fire (Van Wagner I0, Cruz 2005 ROS, crown L/W) and the wind adjustment factor.
"""
import json
import math

import numpy as np

from _common import write, OUT
import firesim as fs
from firesim.fuels import TONS_AC_TO_LB_FT2, FM

P = json.load(open(f'{OUT}/pyre_values.json'))
rows = P['surface']
tables = {int(k): v for k, v in P['tables'].items()}

# ---- table differences
tab_diff = []
for num, t in tables.items():
    i = int(FM['lut'][num])
    mine_w = FM['w0'][i]
    w_o = t['w_o']
    sig = t['sigma']
    d = {}
    # pyretechnics slot order: 1h, 10h, 100h, dead herb, live herb, live woody (same as ours)
    for j, nm in enumerate(['1h', '10h', '100h', 'deadherb', 'liveherb', 'livewoody']):
        if abs(w_o[j] - mine_w[j]) > 6e-5:
            d[f'load_{nm}_lbft2'] = (round(mine_w[j], 5), round(w_o[j], 5))
        if (w_o[j] > 0 or mine_w[j] > 0) and abs(sig[j] - FM['sav'][i][j]) > 0.5:
            d[f'sav_{nm}'] = (FM['sav'][i][j], sig[j])
    if abs(t['delta'] - FM['depth'][i]) > 1e-6:
        d['depth'] = (FM['depth'][i], t['delta'])
    if abs(t['h'][0] - FM['heat'][i][0]) > 1:
        d['heat'] = (FM['heat'][i][0], t['h'][0])
    if abs(t['M_x'][0] - FM['mx'][i]) > 1e-6:
        d['mx'] = (FM['mx'][i], t['M_x'][0])
    if bool(t['dynamic']) != bool(FM['dynamic'][i]):
        d['dynamic'] = (bool(FM['dynamic'][i]), bool(t['dynamic']))
    if d:
        tab_diff.append(dict(number=num, code=fs.code_of(num), diffs=d))
print('fuel table differences:', len(tab_diff))
for t in tab_diff:
    big = {k: v for k, v in t['diffs'].items() if not k.startswith('load') or abs(v[0] - v[1]) > 2e-4}
    if big:
        print('  ', t['code'], big)

# custom fuels from pyretechnics' table (for comparison i)
custom = {}
for num, t in tables.items():
    w = np.array(t['w_o']) / TONS_AC_TO_LB_FT2
    s = t['sigma']
    custom[num] = dict(code=f'P{num}', w1=w[0], w10=w[1], w100=w[2], wherb=w[4] + w[3], wwoody=w[5], sav1=s[0],
                       savherb=s[4] if s[4] > 0 else 1500.0, savwoody=s[5] if s[5] > 0 else 1500.0, depth=t['delta'],
                       mx=t['M_x'][0] * 100, dynamic=bool(t['dynamic']), heat_dead=t['h'][0], heat_live=t['h'][4] or t['h'][0])

DEAD = {'D1': (3, 4, 5), 'D2': (6, 7, 8), 'D3': (9, 10, 11), 'D4': (12, 13, 14)}
LIVE = {'L1': (30, 60), 'L2': (60, 90), 'L3': (90, 120), 'L4': (120, 150)}


def mine(r, table):
    d = DEAD[r['dead']]
    l = LIVE[r['live']]
    cur = min(1.0, max(0.0, 4.0 / 3.0 - (l[0] / 100.0) / 0.9))     # pyretechnics' curing rule, to isolate equations
    kw = dict(m1=d[0], m10=d[1], m100=d[2], mherb=l[0], mwoody=l[1], wind_speed=r['wind_mph'] * 0.44704,
              wind_height='midflame', slope=math.degrees(math.atan(r['slope_pct'] / 100.0)), aspect=180.0,
              wind_dir=(r['wind_rel_deg'] + 180.0) % 360.0, curing=cur, wind_limit='none', lw='behaveplus',
              lw_max=1e9, crown='none')
    if table == 'pyre':
        return fs.behave(r['num'], custom_fuels={r['num']: custom[r['num']]}, **kw)
    return fs.behave(r['num'], **kw)


res = {}
for table in ('pyre', 'own'):
    errs = {k: [] for k in ('r0', 'ros', 'fli', 'fl', 'lw')}
    errs_x = {k: [] for k in ('ros', 'dir')}
    worst = []
    for r in rows:
        b = mine(r, table)
        if r['ros_mmin'] <= 1e-6 and b['ros_mmin'] <= 1e-6:
            continue
        cross = r['slope_pct'] > 0 and r['wind_mph'] > 0     # wind on a slope: 3-D projection differs by design
        e_ros = (b['ros_mmin'] - r['ros_mmin']) / max(r['ros_mmin'], 1e-6)
        if cross:
            errs_x['ros'].append(e_ros)
            dd = (b['direction_max_deg'] - r['dir_deg'] + 180) % 360 - 180
            errs_x['dir'].append(dd)
            continue
        errs['r0'].append((b['r0_mmin'] - r['r0_mmin']) / max(r['r0_mmin'], 1e-6))
        errs['ros'].append(e_ros)
        errs['fli'].append((b['fli_kwm'] - r['fli_kwm']) / max(r['fli_kwm'], 1e-6))
        errs['fl'].append((b['flame_length_m'] - r['fl_m']) / max(r['fl_m'], 1e-6))
        errs['lw'].append((b['lw_ratio'] - r['lw']) / r['lw'])
        if abs(e_ros) > 0.01:
            worst.append((abs(e_ros), r['num'], r['dead'], r['live'], r['wind_mph'], r['slope_pct'], b['ros_mmin'],
                          r['ros_mmin']))
    stats = {}
    for k, v in errs.items():
        a = np.abs(np.array(v))
        stats[k] = dict(n=len(a), median_abs_rel=float(np.median(a)), p99_abs_rel=float(np.quantile(a, 0.99)),
                        max_abs_rel=float(a.max()), frac_within_0p1pct=float(np.mean(a < 1e-3)),
                        frac_within_1pct=float(np.mean(a < 1e-2)))
    stats['wind_on_slope_ros'] = dict(n=len(errs_x['ros']), median_abs_rel=float(np.median(np.abs(errs_x['ros']))),
                                    max_abs_rel=float(np.max(np.abs(errs_x['ros']))))
    stats['wind_on_slope_direction_deg'] = dict(median_abs=float(np.median(np.abs(errs_x['dir']))),
                                              max_abs=float(np.max(np.abs(errs_x['dir']))))
    worst.sort(reverse=True)
    by_model = {}
    for w in worst:
        by_model.setdefault(fs.code_of(w[1]), 0)
        by_model[fs.code_of(w[1])] += 1
    stats['n_cases_over_1pct_by_model'] = by_model
    stats['worst'] = [dict(rel=w[0], model=fs.code_of(w[1]), dead=w[2], live=w[3], wind_mph=w[4], slope_pct=w[5],
                           firesim=w[6], pyretechnics=w[7]) for w in worst[:15]]
    res[table] = stats
    print(table, {k: (v['median_abs_rel'], v['max_abs_rel']) for k, v in stats.items() if isinstance(v, dict) and 'max_abs_rel' in v})
    print('   >1% by model:', by_model)

# ---- crown
ce = dict(i0=[], cros=[], lw=[], ftype_agree=0, n=0)
for c in P['crown']:
    i0 = fs.crown.van_wagner_i0(c['cbh'], c['fmc'] * 100)
    ce['i0'].append((i0 - c['i0']) / c['i0'])
    ra = fs.crown.cruz2005_active(c['u10_kmh'], c['cbd'], c['effm'] * 100)
    rac = fs.crown.r_active_critical(c['cbd'])
    ftype = 3 if ra > rac else 2
    ros = ra if ftype == 3 else fs.crown.cruz2005_passive(c['u10_kmh'], c['cbd'], c['effm'] * 100)
    ce['cros'].append((ros - c['cros']) / c['cros'])
    lw = fs.crown.crown_lw(c['u10_kmh'] / 3.6 / 1.15 * 2.23694)
    ce['lw'].append((lw - c['lw']) / c['lw'])
    ce['ftype_agree'] += int(ftype == c['ftype'])
    ce['n'] += 1
crown = {k: dict(max_abs_rel=float(np.max(np.abs(v))), median_abs_rel=float(np.median(np.abs(v))))
         for k, v in ce.items() if isinstance(v, list)}
crown['fire_type_agreement'] = ce['ftype_agree'] / ce['n']
crown['note'] = ('I0: FireSim uses 25.9 per percent foliar moisture (Van Wagner 1977), pyretechnics 2600 per unit '
                 'fraction (26.0); the I0 difference is that constant.')
print('crown', crown)
# ---- WAF (pyretechnics assumes crown ratio 1, i.e. cbh = 0)
we = []
for w in P['waf']:
    mw, _ = fs.rothermel.waf(np.array(w['depth_ft']), np.array(w['cc'] * 100), np.array(w['ch_ft'] * 0.3048),
                             np.array(0.0))
    we.append(dict(case=w, firesim=float(mw), pyretechnics=w['waf']))
d = np.array([abs(x['firesim'] - x['pyretechnics']) for x in we])
waf = dict(n=len(d), max_abs=float(d.max()), n_agree_1e3=int((d < 1e-3).sum()),
           disagreements=[x for x in we if abs(x['firesim'] - x['pyretechnics']) >= 1e-3])
print('waf', waf['n_agree_1e3'], '/', waf['n'])
write('pyretechnics_crosscheck.json', dict(
    what='FireSim vs pyretechnics 2026.10.20 (PyPI sdist, built in firesim/.refenv with Cython 3.0.12); '
         f'{len(rows)} surface cases = 53 fuel models x 16 moisture scenarios x 4 midflame winds x slopes/wind '
         'directions; crown and WAF grids. Wind limit off, L/W = 1 + 0.25 UE (both), pyretechnics curing rule used '
         'in both. pyretechnics computes in float32.',
    surface_same_table=res['pyre'], surface_own_tables=res['own'], crown=crown, waf=waf, fuel_table_differences=tab_diff,
    notes=['pyretechnics stores Anderson models 4, 5 and 10 live load in the herbaceous slot (live herbaceous '
           'moisture is then used); BehavePlus/GTR-371 treat it as live woody.',
           'pyretechnics FM7 live SAV 1550 vs 1500 in GTR-371 Table 8.',
           'pyretechnics loads are rounded to 4 decimals of lb/ft2.',
           "pyretechnics' 'behave' L/W option uses Anderson (1983) coefficients for m/s with a wind in mi/h; we compare "
           "with the 'rothermel' option (1 + 0.25 UE), which is BehavePlus's documented relation.",
           'Wind-on-slope cases differ by design and are reported separately: pyretechnics projects the horizontal '
           'wind and the slope onto the slope-tangential plane (a horizontal wind gains magnitude 1/cos(slope) '
           'in that plane and ROS is a slope-plane distance); FireSim follows BehavePlus (Rothermel wind factor from '
           'the given midflame wind, horizontal vector addition of wind and slope effects). Flat and slope-only '
           'cases test the same equations.']))
