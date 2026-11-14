"""The vehicle platform's scoreboard: every headline number in one file, so physics versions can be compared.

Reads the check results in this folder (written by run_all) and the three models' cards and ladders, and writes
scoreboard_<PHYSICS_VERSION>.json. With --compare A B it prints the two side by side.

usage:
    python scoreboard.py                  # write scoreboard for the current physics version
    python scoreboard.py --compare roadsim-physics-v1 roadsim-physics-v2
"""
import os, sys, json, glob
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
VEH = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(VEH))
sys.path.insert(0, os.path.join(VEH, 'roadsim'))


sys.path.insert(0, HERE)
import _opts
_opts.init()


def _load(name):
    p = os.path.join(HERE, _opts.out(name))
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None


def checks():
    out = {}
    c = _load('cycles_vs_epa.json')
    if c:
        for arch, cyc in c['summary']['heldout'].items():
            for name, s in cyc.items():
                if name == 'all_main_cycles':
                    continue
                out[f'epa_heldout.{arch}.{name}.median_pct'] = s['median_pct']
                out[f'epa_heldout.{arch}.{name}.iqr_pct'] = round(s['p75_pct'] - s['p25_pct'], 2)
                out[f'epa_heldout.{arch}.{name}.n'] = s['n']
    c = _load('coastdown.json')
    if c:
        for v, e in zip(c['standard_conditions']['speeds_mph'], c['standard_conditions']['abs_err_pct_median']):
            out[f'coastdown.F{v}.abs_err_pct_median'] = e
    c = _load('fastsim_crosscheck.json')
    if c:
        for arch, cyc in c['summary'].items():
            for name, s in cyc.items():
                m = s.get('roadsim_vs_fastsim_matched_components')
                if m:
                    out[f'fastsim_matched.{arch}.{name}.median_pct'] = m['median_pct']
    c = _load('ledger.json')
    if c:
        out['ledger.closure_rel_max'] = c['closure_rel_max']
        out['ledger.n_over_0p5pct'] = c['n_over_0p5pct']
    c = _load('heavy_duty.json')
    if c:
        out['hd.class8_diesel.g_co2_per_ton_mile'] = c['class8_sleeper_diesel']['composite_g_co2_per_ton_mile']
        out['hd.class8_bev.kWh_per_mi'] = c['class8_bev']['composite_kWh_per_mi_dc']
    c = _load('generator.json')
    if c:
        for arch, cyc in c['simulated'].items():
            for name, s in cyc.items():
                out[f'generator.{arch}.{name}.median'] = s['median']
    c = _load('d3_twin__dev.json') or _load('d3_twin.json')
    if c:
        for arch, d in c['summary'].get('dev', {}).items():
            p = ((d or {}).get('all') or {}).get('primary_pct')
            if p:
                out[f'd3_dev.{arch}.primary.median_pct'] = p['median']; out[f'd3_dev.{arch}.primary.iqr_pct'] = p['iqr']
    c = _load('ved_twin.json')
    if c:
        for arch, d in c['summary']['by_arch'].items():
            p = d.get('err_primary')
            if p:
                out[f'ved_dev_twin.{arch}.primary.median_pct'] = p['median']; out[f'ved_dev_twin.{arch}.primary.iqr_pct'] = p['iqr']
    c = _load('epa_pool.json')
    if c:
        for cyc, st in c['summary']['all'].items():
            if st:
                out[f'epa_dev_pool.conv.{cyc}.median_pct'] = st['median']; out[f'epa_dev_pool.conv.{cyc}.iqr_pct'] = st['iqr']
    c = _load('speed.json')
    if c:
        for name, r in c['results'].items():
            if isinstance(r, dict):
                out[f'speed.{name}.x_realtime_udds'] = r.get('x_realtime_udds')
    return out


def models():
    out = {}
    for mid in ('energynet', 'tripdetective', 'hybridpilot'):
        d = os.path.join(REPO, 'models', mid)
        card = yaml.safe_load(open(os.path.join(d, 'model.yaml'), encoding='utf-8'))
        out[f'{mid}.version'] = card['version']
        for r in card['exam']['results']:
            out[f'{mid}.exam.{r["metric"]}'] = {'model': r['model'], 'verdict': r['verdict'],
                                                'baselines': {b['name']: b['value'] for b in r['baselines']}}
        for c in card['exam'].get('coverage', []):
            out[f'{mid}.coverage68.{c["output"]}'] = c['actual']
        lad = yaml.safe_load(open(os.path.join(d, 'ratchet.yaml'), encoding='utf-8'))
        for L in lad['ladders']:
            for r in L['rungs']:
                out[f'{mid}.rung.{L.get("name", "")}.{r["id"]}'] = {'sim': r.get('sim'), 'decision': r.get('decision'),
                                                                   'metrics': r.get('metrics')}
    return out


def write():
    from roadsim import PHYSICS_VERSION
    name = _opts.name()
    sb = {'physics_version': name if name != 'roadsim-physics-v1' else PHYSICS_VERSION, 'checks': checks(), 'models': models()}
    p = os.path.join(HERE, f'scoreboard_{sb["physics_version"]}.json')
    json.dump(sb, open(p, 'w', encoding='utf-8'), indent=1, sort_keys=True)
    print('wrote', os.path.relpath(p, REPO), len(sb['checks']), 'check numbers,', len(sb['models']), 'model numbers')


def compare(a, b):
    A = json.load(open(os.path.join(HERE, f'scoreboard_{a}.json'), encoding='utf-8'))
    B = json.load(open(os.path.join(HERE, f'scoreboard_{b}.json'), encoding='utf-8'))
    for part in ('checks', 'models'):
        print(f'\n{part:<60} {a:>24} {b:>24}')
        for k in sorted(set(A[part]) | set(B[part])):
            x, y = A[part].get(k), B[part].get(k)
            if x != y:
                print(f'{k[:60]:<60} {str(x)[:24]:>24} {str(y)[:24]:>24}')


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--compare':
        compare(sys.argv[2], sys.argv[3])
    else:
        write()
