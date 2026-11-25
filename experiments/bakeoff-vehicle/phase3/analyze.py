"""Vehicle Bake-off, Phase 3: analysis of the once-only exam (reads only the exam's outputs and the arms' own metadata;
chooses nothing, re-scores nothing).

usage: python3 analyze.py                                   (the run named in harness/results/EXAM_DONE.json)
       python3 analyze.py --rehearsal results/rehearsal_<id>.json   (code test on the dev rehearsal)
Writes phase3/results/analysis.json and phase3/results/leaderboards.md.

Headline per board: E16 = the mean of the 16 exam MdAPEs that make up the bake-off's dev score D16 (VED ICE / HEV / EV /
PHEV trips; EPA conv FTP / HWY / US06 / cold FTP / SC03, hybrid FTP / HWY, BEV city / HWY; Mersin diesel / CNG;
ICCT), on the exam. Its interval: a paired cluster bootstrap run per set with shared resamples for every row (same
clusters as the exam: VED cars, EPA test vehicles, Mersin trips, ICCT tests), 2,000 resamples, then averaged over the
16 sets. Per set, the exam's own paired bootstrap (exam.py, vs EnergyNet v2.1, E_phys, the classical baseline) is
reported as is; the comparisons against the in-track controls use score.compare with the exam's seed.
"""
import json
import os
import sys

HARNESS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'harness')  # cloud: /home/claude/bakeoff/vehicle/harness
sys.path.insert(0, HARNESS)
BASE = HARNESS if os.path.isdir(os.path.join(HARNESS, 'results')) else os.path.dirname(HARNESS)  # repo: results/ beside harness/
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import config as C  # noqa: E402
import score as S  # noqa: E402

P3 = os.path.dirname(os.path.abspath(__file__))  # cloud: /home/claude/bakeoff/vehicle/phase3
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work', 'phase3')  # cloud: phase3/results; shipped copies in results/
NBOOT = 2000
E16 = list(S.D16)
VED = ['VED_ICE', 'VED_HEV', 'VED_EV', 'VED_PHEV']
EPA = ['EPA_conv_FTP', 'EPA_conv_HWY', 'EPA_conv_US06', 'EPA_conv_FTPcold', 'EPA_conv_SC03', 'EPA_hev_FTP',
       'EPA_hev_HWY', 'EPA_hev_US06', 'EPA_hev_SC03', 'EPA_hev_FTPcold', 'EPA_bev_FTP', 'EPA_bev_HWY']
HD = ['Mersin_diesel', 'Mersin_cng', 'ICCT', 'ZTBus_legs', 'ZTBus_missions']
ALLSETS = VED + [f'{g}_cars' for g in VED] + EPA + HD
CONDS = ['city', 'highway', 'cold', 'hot', 'hilly', 'short', 'long']
# powertrain groups: (label, sets) -- the board "per powertrain" averages the MdAPEs of these sets
POWERTRAINS = {
    'conventional': ('Conventional (petrol, diesel, CNG)', ['VED_ICE', 'EPA_conv_FTP', 'EPA_conv_HWY', 'EPA_conv_US06',
                                                            'EPA_conv_FTPcold', 'EPA_conv_SC03', 'Mersin_diesel', 'Mersin_cng', 'ICCT']),
    'hybrid': ('Hybrid', ['VED_HEV', 'EPA_hev_FTP', 'EPA_hev_HWY']),
    'plugin': ('Plug-in hybrid', ['VED_PHEV']),
    'bev': ('Battery-electric', ['VED_EV', 'EPA_bev_FTP', 'EPA_bev_HWY']),
    'heavy': ('Heavy duty (Mersin buses, ICCT trucks, ZTBus)', ['Mersin_diesel', 'Mersin_cng', 'ICCT', 'ZTBus_legs']),
}

TABLE_LEARNERS = ['tab_mlp', 'tab_gbm', 'tab_gp', 'formula', 'phys_anchor']
TRACE_LEARNERS = ['trace_gru', 'trace_cnn', 'trace_s4d', 'trace_transformer', 'formula', 'phys_anchor']
CONTROL = {'table': 'tab_mlp', 'trace': 'trace_gru'}
EXCLUDED = ['tab_tabpfn', 'tab_tabpfn_conformal']
REFS = ['ephys', 'label', 'moves_like', 'class_median', 'regression', 'dev_median']
CLASSICAL = {'VED': 'moves_like', 'EPA': 'regression', 'HD': 'dev_median'}

LABEL = {
    'enet_v21': 'EnergyNet v2.1 (as released)', 'climatology': 'Climatology (sim median MJ/km x distance)',
    'tab_mlp': 'MLP (EnergyNet net, retrained)', 'tab_gbm': 'Gradient-boosted trees', 'tab_gp': 'Sparse Gaussian process',
    'tab_tabpfn': 'TabPFN v2 in context (excluded)', 'formula': 'Trip-energy law (symbolic)',
    'phys_anchor': 'Physics anchor (E_phys + sim bands)', 'trace_cnn': '1-D dilated CNN', 'trace_gru': 'Bidirectional GRU',
    'trace_s4d': 'S4D state space', 'trace_transformer': 'Transformer (20-s patches)',
    'ephys': 'E_phys alone (road-load)', 'label': 'Window label x distance (dev-fitted)',
    'moves_like': 'MOVES-like VSP model (dev-fitted)', 'class_median': 'Class median per km (dev)',
    'regression': 'EPA regression (dev-fitted)', 'dev_median': 'Dev median (EPA / HD)',
    'classical': 'Classical composite (MOVES-like / EPA regression / dev median)'}
FAMILY = {'enet_v21': 'mlp', 'tab_mlp': 'mlp', 'tab_gbm': 'gbm', 'tab_gp': 'gp', 'tab_tabpfn': 'tabfm', 'formula': 'symbolic',
          'phys_anchor': 'physics', 'trace_cnn': 'cnn1d', 'trace_gru': 'ssm', 'trace_s4d': 'ssm', 'trace_transformer': 'attn',
          'climatology': 'dumb'}
for _k in list(LABEL):
    LABEL[_k + '_conformal'] = LABEL[_k] + ' + conformal'


def r4(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return round(x, 4) if np.isfinite(x) else None


def setfam(k):
    return 'VED' if k.startswith('VED') else ('EPA' if k.startswith('EPA') else 'HD')


# ============================================================================ loading
def load(fn):
    E = json.load(open(fn))
    od = fn[:-5]
    cols = ['unit', 'cluster', 'y', 'pred', 'v_kmh', 'T', 'climb_per_km', 'dist_km', 'q05', 'q25', 'q75', 'q95', 'ape',
            'crps_log', 'set', 'row', 'track']
    import pyarrow.parquet as pq
    import pyarrow as pa
    t = pq.read_table(os.path.join(od, 'scores.parquet'), columns=cols)
    sc = t.to_pandas(strings_to_categorical=True)
    del t
    pa.default_memory_pool().release_unused()
    for c in ('v_kmh', 'T', 'climb_per_km', 'dist_km', 'q05', 'q25', 'q75', 'q95'):
        sc[c] = sc[c].astype('float32')
    return E, sc, od


_G = {}
_EMPTY = None


def index_scores(sc):
    global _EMPTY
    _G.clear()
    for key, ix in sc.groupby(['track', 'row', 'set'], sort=False, observed=True).indices.items():
        g = sc.iloc[ix].copy()
        _G[key] = g.drop_duplicates('unit') if g.unit.duplicated().any() else g
    _EMPTY = sc.iloc[:0].copy()


def units_of(sc, track, row, st):
    """unit table (unit, cluster, y, pred, q*, strata) of one board row on one set, as the exam scored it."""
    return _G.get((track, row, st), _EMPTY)


def classical_units(sc, track, st):
    return units_of(sc, track, CLASSICAL[setfam(st)], st)


# ============================================================================ shared-resample bootstrap
class SetBoot:
    """per set: one fixed list of cluster resamples (default_rng(SEED + seed offset)); every row's MdAPE / CRPS is computed
    on the same resamples, so any pair of rows is a paired comparison."""

    def __init__(self, base_units, si, nboot=NBOOT):
        u = base_units.reset_index(drop=True)
        self.units = pd.Index(u.unit.astype(str))
        cl = u.cluster.astype(str).values
        codes, inv = np.unique(cl, return_inverse=True)
        order = np.argsort(inv, kind='stable')
        starts = np.searchsorted(inv[order], np.arange(len(codes) + 1))
        self.groups = [order[starts[i]:starts[i + 1]] for i in range(len(codes))]
        self.single = all(len(g) == 1 for g in self.groups)
        rng = np.random.default_rng(C.SEED + 1000 + si)
        G = len(self.groups)
        self.picks = [rng.integers(0, G, G).astype(np.int32) for _ in range(nboot)]

    def stats(self, u):
        return self.stats_many({'x': u})['x']

    def stats_many(self, U):
        """U: {row: unit table} -> {row: (MdAPE, mean crps, boot MdAPE (B,), boot crps (B,))}; all rows share the
        resamples. Units missing in a row are NaN (that row then takes the slower nan-aware path)."""
        keys = [k for k, u in U.items() if len(u)]
        if not keys:
            return {}
        A = np.full((len(keys), len(self.units)), np.nan); Cr = np.full_like(A, np.nan)
        for i, k in enumerate(keys):
            a = U[k].set_index(U[k].unit.astype(str))[['ape', 'crps_log']].reindex(self.units)
            A[i] = a.ape.values.astype(float); Cr[i] = a.crps_log.values.astype(float)
        nanrow = np.isnan(A).any(1) | np.isnan(Cr).any(1)
        B = len(self.picks)
        bm = np.empty((len(keys), B)); bc = np.empty((len(keys), B))
        f, g = ~nanrow, nanrow
        for k in range(B):
            ix = self.picks[k] if self.single else np.concatenate([self.groups[j] for j in self.picks[k]])
            if f.any():
                bm[f, k] = np.median(A[f][:, ix], axis=1); bc[f, k] = Cr[f][:, ix].mean(1)
            if g.any():
                bm[g, k] = np.nanmedian(A[g][:, ix], axis=1); bc[g, k] = np.nanmean(Cr[g][:, ix], axis=1)
        return {k: (float(np.nanmedian(A[i])), float(np.nanmean(Cr[i])), bm[i], bc[i]) for i, k in enumerate(keys)}


def ci_verdict(diff_boot, point):
    lo, hi = np.percentile(diff_boot, [2.5, 97.5])
    v = 'win' if hi < 0 else ('loss' if lo > 0 else 'tie')
    return {'diff': r4(point), 'ci95': [r4(lo), r4(hi)], 'verdict': v}


def e16_board(sc, track, rows, rehearsal=False):
    """E16 point value and bootstrap vector for each row that has all 16 sets; plus mean cov50 / cov90 / crps."""
    sets = E16
    boots = {}
    res = {r: {'md': {}, 'cr': {}, 'bm': [], 'bc': []} for r in rows}
    for si, st in enumerate(sets):
        base = units_of(sc, track, 'enet_v21', st)
        if not len(base):
            base = units_of(sc, track, rows[0], st)
        if not len(base):
            continue
        sb = SetBoot(base, si)
        U = {r: (classical_units(sc, track, st) if r == 'classical' else units_of(sc, track, r, st)) for r in rows}
        for r, (md, cr, bm, bc) in sb.stats_many(U).items():
            res[r]['md'][st] = md; res[r]['cr'][st] = cr
            res[r]['bm'].append(bm); res[r]['bc'].append(bc)
        print(' ', track, st, len(base), flush=True)
    out = {}
    for r, d in res.items():
        full = len(d['md']) == len(sets)
        out[r] = {'n_sets': len(d['md']), 'E16': r4(np.mean(list(d['md'].values()))) if full else None,
                  'E16_crps': r4(np.mean(list(d['cr'].values()))) if full else None,
                  '_bm': np.mean(d['bm'], 0) if full else None, '_bc': np.mean(d['bc'], 0) if full else None}
    return out


def e16_pairs(eb, a, b):
    if eb.get(a, {}).get('_bm') is None or eb.get(b, {}).get('_bm') is None:
        return None
    return {'E16': ci_verdict(eb[a]['_bm'] - eb[b]['_bm'], eb[a]['E16'] - eb[b]['E16']),
            'E16_crps': ci_verdict(eb[a]['_bc'] - eb[b]['_bc'], eb[a]['E16_crps'] - eb[b]['E16_crps'])}


# ============================================================================ boards
def board(E, sc, track):
    B = E['boards'][track]
    arms = [k for k in B['rows'] if B['rows'][k].get('_kind') == 'arm' or k in ('enet_v21', 'climatology')]
    refs = [k for k in REFS if k in B['rows']]
    rows = arms + refs
    eb = e16_board(sc, track, rows + ['classical'])
    tab = {}
    for k in rows + ['classical']:
        s = B['rows'].get(k, {})
        d = {'label': LABEL.get(k, k), 'kind': 'reference' if (k in refs or k == 'classical' or k in ('enet_v21', 'climatology', 'enet_v21_conformal')) else 'arm',
             'family': FAMILY.get(k.replace('_conformal', '')), 'conformal': k.endswith('_conformal'),
             'base': k[:-10] if k.endswith('_conformal') else k, 'row_track': 'table' if k.startswith('enet_v21') else track,
             'E16': eb[k]['E16'], 'E16_crps': eb[k]['E16_crps'], 'n_error_records': s.get('_n_error_records'), 'sets': {}}
        covs = []
        for st in ALLSETS:
            x = s.get(st) if k != 'classical' else E['boards'][track]['rows'].get(CLASSICAL[setfam(st)], {}).get(st)
            if not isinstance(x, dict) or not x.get('n'):
                continue
            e = {m: r4(x.get(m)) for m in ('MdAPE', 'MAPE', 'bias', 'med_log_err', 'crps_log', 'cov50', 'cov90', 'width90_log')}
            e['n'] = x['n']; e['n_clusters'] = x.get('n_clusters')
            if 'conditions' in x:
                e['conditions'] = {c: {m: r4(v.get(m)) for m in ('n', 'MdAPE', 'bias', 'cov90', 'crps_log')} for c, v in x['conditions'].items()}
            d['sets'][st] = e
            if st in E16 and x.get('cov90') is not None:
                covs.append((x['cov50'], x['cov90'], x.get('width90_log')))
        if len(covs) == len(E16):
            d['E16_cov50'] = r4(np.mean([c[0] for c in covs])); d['E16_cov90'] = r4(np.mean([c[1] for c in covs]))
            d['E16_width90_log'] = r4(np.mean([c[2] for c in covs]))
        d['powertrain'] = {}
        for pk, (pl, psets) in POWERTRAINS.items():
            v = [d['sets'][x]['MdAPE'] for x in psets if x in d['sets']]
            d['powertrain'][pk] = r4(np.mean(v)) if len(v) == len(psets) else None
        # the exam's own paired bootstrap per set
        P = B['paired'].get(k, {})
        d['vs_set'] = {st: {anc: {'d_MdAPE': r4(v.get('d_MdAPE')), 'ci': [r4(x) for x in v.get('ci_MdAPE', [])],
                                   'verdict': v.get('verdict_MdAPE'), 'd_crps': r4(v.get('d_crps_log')), 'verdict_crps': v.get('verdict_crps_log')}
                            for anc, v in P.get(st, {}).items() if 'd_MdAPE' in v} for st in P}
        tab[k] = d
    # E16 pairs vs anchors and the in-track control
    ctrl = CONTROL[track]
    for k in tab:
        tab[k]['vs_E16'] = {}
        if k.endswith('_conformal') and k[:-10] in tab:
            p = e16_pairs(eb, k, k[:-10])
            if p:
                tab[k]['vs_E16']['raw'] = p
        for anc in ('enet_v21', 'ephys', 'classical', ctrl, 'climatology'):
            if anc != k:
                p = e16_pairs(eb, k, anc)
                if p:
                    tab[k]['vs_E16'][anc] = p
    # pairwise among the raw learners (+ EnergyNet): the convergence matrix
    learners = [k for k in (TABLE_LEARNERS if track == 'table' else TRACE_LEARNERS) if k in tab] + ['enet_v21']
    pw = {}
    for a in learners:
        for b in learners:
            if a < b:
                p = e16_pairs(eb, a, b)
                if p:
                    pw[f'{a}|{b}'] = p['E16']
    return tab, pw


def shared_boot(U, base, seed):
    """{row: unit table} -> {row: (MdAPE, crps, boot MdAPE, boot crps)} on one set of cluster resamples drawn on the
    base row's units (shared by every row, so differences are paired)."""
    U = {k: u for k, u in U.items() if len(u) >= 3}
    if base not in U:
        return {}
    return SetBoot(U[base], seed).stats_many(U)


def pair(st, a, b):
    """a vs b from shared_boot stats: MdAPE difference (negative = a better) with its 95 % interval."""
    v = ci_verdict(st[a][2] - st[b][2], st[a][0] - st[b][0])
    w = ci_verdict(st[a][3] - st[b][3], st[a][1] - st[b][1])
    return {'d_MdAPE': v['diff'], 'ci': v['ci95'], 'verdict': v['verdict'], 'd_crps': w['diff'], 'verdict_crps': w['verdict']}


def control_per_set(sc, track, rows):
    """paired bootstrap per set against the in-track control (shared cluster resamples, 2,000, exam clusters)."""
    ctrl = CONTROL[track]
    out = {k: {} for k in rows if k != ctrl}
    for si, st in enumerate(VED + EPA + HD):
        if st.endswith('_cars'):
            continue
        U = {k: units_of(sc, track, k, st) for k in rows}
        S_ = shared_boot(U, ctrl, 5000 + si)
        for k in out:
            if k in S_ and ctrl in S_:
                out[k][st] = pair(S_, k, ctrl)
    return out


# ============================================================================ conditions (VED, pooled over powertrains)
def ved_pool(sc, track, row):
    fr = [units_of(sc, track, row, st) for st in VED]
    fr = [x for x in fr if len(x)]
    return pd.concat(fr) if fr else _EMPTY


def cond_mask(u, c):
    return {'city': u.v_kmh < 40, 'highway': u.v_kmh > 55, 'cold': u['T'] < 0, 'hot': u['T'] > 25,
            'hilly': u.climb_per_km >= 15, 'short': u.dist_km < 2, 'long': u.dist_km > 15, 'all': u.v_kmh > -1e9}[c]


def conditions_board(sc, track, rows):
    out = {}
    U0 = {k: ved_pool(sc, track, k) for k in rows}
    for ci, c in enumerate(['all'] + CONDS):
        out[c] = {}
        U = {k: u[cond_mask(u, c)] for k, u in U0.items() if len(u)}
        for k, m in U.items():
            if len(m) < 3:
                continue
            d = {'n': int(len(m)), 'MdAPE': r4(np.median(m.ape)), 'bias': r4(np.median(m.pred / m.y - 1)),
                 'crps_log': r4(m.crps_log.mean()), 'by_group': {g: r4(np.median(x.ape)) for g, x in m.groupby('set')}}
            if not np.allclose(m.q05, m.q95):
                d['cov90'] = r4(np.mean((m.y >= m.q05) & (m.y <= m.q95)))
            out[c][k] = d
        S_ = shared_boot(U, 'enet_v21', 6000 + ci)
        for k in out[c]:
            if k != 'enet_v21' and k in S_:
                v = pair(S_, k, 'enet_v21')
                out[c][k]['vs_enet'] = {'d': v['d_MdAPE'], 'ci': v['ci'], 'verdict': v['verdict']}
    return out


# ============================================================================ where each assumption wins
def wins_map(sc, cands, refs):
    """columns: powertrain sets and VED conditions; candidates are raw learners from both tracks (key 'arm|track');
    best by MdAPE, every candidate the paired bootstrap cannot separate from it, and the best reference."""
    cols = [(st, None) for st in VED + ['EPA_conv_FTP', 'EPA_conv_HWY', 'EPA_conv_US06', 'EPA_conv_FTPcold', 'EPA_conv_SC03',
                                       'EPA_hev_FTP', 'EPA_hev_HWY', 'EPA_bev_FTP', 'EPA_bev_HWY', 'Mersin_diesel',
                                       'Mersin_cng', 'ICCT', 'ZTBus_legs']] + [('VED_all', c) for c in CONDS]
    pools = {}
    out = {}
    for ci, (st, c) in enumerate(cols):
        key = st if c is None else c
        U = {}
        for k in cands + refs:
            row, tr = k.split('|')
            if c is None:
                u = units_of(sc, tr, row, st)
            else:
                if k not in pools:
                    pools[k] = ved_pool(sc, tr, row)
                u = pools[k]
                u = u[cond_mask(u, c)] if len(u) else u
            if len(u) >= 3:
                U[k] = u
        md = {k: float(np.median(U[k].ape)) for k in cands if k in U}
        if not md:
            continue
        best = min(md, key=md.get)
        S_ = shared_boot(U, best, 7000 + ci)
        tied, beaten = [], []
        for k in md:
            if k == best:
                continue
            (tied if pair(S_, best, k)['verdict'] == 'tie' else beaten).append(k)
        d = {'n': int(len(U[best])), 'best': best, 'best_MdAPE': r4(md[best]), 'tied': tied, 'beaten': beaten,
             'MdAPE': {k: r4(v) for k, v in md.items()}}
        rr = {k: float(np.median(U[k].ape)) for k in refs if k in U}
        if rr:
            br = min(rr, key=rr.get)
            v = pair(S_, best, br)
            d.update(best_ref=br, best_ref_MdAPE=r4(rr[br]), ref_MdAPE={k: r4(x) for k, x in rr.items()},
                     best_vs_best_ref={'d': v['d_MdAPE'], 'ci': v['ci'], 'verdict': v['verdict']})
        out[key] = d
        print('  wins', key, flush=True)
    return out


# ============================================================================ compute
PARAMS_NOTE = {'enet_v21': '5 MLPs (151-256-256-256-6 each)', 'tab_mlp': '5 MLPs, EnergyNet architecture',
               'tab_gbm': 'tree nodes (mean + 5 quantile models)', 'tab_gp': '500 inducing points x 46 inputs + matrices',
               'formula': '3 laws, 20-22 nodes each', 'phys_anchor': 'band quantiles only', 'climatology': 'rate table'}


def count_params(arm_dir, k):
    import glob
    try:
        if k in ('enet_v21', 'tab_mlp'):
            return int(sum(sum(np.load(f)[x].size for x in np.load(f).files) for f in glob.glob(f'{arm_dir}/weights/*.npz')))
        if k.startswith('trace_'):
            z = np.load(f'{arm_dir}/weights.npz')
            return int(sum(z[x].size for x in z.files if x != 'cfg_json'))
        if k == 'tab_gbm':
            z = np.load(f'{arm_dir}/gbm_trees.npz')
            return int(sum(z[x].size for x in z.files if x.endswith('__feat')))
        if k == 'tab_gp':
            z = np.load(f'{arm_dir}/gp.npz')
            return int(sum(z[x].size for x in z.files))
        if k == 'formula':
            return 3 * 22
    except Exception:  # noqa: BLE001
        return None
    return None


def arm_dir(r):
    """the arm folder of a registry record; records hold the cloud path (/home/claude/bakeoff/vehicle/...)."""
    d = r.get('arm_dir', '')
    pre = '/home/claude/bakeoff/vehicle/'
    return os.path.join(os.path.dirname(HARNESS), d[len(pre):]) if d.startswith(pre) else d


def compute_table(E, boards):
    out = {}
    regs = {}
    import glob
    for f in glob.glob(f'{HARNESS}/registry/*.json'):
        if f.endswith('excluded.json'):
            continue
        r = json.load(open(f)); regs[r['arm_id']] = r
    for k, a in E['arms'].items():
        cpu = a.get('exam_cpu', {})
        rec = regs.get(k, {})
        meta = rec.get('meta', {})
        e16 = {tr: boards[tr][0].get(k, {}).get('E16') for tr in boards if k in boards[tr][0]}
        if k.startswith('enet_v21'):
            e16 = {'table': boards['table'][0].get(k, {}).get('E16')}
        train_h = (a.get('compute') or meta.get('compute') or {}).get('train_cpu_hours')
        out[k] = {'label': LABEL.get(k, k), 'family': FAMILY.get(k.replace('_conformal', '')), 'train_cpu_hours': train_h,
                  'exam_predict_cpu_s': {tr: v.get('predict_cpu_s') for tr, v in cpu.items()},
                  'n_errors': {tr: v.get('n_errors') for tr, v in cpu.items()},
                  'params': None if k.endswith('_conformal') else count_params(arm_dir(rec), k),
                  'params_note': PARAMS_NOTE.get(k), 'numpy_only': a.get('numpy_only'), 'released': a.get('released'),
                  'unchanged_after': a.get('unchanged_after'), 'hash': (a.get('hash') or '')[:16], 'E16': e16}
    return out


def simval_numbers():
    """the released rung's sim val skill / MdAPE and dev D16 from each arm's ratchet.json (pre-exam numbers)."""
    import glob
    out = {}
    for f in glob.glob(f'{HARNESS}/registry/*.json'):
        if f.endswith('excluded.json'):
            continue
        r = json.load(open(f))
        try:
            m = json.load(open(os.path.join(arm_dir(r), 'ratchet.json')))
        except Exception:  # noqa: BLE001
            continue
        rel = next((x for x in m.get('rungs', []) if x.get('id') == m.get('released')), {})
        out[r['arm_id']] = {'simval_skill': r4(rel.get('simval_skill')), 'simval_MdAPE': r4(rel.get('simval_MdAPE')),
                            'dev_D16': r4(rel.get('dev_D16')), 'rungs': [{k: x.get(k) for k in ('id', 'change', 'decision', 'simval_skill', 'simval_MdAPE', 'dev_D16')} for x in m.get('rungs', [])],
                            'released': m.get('released')}
    return out


# ============================================================================ markdown
def f3(x, p=3):
    return '' if x is None else f'{x:.{p}f}'


def md_board(tab, track):
    order = sorted(tab, key=lambda k: (tab[k]['kind'] == 'reference' and k not in ('enet_v21',), tab[k]['E16'] if tab[k]['E16'] is not None else 9))
    L = [f'### {track} track: E16 (mean exam MdAPE over the 16 D16 sets) and selected sets', '',
         '| Row | E16 | vs EnergyNet v2.1 (95 % CI) | vs control | VED ICE / HEV / EV / PHEV | EPA conv FTP / HWY | EPA hev FTP / HWY | EPA BEV city / HWY | Mersin d / CNG | ICCT | ZTBus legs | cov50 / cov90 | CRPS(log) |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    ctrl = CONTROL[track]
    for k in order:
        d = tab[k]; s = d['sets']
        g = lambda st: f3(s.get(st, {}).get('MdAPE'))  # noqa: E731
        v = d['vs_E16'].get('enet_v21', {}).get('E16')
        vs = '' if not v else f"{v['diff']:+.3f} ({v['ci95'][0]:+.3f}..{v['ci95'][1]:+.3f}) {v['verdict']}"
        c = d['vs_E16'].get(ctrl, {}).get('E16')
        vc = '' if not c else f"{c['diff']:+.3f} {c['verdict']}"
        cov = '' if d.get('E16_cov50') is None else f"{d['E16_cov50']:.2f} / {d['E16_cov90']:.2f}"
        L.append(f"| {d['label']} | {f3(d['E16'])} | {vs} | {vc} | {g('VED_ICE')} / {g('VED_HEV')} / {g('VED_EV')} / {g('VED_PHEV')} | "
                 f"{g('EPA_conv_FTP')} / {g('EPA_conv_HWY')} | {g('EPA_hev_FTP')} / {g('EPA_hev_HWY')} | {g('EPA_bev_FTP')} / {g('EPA_bev_HWY')} | "
                 f"{g('Mersin_diesel')} / {g('Mersin_cng')} | {g('ICCT')} | {g('ZTBus_legs')} | {cov} | {f3(d['E16_crps'])} |")
    return '\n'.join(L) + '\n'


def main():
    a = sys.argv[1:]
    rehearsal = '--rehearsal' in a
    a = [x for x in a if x != '--rehearsal']
    fn = a[0] if a else os.path.join(BASE, json.load(open(os.path.join(BASE, 'results', 'EXAM_DONE.json')))['result'])
    if not os.path.isabs(fn):
        fn = os.path.join(BASE, fn)
    E, sc, od = load(fn)
    index_scores(sc)
    sc = None          # only the per-(track, row, set) slices are kept
    import gc
    gc.collect()
    R = {'source': os.path.relpath(fn, BASE).replace(os.sep, '/'), 'run_id': E['run_id'], 'mode': E['mode'], 'notes': E['notes'],
         'excluded': EXCLUDED, 'boards': {}, 'pairwise': {}, 'vs_control_per_set': {}, 'conditions': {},
         'reproduction_enet_v21': E.get('reproduction_enet_v21'), 'reproduction_refs': E.get('reproduction_refs')}
    boards = {}
    for track in ('table', 'trace'):
        tab, pw = board(E, sc, track)
        boards[track] = (tab, pw)
        R['boards'][track] = {k: {kk: vv for kk, vv in d.items()} for k, d in tab.items()}
        R['pairwise'][track] = pw
        R['vs_control_per_set'][track] = control_per_set(sc, track, [k for k in tab if tab[k]['kind'] == 'arm' or k == 'enet_v21'])
        rows = [k for k in tab if k != 'classical']
        R['conditions'][track] = conditions_board(sc, track, rows)
        print('board', track, 'done', flush=True)
    cands = [f'{k}|table' for k in TABLE_LEARNERS + ['enet_v21'] if k in boards['table'][0]] + \
            [f'{k}|trace' for k in TRACE_LEARNERS if k in boards['trace'][0] and k not in ('formula', 'phys_anchor')]
    refs = [f'{k}|table' for k in REFS + ['climatology'] if k in boards['table'][0]]
    R['wins'] = wins_map(sc, cands, refs)
    R['compute'] = compute_table(E, boards)
    R['simval'] = simval_numbers()
    os.makedirs(OUT, exist_ok=True)
    tag = 'rehearsal_' if rehearsal else ''
    md = [f"# Vehicle Bake-off exam leaderboards (run {E['run_id']})", '',
          'E16 = mean over the 16 D16 sets of the exam MdAPE (lower is better). Intervals: paired cluster bootstrap with '
          'shared resamples per set (2,000), averaged over the 16 sets; win / loss only when the 95 % interval excludes 0. '
          'cov50 / cov90 and CRPS(log) are means over the 16 sets. Control: MLP (table), GRU (trace).', '']
    for track in ('table', 'trace'):
        md.append(md_board(boards[track][0], track))
    md.append('### Where each assumption wins (lowest MdAPE among the raw learners of both tracks + EnergyNet v2.1)\n')
    md.append('| Column | n | Best | MdAPE | Tied with it | Best reference (MdAPE) | best vs best reference |')
    md.append('|---|---|---|---|---|---|---|')
    for c, d in R['wins'].items():
        v = d.get('best_vs_best_ref') or {}
        md.append(f"| {c} | {d['n']} | {d['best']} | {f3(d['best_MdAPE'])} | {', '.join(d['tied']) or 'none'} | "
                  f"{d.get('best_ref', '')} ({f3(d.get('best_ref_MdAPE'))}) | {f3(v.get('d'))} {v.get('verdict', '')} |")
    md.append('\n### Compute\n')
    md.append('| Arm | Train CPU-h | Exam predict CPU-s | Stored numbers | E16 |')
    md.append('|---|---|---|---|---|')
    for k, d in sorted(R['compute'].items()):
        md.append(f"| {d['label']} | {d['train_cpu_hours']} | {d['exam_predict_cpu_s']} | {d['params'] or ''} | {d['E16']} |")
    # strip bootstrap vectors
    for track in R['boards']:
        for k in R['boards'][track]:
            R['boards'][track][k].pop('_bm', None); R['boards'][track][k].pop('_bc', None)
    json.dump(R, open(os.path.join(OUT, f'{tag}analysis.json'), 'w'), indent=1, default=float)
    open(os.path.join(OUT, f'{tag}leaderboards.md'), 'w').write('\n'.join(md) + '\n')
    print('wrote', os.path.join(OUT, f'{tag}analysis.json'))


if __name__ == '__main__':
    main()
