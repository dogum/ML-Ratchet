"""Audit repair rung r10 (27 Nov 2026): judged by the ratchet on the frozen simulated set val_v5 and the dev gate.

The audit (25 Nov 2026) found inputs the simulator never varied: reported solar at hours the physics calls dark, and
missing hours in the demand history (miss_frac's simulated sd was ~0, so one missing hour reached |z| ~286). r10:
  - simulator v5 (sim.py): v4 + twilight output at dark edge hours, -2..+2 h solar offsets, demand-history gaps
  - solar: offset read up to +-2 h, history aligned, reported output zeroed where the physics is dark (training and
    inference: train.py --repair, demandcast.mask_dark_solar)
  - standardisation floor 0.05 on count-like inputs (train.py COUNT_COLS); inputs clipped at |z| 8 (as v1.1.1)
  - three 512-512 MLPs on base features, 180,000 region-days of simulator v5 (train_v5a-c), 40 epochs, as r8

    python repair.py val <ckpt>     # v1.1, v1.1.1 and <ckpt> on val_v5 (and val_v4, for reference)
    python repair.py add <ckpt>     # val_v5 + dev gate -> decision appended to training/ratchet.json
    python repair.py export <ckpt>  # numpy weights -> work/release_v12/ (only for a promoted rung)
    python repair.py events <ckpt>  # events.py for v1.1.1 and the rung (after the exam; reported only)
    python repair.py compare x      # v1.1 / v1.1.1 / v1.2 on every exam bank -> training/exam_v1_2.json
    python repair.py exam <ckpt>    # the exam, run ONCE on a promoted rung after its release choice is committed:
                                    # actual weather (work/data/banks), forecast weather 2024-2025 (forecast_full) and
                                    # the released 2024 forecast row (forecast_release2024) -> exam/v1.2.json each

The incumbent is v1.1.1 (the released r8 network with z_clip=8, the default since the dated re-score of 27 Nov 2026).
The exam is never computed here.
"""
import json
import os
import sys

import numpy as np

from common import DATA, SRC, ROOT, BAS, jdump
from train import XK, PK, load_banks
import evaluate as E
from ratchet import gate_rows

MODEL = os.path.dirname(SRC)
sys.path.insert(0, MODEL)
import demandcast as DC  # noqa: E402

LOG = os.path.join(SRC, 'ratchet.json')
REL = os.path.join(ROOT, 'release_v12')
META = json.load(open(DC.META_V1_1))      # solar_offset_ref (dev reference; the same in every version)
REF = META['solar_offset_ref']


def weights(ckpt):
    if ckpt in ('v1.1', 'v1.1.1'):
        return DC.DemandCast(DC.WEIGHTS_V1_1, DC.META_V1_1).W
    import torch
    ck = torch.load(f'{ROOT}/ckpt/{ckpt}.pt', weights_only=False)
    return {k: {n: v.cpu().numpy().astype(np.float64) for n, v in ck[k].items()} for k in ('d', 'w', 's')}


CFG = {'v1.1': dict(rmax=1, mask=False, zc=None), 'v1.1.1': dict(rmax=1, mask=False, zc=8.0)}
CAND = dict(rmax=2, mask=True, zc=8.0)


def predict(B, W, cfg, k_only=None):
    """evaluate.predict with the numpy forward and the v1.1 solar alignment (raw sigmas)."""
    import warnings
    out = {}
    ref = 0.0 if 'meta_region' in B or 'meta_ba' not in B else np.array([REF.get(b, 0.0) for b in B['meta_ba']])
    for k in ('d', 'w', 's') if k_only is None else (k_only,):
        X = np.nan_to_num(B[XK[k]].astype(np.float32)).astype(np.float64)
        r = np.zeros(len(X), int)
        if k == 's':
            r, _ = DC.solar_offset(X, ref, rmax=cfg['rmax']); X = DC.align_solar_x(X, r)
            if cfg['mask']:
                X = DC.mask_dark_solar(X)
        P = B[PK[k]].astype(np.float64) if k != 'd' else None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mu, sig = DC._forward(W[k], X, k, P, z_clip=cfg['zc'])
        if k != 'd':
            mu = np.clip(mu, 0, 1.2)
        if k == 's':
            night = B['night'] > 0.5
            mu = np.where(night, 0.0, mu); sig = np.where(night, DC.FLOOR['s'], sig)
            if r.any():
                mu = DC._shift(mu, r, 0.0); sig = DC._shift(sig, r, DC.FLOOR['s'])
        out[k] = (mu.astype(np.float32), sig.astype(np.float32))
    return out


def models(ckpt):
    return {'v1.1': (weights('v1.1'), CFG['v1.1']), 'v1.1.1': (weights('v1.1'), CFG['v1.1.1']), ckpt: (weights(ckpt), CAND)}


def val(ckpt, name='val_v5'):
    Bv = load_banks([name])
    res = {}
    for nm, (W, cfg) in models(ckpt).items():
        p = predict(Bv, W, cfg)
        res[nm] = E.sim_score(Bv, p)
        if 'meta_dgap' in Bv:
            g = Bv['meta_dgap'] > 0
            res[nm]['demand_err_gap_rows'] = E.errors(Bv, p, 'd', rows=g)['err']
            res[nm]['demand_err_no_gap_rows'] = E.errors(Bv, p, 'd', rows=~g)['err']
        if 'meta_soff' in Bv:
            res[nm]['solar_err_by_true_offset'] = {int(a): E.errors(Bv, p, 's', rows=Bv['meta_soff'] == a)['err']
                                                   for a in sorted(set(Bv['meta_soff'].tolist()))}
        print(f"{nm} on {name}: objective {res[nm]['objective']:.4f} | skill d {res[nm]['d']['skill']:.4f} "
              f"w {res[nm]['w']['skill']:.4f} s {res[nm]['s']['skill']:.4f} | err d {res[nm]['d']['err']:.2f}% "
              f"w {res[nm]['w']['err']:.1f}% s {res[nm]['s']['err']:.1f}%", flush=True)
        for kk in ('demand_err_gap_rows', 'demand_err_no_gap_rows', 'solar_err_by_true_offset'):
            if kk in res[nm]:
                print('   ', kk, res[nm][kk])
    if 'meta_soff' in Bv:
        X = np.nan_to_num(Bv['xs'].astype(np.float64)); t = Bv['meta_soff']
        r, _ = DC.solar_offset(X, 0.0, rmax=2)
        res['confusion_rmax2'] = {f'{a:+d}': [int(((t == a) & (r == b)).sum()) for b in range(-2, 3)] for a in range(-2, 3)}
        print('true offset -> read -2..+2:', res['confusion_rmax2'])
    return res


def dev(ckpt):
    Bd = dict(np.load(f'{DATA}/banks/real_dev.npz')); rows = gate_rows(Bd)
    out = {}
    for nm, (W, cfg) in models(ckpt).items():
        p = predict(Bd, W, cfg)
        out[nm] = {k: E.errors(Bd, p, k, rows=rows) for k in ('d', 'w', 's')}
        print(f"dev gate {nm}: d {out[nm]['d']['err']:.3f}% w {out[nm]['w']['err']:.2f}% s {out[nm]['s']['err']:.2f}% "
              f"| cov68 d {out[nm]['d']['cov68']:.2f} s {out[nm]['s']['cov68']:.2f}", flush=True)
    return out


def add(ckpt, note):
    v = val(ckpt); v4 = val(ckpt, 'val_v4'); d = dev(ckpt)
    inc = 'v1.1.1'
    s_ok = v[ckpt]['objective'] > v[inc]['objective'] + 0.01
    rel = float(np.mean([d[ckpt][k]['err'] / max(d[inc][k]['err'], 1e-9) for k in ('d', 'w', 's')]))
    d_ok = rel <= 1.02
    why = (f"objective {v[ckpt]['objective']:.4f} vs {v[inc]['objective']:.4f} on val_v5" + ('' if s_ok else ' (needs +0.01)')
           + f"; dev errors {100 * (rel - 1):+.1f}% vs incumbent" + ('' if d_ok else ' (gate: <= +2%)'))
    rec = dict(ckpt=ckpt, date='2026-11-27', note=note, val_set='val_v5', val=v[ckpt], objective=v[ckpt]['objective'],
               dev=d[ckpt], incumbent_rescored={nm: dict(val=v[nm], objective=v[nm]['objective'], dev=d[nm])
                                                for nm in ('v1.1', 'v1.1.1')},
               val_v4={nm: v4[nm]['objective'] for nm in ('v1.1', 'v1.1.1', ckpt)},
               confusion_rmax2=v.get('confusion_rmax2'), dev_rel=rel, vs=inc, accepted=bool(s_ok and d_ok), why=why)
    log = json.load(open(LOG)); log.append(rec); jdump(log, LOG)
    print('ACCEPTED' if rec['accepted'] else 'rejected', '-', why)


def export(ckpt):
    """Numpy weights of a promoted rung -> work/release_v12/demandcast_weights.npz (as export.py)."""
    import torch
    ck = torch.load(f'{ROOT}/ckpt/{ckpt}.pt', weights_only=False)
    os.makedirs(REL, exist_ok=True)
    arrs = {f'{k}.{n}': v.cpu().numpy().astype(np.float32) for k in ('d', 'w', 's') for n, v in ck[k].items()}
    np.savez(f'{REL}/demandcast_weights.npz', **arrs)
    print('wrote', f'{REL}/demandcast_weights.npz', {k: ck[k]['xm'].shape[0] for k in ('d', 'w', 's')})


def run_exam(ckpt):
    """exam.main() unchanged, with rescore_release.py's numpy stand-in for evaluate.py running the rung's arithmetic
    (sigma scales re-fitted on the dev bank inside exam.main, as for every release)."""
    import rescore_release as RR
    import warnings
    W = weights(ckpt)
    RR.shim.load = lambda name: W
    RR.shim.predict = lambda W_, B: predict(B, W_, CAND)
    for data in (DATA, f'{DATA}/forecast_full', f'{DATA}/forecast_release2024'):
        RR.exam.DATA = data
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            RR.exam.main('v1.2')
        print('exam written', f'{data}/exam/v1.2.json', flush=True)


def run_events(ckpt):
    """events.py (named extreme periods, demand only, reported and never used for choosing) for v1.1.1 (exam name
    v1.1_zclip8) and the rung (v1.2), next to the released r8 numbers in exam_r8_wide_events.json."""
    import rescore_release as RR  # noqa: F401  (installs the evaluate shim)
    import events
    for name, W, cfg in (('v1.1_zclip8', weights('v1.1'), CFG['v1.1.1']), ('v1.2', weights(ckpt), CAND)):
        events.E.load = lambda n, W=W: W
        events.E.predict = lambda W_, B, cfg=cfg: predict(B, W_, cfg, k_only='d')
        events.main(name)


def compare():
    """v1.1 / v1.1.1 / v1.2 side by side on every exam bank -> training/exam_v1_2.json (and the events)."""
    import rescore_release as RR
    out = {}
    banks = (('actual', DATA, 'actual', ('v1.1', 'v1.1_zclip8', 'v1.2')),
             ('forecast_2024_2025', f'{DATA}/forecast_full', 'forecast', ('v1.1', 'v1.1_zclip8', 'v1.2')),
             ('forecast_2024_released_row', f'{DATA}/forecast_release2024', 'forecast', ('release', 'release_zclip8', 'v1.2')))
    for tag, data, mode, names in banks:
        B = dict(np.load(f'{data}/banks/real_exam.npz'))
        fcb = B['meta_fc']
        rows = fcb == 0 if mode == 'actual' else (fcb & 7) == 7
        wr = rows if mode == 'actual' else (fcb & 8 > 0) & (fcb > 0)
        sr = rows if mode == 'actual' else (fcb & 4 > 0) & (fcb > 0)
        res = {}
        for nm, lab in zip(names, ('v1.1', 'v1.1.1', 'v1.2')):
            J = json.load(open(f'{data}/exam/{nm}.json')); Pz = dict(np.load(f'{data}/exam/{nm}_pred.npz'))
            pred = {k: (Pz[f'{k}_mu'], Pz[f'{k}_sig']) for k in ('d', 'w', 's')}
            one = {'d': 1.0, 'w': 1.0, 's': 1.0}
            r = dict(sigma_scale_dev=J['sigma_scale_dev'], pooled=J['overall'][mode],
                     by_grid={ba: J['table'][ba]['all'][mode] for ba in J['table'] if mode in J['table'][ba]['all']})
            six = rows & ~np.isin(B['meta_ba'], ['CISO', 'SWPP'])
            r['pooled_without_CISO_SWPP'] = RR.exam.block(B, pred, one, six, wr & six, sr & six)
            r['grids_beating_grid_forecast'] = sorted(ba for ba, v in r['by_grid'].items()
                                                      if v['demand']['model'] < v['demand']['ba_day_ahead'])
            if mode == 'actual':
                r['temp_bands'] = J['temp_bands']
            res[lab] = r
        out[tag] = res
    out['events'] = {lab: json.load(open(f'{DATA}/exam/{nm}_events.json')) for lab, nm in (('v1.1.1', 'v1.1_zclip8'), ('v1.2', 'v1.2'))}
    out['events']['v1.1'] = json.load(open(os.path.join(SRC, 'exam_r8_wide_events.json')))
    jdump(out, os.path.join(SRC, 'exam_v1_2.json'))
    for tag in ('actual', 'forecast_2024_2025', 'forecast_2024_released_row'):
        for lab in ('v1.1', 'v1.1.1', 'v1.2'):
            r = out[tag][lab]; po = r['pooled']; six = r['pooled_without_CISO_SWPP']
            print(f"{tag:28s} {lab:7s} demand {po['demand']['model']:.3f} (grid {po['demand']['ba_day_ahead']:.2f}; 6 grids "
                  f"{six['demand']['model']:.2f} vs {six['demand']['ba_day_ahead']:.2f}) cov {po['demand']['cov68']:.2f}/{po['demand']['cov95']:.2f}"
                  f" | wind {po['wind']['model']:.2f} cov {po['wind']['cov68']:.2f}/{po['wind']['cov95']:.2f} | solar {po['solar']['model']:.2f} "
                  f"(pers {po['solar']['persistence_24h']:.1f}) cov {po['solar']['cov68']:.2f}/{po['solar']['cov95']:.2f} | wins {len(r['grids_beating_grid_forecast'])}"
                  f" {r['grids_beating_grid_forecast']}")
            print('     ', {ba: (round(v['demand']['model'], 2), round(v['demand']['ba_day_ahead'], 2), round(v['wind']['model'] or 0, 1),
                             round(v['solar']['model'], 1) if v['solar']['model'] == v['solar']['model'] else None,
                             round(v['solar']['persistence_24h'], 1) if v['solar']['persistence_24h'] == v['solar']['persistence_24h'] else None,
                             round(v['demand']['cov68'], 2)) for ba, v in r['by_grid'].items()})


if __name__ == '__main__':
    cmd, ck = sys.argv[1], sys.argv[2]
    if cmd == 'val':
        val(ck); dev(ck)
    elif cmd == 'add':
        add(ck, sys.argv[3] if len(sys.argv) > 3 else '')
    elif cmd == 'export':
        export(ck)
    elif cmd == 'exam':
        run_exam(ck)
    elif cmd == 'events':
        run_events(ck)
    elif cmd == 'compare':
        compare()
