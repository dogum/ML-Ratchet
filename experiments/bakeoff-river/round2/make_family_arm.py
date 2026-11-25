"""Write ratchet.json and card.md of a KEPT round-2 family arm (round2/arms/<arm>_b2/) from results/family_rungs.json and
the evaluation summaries. usage: python make_family_arm.py <v1 arm id>    (tab_gbm | seq_lstm | seq_tcn | seq_ssm)
"""
import json
import os
import sys

import r2

INFO = {
    'seq_lstm': dict(label="LSTM (FlowCast's network) retrained on bank b2", family='ssm',
                     family_label='State-space and recurrent models (LSTM)', cfg='arms_seq/train/configs/lstm_r1.json',
                     recipe='seq_lstm r1: H 64, dropout 0.1, 6,000 steps x 32 windows, lr 0.002 cosine, seed 0',
                     extra=['../../../arms_seq/common']),
    'seq_tcn': dict(label='TCN (dilated 1-D CNN) retrained on bank b2', family='cnn1d',
                    family_label='1-D convolutional network', cfg='arms_seq/train/configs/tcn_r1.json',
                    recipe='seq_tcn r1: 26 channels, kernel 3, dilations 1-256, dropout 0.1, 6,000 steps x 32, seed 0',
                    extra=['../../../arms_seq/common']),
    'seq_ssm': dict(label='S4D state space + MLP encoder retrained on bank b2', family='ssm',
                    family_label='State-space models (S4D)', cfg='arms_seq/train/configs/ssm_r2.json',
                    recipe='seq_ssm r2: d 48, N 16, 3 layers, pointwise MLP encoder 48, 6,000 steps x 32, seed 0',
                    extra=['../../../arms_seq/common']),
    'tab_gbm': dict(label='Gradient-boosted trees (LightGBM -> numpy trees) retrained on bank b2', family='gbm',
                    family_label='Gradient-boosted decision trees', cfg='arms_tab/src/train_gbm.py, monotone off',
                    recipe='tab_gbm r2: 47 features (feats.py), 1 window x 100 days per train run (1.2 M rows), '
                           'L2 mean booster (63 leaves, lr 0.05, early stop on train runs >= 11,400) + 5 quantile '
                           'boosters (31 leaves, lr 0.1); no monotone constraints. LightGBM 4.7.0 here (round 1: '
                           '4.6.0); inference walks the exported numpy trees', extra=[]),
}


def main(arm):
    fr = json.load(open(os.path.join(r2.RESULTS, 'family_rungs.json')))['families'][arm]
    I = INFO[arm]
    new = fr['b2_arm_id']
    run = fr['run']
    s = json.load(open(os.path.join(r2.WORK, 'eval', run, 'own_summary.json')))
    d = s['dev']['ungauged']
    v1d = json.load(open(os.path.join(r2.WORK, 'eval', f'v1_{arm}', 'dev_summary.json')))['dev']['ungauged']
    p1v = None
    if os.path.exists(os.path.join(r2.WORK, 'eval', run, 'p1val_summary.json')):
        p1v = json.load(open(os.path.join(r2.WORK, 'eval', run, 'p1val_summary.json')))['simval']['ungauged']['nse']
    if arm == 'tab_gbm':
        g = json.load(open(os.path.join(r2.WORK, 'gbm', 'b2b', 'gbm2.json')))
        cpu = round((g['table_cpu_s'] + g['train_cpu_s']) / 3600, 3)
    else:
        cpu = json.load(open(os.path.join(r2.WORK, 'cand', run, 'train.json')))['train_cpu_h']
    cls = {k: round(v['nse'], 3) for k, v in d['by_class_nse'].items()}
    v1cls = {k: round(v['nse'], 3) for k, v in v1d['by_class_nse'].items()}
    vs = {k: f"{v['paired']['nse']['diff']:+.3f} {v['paired']['nse']['verdict']}" for k, v in d['vs'].items()}
    why = (f"sim val on b2's frozen set {fr['b2_simval_on_b2']:.4f} vs v1.0 {fr['v1_simval_on_b2']:.4f} "
           f"({fr['sim_gain']:+.4f} > 0.01) and dev {fr['b2_dev']:.4f} >= 0.95 x {fr['v1_dev']:.4f}: kept")
    rj = {'arm_id': new, 'label': I['label'], 'family': I['family'], 'family_label': I['family_label'],
          'group': 'round2', 'protocols': ['ungauged'], 'numpy_only': True, 'numpy_only_reason': '',
          'compute': {'train_cpu_hours': cpu, 'notes': 'one rung: the v1.0 recipe retrained on bank b2, one thread '
                                                       '(process_time), shared 16-core Windows machine'},
          'feature_engineering': f'identical to {arm} (round 1); only the training bank differs',
          'extra_files': I['extra'], 'compare_to': [arm], 'bank': 'b2 (= b2b: p1 + the reconciled water-yield prior)',
          'margin': 0.01,
          'rule': "PREREG section 4: keep if sim val median NSE (both on b2's frozen val) > v1.0 + 0.01 AND dev median "
                  "NSE >= 0.95 x v1.0's",
          'rungs': [
              {'id': 'v1.0', 'change': f'{arm} as registered in round 1 (bank p1)', 'decision': 'base',
               'why': 'the incumbent', 'simval_nse_b2': round(fr['v1_simval_on_b2'], 4),
               'dev_nse': round(fr['v1_dev'], 4), 'dev_kge': round(fr['v1_dev_kge'], 4), 'dev_by_class_nse': v1cls},
              {'id': 'b2', 'change': f'same recipe ({I["recipe"]}) trained on bank b2', 'decision': 'released',
               'why': why, 'simval_nse_b2': round(fr['b2_simval_on_b2'], 4),
               'simval_nse_p1': round(p1v, 4) if p1v is not None else None, 'dev_nse': round(d['nse'], 4),
               'dev_kge': round(d['kge'], 4), 'dev_fhv': round(d['fhv'], 1), 'dev_cov50': round(d['cov50'], 3),
               'dev_cov90': round(d['cov90'], 3), 'dev_crps_rel': round(d['crps_rel'], 3), 'dev_by_class_nse': cls,
               'dev_vs': vs}],
          'released': 'b2'}
    dst = os.path.join(r2.ROUND2, 'arms', new)
    json.dump(rj, open(os.path.join(dst, 'ratchet.json'), 'w'), indent=1)
    card = f"""# {new}: {I['label']} (River Bake-off round 2)

**What it is.** `{arm}` from round 1 with its released recipe and hyperparameters unchanged ({I['recipe']}),
trained on bank **b2** instead of p1. Ungauged. numpy only at inference. Zero real labels in training.

**Bank b2** = FlowCast's p1 recipe with one dev-calibrated change (round2/PREREG.md, rung b2b): the ERA5-Land
water-yield multiplier m is drawn from a constant prior, log m = -0.026 + N(0, 0.153), instead of p1's regression on
elevation, snow fraction and wetness. The constant form won the pre-registered leave-one-basin-out test on the 52 dev
basins (robust RMSE 0.153 vs 0.159 for p1's form). Dry basins no longer get m well below 1, so the bank holds less
"lost" water there: mean simulated flow 1.556 -> 1.411 mm/day.

**Rung** (rule: sim val on b2's frozen val > v1.0 + 0.01 and dev >= 0.95 x v1.0):

| | Sim val (b2 frozen) | Dev NSE | Dev KGE | Dev by class (snow / arid / humid / regulated / large) |
|---|---|---|---|---|
| v1.0 (`{arm}`, bank p1) | {fr['v1_simval_on_b2']:.3f} | {fr['v1_dev']:.3f} | {fr['v1_dev_kge']:.3f} | {' / '.join(str(v1cls.get(c)) for c in ('snow', 'arid', 'humid', 'regulated', 'large'))} |
| b2 (this arm) | {fr['b2_simval_on_b2']:.3f} | {d['nse']:.3f} | {d['kge']:.3f} | {' / '.join(str(cls.get(c)) for c in ('snow', 'arid', 'humid', 'regulated', 'large'))} |

Decision: **kept** ({why}). Sim val on p1's frozen set: {p1v if p1v is None else round(p1v, 3)}.
The sim criterion mostly measures the bank shift itself: a net trained on b2 is scored on b2's own val set,
which the v1.0 net never saw. The dev criterion is the one that speaks to real basins, and it is a non-inferiority
check (>= 95 %), not a gain.

**Dev, other numbers.** FHV {d['fhv']:.1f} %, 50 / 90 % coverage {d['cov50']:.2f} / {d['cov90']:.2f}, CRPS / mean
flow {d['crps_rel']:.3f}. Paired against the dev references: {', '.join(f'{k} {v}' for k, v in vs.items())}.

**Compute.** {cpu} single-thread CPU-hours (this rung).

**Limits.** One seed (the round-1 recipe's seed 0); the LSTM control's dev seed spread on this bank was 0.439-0.443.
Dev is 52 basins; class medians rest on 5-14 basins each.
"""
    open(os.path.join(dst, 'card.md'), 'w').write(card)
    print('wrote', dst)


if __name__ == '__main__':
    main(sys.argv[1])
