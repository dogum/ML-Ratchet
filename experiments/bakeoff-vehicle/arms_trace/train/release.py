"""Write a released arm folder from train/decisions.json: predict.py, weights.npz (copied from the released rung's
candidate), ratchet.json (every rung with its numbers and decision) and card.md.
usage: python3 release.py <arm>"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRC = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ladder  # noqa: E402

PREDICT = '''"""{label} (`{arm_id}`): Vehicle Bake-off trace track, arms_trace (agent B).
numpy-only inference: the shared trace pipeline + numpy core in ../common/tracelib.py, weights in weights.npz."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'common'))
import tracelib  # noqa: E402

_M = tracelib.Model(os.path.join(HERE, 'weights.npz'))


def predict(batch):
    return _M.predict(batch)


def predict_many(batches):
    return _M.predict_many(batches)
'''

META = {
    'cnn': dict(arm_id='trace_cnn', label='1-D dilated CNN on the trip trace', family='cnn1d',
                family_label='1-D convolutional network (dilated residual CNN)'),
    'gru': dict(arm_id='trace_gru', label='Bidirectional GRU on the trip trace', family='ssm',
                family_label='State-space and recurrent models (bidirectional GRU)'),
    'transformer': dict(arm_id='trace_transformer', label='Transformer on the trip trace ({ps}-s patch tokens)',
                        family='attn', family_label='Attention and transformers'),
    's4d': dict(arm_id='trace_s4d', label='S4D state-space model on the trip trace', family='ssm',
                family_label='State-space and recurrent models (S4D diagonal state space)'),
}
RULE = ('keep a rung if its full sim val skill beats the incumbent by > 0.01 AND its dev D16 is not worse than the '
        "incumbent's by more than 0.01 (HARNESS.md section 7, EnergyNet's model-rung rule)")


def main(arm):
    dec = json.load(open(os.path.join(HERE, 'decisions.json')))[arm]
    lad = {r['rung']: r for r in ladder.rows(arm)}
    rel = dec['released']
    out = os.path.join(TRC, arm)
    shutil.copy(os.path.join(out, 'work', f'cand_{rel}', 'weights.npz'), os.path.join(out, 'weights.npz'))
    m = meta_for(arm, rel)
    if not os.path.exists(os.path.join(out, 'work', 'npcheck.json')) or dec.get('recheck'):
        import npcheck
        npcheck.main(arm, rel)
    open(os.path.join(out, 'predict.py'), 'w').write(PREDICT.format(label=m['label'], arm_id=m['arm_id']))
    rungs, cpu = [], 0.0
    for r in dec['rungs']:
        L = lad[r['id']]
        cpu += L['cpu_h']
        rungs.append({'id': r['id'], 'change': r['change'], 'decision': r['decision'], 'why': r['why'],
                      'simval_skill': round(L['skill'], 4), 'simval_MdAPE': round(L['MdAPE'], 4),
                      'dev_D16': round(L['D16'], 4), 'train_cpu_hours': L['cpu_h']})
    cpu += dec.get('extra_cpu_h', 0.0)
    meta = {'arm_id': m['arm_id'], 'label': m['label'], 'family': m['family'], 'family_label': m['family_label'],
            'group': 'trace', 'tracks': ['trace'], 'numpy_only': True, 'numpy_only_reason': '',
            'real_labels_in_training': False,
            'compute': {'train_cpu_hours': round(cpu, 3), 'notes': dec.get('compute_notes', '')},
            'feature_engineering': dec['feature_engineering'], 'extra_files': ['../common'], 'margin': 0.01,
            'rule': RULE, 'rungs': rungs, 'released': rel}
    json.dump(meta, open(os.path.join(out, 'ratchet.json'), 'w'), indent=1)
    open(os.path.join(out, 'card.md'), 'w').write(card(arm, dec, meta, lad))
    print('released', arm, rel, 'cpu', round(cpu, 3))


ASSUME = {
    'cnn': ("Energy is read from local patterns. Each 5-s block sees a fixed window of its neighbours (7 residual "
            "layers, kernel 3, dilations 1-64: a receptive field of 255 blocks, about 21 minutes either side), with the "
            "same weights everywhere along the trip. There is no state that runs the whole trip: anything longer than "
            "the window (a long warm-up, a battery running down) has to come from the mean pooling and the statics."),
    'gru': ("A learned state is carried along the trip, once forwards and once backwards (one GRU of 48 units per "
            "direction). Gates decide at every block what to keep and what to overwrite, so it can integrate slow "
            "things such as engine warm-up, the battery's state of charge or the cumulative climb, in trip order."),
    'transformer': ("Any part of the trip can look at any other part: {ps}-s patch tokens ({pb} blocks each) attend to every "
                    "other token of the trip (3 pre-LN layers, 4 heads of 12), with only a sinusoidal position code for "
                    "order. Nothing is built in about locality or time direction; it has to be learned."),
    's4d': ("Memory is linear and time-invariant: each channel is filtered by a sum of decaying and oscillating modes "
            "(S4D, 16 complex modes per channel, ZOH, step sizes 1e-3 to 1e-1 per block, i.e. memories of about "
            "10 to 1,000 blocks), one filter running forwards and one backwards over the trip; the nonlinearity sits "
            "only between layers (GELU, GLU) and in the shared input MLP stem."),
}
NET = {
    'cnn': "core: 7 x [LayerNorm -> Conv1d(48, 48, k 3, dilation 1/2/4/8/16/32/64, zero padding) -> GELU -> Linear(48, 48)], residual",
    'gru': "core: GRU(48 -> 48) forwards + GRU(48 -> 48) on the trip reversed -> Linear(96, 48), residual",
    'transformer': ("core: {ps}-s patch embedding Linear({pb} x 48 -> 48) + sinusoidal positions -> 3 x pre-LN encoder "
                    "block (4 heads, d 48, feed-forward 96, GELU, non-causal, padding masked) -> tokens repeated back to "
                    "their {pb} blocks and added to the stem output"),
    's4d': ("core: 3 x [LayerNorm -> bidirectional S4D (N2 16, forward and backward kernels, D skip) -> GELU -> "
            "Linear(48, 96) -> GLU], residual"),
}


def meta_for(arm, rel):
    """META / ASSUME / NET of the arm, with the released rung's patch size filled in (transformer)."""
    cfg = json.load(open(os.path.join(HERE, 'configs', f'{arm}_{rel}.json')))
    pb = cfg.get('patch', 4); f = dict(ps=5 * pb, pb=pb)
    m = {k: v.format(**f) for k, v in META[arm].items()}
    m['assume'] = ASSUME[arm].format(**f); m['net'] = NET[arm].format(**f)
    return m


def card(arm, dec, meta, lad):
    m = meta_for(arm, dec['released'])
    tj = {r['id']: json.load(open(os.path.join(TRC, arm, 'work', f"train_{r['id']}.json"))) for r in dec['rungs']}
    rel = dec['released']
    npc = json.load(open(os.path.join(TRC, arm, 'work', 'npcheck.json')))
    S = json.load(open(os.path.join(TRC, arm, 'work', 'eval', f'{rel}_trace_summary.json')))
    sv, dv = S['simval'], S['dev']['sets']
    L = []
    L.append(f"# {m['label']} (`{m['arm_id']}`)\n")
    L.append(f"Vehicle Bake-off, **trace track**, group B (trace learners, `arms_trace/`). Family: **{m['family_label']}** "
             f"(`{m['family']}` on the Model families page). numpy-only inference (`predict.py` -> `../common/tracelib.py` "
             f"+ `weights.npz`). Released rung: **{rel}**. Zero real labels in training.\n")
    L.append("## The built-in assumption\n" + m['assume'] + "\n")
    L.append(f"## Network\n{m['net']}. Parameters: {tj[rel]['n_params']:,} (the four trace arms hold 49k-86k).\n")
    L.append("""## Shared with every trace arm (only the core differs)
- **Inputs (the trace track's, nothing else):** the 1 Hz trace (8 channels), `spec` (49) and `rl` (5).
- **Resampling:** the harness's shared rule, `bake.block_means(trace, 5)`: 5-s blocks of the 8 channel means + mean v^2,
  mean v^3 and the summed positive kinetic-energy increment per unit mass, + the block's valid-second fraction ->
  12 scaled channels per block. Traces longer than 1,440 blocks (7,200 s, the sim bank's cap; only real heavy-duty legs,
  up to 37,919 s) are cut into <= 1,440-block chunks; each chunk is encoded, and the chunk means are averaged weighted by
  their block counts (the mean pooling is intensive, so this is the harness's "mean for intensive quantities").
  Block encodings are cached on `trace_key` (EPA rows repeat 5 cycle traces).
- **Statics (57), repeated at every block (early fusion) and fed again to the head:** spec standardised with the train
  bank's mean / sd (tiers drawn with seed 12345); log mass, log CdA, log Crr, Crr speed coefficient and rotating-mass
  fraction from `rl` (standardised); log distance and log duration of the trip (sum of the 1 Hz speeds, length);
  log(E_phys / A0) from r1 on (0 in r0).
- **Stem:** Linear(69 -> 48) + a residual pointwise MLP (48 -> 96 -> 48, GELU): the input-side nonlinearity the River
  Bake-off found S4D needs, given to every arm from the start.
- **Pooling and head:** final LayerNorm, mean over the valid blocks, [pool | statics] -> Linear(105 -> 64) -> GELU ->
  Linear(64 -> 10): the median log(E / anchor), the 25/5 % and 75/95 % quantiles as cumulative softplus offsets
  (monotone, never cross), and 5 log component ratios (aero, rolling, brakes, aux, powertrain losses) / anchor ->
  `components` (MJ) for the page.
- **Anchor:** r0: A0 = the train bank's median energy per km for the vehicle's (architecture, class)
  (`bake.dumb_rates()`, the climatology rate) x the trip distance: no physics. r1 on: E_phys = EnergyNet's generic
  road-load estimate recomputed from the trace-track inputs by the harness's `bake.e_phys_from_trace` (declared
  feature engineering); the network predicts log(E / E_phys), as EnergyNet does.
- **Loss:** pinball loss of the 5 log-quantiles + 0.2 x MSE of the log component ratios (EnergyNet's weight); trips with
  y <= 0.02 MJ masked (EnergyNet).
- **Training:** the harness's train bank only (28,800 simulated trips, one of EnergyNet v2.1's three banks; zero real
  labels), every trip whole, fresh tiers every epoch (`bake.draw_tiers`), target `bake.target`; batches of trips of
  similar length (<= 8,192 blocks, <= 64 trips; ~929 steps per epoch); AdamW (wd 1e-4), one-cycle cosine (3 % warm-up,
  peak 2e-3 -> 1e-5), gradient clip 1.0, 12 epochs; last step used (no early stopping); same data stream for every arm.
  One thread (`torch.set_num_threads(1)`), nice 5, shared 2-core machine.
""")
    L.append("## Ratchet\n")
    L.append(f"Rule: {RULE}. Sim val = the harness's `evaluate.py` on all 3,985 frozen sim val trips (frozen tiers); dev = "
             "the dev gate's D16. Paired bootstrap = sim val skill difference vs the incumbent, 2,000 resamples of the 400 "
             "val vehicles.\n")
    L.append("| Rung | Change | Sim val skill | Sim val MdAPE | Dev D16 | VED ICE / HEV / EV / PHEV | EPA conv FTP / HWY | Mersin diesel / CNG | ICCT | CPU h | Decision | Why |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in dec['rungs']:
        x = lad[r['id']]
        L.append(f"| {r['id']} | {r['change']} | {x['skill']:.3f} | {x['MdAPE']:.4f} | {x['D16']:.4f} | "
                 f"{x['VED_ICE']:.3f} / {x['VED_HEV']:.3f} / {x['VED_EV']:.3f} / {x['VED_PHEV']:.3f} | "
                 f"{x['EPA_conv_FTP']:.3f} / {x['EPA_conv_HWY']:.3f} | {x['Mersin_diesel']:.3f} / {x['Mersin_cng']:.3f} | "
                 f"{x['ICCT']:.3f} | {x['cpu_h']:.2f} | **{r['decision']}** | {r['why']} |")
    L.append("\nReferences (same harness): EnergyNet v2.1 (table track, 3x the training data, 5-member ensemble) sim val "
             "skill 0.680, D16 0.1133; E_phys alone 0.048 / 0.2278; climatology 0 / 0.3093.\n")
    ba = ', '.join(f"{k} {v:.3f}" for k, v in sv['by_arch'].items())
    L.append(f"## Released rung {rel}\nSim val (3,985 trips): skill {sv['skill']:.3f}, MdAPE {sv['MdAPE']:.4f}, bias "
             f"{sv['bias']:+.4f}, cov50 / cov90 {sv['cov50']:.2f} / {sv['cov90']:.2f}, crps_log {sv['crps_log']:.4f}. "
             f"MdAPE by architecture: {ba}.\n")
    L.append("Dev gate (MdAPE, cov50 / cov90; raw bands, no dev widening):\n")
    L.append("| Set | n | MdAPE | bias | cov50 | cov90 |")
    L.append("|---|---|---|---|---|---|")
    for k in ['VED_ICE', 'VED_HEV', 'VED_EV', 'VED_PHEV', 'EPA_conv_FTP', 'EPA_conv_HWY', 'EPA_conv_US06',
              'EPA_conv_FTPcold', 'EPA_conv_SC03', 'EPA_hev_FTP', 'EPA_hev_HWY', 'EPA_bev_FTP', 'EPA_bev_HWY',
              'Mersin_diesel', 'Mersin_cng', 'ICCT', 'ZTBus_legs', 'ZTBus_missions']:
        v = dv.get(k)
        if v and v.get('n'):
            L.append(f"| {k} | {v['n']} | {v['MdAPE']:.3f} | {v['bias']:+.3f} | {v.get('cov50', float('nan')):.2f} | "
                     f"{v.get('cov90', float('nan')):.2f} |")
    L.append(f"\n## Compute\nTraining {meta['compute']['train_cpu_hours']:.2f} single-thread CPU-hours in all "
             f"({meta['compute']['notes']}), measured with time.process_time() in the training process. Budget: about 4. "
             f"Exam inference: see the registry record's estimate.\n")
    a64, a32 = npc['float64']['all'], npc['float32']['all']
    L.append(f"## numpy release check\nnumpy float64 (`predict.py`, each trip unpadded, chunked) vs the torch checkpoint of "
             f"{rel}, row by row, on {a64['rows']} rows (16 sim val trips, 24 VED and 24 EPA dev records, every HD dev record; "
             f"longest {a64['max_blocks']:,} blocks = {a64['max_blocks'] // 1440 + 1} chunks): max |log diff| "
             f"{a64['max_abs_log_diff']:.1e} against torch in float64, {a32['max_abs_log_diff']:.1e} against torch in float32 "
             f"(`work/npcheck.json`). Training-time check (padded torch batch vs unpadded numpy, 5 val trips): "
             f"{tj[rel]['export_check']['max_abs_diff_logq']:.1e}.\n")
    L.append("## Limits\n" + dec['limits'] + "\n")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main(sys.argv[1])
