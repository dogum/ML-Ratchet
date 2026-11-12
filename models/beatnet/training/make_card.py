"""Write release/MODEL_CARD.md from exam.json, ratchet.json and physics_check.json (numbers never typed by hand)."""
import _paths as _P  # portable paths (see _paths.py)
import json
B = _P.WORK + '/'
E = json.load(open(B + 'exam.json')); R = json.load(open(B + 'ratchet.json')); P = json.load(open(B + 'physics_check.json'))
pc = lambda x: f'{100 * x:.1f}'
def row(nm, s): return f"| {nm} | " + ' | '.join(f"{pc(s['se'][c])} / {pc(s['ppv'][c])}" for c in range(4)) + f" | {s['macro_f1']:.3f} |"
def exam_table(sp):
    d = E[sp]; L = ['| Method | N Se / +P | S Se / +P | V Se / +P | F Se / +P | macro-F1 |', '|---|---|---|---|---|---|']
    L += [row('**BeatNet** (simulation only, dev prior)', d['beatnet']), row('BeatNet without the dev prior', d['beatnet_raw']), row('Linear discriminant (de Chazal features) trained on our simulations', d['ld_sim']),
          row('Textbook rules (RR prematurity, QRS width, shape)', d['rules']), row('Majority class (always N)', d['majority'])]
    return '\n'.join(L)
def det_table(sp):
    d = E[sp]; L = ['| Method | N Se / +P | S Se / +P | V Se / +P | F Se / +P | macro-F1 |', '|---|---|---|---|---|---|']
    L += [row('**BeatNet**, own QRS detector', d['beatnet']), row('Textbook rules, own QRS detector', d['rules']), row('Always N, own QRS detector', d['majority'])]
    return '\n'.join(L)
cm = E['ds2']['beatnet']['cm']; cmt = '| true \\ called | N | S | V | F |\n|---|---|---|---|---|\n' + '\n'.join(f"| {c} | " + ' | '.join(f'{v:,}' for v in cm[i]) + ' |' for i, c in enumerate('NSVF'))
pub = '\n'.join(f"| {k} (trained on real DS1) | " + ' | '.join(f"{pc(v['se'][c])} / {pc(v['ppv'][c])}" for c in range(4)) + ' | |' for k, v in E['published'].items())
rt = ['| # | Rung | frozen-sim macro-F1 | dev S/V F1 | DS2 macro-F1 (not used) | kept | why |', '|---|---|---|---|---|---|---|']
rt += [f"| {i + 1} | {r['note']} | {r['objective']:.3f} | {r['dev']['sv_f1']:.3f} | {r['exam']['macro_f1']:.3f} | {'yes' if r['accepted'] else 'no'} | {r['why']} |" for i, r in enumerate(R)]
q = E['qrs']; ph = P['physiology']; A = E['ds2'].get('af'); dd = E['ds2_detector']['detector']
other = '\n'.join(f"| {nm} | {E[k]['beatnet']['n']:,} | {pc(E[k]['beatnet']['se'][1])} / {pc(E[k]['beatnet']['ppv'][1])} | {pc(E[k]['beatnet']['se'][2])} / {pc(E[k]['beatnet']['ppv'][2])} | {E[k]['beatnet']['macro_f1']:.3f} | {E[k]['rules']['macro_f1']:.3f} |"
                  for k, nm in (('ds2', 'MIT-BIH DS2 (exam)'), ('ds1rest', 'MIT-BIH DS1 minus dev (held out)'), ('svdb', 'MIT-BIH Supraventricular Arrhythmia DB (held out)'), ('dev', 'dev (6 records; used for choosing)')))
af = '' if not A else f"""
## Rhythm output: atrial fibrillation (DS2, beat by beat)

| Method | AF Se | AF +P | non-AF specificity | AUC |
|---|---|---|---|---|
| BeatNet AF output (simulation only) | {pc(A['beatnet']['se'])} | {pc(A['beatnet']['ppv'])} | {pc(A['beatnet']['spec'])} | {A['beatnet']['auc']:.3f} |
| RR-irregularity rule, threshold fitted on simulations | {pc(A['rr_rule']['se'])} | {pc(A['rr_rule']['ppv'])} | {pc(A['rr_rule']['spec'])} | {A['rr_rule']['auc']:.3f} |
| Never AF | 0.0 | 0.0 | 100.0 | 0.5 |

{A['beatnet']['n_af']:,} of {A['beatnet']['n']:,} DS2 beats lie inside a reference AF episode. Flutter beats are left out.
"""
VX = json.load(open(B + 'verify_export.json')); CC = json.load(open(B + 'calib_coupling.json'))
FAILS = json.load(open(B + 'fails.json')) if __import__('os').path.exists(B + 'fails.json') else dict(fails=[], limits=[])

card = f"""# BeatNet v1.0 model card

**Not a medical device.** BeatNet is a research and teaching model. It has not been validated for any clinical use. Do not
use it to diagnose, monitor or treat anyone.

## What it does
For every heartbeat in a single-lead ECG (modified lead II, any sampling rate), BeatNet gives the AAMI EC57 class:
**N** normal (bundle-branch-block beats included), **S** supraventricular ectopic, **V** ventricular ectopic, **F** fusion of
a normal and a ventricular beat. It also gives class probabilities, the QRS width in ms with a 68% error bar, and the
probability that the beat sits inside atrial fibrillation. Class Q (unclassifiable) is not predicted; Q beats are left
out of every score, as is usual.

It was trained **only on simulated heartbeats**. No real patient's labelled beat was ever a training sample. The question
it answers: how far does a physics simulator that spans patient variation get on the hard inter-patient problem?

## Inputs and outputs
- Input: one ECG lead in mV, its sampling rate, and beat positions (R-peaks). If you have no positions, a simple QRS
  detector in `pipeline.py` finds them.
- Front end (`pipeline.py`, numpy + scipy, identical for simulated and real data): resample to 360 Hz, band-pass 0.5-40 Hz,
  decimate to 180 Hz; per beat a 2.5 s window divided by the local QRS size, the median of its 40 neighbouring beats, and six
  rhythm features (pre-RR, post-RR and local mean RR as ratios, RR irregularity, relative QRS size).
- Outputs per beat: `label`, `probs` (N, S, V, F), `qrs_ms`, `qrs_sig_ms`, `af_prob`. Per record: `heart_rate_bpm`, `af_share`.

## The physics (simulator)
- **Heart**: a 2 mm voxel model with a thick left ventricle, a crescent right ventricle and thin atria; random size,
  orientation (frontal axis, rotation about the long axis, tilt) per patient.
- **Activation**: shortest-path (eikonal) spread. Muscle 0.2-0.4 mm/ms, Purkinje-lined endocardium 1.2-2.2 mm/ms, atria
  0.75-1.1 mm/ms. Normal beats start at His-Purkinje root sites (LV septum, LV anterior and posterior, RV moderator band) after
  the sinus P and an AV delay of 125-205 ms. Bundle-branch block removes one side's roots. A PVC starts from a random
  ventricular voxel with little Purkinje engagement, so the QRS is wide and the T is discordant. Atrial premature beats start
  from an ectopic atrial focus (normal QRS, aberrant right-bundle-like QRS when very early). Fusion: tau = min(tau_normal,
  tau_ectopic + delta).
- **Dipole**: each voxel's membrane potential rises at activation and falls one action-potential duration later (transmural and
  apico-basal APD gradients, electrotonic coupling, restitution with RR). Heart vector = volume sum of -grad Vm.
- **Leads**: Dower lead vectors with random electrode placement and breathing rotation; MIT-BIH records modified lead II.
- **Rhythm**: ECGSYN RR spectrum (McSharry et al. 2003) for HRV; PVCs keep the sinus clock (fully compensatory pause); APBs reset
  it; bigeminy, trigeminy, couplets, runs, interpolated PVCs; atrial fibrillation (irregular RR, 4-9 Hz f-waves) and flutter.
  Atrial premature beats come at 0.28-0.60 of the RR (physics v4; v1-v3 used 0.50-0.88, see Calibration).
- **Noise**: REAL baseline wander, electrode motion and muscle noise from the MIT-BIH Noise Stress Test Database at the
  record's SNR, plus (from physics v3) quiet real background taken from the six dev records, mains hum, amplifier noise, Holter
  filters and 11-bit quantisation.

### Physics checks (`physics_check.json`, `training/physics_check.py`)
| Check | Result | Reference |
|---|---|---|
| Eikonal shortest path vs exact straight-line travel time | +{100 * P['eikonal']['median_rel_err']:.1f}% median, {100 * P['eikonal']['max_rel_err']:.1f}% max | geometry bound of 26-neighbour paths |
| Dipole: volume form vs surface form | r = {P['dipole']['volume_vs_surface_corr']:.3f} | Gauss theorem |
| Lead II: Dower vector vs infinite-medium lead field | {P['lead_field']['angle_deg_dower_vs_infinite_medium']['II']:.0f} deg (median of 8 leads {P['lead_field']['median_angle']:.0f} deg) | torso boundary ignored, tens of degrees expected |
| RR spectrum LF/HF asked 0.5 / 1 / 2 | {' / '.join(f"{x['measured_median']:.2f}" for x in P['ecgsyn']['rr_lfhf'])} | ECGSYN method |
| Average beat vs ECGSYN lead II | r = {P['ecgsyn']['beat_corr_leadII']:.2f} | McSharry et al. 2003 |
| QRS ms (p10/p50/p90) normal / LBBB / RBBB / PVC | {'/'.join(f'{v:.0f}' for v in ph['qrs_ms_normal_p10_50_90'])} · {'/'.join(f'{v:.0f}' for v in ph['qrs_ms_LBBB'])} · {'/'.join(f'{v:.0f}' for v in ph['qrs_ms_RBBB'])} · {'/'.join(f'{v:.0f}' for v in ph['qrs_ms_PVC'])} | textbook: normal 70-110 (vector definition reads shorter), BBB >= 120, PVC 120-200 |
| Frontal QRS axis (p10/p50/p90) | {'/'.join(f'{v:.0f}' for v in ph['frontal_axis_deg'])} deg | normal -30..+90 |
| Pause around PVC / APB (x 2 RR) | {ph['pause_ratio_PVC_median']:.2f} / {ph['pause_ratio_APB_median']:.2f} | full compensatory = 1.0; APB < 1 |

## Calibration on the dev records (physics versions)
Each correction was measured by putting the six real dev records next to simulations of the same measurement (`calib.py`,
`calib_coupling.py`), then built into the simulator as a new physics version with a new frozen validation set.
- v2: T wave too early and too small on sims (APD +47 ms, weaker electrotonic smoothing), PVC T waves too weak (gain 1.35), noise too high.
- v3: noise floor still 1.7x the dev floor at every percentile: NSTDB noise -10 dB plus quiet real background cut from the dev
  records; a fusion beat is labelled F only when each front takes 25-75% of the ventricles.
- v4: prematurity of S beats (pre-RR / local mean RR, median and IQR): dev {CC['v4']['pre_rr_ratio']['S']['dev'][2]:.2f} ({CC['v4']['pre_rr_ratio']['S']['dev'][1]:.2f}-{CC['v4']['pre_rr_ratio']['S']['dev'][3]:.2f}), v3 sims {CC['v3']['pre_rr_ratio']['S']['sim'][2]:.2f} ({CC['v3']['pre_rr_ratio']['S']['sim'][1]:.2f}-{CC['v3']['pre_rr_ratio']['S']['sim'][3]:.2f}),
  v4 sims {CC['v4']['pre_rr_ratio']['S']['sim'][2]:.2f} ({CC['v4']['pre_rr_ratio']['S']['sim'][1]:.2f}-{CC['v4']['pre_rr_ratio']['S']['sim'][3]:.2f}).

## Network
A 1-D residual CNN (5 blocks, 16-64 channels, about 172k weights) over the 2.5 s window (180 Hz), with the six rhythm
features joined before a 3-layer head. Outputs: 4 class logits, log QRS width with log-sigma (Gaussian NLL), AF logit.
Loss: cross-entropy with 0.05 label smoothing + 0.3 x QRS NLL + 0.3 x AF BCE. AdamW, cosine schedule, EMA weights.
Banks: 40,000 simulated beats (2,500 simulated patients) for r1-r4 and 24,000 (1,500 patients, physics v4) for r5, rare classes
over-sampled; rungs after r1 start from the incumbent's weights. One CPU thread.
The release runs in numpy (BatchNorm folded into the convolutions); numpy matches torch to {VX['max_abs_diff']:.1e} (largest raw-output difference on {VX['n']:,} DS2 beats; labels agree on {100 * VX['label_agreement']:.0f}%).

## Ratchet
Objective: macro-F1 over N/S/V/F on a frozen simulated set (400 simulated patients, versioned with the physics; when the
physics changed every rung was re-scored on the new set and every decision re-taken in order). A rung is kept only if it
beats the incumbent by more than 0.01 AND, on the six real dev records (MIT-BIH DS1 101, 112, 119, 201, 203, 220, fixed seed
2026, never in any exam), the mean S/V F1 does not drop by more than 0.02. The DS2 column was logged but never used.

{chr(10).join(rt)}

## Final exam: MIT-BIH Arrhythmia Database, inter-patient DS2 (de Chazal et al. 2004)
22 records, {sum(E['ds2']['counts'][:4]):,} beats (N {E['ds2']['counts'][0]:,}, S {E['ds2']['counts'][1]:,}, V {E['ds2']['counts'][2]:,}, F {E['ds2']['counts'][3]:,}),
MLII, paced records excluded. Beat positions = reference annotations. Se = sensitivity, +P = positive predictivity, in %.
The only real-data choice is one prior scale lambda = {E['prior']['lam']} picked on the dev records.

{exam_table('ds2')}
{pub}

Published methods were trained on real labelled DS1 beats; they are listed for scale only.

BeatNet confusion matrix on DS2:

{cmt}

### Same exam, beats found by our own QRS detector
The detector (Pan-Tompkins style, `pipeline.detect`) found the beats; a detection within 150 ms of a reference beat takes its
class. Missed beats ({sum(dd['missed'][:4]):,} of {dd['n_ref']:,}) count against sensitivity, false detections ({dd['false_det']}) against +P.

{det_table('ds2_detector')}

### Other held-out sets
| Set | beats | S Se / +P | V Se / +P | BeatNet macro-F1 | rules macro-F1 |
|---|---|---|---|---|---|
{other}
{af}
## Error bars and coverage
- **QRS width** (QT Database, cardiologists' onset/offset marks, beats from our detector): a linear map from the simulator's
  vector definition to calipers (width = {q['a']:.1f} + {q['b']:.2f} x raw) and a widen-only sigma factor ({q['sig_factor']:.2f}) were fitted
  on 8 dev records only. Exam (MLII records, {q['exam_MLII']['n']} beats, {q['exam_MLII']['records']} records): mean error {q['exam_MLII']['beatnet_mae']:.1f} ms vs classical slope rule
  {q['exam_MLII']['classical_mae']:.1f} ms vs median guess {q['exam_MLII']['median_guess_mae']:.1f} ms; 68% bar covers {pc(q['exam_MLII']['coverage_68'])}% (all leads: {pc(q['exam_all']['coverage_68'])}%).
- **Class probabilities**: expected calibration error {pc(E['ds2']['ece'])}% on DS2 (10 bins).

## Where it fails
{chr(10).join('- ' + x for x in FAILS['fails'])}

## Limits
{chr(10).join('- ' + x for x in FAILS['limits'])}

## Data and licences
- MIT-BIH Arrhythmia Database: Moody GB, Mark RG. IEEE Eng Med Biol 20(3):45-50 (2001).
- MIT-BIH Noise Stress Test Database: Moody GB, Muldrow WE, Mark RG. Computers in Cardiology 11:381-384 (1984).
- MIT-BIH Supraventricular Arrhythmia Database: Greenwald SD, Patil RS, Mark RG. Computers in Cardiology 461-464 (1990).
- QT Database: Laguna P, Mark RG, Goldberger AL, Moody GB. Computers in Cardiology 24:673-676 (1997).
- PhysioNet: Goldberger AL et al. Circulation 101(23):e215-e220 (2000). All four databases: Open Data Commons Attribution License v1.0.
- Inter-patient split: de Chazal P, O'Dwyer M, Reilly RB. IEEE TBME 51(7):1196-1206 (2004).
- ECGSYN RR model: McSharry PE, Clifford GD, Tarassenko L, Smith LA. IEEE TBME 50(3):289-294 (2003). Lead vectors: Dower GE et al. (1980).
- Code and weights: MIT licence. The example excerpts in `examples/` are MIT-BIH data (ODC-By).
"""
open(B + 'release/MODEL_CARD.md', 'w').write(card); print('MODEL_CARD.md', len(card), 'chars')
