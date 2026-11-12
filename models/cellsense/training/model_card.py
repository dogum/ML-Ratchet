"""MODEL_CARD.md and README.md for CellSense, written from the logs and the page data."""
import _paths as _P  # portable paths (see _paths.py)
import json, numpy as np
R = json.load(open(_P.WORK + '/ratchet.json')); D = json.load(open(_P.LAB + '/data.json')); ph = json.load(open(_P.WORK + '/physics_check.json'))
CAL = json.load(open(_P.WORK + '/calibration.json')); W = D['real']['with_ref']; N = D['real']['no_ref']; L = D['life']
pc = lambda x, d=1: f"{100 * x:.{d}f}%"
rel = next(i for i, r in enumerate(R) if r['ckpt'] == D['release']['ckpt'] and r.get('life_ckpt') == D['release']['life_ckpt'] and r.get('partial_ckpt') == D['release'].get('partial_ckpt')) + 1
PT = D.get('partial'); RR = R[rel - 1]
rows = []
for i, r in enumerate(R):
    v = r['val']; lf = r.get('life') or {}
    rows.append(f"| {i + 1} | {r['note'].split(':')[0]} | {'kept' if r['accepted'] else 'rejected'} | {r['objective']:.3f} | {v['health']:.3f} | {v['life']['skill']:.3f} | {r['dev']['q_total_mae_mAh']:.1f} | "
                f"{r['real']['with_ref']['q_total_mae_mAh']:.1f} | {r['real']['with_ref']['fade_corr']:.3f} | {(lf.get('cycle100', {}).get('primary', {}).get('mpe', float('nan'))):.1f} | "
                f"{(r.get('dev_life') or {}).get('mpe_all', float('nan')):.1f} | {(r.get('partial') or {}).get('0.75', {}).get('full_ref', {}).get('q_total_mae_mAh', float('nan')):.1f} |")
spm = '\n'.join(f"| {r['case']} | {r['C']}C | {r['cap_err_pct']:.3f}% | {r['V_mederr_mV']:.2f} | {r['V_maxerr_2_90pct_mV']:.1f} |" for r in ph['spm'])
spme = '\n'.join(f"| {r['case']} | {r['C']}C | {r['cap_err_pct']:.3f}% | {r['V_mederr_mV']:.2f} | {r['V_maxerr_2_90pct_mV']:.1f} |" for r in ph['spme']['rows'])
cv = '\n'.join(f"| {r['case']} | {r['cc_ours']:.4f} / {r['cc_pybamm']:.4f} | {r['total_ours']:.4f} / {r['total_pybamm']:.4f} |" for r in ph['cv_hold'])
cal = '\n'.join(f"| {k} | {f['v_rms_mV']:.1f} | {f['t_rms_C']:.2f} | {f['q_cc_err_mAh']:+.0f} | {f['q_tot_err_mAh']:+.0f} | {f['ir_err_mohm']:+.2f} |" for k, f in CAL.items())
hid = '\n'.join(f"| {h['stage'] if h['stage'] != 'all' else 'All readings'} | {h['n']:,} | {1000 * h['mae_net']:.1f} | {1000 * h['mae_base']:.1f} |" for h in D['hidTable'])
S = L['severson']; O = L['ours']; M = L['median']
stage = '\n'.join(f"| {s['label']} | {s['n']} | {s['mpe']:.1f}% | {s['left_mae']:.0f} |" for s in L['stage'])
part = '\n'.join(f"| {r['label']} | {100 * r['frac_cut']:.0f}% | {r['net']:.1f} | {r['net_cut']:.1f} | {r['base']:.1f} |" for r in PT['rows']) if PT else ''
card = f"""# CellSense model card (v1.1, rung {rel})

**What it is.** Three 1-D convolutional networks ({D['release']['params']:,} weights in all) that read one 4C constant-current
discharge of an LFP/graphite cell (the A123 APR18650M1A, 1.1 Ah, of Severson et al. 2019) as capacity Q(V) and can
temperature T(V) at 250 voltages from 3.5 V to 2.0 V, optionally next to the same cell's discharge from its first cycles.
The health network returns, each with a 68% error bar:

- `q_total`: capacity including the 2.0 V hold to C/50 (what Severson's 0.88 Ah end-of-life rule measures)
- `q_cc`: capacity of the 4C part; `ir`: 30 ms pulse resistance at 80% SOC (mOhm)
- `lli`, `lam_ne`, `lam_pe`: lithium, graphite and LFP lost since new (Dubarry's degradation modes)
- `r_growth`, `fade`: resistance growth and capacity fade since new

The lifetime network also takes the cycle number and the charging time from 0 to 80% SOC and returns `log_life` (log10
of the cycle at which `q_total` reaches 0.88 Ah) and `left` (cycles left). A discharge that stops before 2.0 V (a partial
discharge from full) goes to the third network, which returns the health outputs from what is there. The lifetime and
partial-discharge networks read the curves smoothed over 24 mV, because the simulator's 4C curves carry sharper
graphite-staging features than real cells.

**Trained on no real cell.** Six real cells (two per batch) calibrated the new-cell physics and later served as a real-data
dev set; they are excluded from every exam. Every other number below is on cells the networks never saw.

## The physics (all training data)

A vectorized single-particle model of each electrode with PyBaMM's Prada2013 open-circuit curves (LFP: Afshar 2017,
graphite: Chen 2020), Fickian diffusion solved exactly between time steps in the eigenbasis of a 10-shell finite-volume
operator, Butler-Volmer kinetics, a lagged electrolyte polarization, a lumped thermal model with Arrhenius temperature
dependence, electrode heterogeneity (open-circuit curves averaged over a spread of local states of charge), and a slow,
poorly connected graphite population. Discharge: 4.4 A to 2.0 V, then a hold at 2.0 V to 22 mA, as in Severson et al.

Checked against PyBaMM 26.10 (Prada2013 set), single-particle model (SPM), constant current:

| Case | Rate | Capacity error | Median V error (mV) | Max V error, 2-90% (mV) |
|---|---|---|---|---|
{spm}

Electrolyte polarization against the SPMe (two constants fitted on the fresh cell at 4C, then held fixed):

| Case | Rate | Capacity error | Median V error (mV) | Max V error, 2-90% (mV) |
|---|---|---|---|---|
{spme}

The hold at 2.0 V against PyBaMM's experiment runner (SPM, 9.2 A, to 46 mA), capacity ours / PyBaMM (Ah):

| Case | 4C part | Total |
|---|---|---|
{cv}

**Calibration to the real cell** (cycles 5-10 of six cells; least squares on the 4C voltage curve, can temperature,
both capacities and the pulse resistance):

| Cell | V rms (mV) | T rms (C) | 4C capacity (mAh) | Total (mAh) | Pulse R (mOhm) |
|---|---|---|---|---|---|
{cal}

The fits favour the curve's shape and read 1-3% high on capacity, so new cells' lithium inventory is trimmed by the six
cells' mean error. New training cells are drawn around the six fits (a mixture in log-parameter space).

**Aging.** Health data: each new cell is paired with an aged copy (lithium lost up to 28%, graphite up to 30% with part
of it lithiated, LFP up to 20%, ohmic and kinetic resistance up to 1.5x, slow polarization up to 10x, poorly connected
graphite up to 30% behind 0.03-1 ohm). Lifetime data (v2.1): whole trajectories at a charging rate of 3.6-8C: SEI lithium
loss linear in cycles (these cells do not follow sqrt(t); Attia et al. 2020), graphite loss accelerating as cycles^beta,
slow LFP loss, lithium plating once the graphite is filled past a rate-dependent threshold at the top of charge, and
resistance and slow polarization growing late and with wear. Early life: a share of slow (poorly connected) graphite grows
from the first cycles (~cycles^0.8, levelling off at 8-14%), and cells that grow more of it lose graphite sooner (graphite
loss rate ~ share^3 around each calibration cell). End of life is the first cycle under 0.88 Ah; each cell is read once at
a random cycle (a quarter of them near cycle 100), next to its own curve at cycle 10. {D['release']['samples']:,} simulated cells in all.

**Early life, calibrated on the six cells.** The change in the 4C curve between cycles 10 and 100 (Severson's dQ_100-10(V))
is fitted on every calibration cell by one mechanism, the slow graphite share (4-10% by cycle 100), with small trims to
the slow polarization and lithium; the fits leave 1-4 mAh of a 10-41 mAh signal. Each cell's share is then matched so
its simulated dQ has the measured variance, and the strength of the link to graphite loss was chosen so that simulated
cells show roughly the six cells' slope of log10 cycle life against log10 var(dQ) (-0.38; simulated, -0.33 to -0.40). The
six cells' own relation scatters by 0.06 in log10 life (Severson's training cells, checked but not used: 0.07); the
simulated one scatters more (0.15), so the network's error bars are wider than the real relation needs.

## The ratchet

Objective = 0.5 x health + 0.5 x lifetime. Health: on a frozen set of 4,000 simulated new/aged pairs, the mean skill
(1 - MAE / MAE of guessing the median, floored at 0) over total capacity, pulse resistance, lithium lost, graphite lost, LFP
lost, resistance growth and fade, averaged over three readings: with the early curve, without it, and (from rung 6) cut
short as partial discharges at 30-95% depth. Lifetime: the same skill on log10 cycle life over 3,000 simulated cells (zero
without a lifetime network). Kept only if the objective rises by more than 0.01 **and**, on the six calibration cells (the
dev set), the total-capacity error is no more than 1 mAh worse **and** (from rung 6) the lifetime error, read every 50
cycles through life, is no more than 2 points worse.

| Rung | Change | Decision | Objective | Health | Lifetime | Dev cells (mAh) | Exam capacity error (mAh) | Exam fade corr. | Life error at cycle 100, primary (%) | Dev lifetime error (%) | Partial to 0.75 Ah (mAh) |
|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

History, in order: twice the calibration cells' worn-out curves showed something the simulator lacked (slow polarization
that grows several-fold while the pulse resistance barely moves; then charge that only comes out during the hold), and each
time the frozen sets were regenerated and every rung re-scored on them. Rungs 2 and 3 improved the simulated exam but
made the calibration cells' capacity worse (13.3 and 21.6 vs 11.4 mAh). The dev gate was added after rung 3 and applied to
every rung; it rejects both. The exam cells told the same story, but were never used for the decision.

Version 1.1 (rungs 6-9). Rung 6 added the early-life physics; the simulated lifetime exam improved, but the calibration
cells' lifetime error read through life got worse (45% vs 31%), and the new lifetime gate rejected it. A whole-life check
of simulated against real curves on the same six cells at 50, 80 and 100% of life showed too much slow graphite and too
little late transport polarization; with those set from the check (rung 7), the calibration cells' lifetime error fell
to {R[6]['dev_life']['mpe_all']:.0f}%, and it was kept. A twin of rung 7 without smoothing scored the same on the simulated set and worse on
the calibration cells, and was not entered. Rung 8 retrained the health network on half-cut curves: better on the
simulated exam, worse on the calibration cells' full curves (16.6 vs 11.4 mAh), rejected. Rung 9 gave partial discharges
their own network and kept the full-curve network: kept.

## Final exam 1: health now, {W['cells']} cells, {W['n']:,} readings

Every 25 cycles through life, read with the same cell's mean curve from cycles 5-10. Answer keys from the cycler:
total discharge capacity (including the hold), 30 ms pulse resistance, and both relative to cycles 5-10. Readings with
corrupted records (a curve or thermocouple glitch) are excluded.

| Stage of life | Readings | CellSense total-capacity error (mAh) | Reading the 4C curve's end (mAh) |
|---|---|---|---|
{hid}

With the early curve: capacity bias {W['q_total_bias_mAh']:+.1f} mAh, correlation {W['q_total_corr']:.3f}, inside 2 sigma {pc(W['q_total_in_2sigma'], 0)};
fade error {W['fade_mae_pct']:.2f} points, correlation {W['fade_corr']:.3f}; pulse resistance error {W['ir_mae_mohm']:.2f} mOhm, correlation {W['ir_corr']:.2f};
resistance growth correlation {W['r_growth_corr']:.2f}; past or not past end of life called right {pc(W['eol_called_right'], 1)}. From the curve alone:
capacity error {N['q_total_mae_mAh']:.1f} mAh, fade correlation {N['fade_corr']:.3f}. Lost lithium and lost electrode material have no answer key.

## Final exam 2: cycle life from cycle 100

Severson et al.'s split (cells ordered as their loading notebook; primary test = even indices of batches 1-2 plus the
last, secondary = batch 3); calibration cells left out. Their Table 2, models trained on 41 real cells:

| Model | Primary RMSE (cycles) | Primary error | Secondary RMSE | Secondary error |
|---|---|---|---|---|
| Severson variance | {S['variance']['primary_rmse']} | {S['variance']['primary_mpe']}% | {S['variance']['secondary_rmse']} | {S['variance']['secondary_mpe']}% |
| Severson discharge | {S['discharge']['primary_rmse']} | {S['discharge']['primary_mpe']}% | {S['discharge']['secondary_rmse']} | {S['discharge']['secondary_mpe']}% |
| Severson full | {S['full']['primary_rmse']} | {S['full']['primary_mpe']}% | {S['full']['secondary_rmse']} | {S['full']['secondary_mpe']}% |
| **CellSense (no real cells)** | {O['primary_rmse']:.0f} | {O['primary_mpe']:.1f}% | {O['secondary_rmse']:.0f} | {O['secondary_mpe']:.1f}% |
| Guess the test cells' median | {M['primary_rmse']:.0f} | {M['primary_mpe']:.1f}% | {M['secondary_rmse']:.0f} | {M['secondary_mpe']:.1f}% |

The median row is a no-skill reference that cheats (it uses the test cells' own lives). Readings later in life:

| Read at | Readings | Life error | Cycles-left error |
|---|---|---|---|
{stage}

## Final exam 3: partial discharges

The exam readings of Final exam 1, cut short once they have delivered a given charge from full, everything below thrown
away, and read by the partial-discharge network. Total-capacity error (mAh), with the cell's full early curve as reference
and with that curve cut at the same depth; the no-model baseline adds the shift between the two curves at the cut to the
early total. Assuming no fade at all errs by {PT['nofade'] if PT else float('nan'):.1f} mAh; the full-curve network on full discharges, {PT['full'] if PT else float('nan'):.1f} mAh.

| Stopped after | Readings cut short | CellSense | Reference cut too | No-model baseline |
|---|---|---|---|---|
{part}

## Limits

{chr(10).join('- ' + x for x in D['limits'])}
"""
import re; card = re.sub(r'\| nan(?= \|)', '| —', card)
open(_P.WORK + '/release/MODEL_CARD.md', 'w').write(card)
readme = f"""# CellSense v1.1

Battery health from one fast discharge, learned from physics alone. Three small networks read a 4C discharge of an
LFP/graphite cell (trained on the A123 APR18650M1A of Severson et al. 2019) and return its true capacity (including what
a hold at 2.0 V recovers), pulse resistance, lithium and electrode material lost, fade, and, given the cycle number and
charging time, a forecast of its cycle life. A discharge cut short goes to its own network. Every output has an error
bar. Trained on zero real cells.

New in 1.1: early-life physics (a growing share of slow graphite, tied to the later loss of graphite) makes the forecast
from cycle 100 useful; partial discharges from full are read down to about 0.75 Ah.

    pip install numpy
    python demo.py

    import cellsense, numpy as np
    cs = cellsense.CellSense()
    d = np.load('examples/b3c33.npz')            # a real exam cell: row 0 is its mean curve from cycles 5-10
    r = cs.diagnose(d['Q'][-1], d['T'][-1], d['Q'][0], d['T'][0], cycle=int(d['cycle'][-1]), charge_time_min=float(d['charge_time']))
    print(r['q_total'], r['q_total_sig'], r['life'])

Your own data: record a 4C constant-current discharge to 2.0 V (time, volts, amp-hours out, can temperature) and convert
it with `cellsense.q_of_v(volts, amp_hours, temp)`; a discharge that stops early leaves NaN below where it stopped and is
read by the partial-discharge network. Other chemistries, cell sizes, rates or cut-offs are outside what it learned.

MODEL_CARD.md: the physics and its PyBaMM checks, the calibration, every rung, both exams and the limits.
`training/` holds every script (PyTorch, PyBaMM, scipy).

Data: Severson, Attia et al., Nature Energy 4, 383-391 (2019), via data.matr.io (Toyota Research Institute).
"""
open(_P.WORK + '/release/README.md', 'w').write(readme); print('card', len(card), 'readme', len(readme))
