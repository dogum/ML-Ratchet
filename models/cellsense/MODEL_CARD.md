# CellSense model card (v1.1, rung 9)

**What it is.** Three 1-D convolutional networks (1,934,516 weights in all) that read one 4C constant-current
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
| fresh | 1C | 0.021% | 0.31 | 2.6 |
| fresh | 4C | 0.149% | 0.34 | 2.3 |
| lost lithium 15% | 1C | 0.017% | 0.30 | 2.7 |
| lost lithium 15% | 4C | 0.211% | 0.35 | 3.8 |
| lost graphite 20% | 1C | 0.032% | 0.32 | 3.9 |
| lost graphite 20% | 4C | 0.082% | 0.34 | 4.3 |
| lost LFP 15% | 1C | 0.003% | 0.30 | 3.1 |
| lost LFP 15% | 4C | 0.072% | 0.35 | 2.8 |

Electrolyte polarization against the SPMe (two constants fitted on the fresh cell at 4C, then held fixed):

| Case | Rate | Capacity error | Median V error (mV) | Max V error, 2-90% (mV) |
|---|---|---|---|---|
| fresh | 4C | 0.042% | 0.12 | 2.4 |
| fresh | 1C | 0.003% | 0.30 | 5.4 |
| fresh | 2C | 0.038% | 0.10 | 3.0 |
| lost lithium 15% | 4C | 0.082% | 0.13 | 3.5 |
| lost graphite 20% | 4C | 0.210% | 0.11 | 5.7 |

The hold at 2.0 V against PyBaMM's experiment runner (SPM, 9.2 A, to 46 mA), capacity ours / PyBaMM (Ah):

| Case | 4C part | Total |
|---|---|---|
| fresh | 1.5075 / 1.5074 | 2.7331 / 2.7518 |
| lost lithium 15% | 1.1576 / 1.1575 | 2.3097 / 2.3285 |
| lost graphite 20% | 1.0665 / 1.0665 | 2.1919 / 2.2106 |

**Calibration to the real cell** (cycles 5-10 of six cells; least squares on the 4C voltage curve, can temperature,
both capacities and the pulse resistance):

| Cell | V rms (mV) | T rms (C) | 4C capacity (mAh) | Total (mAh) | Pulse R (mOhm) |
|---|---|---|---|---|---|
| b1c7 | 7.2 | 0.15 | +13 | +11 | +0.27 |
| b1c30 | 8.0 | 0.12 | +22 | +19 | +0.03 |
| b2c12 | 8.1 | 0.20 | +21 | +23 | +0.28 |
| b2c30 | 8.1 | 0.17 | +20 | +15 | +0.14 |
| b3c10 | 5.0 | 0.10 | +17 | +11 | +0.13 |
| b3c25 | 14.8 | 0.17 | +32 | +27 | -0.27 |

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
a random cycle (a quarter of them near cycle 100), next to its own curve at cycle 10. 190,000 simulated cells in all.

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
| 1 | Baseline | kept | 0.037 | 0.074 | 0.000 | 11.4 | 13.2 | 0.979 | — | — | 153.7 |
| 2 | Transport aging | rejected | 0.170 | 0.340 | 0.000 | 13.3 | 14.4 | 0.979 | — | — | 283.2 |
| 3 | Slow graphite | rejected | 0.211 | 0.421 | 0.000 | 21.6 | 22.3 | 0.975 | — | — | 237.7 |
| 4 | Lifetime network | rejected | -0.124 | 0.074 | -0.322 | 11.4 | 13.2 | 0.979 | 133.3 | 66.3 | 153.7 |
| 5 | Lifetime, calibrated | kept | 0.190 | 0.074 | 0.306 | 11.4 | 13.2 | 0.979 | 32.8 | 30.7 | 153.7 |
| 6 | Early-life physics | rejected | 0.316 | 0.074 | 0.557 | 11.4 | 13.2 | 0.979 | 21.5 | 45.3 | 153.7 |
| 7 | Early-life physics, whole-life check | kept | 0.375 | 0.074 | 0.676 | 11.4 | 13.2 | 0.979 | 22.4 | 22.4 | 153.7 |
| 8 | Partial discharges, one network | rejected | 0.625 | 0.574 | 0.676 | 16.6 | 19.9 | 0.967 | 22.4 | 22.4 | 53.1 |
| 9 | Partial discharges, a specialist | kept | 0.466 | 0.256 | 0.676 | 11.4 | 13.2 | 0.979 | 22.4 | 22.4 | 28.7 |

History, in order: twice the calibration cells' worn-out curves showed something the simulator lacked (slow polarization
that grows several-fold while the pulse resistance barely moves; then charge that only comes out during the hold), and each
time the frozen sets were regenerated and every rung re-scored on them. Rungs 2 and 3 improved the simulated exam but
made the calibration cells' capacity worse (13.3 and 21.6 vs 11.4 mAh). The dev gate was added after rung 3 and applied to
every rung; it rejects both. The exam cells told the same story, but were never used for the decision.

Version 1.1 (rungs 6-9). Rung 6 added the early-life physics; the simulated lifetime exam improved, but the calibration
cells' lifetime error read through life got worse (45% vs 31%), and the new lifetime gate rejected it. A whole-life check
of simulated against real curves on the same six cells at 50, 80 and 100% of life showed too much slow graphite and too
little late transport polarization; with those set from the check (rung 7), the calibration cells' lifetime error fell
to 22%, and it was kept. A twin of rung 7 without smoothing scored the same on the simulated set and worse on
the calibration cells, and was not entered. Rung 8 retrained the health network on half-cut curves: better on the
simulated exam, worse on the calibration cells' full curves (16.6 vs 11.4 mAh), rejected. Rung 9 gave partial discharges
their own network and kept the full-curve network: kept.

## Final exam 1: health now, 118 cells, 3,771 readings

Every 25 cycles through life, read with the same cell's mean curve from cycles 5-10. Answer keys from the cycler:
total discharge capacity (including the hold), 30 ms pulse resistance, and both relative to cycles 5-10. Readings with
corrupted records (a curve or thermocouple glitch) are excluded.

| Stage of life | Readings | CellSense total-capacity error (mAh) | Reading the 4C curve's end (mAh) |
|---|---|---|---|
| New to 20% of life | 690 | 5.8 | 19.2 |
| 20 to 60% | 1,518 | 7.3 | 35.3 |
| 60 to 90% | 1,074 | 15.2 | 53.1 |
| Last 10% and beyond | 489 | 37.2 | 82.4 |
| All readings | 3,771 | 13.2 | 43.5 |

With the early curve: capacity bias -7.8 mAh, correlation 0.978, inside 2 sigma 70%;
fade error 1.82 points, correlation 0.979; pulse resistance error 1.24 mOhm, correlation 0.76;
resistance growth correlation 0.83; past or not past end of life called right 94.6%. From the curve alone:
capacity error 17.0 mAh, fade correlation 0.973. Lost lithium and lost electrode material have no answer key.

## Final exam 2: cycle life from cycle 100

Severson et al.'s split (cells ordered as their loading notebook; primary test = even indices of batches 1-2 plus the
last, secondary = batch 3); calibration cells left out. Their Table 2, models trained on 41 real cells:

| Model | Primary RMSE (cycles) | Primary error | Secondary RMSE | Secondary error |
|---|---|---|---|---|
| Severson variance | 138 | 14.7% | 196 | 11.4% |
| Severson discharge | 91 | 13.0% | 173 | 8.6% |
| Severson full | 118 | 14.1% | 214 | 10.7% |
| **CellSense (no real cells)** | 145 | 22.4% | 249 | 13.1% |
| Guess the test cells' median | 415 | 33.7% | 326 | 19.2% |

The median row is a no-skill reference that cheats (it uses the test cells' own lives). Readings later in life:

| Read at | Readings | Life error | Cycles-left error |
|---|---|---|---|
| First quarter of life | 421 | 17.6% | 193 |
| Second quarter | 464 | 22.5% | 269 |
| Third quarter | 473 | 33.3% | 376 |
| Last quarter | 497 | 35.9% | 367 |

## Final exam 3: partial discharges

The exam readings of Final exam 1, cut short once they have delivered a given charge from full, everything below thrown
away, and read by the partial-discharge network. Total-capacity error (mAh), with the cell's full early curve as reference
and with that curve cut at the same depth; the no-model baseline adds the shift between the two curves at the cut to the
early total. Assuming no fade at all errs by 45.7 mAh; the full-curve network on full discharges, 13.2 mAh.

| Stopped after | Readings cut short | CellSense | Reference cut too | No-model baseline |
|---|---|---|---|---|
| 0.90 Ah (82% of rated) | 86% | 17.1 | 17.3 | 57.6 |
| 0.75 Ah (68% of rated) | 98% | 28.7 | 27.5 | 65.3 |
| 0.60 Ah (55% of rated) | 100% | 46.5 | 39.2 | 62.2 |
| 0.45 Ah (41% of rated) | 100% | 52.0 | 47.6 | 56.4 |
| 0.30 Ah (27% of rated) | 100% | 41.8 | 41.3 | 51.2 |

## Limits

- It knows one cell. Everything was calibrated on the A123 APR18650M1A at Severson's protocol: a 4C discharge to 2.0 V in a 30 °C chamber. Other chemistries (the nickel-rich cells in most long-range cars, for one), sizes, rates or temperatures are outside what it learned.
- Near end of life it reads total capacity low: real worn cells hide more slow charge behind the 4C curve than the simulator makes. Two attempts to model why (growing slow polarization, then poorly connected graphite) improved the simulated exam but made the calibration cells worse, and the ratchet threw both out. Typical capacity error is 13 mAh overall and largest in the last tenth of life.
- Lost lithium, lost graphite and lost LFP have no answer key. The cells were never opened, so these are the physics model's explanation of each curve. Lost LFP barely changes these curves at all, and the error bars say so.
- The lifetime forecast is useful from cycle 100 but not yet Severson's equal: 22% error on their primary test cells and 13% on the secondary ones, against 9 to 15% for models trained on 41 real cells. How strongly early slow graphite predicts the knee was set on six cells. Late in life it still reads worn cells as healthier than they are and overestimates the cycles left (36% life error in the last quarter).
- The aging rates were calibrated on six real cells' trajectories, all charged in about 10 minutes. Charging recipes much faster or slower than that rely on the physics alone.
- Pulse resistance follows the real cells (correlation 0.76) with about 1.2 mΩ error. The model's 30 ms pulse is ohmic plus charge transfer with no double layer, so absolute values carry the calibration's error.
- Partial discharges from full read well down to about 0.75 Ah (29 mAh capacity error) and no better than assuming no fade at 0.6 Ah and below. Discharges that start partway (cycling between 20% and 80%, say), other cut-off voltages and other rates are untested.
