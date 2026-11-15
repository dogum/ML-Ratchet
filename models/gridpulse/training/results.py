"""results.json for the page: exam, ratchet ladder, method, limits, credits."""
from common import *
X = json.load(open(f'{ROOT}/exam.json')); R = json.load(open(f'{ROOT}/ratchet.json'))
G = X['GB']; I = G['inertia']; E = X['ERCOT']
short = {'r1': 'Baseline MLP on the 1 s window (p1 physics, 1 s-mean measurement)',
         'r2': 'Measurement: NESO 1 s values as instantaneous samples (dev check favoured it)',
         'r3': 'Add transmission demand as a conditioned input',
         'r4': 'Twice the simulated data (48k GridSim states)',
         'r5': 'Wider network (512-512-256), 120 epochs',
         'r6': 'Physics p2: 30 % of trips ramp out over 0.5-3 s'}
lad = [dict(rung=r['rung'], note=short[r['rung']], objective=r['objective'], skills=r['skills'], dev_rel=r.get('dev_rel'),
            accepted=r['accepted'], why=r['why']) for r in R]
verdict = (f"<b>Verdict.</b> On inertia GridPulse beats the constant guess ({I['gridpulse_N60']['MdAPE']:.1f} % vs "
           f"{I['baseline_constant_230']['MdAPE']:.1f} %) but not a simple demand-scaling rule fitted on the four dev events "
           f"({I['baseline_k_x_demand']['MdAPE']:.1f} %); without the demand input the trace alone gives {I['trace_only_N60']['MdAPE']:.1f} %, "
           f"no better than the constant. It overestimates 2023-2025 inertia by about 30 % (it learned the 2018-2022 inertia per MW of demand). "
           f"MW lost is marginally better than the RoCoF formula on medium labels ({G['loss']['medium']['gridpulse_N60']:.1f} % vs "
           f"{G['loss']['medium']['rocof_assumed_E230']:.1f} %) but its error bars are too narrow (48 % inside 1σ). The early-warning forecasts "
           f"only help at 5 s; by 10 s, simply holding the lowest value seen beats the nadir forecast, and holding the current value beats the "
           f"60 s forecast. On ERCOT the sim-trained summary model reads MW lost from the reported drop better than ERCOT's own MW/Hz constant "
           f"({E['loss']['MdAPE']:.1f} % vs {E['loss']['baseline_k_delta_f']:.1f} %), most clearly in 2023-2025, while GridSim's forward drop "
           f"forecast is {E['dfnad']['bias_med_pct']:.0f} % too deep on the exam years.")
how = [
 ('Physics', '<p>GridSim (Phase 1) solves the aggregated swing equation with per-technology governors (deadband, lag, rate limit, headroom, reheat / CCGT / hydro lead-lags), load damping, fast frequency response and under-frequency load shedding. Checked by Phase 1 against closed forms (RoCoF to 4e-14, nadir to 0.03 mHz) and ANDES (nadir 24 mHz off on a 200 MW step). GridPulse wraps <code>generate_states</code>: each random fleet is rescaled to a real GB or ERCOT operating point and given that grid\'s services. Initial RoCoF of the rescaled states matches ΔP·f0/2E to 0.1 %.</p>'),
 ('Realistic systems', '<p>GB demand, inertia and wind/solar share are drawn together from NESO half-hours of the training-input years 2018, 2020 and 2022 (70 %) or from wide ranges (30 %). ERCOT load comes from EIA-930 in the same years, inertia ≈ 5 s × load. Losses 150-2,000 MW (GB) and 250-2,800 MW (ERCOT). GB response: governors 300-1,500 MW plus, in 60 % of states, Dynamic Containment built from NESO\'s service curve (nothing until 0.2 Hz, all of it at 0.5 Hz), Moderation and Regulation. ERCOT: load resources tripping at 59.7 Hz, FFR at 59.85 Hz, battery droop in half the states.</p>'),
 ('Dev calibration', '<p>Prior-predictive twins of the 4 GB dev events (each event\'s own NESO inertia, demand and loss, everything else drawn): with generic services the real nadirs sat at the 2-6th percentile (the sims were too shallow); trimming 2019-era response put them at the 13-20th. ERCOT twins were 8-29 % too shallow with Phase 1\'s effective damping of 2.5 %/%; 1.0 %/% fits these random states. The 3 Dec 2019 dev label (1,788 MW) was replaced by Elexon metered output (854 MW) after the twins confirmed it was about twice too large. A dev check of the first-second drop favoured instantaneous 1 s samples over 1 s means, but the rung trained that way was not better.</p>'),
 ('Noise and network', '<p>Every simulated trip is sampled at 1 s with a random sub-second phase and added to a real quiet 140 s slice of NESO 1 s frequency (56,000 slices from 2018-07, 2020-07, 2022-01, at least 1 h from any event), then cut with the same onset rule griddata used on the real events and kept only if it would have been detected (≥0.15 Hz fall in 10 s for GB; ERCOT report criteria). A 0.44 M-parameter MLP with Gaussian heads reads the window −20…+60 s, masked after N s, plus the grid flag and demand. 28,201 detected events from 48,000 simulated states. ERCOT has no public traces, so a separate small sim-trained model reads its reported summary numbers.</p>'),
]
limits = [
 f"<b>Inertia mostly comes from demand, not from the trace.</b> Trace-only inertia error is {I['trace_only_N60']['MdAPE']:.0f} %, the same as guessing 230 GVA·s. One-second data blur the first-second RoCoF, and response arrives within 1-2 s, so the slope that carries inertia is barely sampled.",
 f"<b>It learned 2018-2022 GB.</b> In 2023-2025 inertia per MW of demand fell (NESO median 160-189 GVA·s); GridPulse still reads 216-244, so its exam-year error is 20-31 %.",
 "<b>Error bars are too narrow on real MW lost</b> (48 % of medium-label events inside 1σ, 79 % inside 2σ) and slightly narrow on inertia (63 % / 89 %). Four dev events are too few to widen them properly. Nadir and 60 s bands are about right (60-75 % / 87-95 %).",
 "<b>The 60 s forecast is worse than holding the current value.</b> Real GB frequency 30-60 s after a trip is shaped by secondary response, market actions and the next disturbance; GridSim models secondary control as a single ramp (Phase 1 limit 6).",
 f"<b>ERCOT exam years are stiffer than GridSim.</b> Forward drop forecasts are {E['dfnad']['bias_med_pct']:.0f} % too deep (2023-2025 have gigawatts of battery FFR and ECRS). No ERCOT 1 s data exist, so the lab's ERCOT noise is scaled GB noise.",
 "<b>MW/Hz, headroom and FFR are weakly identifiable</b> (sim skill 0.5-0.7 for MW/Hz, about 0.5 for headroom and FFR) and have no real truth to grade them on.",
 "<b>Not every detected GB event is a trip</b>: some are fast demand or interconnector ramps; GridPulse reads them as trips. Loss labels themselves are noisy (REMIT timing, capacity vs output).",
]
credits = ('<p><b>Data.</b> NESO system frequency (1 s), outturn inertia and historic generation mix (NESO Open Data Licence); Elexon BMRS REMIT and B1610 (loss labels, by GridData); '
           'ERCOT monthly Operations Reports (frequency events); EIA-930 hourly demand (public domain); National Grid ESO technical report on 9 August 2019. '
           'Simulator: GridSim (Phase 1 of this project; IEEE PES-TR1 governor models, Kundur 1994). Splits frozen by GridData (seed 20261113): dev = 4 GB + 5 ERCOT events, exam = 412 GB + 301 ERCOT events.</p>'
           '<p>Part of ML Ratchet, github.com/dogum/ML-Ratchet · "Trained on physics. Graded on reality."</p>')
note = ("The frozen set was rebuilt twice as the simulated measurement and then the loss shapes changed (p1 mean → p1 sample → p2 with ramped losses); every rung is re-scored here on the latest set, p2. "
        "r6 had the best simulated score but made inertia on the dev events worse (MdAPE 15 → 20 %), so the release is r5.")
json.dump(dict(exam=X, ratchet=lad, ratchet_note=note, verdict=verdict, how=how, limits=limits, credits=credits,
               n_train_states='48,000'), open(f'{ROOT}/web/results.json', 'w'), indent=0, default=float)
print('ok')
