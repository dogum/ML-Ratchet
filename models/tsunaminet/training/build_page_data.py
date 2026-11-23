"""Page text and tables (web/page_text.json) from the exam, ratchet and checks."""
import os, json, pickle, base64, numpy as np
from _paths import ROOT, DATA
W = os.path.join(ROOT, 'web')
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()
rows = json.load(open(os.path.join(ROOT, 'exam', 'exam_rows.json'))); INP = pickle.load(open(os.path.join(ROOT, 'exam', 'exam_inputs.pkl'), 'rb'))
S = json.load(open(os.path.join(ROOT, 'exam', 'exam_summary.json'))); A = S['all']
short = {5413: 'Tohoku 2011', 4682: 'Maule 2010', 5557: 'Iquique 2014', 5590: 'Illapel 2015', 6044: 'Kamchatka 2025'}
EV = []
for r in sorted(rows, key=lambda r: r['name']):
    if 'net' not in r: continue
    I = INP[str(r['id'])]
    ins = []
    for k, site in enumerate(I['inputs']):
        ins.append(dict(site=site, ta=int(I['input_ta'][k]), scale=float(10 ** (2 * I['S'][0, k, 1])), s=[float(v) for v in I['S'][0, k]], x=b64(I['X'][0, k].ravel())))
    nm = r['name'][11:].replace(', Russia', '').replace(' Island', '').replace(' Islands', '').replace('Off Coast', 'Ecuador').replace(', Kermadec', '')
    EV.append(dict(id=r['id'], short=short.get(r['id'], f"{nm[:16].strip()} {r['name'][:4]}"), name=r['name'], mw_true=r['mw_true'], interface=r['interface'], sc_true=r['sc_true'],
                   cols_s=I['cols_s'], inputs=ins, nnls=dict(mw=r['nnls']['mw'], sc=r['nnls']['sc']),
                   targets=[dict(site=t['site'], ta=t['ta'], obs=t['obs'], net=t['net'], lo=t['net68'][0], hi=t['net68'][1], nnls=t['nnls'], seismic=t['seismic'], ffm=t['ffm_twin']) for t in r['targets']]))
order = [5413, 6044, 4682, 5590, 5557]
EV.sort(key=lambda e: (order.index(e['id']) if e['id'] in order else 9, -e['mw_true']))
f = lambda v, d=2: f'{v:.{d}f}'
def cell(v, best=False): return dict(v=v, cls='best') if best else v
L = json.load(open(os.path.join(ROOT, 'ratchet.json')))
D = dict(
  stats=[dict(v=f(A['mw_mae_net'][0]), l='Mw error on 28 real tsunamis (SIFT-style inversion 0.25, median guess 0.30)'),
         dict(v=f"{A['amp_f2_net'][0] * 100:.0f}%", l='of 119 held-out buoy forecasts within a factor 2 (inversion 71 %, seismic 56 %)'),
         dict(v=f"{A['centroid_km_median_net']:.0f} km", l='median centroid miss vs USGS finite fault (inversion 82 km)'),
         dict(v='0', l='real tsunamis seen in training: 40,000 simulated ones in real DART noise')],
  heroNote='Research model, not a warning system. Built on NOAA SIFT unit sources, ETOPO 2022 and NDBC DART records.',
  exam=dict(title='The exam: 28 real tsunamis, run once',
    text=('<p>Every Pacific tsunami from 2006 to 2025 caused by an earthquake of magnitude 7.5 or more near a subduction zone, '
          'that NOAA\'s tsunami database lists with at least one deep-ocean reading. Five were set aside as the dev set by a seeded draw; '
          'the other 28 are the exam, named in advance. Truth is the USGS W-phase magnitude and the centre of the USGS finite-fault slip. '
          'Forecasts are graded at every other buoy with a clear signal (119 forecasts in 15 events). Brackets are 95 % bootstrap ranges over events.</p>'
          '<p>TsunamiNet wins the pooled rows and loses two honestly: weighted event by event, a constant 2.8 cm beats every method on amplitude, '
          'because most events are small with one or two buoys; and the centroid along the trench only ties the epicentre\'s own column. '
          f"It reads magnitudes low on average ({A['mw_bias_net']:+.2f}); Tohoku comes out at 8.67 against 9.1.</p>"),
    head=['Metric', 'TsunamiNet', 'SIFT-style NNLS', 'NNLS, calibrated GFs', 'Seismic Mw scaling', 'Dumb', 'Verdict'],
    rows=[['Mw mean |error|', cell(f"{f(A['mw_mae_net'][0])} [{f(A['mw_mae_net'][1])}, {f(A['mw_mae_net'][2])}]", True), f(A['mw_mae_nnls'][0]), f(A['mw_mae_nnls_p1'][0]), 'is the truth', f(A['mw_mae_dumb'][0]), '<span class="pill win">win</span>'],
          ['Centroid miss, median (km)', cell(f"{A['centroid_km_median_net']:.0f}", True), f"{A['centroid_km_median_nnls']:.0f}", f"{A['centroid_km_median_nnls_p1']:.0f}", '–', f"{A['centroid_km_median_dumb']:.0f}", '<span class="pill win">win</span>'],
          ['Centroid along the trench, mean (km)', f"{A['sc_km_mae_net'][0]:.0f} [{A['sc_km_mae_net'][1]:.0f}, {A['sc_km_mae_net'][2]:.0f}]", f"{A['sc_km_mae_nnls'][0]:.0f}", f"{A['sc_km_mae_nnls_p1'][0]:.0f}", '–', cell(f"{A['sc_km_mae_dumb'][0]:.0f} [{A['sc_km_mae_dumb'][1]:.0f}, {A['sc_km_mae_dumb'][2]:.0f}]", True), '<span class="pill loss">loss</span>'],
          ['Buoy amplitude, mean |log10 error|, 119 forecasts', cell(f"{f(A['amp_mae_net'][0])} [{f(A['amp_mae_net'][1])}, {f(A['amp_mae_net'][2])}]", True), f(A['amp_mae_nnls'][0]), f(A['amp_mae_nnls_p1'][0]), f(A['amp_mae_seismic'][0]), f(A['amp_mae_dumb'][0]), '<span class="pill win">win</span>'],
          ['Buoy amplitude within factor 2', cell(f"{A['amp_f2_net'][0] * 100:.0f} %", True), f"{A['amp_f2_nnls'][0] * 100:.0f} %", f"{A['amp_f2_nnls_p1'][0] * 100:.0f} %", f"{A['amp_f2_seismic'][0] * 100:.0f} %", f"{A['amp_f2_dumb'][0] * 100:.0f} %", '<span class="pill win">win</span>'],
          ['Buoy amplitude, each event once (15)', f(A['amp_mae_evw_net']), f(A['amp_mae_evw_nnls']), f(A['amp_mae_evw_nnls_p1']), f(A['amp_mae_evw_seismic']), cell(f(A['amp_mae_evw_dumb']), True), '<span class="pill loss">loss</span>'],
          ['Coverage of 68 % ranges (dev-widened)', f"Mw {A['mw_cov68']:.2f} · centroid {A['sc_cov68']:.2f} · amplitude {A['amp_cov68_net']:.2f}", '', '', '', '', '<span class="pill mixed">info</span>']],
    ehead=['Event', 'USGS Mw', 'TsunamiNet Mw (68 %)', 'NNLS Mw', 'Centroid miss (km)', 'Buoys read', 'Forecasts'],
    erows=[[r['name'] + ('' if r['interface'] else ' <span class="pill mixed">not interface</span>'), f(r['mw_true']), f"{f(r['net']['mw'])} [{f(r['net']['mw68'][0])}, {f(r['net']['mw68'][1])}]", f(r['nnls']['mw']),
            f"{r['net'].get('centroid_err_km', float('nan')):.0f}", ', '.join(s.split('_')[0] for s in r['inputs']), str(len(r['targets']))] for r in sorted(rows, key=lambda r: r['name']) if 'net' in r]),
  ratchet=dict(text=('<p>A rung stays only if it beats the incumbent on a frozen set of 3,000 simulations by more than 0.01 (mean skill of magnitude and centroid) '
                     'and does not get more than 5 % worse on the five real dev events. When the physics changed (p0 to p1), a new frozen set was built and r1 re-scored on it. '
                     'Two of five rungs were rejected.</p>'),
               head=['Rung', 'Change', 'Frozen score', 'Sim Mw MAE', 'Dev Mw MAE', 'Dev centroid (km)', 'Dev amplitude (dex)', 'Decision'],
               rows=[['r1', 'Textbook physics p0: linear shallow water, instantaneous rupture', '0.327 (0.438 on p0 set)', '0.155', '0.21', '91', '0.40', '<span class="pill kept">kept</span>'],
                     ['r2', 'Physics p1 from the twin test: arrivals 1.65 % late, waves ×1.6, rupture at 2.5 km/s', '0.663', '0.092', '0.21', '79', '0.24', '<span class="pill kept">kept</span>'],
                     ['r4', 'Mixed p0 + p1 simulations', '0.641', '0.104', '0.28', '54', '0.28', '<span class="pill rejected">rejected</span>'],
                     ['r3', 'Physics anchor: the SIFT-style inversion on the same windows feeds the flow', '0.718', '0.091', '0.23', '57', '0.26', '<span class="pill released">released</span>'],
                     ['r5', 'r3 polished at a lower learning rate', '0.724', '0.090', '0.22', '59', '0.25', '<span class="pill rejected">rejected</span>']],
               twin=('<p><b>The twin test.</b> Each dev event was re-simulated from its USGS finite-fault slip and read at every buoy that recorded it. '
                     'Real first waves arrived later than the linear model by 1.65 % of the travel time (about 3 minutes after 3 hours), close to the 1–2 % '
                     'that elastic loading and seawater compressibility cause, and were 1.6 times larger. Building both into the simulator (p1) doubled the frozen-set score '
                     'and cut the dev forecast error from 0.40 to 0.24 dex. Before that, the twin test also caught a detiding problem: tide-fit errors of several centimetres '
                     'looked like waves, so records now pass a causal 3-hour high-pass, applied identically to every Green\'s function.</p>')),
  flow=[['Okada sources', '1,317 NOAA SIFT unit sources, 100 × 50 km, 1 m of thrust each'], ['Shallow water', 'Linear equations on ETOPO 2022 at 8′, one reciprocal run per buoy'],
        ['Real noise', '955 quiet DART windows with their real gaps and glitches'], ['Flow', 'CNN per buoy, pooled, plus the NNLS anchor, into a RealNVP flow'],
        ['Forecast', 'Posterior samples summed through the Green\'s functions at other buoys']],
  checks=['Okada vs GeoClaw\'s independent code: 4 × 10⁻¹³; vs NOAA MOST\'s own deformation for unit source acsza1: r = 0.995, peak 0.352 vs 0.349 m.',
          '1-D flat basin: speed 198.5 vs √(gh) = 198.1 m/s.', 'Green\'s law: ×1.416 vs 1.414 at 1,000 m; ×1.868 vs 1.943 at 281 m.',
          '2-D basin vs the exact Hankel solution: front 607 km in both, peak −2.1 %.', 'Reciprocity on the real grid: 2.4 × 10⁻⁶.',
          '8′ vs NOAA\'s 4′: peak ratio 0.97–0.99; first-wave correlation 0.98 near, 0.73–0.81 beyond 3,000 km.',
          'numpy vs PyTorch: 2.2 × 10⁻⁶. This page\'s JavaScript vs numpy: see the footer.'],
  limits=['Reads magnitudes low on average (−0.22); Tohoku 8.67 against 9.1. Ruptures longer than the 9 columns it reads are under-read.',
          'Thrust slip on the two shallowest unit-source rows only: outer-rise, strike-slip and intraslab events are read as if they were thrusts (Mw coverage 44 % on those).',
          '8′ linear propagation, coarser than NOAA\'s 4′; no dispersion or Coriolis.',
          'Five dev events set the widening factors (×2–4.4); the centroid ranges came out too wide (89 % at nominal 68 %).',
          'Green\'s functions for 68 DART sites; no coastal tide gauges.',
          'NOAA\'s own propagation database could not be compared waveform by waveform: its compressed format is undocumented and its documented OPeNDAP access returns 404.'],
  use=dict(text='The release runs on numpy and scipy. <code>demo.py</code> reads four exam events (a seeded draw, one per stratum) in about two seconds.',
           code='pip install numpy scipy\npython demo.py\n\nimport tsunaminet as T\nnet = T.TsunamiNet(); gf = T.GreensFunctions(); gf.cal = net.cal\nev = gf.event(lon, lat, records)   # {site: (eta, mask)} minutes from origin\npost = net.posterior(ev)            # Mw, centroid, ranges, samples\nfc = gf.forecast(ev, post, site, a, b, mask, widen=net.cal["widen"]["amp"])'),
  credits=['DART bottom-pressure records and station history: NOAA National Data Buoy Center.', 'Tsunami events and deep-ocean readings: NCEI/WDS Global Historical Tsunami Database.',
           'Magnitudes, W-phase moment tensors and finite-fault models: USGS ComCat.', 'Bathymetry: ETOPO 2022, NOAA NCEI.', 'Unit sources and MOST deformation: NOAA Center for Tsunami Research (SIFT).',
           'Okada check against GeoClaw (BSD-3). Flow pattern from ChirpNet (ML Ratchet).'],
  foot='TsunamiNet v1.0 · ML Ratchet · trained on physics, graded on reality · research only.',
  events=EV)
json.dump(D, open(os.path.join(W, 'page_text.json'), 'w'))
print('events', len(EV), 'bytes', os.path.getsize(os.path.join(W, 'page_text.json')))
