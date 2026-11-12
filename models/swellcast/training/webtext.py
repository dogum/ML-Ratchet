"""Text and tables for the page (web/text.json) and markdown tables for the model card (logs/tables.md), all from the
result files (final.json, ratchet.json, physics_check.json, calib_v2_record.json)."""
import os, json, numpy as np
import evaluate as E
ROOT = E.ROOT
fin = json.load(open(os.path.join(ROOT, 'final.json'))); rat = json.load(open(os.path.join(ROOT, 'ratchet.json'))); phys = json.load(open(os.path.join(ROOT, 'physics_check.json')))
H = fin['exam_confirmed']; N, C, Dm = H['network'], H['classical'], H['dumb']; q = fin['quiet']; cd = fin['classical_det']
f0 = lambda x: '&ndash;' if x is None or not np.isfinite(x) else f'{x:.0f}'
pc = lambda x: '&ndash;' if x is None or not np.isfinite(x) else f'{100 * x:.0f}%'

def rows(key):
    h = fin[key]; n, c, d = h['network'], h['classical'], h['dumb']
    return [('SwellCast network', n, h.get('network', {}).get('detect_rate'), q['false_alarm']),
            ('Classical ridge fit (Munk)', c, h.get('classical_detect_rate'), cd['quiet_false_alarm']),
            ('Dumb: dev median, always', d, None, None)]

def table(key):
    hd = '<thead><tr><th>Method</th><th>Distance error, median</th><th>Within 20%</th><th>Storm time error, median</th><th>mean</th><th>Direction error, median</th><th>mean</th><th>Detected</th><th>False alarms, quiet 2025</th></tr></thead>'
    b = ''
    for name, m, det, fa in rows(key):
        b += f"<tr class=\"{'us' if name.startswith('Swell') else ''}\"><td>{name}</td><td>{f0(m['D_med_pct'])}%</td><td>{pc(m['D_within_20pct'])}</td><td>{f0(m['t0_med_h'])} h</td><td>{f0(m['t0_mae_h'])} h</td><td>{f0(m.get('dir_med_deg'))}&deg;</td><td>{f0(m.get('dir_mean_deg'))}&deg;</td><td>{pc(det) if det is not None else 'always'}</td><td>{pc(fa) if fa is not None else '100%'}</td></tr>"
    return hd + '<tbody>' + b + '</tbody>'

def md_table(key):
    s = '| Method | D error median | D within 20% | log10 D MAE | t0 error median | t0 MAE | dir error median | dir mean | detected | false alarms (quiet 2025) |\n|---|---|---|---|---|---|---|---|---|---|\n'
    for name, m, det, fa in rows(key):
        s += f"| {name} | {m['D_med_pct']:.0f}% | {100 * m['D_within_20pct']:.0f}% | {m['logD_mae']:.3f} | {m['t0_med_h']:.0f} h | {m['t0_mae_h']:.0f} h | {m.get('dir_med_deg', np.nan):.0f} deg | {m.get('dir_mean_deg', np.nan):.0f} deg | {'always' if det is None else f'{100 * det:.0f}%'} | {'100%' if fa is None else f'{100 * fa:.0f}%'} |\n"
    return s

def devrel(r): return f" (dev {100 * (r['dev_rel'] - 1):+.0f}%)"

def rat_rows():
    out = []
    for r in rat:
        out.append(dict(ck=r['ckpt'].replace('.pt', ''), note=r['note'], val=r.get('val_set'), obj=r['objective'], sk=r['val']['skill'], acc=r['accepted'], why=r['why'],
                        dev=r['dev'], exam=r['exam']))
    return out

def rat_html():
    s = '<thead><tr><th>Rung</th><th>What changed</th><th>Frozen set</th><th>Score</th><th>Dev: D / time / dir</th><th>Exam (not used): D / time / dir</th><th>Decision</th></tr></thead><tbody>'
    for r in rat_rows():
        s += (f"<tr><td>{r['ck']}</td><td style=\"white-space:normal;min-width:240px;text-align:left\">{r['note']}</td><td>{r['val']}</td><td>{r['obj']:.3f}</td>"
              f"<td>{r['dev']['D_med_pct']:.0f}% / {r['dev']['t0_med_h']:.0f} h / {r['dev'].get('dir_med_deg', np.nan):.0f}&deg;</td><td>{r['exam']['D_med_pct']:.0f}% / {r['exam']['t0_med_h']:.0f} h / {r['exam'].get('dir_med_deg', np.nan):.0f}&deg;</td>"
              f"<td><span class=\"pill {'k' if r['acc'] else 'r'}\">{'kept' if r['acc'] else 'rejected'}</span> <span class=\"small\" style=\"white-space:normal\">{r['why']}</span></td></tr>")
    return s + '</tbody>'

def rat_md():
    s = '| Rung | What changed | Frozen set | Objective | skills D/t0/dir/Hs/det | Dev D med / t0 med / dir med | Exam (not used) | Decision |\n|---|---|---|---|---|---|---|---|\n'
    for r in rat_rows():
        sk = r['sk']; s += (f"| {r['ck']} | {r['note']} | {r['val']} | {r['obj']:.3f} | {sk['D']:.2f}/{sk['t0']:.2f}/{sk['dir']:.2f}/{sk['hs']:.2f}/{sk.get('det', np.nan):.2f} | "
                            f"{r['dev']['D_med_pct']:.0f}% / {r['dev']['t0_med_h']:.0f} h / {r['dev'].get('dir_med_deg', np.nan):.0f} deg | {r['exam']['D_med_pct']:.0f}% / {r['exam']['t0_med_h']:.0f} h / {r['exam'].get('dir_med_deg', np.nan):.0f} deg | {'KEPT' if r['acc'] else 'rejected'}: {r['why']} |\n")
    return s

cov = N; ridge = phys['ridge']
txt = dict(
  stats=[[f"{H['network']['n']}", 'real hurricane and typhoon swells in the exam, never seen in training'],
         [f"{N['D_med_pct']:.0f}%", f"median distance error (ridge fit {C['D_med_pct']:.0f}%, median guess {Dm['D_med_pct']:.0f}%)"],
         [f"{N['dir_med_deg']:.0f}&deg;", f"median direction error (ridge fit {C['dir_med_deg']:.0f}&deg;, median guess {Dm['dir_med_deg']:.0f}&deg;)"],
         [f"{100 * N['detect_rate']:.0f}%", f"of exam swells detected, with {100 * q['false_alarm']:.0f}% alarms on quiet 2025 records"]],
  examText=(f"<p>The exam is every hurricane or typhoon (peak wind 64 kt or more, IBTrACS 2008&ndash;2025) that passed 1,000&ndash;8,000 km from one of 27 NOAA buoys over open water, "
            f"where the buoy record has a visible streak along the ridge the real storm predicts and the buoy's measured swell direction points back to within 45&deg; of the storm. That leaves "
            f"{N['n']} storm&ndash;buoy pairs from storms not in the dev set. The ten-day window is centred near the expected arrival with a random shift of up to &plusmn;36 h, the same rule used in training, "
            f"so every method gets the same hint about timing. Storm time is the time of peak wind in the best track; distance and direction are measured from that point.</p>"),
  examTable=table('exam_confirmed'),
  examMore=(f"<p class=\"note\">68% error bars, widened on the dev set only (factors D &times;{fin['sigma_factor']['logD']:.2f}, time &times;{fin['sigma_factor']['t0']:.2f}, direction &times;{fin['sigma_factor']['dir']:.2f}): on the exam the true distance falls inside the bar {100 * N['logD_in_1sig']:.0f}% of the time, the storm time {100 * N['t0_in_1sig']:.0f}%, the direction {100 * N.get('dir_in_1sig', np.nan):.0f}% (ideal: 68%).</p>"
            f"<h3 style=\"margin-top:22px\">The same scores on looser event sets</h3><div class=\"tablewrap\"><table class=\"exam\">{table('exam_True')}</table></div><p class=\"note\">All {fin['exam_True']['network']['n']} visible pairs (streak present, direction not checked): some of these streaks belong to a different storm.</p>"
            f"<div class=\"tablewrap\"><table class=\"exam\">{table('exam_None')}</table></div><p class=\"note\">All {fin['exam_None']['network']['n']} pairs, including storms the buoy never recorded (blocked by islands, aimed elsewhere, or too weak). Here no method can do well; the detection column is the honest number.</p>"),
  ratchetTable=rat_html(),
  steps=[['Physics first', f"Deep-water swell travels at group speed g/(4&pi;f). The simulator sends a JONSWAP sea from a moving, growing and decaying storm patch along great circles, spreads it over the sphere and removes energy by air friction (Ardhuin et al. 2009). Check: the simulated ridge slope matches g/(4&pi;D) to {100 * max(abs(r['rel_err']) for r in ridge):.1f}% from 1,500 to 9,000 km, and energy is conserved to {100 * abs(phys['energy_conservation']['received_over_emitted'] - 1):.2f}%."],
         ['Real noise', 'Every training window is a real ten-day NOAA buoy record (27 buoys, 2008&ndash;2025) from a quiet-ish period with no hurricane nearby; the simulated swell is added on top. Local wind sea, other swells, gaps and the instrument\'s quirks are all real.'],
         ['Network', 'A 300k-parameter 2-D convolutional network reads the spectrogram (frequency by time, five channels: energy, energy anomaly, the two directional moments, and a data mask). It outputs distance, storm time, storm wave height and direction, each with its own error bar, plus a detection score.'],
         ['Calibrate on dev', 'Eighteen dev storms (29 confirmed arrivals) were compared with simulations, aligned on the true ridge. The first simulator made swells too strong, too long-period and far too narrow in direction (r1 0.79 against 0.47 in real data). Version 2 fixed these and added crossing swells and the buoy\'s low-frequency cutoff.'],
         ['Twin test', 'Each dev storm was re-simulated from its real best track and injected into quiet records of the same buoy. The network&rsquo;s errors on these twins line up with its errors on the real records (correlation 0.60 for distance, 0.65 for time), so a large part of the error comes from how real hurricanes move and grow, not from the buoy data. That led to simulator v3, which draws 65% of its storms from real 1985&ndash;2007 hurricane tracks (never in dev or exam).'],
         ['Ratchet', 'Each change is scored on a frozen simulated set and checked on the dev arrivals; it is kept only if it helps both. ' + ' '.join(f"{r['ckpt'].replace('.pt', '')}: {'kept' if r['accepted'] else 'rejected'}{'' if r.get('dev_rel') is None else devrel(r)}." for r in rat) + ' Versions 2 and 3 of the simulator alone made the real dev arrivals worse; a mix of all three versions helped.'],
         ['Exam', 'Real storms never used to choose anything, against the classical ridge fit and a dumb guess, with error bars widened on dev only.']],
  limits=[['It does not beat a median guess on distance or time', f"Guessing the dev median ({10 ** json.load(open(os.path.join(ROOT, 'baseline.json')))['dumb_values']['logD']:,.0f} km, 48 h before the window centre) gives a {Dm['D_med_pct']:.0f}% median distance error and {Dm['t0_med_h']:.0f} h on time; the network gets {N['D_med_pct']:.0f}% and {N['t0_med_h']:.0f} h. Most exam pairs sit at similar distances (Mexican hurricanes to California buoys, Atlantic hurricanes to the US east coast), so the median is a strong guess. The network does beat the classical ridge fit on distance, time and detection, matches it on direction, and its error bars hold the truth about as often as they claim."],
          ['Truth is only for tropical cyclones', 'IBTrACS lists hurricanes and typhoons, so the exam has no Southern Ocean or North Pacific winter storms, which make most of the long-period swell at Hawaii and California. The network was trained on generic storms and has not been graded on those.'],
          ['The window hint matters', 'Every method gets a window centred within &plusmn;36 h of the arrival, which is why the dumb median guess is hard to beat on distance. In a real monitoring setting the window would slide, and detection would have to carry more of the load.'],
          ['One-point truth for a moving storm', 'A hurricane makes swell for days along a track hundreds of kilometres long. The truth is a single point (peak wind), so some of the measured error is the storm&rsquo;s own size and motion.'],
          ['Quiet is not empty', 'The false-alarm test uses quiet buoy windows with no nearby hurricane. A winter storm the catalogue does not list can still send real swell into those windows, so the false-alarm rate is an upper bound.']],
  usage=("unzip swellcast-v1.0.zip && cd swellcast-v1.0\npython3 demo.py            # raw NDBC files -> window -> answer, plus 5 exam windows\n\n"
         "import pipeline as PL, swellcast as SC\nrec = PL.read_ndbc('51004w2018.txt.gz', '51004d2018.txt.gz', '51004j2018.txt.gz')\n"
         "win, tc = PL.window(rec, PL.hour('2018-09-17T12'))   # 10 days centred on this hour\nprint(SC.SwellCast().read(win))\n"
         "# -> D_km, logD_sig, t0 (h from tc), t0_sig, dir (deg, from), kappa, hs, p_swell, detected"),
  credits=("Data: NOAA National Data Buoy Center spectral wave data (swden, swdir, swr1; public domain). Storm truth: IBTrACS v04r01, Knapp et al. 2010, NOAA NCEI (public domain). Coastlines: Natural Earth via world-atlas (public domain). "
           "Physics: Munk et al. 1963 and Snodgrass et al. 1966 (swell from distant storms), Hasselmann et al. 1973 (JONSWAP), Ardhuin et al. 2009 (swell dissipation). "
           "Built for Greg&rsquo;s ML Ratchet collection; code and weights in swellcast-v1.0.zip."))
json.dump(txt, open(os.path.join(ROOT, 'web/text.json'), 'w'))
open(os.path.join(ROOT, 'logs/tables.md'), 'w').write('## exam confirmed\n' + md_table('exam_confirmed') + '\n## exam visible\n' + md_table('exam_True') + '\n## exam all\n' + md_table('exam_None') + '\n## dev confirmed\n' + md_table('dev_confirmed') + '\n## ratchet\n' + rat_md())
print(open(os.path.join(ROOT, 'logs/tables.md')).read())
