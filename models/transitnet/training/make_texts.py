"""Page and model-card text for TransitNet, written from the released rung's numbers."""
import _paths as _P  # portable paths (see _paths.py)
import json, numpy as np
D = _P.WORK + '/'; R = json.load(open(D + 'ratchet.json')); rel = [r for r in R if r['accepted']][-1]; ex = rel['exam']; PA = ex['params_all']; BS = ex['params_by_snr']
PH = json.load(open(D + 'physics_check.json')); BL = json.load(open(D + 'data/baseline_fits.json')); Mt = json.load(open(D + 'data/real_meta.json')); S = json.load(open(D + 'data/split.json'))
p = lambda x, d=0: f"{100 * x:.{d}f}%"
n_exam = len([i for i in S['exam'] if Mt[i]['info']['snr'] >= 7]); hi = BS['100-1000000000.0']; mid = BS['20-100']
T = {}
T['version'] = 'v1.0 · rung %d' % (R.index(rel) + 1)
T['lede'] = (f"TransitNet never saw a labelled planet. It learned from 60,000 light curves of ordinary stars recorded by NASA's TESS satellite, with transiting planets, eclipsing binary stars and "
             f"background binaries added by a physics model of a star's limb-darkened disk being covered. Then it read {n_exam:,} real signals: {ex['n_planets']:,} confirmed planets and {ex['n_fp']:,} candidates "
             f"that follow-up observations showed were not planets. On clear transits it measures a planet's size to {p(hi['k_frac']['median'], 1)}. It ranks a real planet above a real false positive {p(ex['auc'])} of the time, "
             f"but on its own it would still pass {p(ex['fp_called_planet'])} of those false positives as planets.")
T['heroStats'] = [[p(hi['k_frac']['median'], 1), f"typical error in planet size (Rp/R*) on clear transits (SNR 100+), against {p(hi['k_frac_naive']['median'], 1)} from the dip's depth alone"],
                  [f"{ex['auc']:.2f}", 'chance that a real planet scores higher than a real false positive (1 is perfect, 0.5 is a coin)'],
                  [p(PA['k_in_1sigma']), 'of published radius ratios inside its own 68% error bar (with the published error added)'],
                  [f"{ex['n_planets']:,}", 'confirmed planets in the exam, none used for training or choosing']]
T['starsSub'] = ("Each star's TESS light curve goes through the same pipeline as the training data: the star's slow variability removed, the catalogue ephemeris checked sector by sector, every transit folded on top "
                 "of the others. TransitNet reads the folded transit, the half-orbit point, odd against even transits and the motion of the light's centre, and returns what it thinks the dip is and, for a planet, its size, "
                 "tilt and orbit. Gold: the published values, from papers that used many more telescopes and hours of fitting.")
T['labSub'] = ("Pick one of four ordinary stars TESS watched for a month, and make something cross it. The physics model draws the dip into the star's real noise, the same folding as above runs here in your "
               "browser, and TransitNet reads it. It is never told what you made.")
T['ratchetSub'] = ("Every candidate network was scored on a frozen set of 5,000 simulated readings: half the score for telling planets from everything else (2 AUC - 1), half for the four planet parameters (skill against "
                   "guessing the median, floored at 0). It was kept only if that rose by more than 0.01 and it did no worse on 40 real dev stars that no exam uses: 20 confirmed planets and 20 false positives "
                   "(log-loss within 0.05, radius-ratio error within 1 point). When the dev planets showed the simulated limb darkening was wrong, the frozen set was regenerated and every rung re-scored. The exam is reported, never used to choose.")
T['exam'] = {'sub': (f"Every confirmed transiting planet with an orbit under 25 days that TESS watched at 2-minute cadence and saw at signal-to-noise 7 or more, and every TESS Object of Interest that follow-up "
                     f"proved a false positive. The 40 dev stars are left out. Published answers come from the NASA Exoplanet Archive's composite table; they were fitted with more data than one TESS light curve."),
             'rocNote': (f"A false positive is a candidate that passed TESS's own vetting and was ruled out later, often with telescopes on the ground. Many look exactly like planets in TESS photometry: "
                         f"a dim eclipsing binary in the same pixels, or a small star orbiting a big one. The AUC is {ex['auc']:.2f}."),
             'parNote': (f"Radius ratio Rp/R*, TransitNet against published, for {PA['k_frac']['n']:,} planets. Median errors by signal-to-noise against a least-squares transit fit to the same folded data "
                         f"(fixed limb darkening, no error bars, {BL['sec_per_fit']:.1f} s per star) and the square root of the depth.")}
T['how'] = [["The physics", f"A star's disk darkens toward its edge (quadratic limb darkening). A planet or a second star covers part of it; the light lost is integrated exactly over the covered area. Checked against batman (Kreidberg 2015): within {max(r['max_err_ppm'] for r in PH['transit']):.1f} ppm on seven test transits, and against pixel-by-pixel integration for one star eclipsing a bigger one. Orbits are Keplerian, eccentric where long."],
            ["The data", "2,200 ordinary stars TESS watched at 2-minute cadence, ten sectors spread over 2018-2023, give the real noise: spots, flares, jitter, gaps. Into them go simulated planets (radius ratio 0.01-0.3), eclipsing binaries, background binaries diluted 3 to 3,000 times, or nothing. 1 to 3 sectors per reading. Background binaries also move the light's centre, the way a dimming neighbour does."],
            ["The pipeline", "Exactly the same for simulated and real stars: running-median detrending that skips the transits, a per-sector search for the transit times (catalogue periods drift), a fine ephemeris refit, then five views: the transit, the half-orbit point, the whole orbit, odd and even transits, and the centroid. The network reads them twice, forwards and mirrored, and averages."]]
T['use'] = {'sub': "Give it a TESS light curve (one or more sectors), a candidate's period, epoch and duration, and the star's TIC values if you have them. numpy and scipy only; astropy if you read FITS files.",
            'code': "import transitnet as tn\nnet = tn.TransitNet()\nsector, star = tn.load_spoc('tess2019...-s0012-..._lc.fits')        # or your own dict of t, f\nr = net.read([sector], period=5.6606, epoch=2458463.08 - 2457000, duration_hours=1.68, star=star)\nr['p_planet'], r['k'], r['k_lo'], r['k_hi'], r['Rp_earth'], r['b'], r['aR']",
            'files': [['transitnet.py', 'the network in numpy, the reader and a FITS loader'], ['pipeline.py', 'detrending, ephemeris alignment, folding (the same file that made the training data)'],
                      ['transitnet_weights.npz', 'the weights, BatchNorm folded in'], ['demo.py', 'reads three real stars from examples/'], ['MODEL_CARD.md', 'the physics checks, the data, every rung, the exam and the limits'],
                      ['training/', 'every script: simulator, pipeline, training, ratchet, exam']]}
fpc = rel['exam']['fp_called_planet']
T['limits'] = [f"It cannot rule out most false positives from TESS photometry alone: it calls {p(fpc)} of the exam's false positives planets. Many need ground-based images, spectra or radial velocities; TransitNet's job is to rank them, and its score is best used that way.",
               "It knows TESS 2-minute light curves of single stars, orbits under 25 days, and dips the pipeline can fold. Single transits, very long periods, 20-second or 30-minute data, and circumbinary planets are outside what it learned.",
               f"Limb darkening is a broad prior, set from the dev planets' own fits, not from stellar atmosphere tables (the Claret tables could not be downloaded here). Its error bars include that; very hot or very cool stars may be biased.",
               "Published answers are not ground truth either: they come from different papers, bandpasses and dilution corrections. Part of the scatter in the exam is theirs.",
               f"Impact parameter is the hardest number: {PA['b_abs']['median']:.2f} typical error, worse near grazing. a/R* assumes the orbit's shape is unknown to it; eccentric orbits shift it.",
               "Stellar radius comes from the TIC catalogue. Planet sizes in Earth radii carry its error, which TransitNet does not see."]
T['foot'] = "Data: NASA TESS mission (SPOC light curves via MAST), NASA Exoplanet Archive (pscomppars), ExoFOP TESS TOI list. Physics checked against batman (Kreidberg 2015). Built with the ratchet method; trained on zero real planets."
json.dump(T, open(D + 'web/texts.json', 'w'), indent=1); print(json.dumps(T['heroStats'], indent=1)); print(T['lede'])
