"""Inject web/page_data.json + page texts into web/template.html -> web/firesize.html (after build_release.py)."""
import json, re
import numpy as np
from common import ROOT

WEB = f'{ROOT}/web'


def clean(n):
    n = re.sub(r'[^A-Za-z0-9 ]+$', '', re.sub(r'^Fy\d{4}-', '', str(n))).strip()
    return n if re.search('[A-Za-z]', n) else f'Fire {n}'.strip()


def main():
    D = json.load(open(f'{WEB}/page_data.json'))
    for e in D['examples']:
        e['name'] = clean(e['name']); e['label'] = f"{e['name'][:14]} ({e['label'][-5:-1]})"
    a = D['exam']['periods']['all']; b = D['exam']['baselines']['all']; p = D['post']['bootstrap_auc_diff']
    rc = D['exam']['periods']['recent_2020']
    D['resp_med_h'] = float(np.exp(-2.302585))
    D['exam_text'] = (
        f"<p>The exam script ran once, on {D['exam']['n']:,} FPA-FOD ignitions from 2012 to 2020. It used the model frozen by the "
        f"ratchet, and nothing was changed afterwards. The baselines are B's, on the same records. The GBM is the strongest of them, "
        f"and it was fitted on the real labels of the 8,913 dev ignitions.</p>"
        f"<p>FireSize ranks escapes better than climatology (AUC {a['escape']['auc']:.3f} vs {b['clim']['escape']['auc']:.3f}, "
        f"difference {p['firesize-clim']['mean']:+.3f}, 95 % bootstrap {p['firesize-clim']['lo']:+.3f} to {p['firesize-clim']['hi']:+.3f}) "
        f"and better than the fire-danger indices. It is worse than the GBM ({p['firesize-gbm']['mean']:+.3f}). "
        f"On fires that reach 1,000 acres it beats every baseline (AUC {a['large']['auc']:.3f} vs GBM {b['gbm']['large']['auc']:.3f}). "
        f"In the held-out 2020 season its AUC is {rc['escape']['auc']:.3f}.</p>"
        f"<p>Two things are poor. The probabilities are about twice too high: mean predicted chance {a['mean_p_escape']:.3f} against "
        f"a real rate of {a['escape']['prevalence']:.3f}, so the Brier score is worse than predicting the prevalence for everyone. "
        f"The size point estimate is worse than climatology ({a['size']['mae_log10']:.3f} vs {b['clim']['size']['mae_log10']:.3f}). "
        f"The 80 % size range contains the real size class {100 * a['cover80']:.0f} % of the time.</p>")
    D['ratchet_text'] = (
        "<p>The two biggest steps were physical, not statistical. In r2, agricultural fuel was allowed to burn as short grass; FireSim "
        "treats it as non-burnable, yet thousands of FPA-FOD fires start in fields. In r5, initial attack became two modes: most fires "
        "meet a fast, strong attack, while a share are only confined or monitored. Regional tweaks (r3, r3b) improved the simulator's "
        "fit to dev sizes but lost escape ranking on the dev gate, so the ratchet refused them. "
        "The incumbent r7 then got one widen-only temperature (T = 1.46) fitted on the dev calibration half.</p>"
        "<p>A twin test on 911 dev ignitions shows where the ceiling is. The network predicts what the simulator does with the "
        "same ignition at AUC 0.91, but the simulator itself predicts the real outcome at only 0.67. The gap is in the physics and the "
        "suppression model, not the network. The same test shows that the widening doubled the predicted escape rate (0.025 to 0.050 "
        "against a real 0.026). That was visible on dev before the exam, and it is the first thing to fix.</p>")
    D['built_text'] = (
        "<ul class='plain'>"
        "<li><b>Ignitions.</b> 2,400 real 18 km landscape windows near dev ignitions, 8 ignitions each, with cause, agency, hour and "
        "date borrowed from a dev ignition within 150 km. Every window and weather period passed the exam guard, and 304 windows are "
        "frozen for validation.</li>"
        "<li><b>Free burn.</b> FireSim on 90 m LANDFIRE: Rothermel surface fire, crown fire, minimum-travel-time spread, hourly "
        "weather from real gridMET, 96 hours. Fires smaller than one cell are tracked as an analytic ellipse.</li>"
        "<li><b>Suppression.</b> A containment race, calibrated on 911 dev twins (real dev ignitions simulated in their own "
        "landscape and weather) by the likelihood of each one's real size class. About 30 parameters; held-out twin log loss "
        "fell from 3.6 to 2.7.</li>"
        "<li><b>Network.</b> Three small MLPs (75 inputs, 2 × 128 hidden, 12 size classes) trained on the simulated size "
        "distributions of 8,500 ignitions, on one CPU thread in minutes. The network you are running here is that one, "
        "with numpy-equal weights.</li>"
        "<li><b>Inputs.</b> Fuel mix, canopy and slope from 250 m LANDFIRE around the point; two days of gridMET; "
        "Rothermel spread rate of the local fuels under the afternoon weather; cause, agency, hour, season, region.</li></ul>")
    D['limits_text'] = (
        "<ul class='plain'>"
        "<li><b>The Southeast</b> (AUC 0.57). Agricultural and debris burning, small parcels and dense, fast suppression are "
        "outside what the simulator knows.</li>"
        "<li><b>Probabilities are too high.</b> The widening fitted to size classes raised the escape probability further. "
        "A recalibration aimed at escape would fix the Brier score, but the exam can't be used for that.</li>"
        "<li><b>Size of small fires.</b> Reported sizes cluster on round numbers (0.1, 1, 0.01 acres). Climatology predicts those "
        "values; a smooth median doesn't.</li>"
        "<li><b>The suppression model is effective, not measured.</b> Several calibrated values sit at their limits "
        "(6-minute response, line rates 30 times the handbook values, half of non-federal fires only confined).</li>"
        "<li><b>Missing physics.</b> No spotting, no re-burn after containment, LANDFIRE 2016 fuels for 2017–2020 fires.</li></ul>")
    D['credits_text'] = ("FireSize v1.0 · ML Ratchet Wildfire Phase 2 · Data: FPA-FOD (Short 2022), LANDFIRE 2016, gridMET, "
                         "FireSim (Phase 1). Exam run once; scores computed against B's FireSize baselines.")
    html = open(f'{WEB}/template.html').read().replace('/*DATA*/', json.dumps(D, separators=(',', ':')))
    open(f'{WEB}/firesize.html', 'w').write(html)
    print('page KB', len(html) // 1024)


if __name__ == '__main__':
    main()
