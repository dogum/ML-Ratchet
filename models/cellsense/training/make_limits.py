"""The page's and model card's limits, written from the released rung's numbers."""
import _paths as _P  # portable paths (see _paths.py)
import json, numpy as np
R = json.load(open(_P.WORK + '/ratchet.json')); rel = [r for r in R if r['accepted']][-1]
W = rel['real']['with_ref']; L = rel.get('life') or {}; c100 = L.get('cycle100', {}); st = L.get('by_stage', {})
p = lambda x: f"{100 * x:.0f}%"
late = st.get('0.75-1.05', {}); prim = c100.get('primary', {}); sec = c100.get('secondary', {})
PX = rel.get('partial') or {}; sec_ = c100.get('secondary', {})
lim = [
    "It knows one cell. Everything was calibrated on the A123 APR18650M1A at Severson's protocol: a 4C discharge to 2.0 V in a 30 °C chamber. Other chemistries (the nickel-rich cells in most long-range cars, for one), sizes, rates or temperatures are outside what it learned.",
    f"Near end of life it reads total capacity low: real worn cells hide more slow charge behind the 4C curve than the simulator makes. Two attempts to model why (growing slow polarization, then poorly connected graphite) improved the simulated exam but made the calibration cells worse, and the ratchet threw both out. Typical capacity error is {W['q_total_mae_mAh']:.0f} mAh overall and largest in the last tenth of life.",
    "Lost lithium, lost graphite and lost LFP have no answer key. The cells were never opened, so these are the physics model's explanation of each curve. Lost LFP barely changes these curves at all, and the error bars say so.",
    (f"The lifetime forecast is useful from cycle 100 but not yet Severson's equal: {prim.get('mpe', float('nan')):.0f}% error on their primary test cells and {sec_.get('mpe', float('nan')):.0f}% on the secondary ones, against 9 to 15% for models trained on 41 real cells. "
     f"How strongly early slow graphite predicts the knee was set on six cells. Late in life it still reads worn cells as healthier than they are and overestimates the cycles left ({late.get('mpe', float('nan')):.0f}% life error in the last quarter).") if L else "No lifetime forecast in this release.",
    "The aging rates were calibrated on six real cells' trajectories, all charged in about 10 minutes. Charging recipes much faster or slower than that rely on the physics alone.",
    f"Pulse resistance follows the real cells (correlation {W['ir_corr']:.2f}) with about {W['ir_mae_mohm']:.1f} mΩ error. The model's 30 ms pulse is ohmic plus charge transfer with no double layer, so absolute values carry the calibration's error.",
    (f"Partial discharges from full read well down to about 0.75 Ah ({PX['0.75']['full_ref']['q_total_mae_mAh']:.0f} mAh capacity error) and no better than assuming no fade at 0.6 Ah and below. " if PX else '')
    + "Discharges that start partway (cycling between 20% and 80%, say), other cut-off voltages and other rates are untested.",
]
json.dump(lim, open(_P.LAB + '/limits.json', 'w'), indent=1); print('\n'.join(lim))
