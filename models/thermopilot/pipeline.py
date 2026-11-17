"""ThermoPilot on a BuildingSim building (needs the ML Ratchet BuildingSim + BuildingData packages and training/tp).

    python3 pipeline.py                      # a generated heat-pump house, January 2017, Minneapolis weather, Xcel TOD
Builds the building card from the house's nominal BuildingSim model, runs ThermoPilot and the fixed / setback
thermostats for the month, and prints bills and comfort.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'training'))
import numpy as np


def main():
    import tp                                    # adds BuildingSim / BuildingData to the path
    import buildingsim as bs
    from tp import features as F, policy as P
    from tp.episode import Episode
    W = P.load(os.path.join(HERE, 'thermopilot_v1.npz'))
    spec = bs.generate_buildings(1, seed=11, primary_use='residential_single', climate_zone='6A', system='heat_pump_air')[0]
    pref = dict(lo=20.5, hi=24.0, w_sleep_h=2.0, w_away_h=3.0, w_sleep_c=0.0, w_away_c=2.0)
    ep = Episode(spec, spec, 'site:Hog', 2017, 1, 'xcel_mn_a04_tod', pref, seed=1)
    card = F.card_from_model(ep.model, pref['hi'])
    for name, st in [('fixed', ep.fixed()), ('setback', ep.setback()),
                     ('thermopilot', P.ThermoPilot(W, card, ep.ctx, ep.lam, ep.model.base_W, ep.fc_noisy).act)]:
        m, _ = ep.run(st)
        print(f"{name:12s} bill ${m['bill']:7.2f}  kWh {m['kWh']:6.0f}  outside band {m['DH']:5.1f} K h  peak {m['peak_kW']:.1f} kW")


if __name__ == '__main__':
    main()
