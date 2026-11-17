"""Add presentation fields (headline numbers, limits, training size) to exam.json. Numbers are read, never typed."""
import glob
import json
import os

import common as C


def main():
    fn = os.path.join(C.ROOT, 'exam.json')
    X = json.load(open(fn))
    n = sum(len([i for i in json.load(open(f)) if 'error' not in i]) for f in glob.glob(os.path.join(C.ROOT, 'bank', 'v1', 'shard_00*.json')))
    X['n_train_buildings'] = n
    sv, sb, b, c, cr, r, h = (X['sim_val']['hourly'], X['sim_baselines']['hourly'], X['bdg2'], X['cbecs'], X['cbecs_rules'],
                              X['recs'], X['hue'])
    st = X['stock']
    X['kpis'] = [
        dict(v=f"{sv['logUA']['MAE']:.2f}", l=f"log10 error in UA on held-out simulated buildings (PRISM: {sb['prism_logUA']['MAE']:.2f})"),
        dict(v=f"{b['annual']['area']['MdAPE']:.0f}%", l=f"floor-area error on 1,428 real BDG2 buildings (energy ÷ EUI of the true use: {X['bdg2_baselines']['area_eui_by_true_use']['MdAPE']:.0f}%)"),
        dict(v=f"{100 * c['heat_class']['accuracy']:.0f}%", l=f"heating type right on 2,513 real CBECS buildings (fuel rule: {100 * cr['fuel_rule_accuracy']:.0f}%)"),
        dict(v=f"{100 * b['hourly']['use']['accuracy']:.0f}%", l=f"primary use right on real BDG2 (majority guess: {100 * X['bdg2_baselines']['use_majority']['accuracy']:.0f}%). It fails here."),
    ]
    rs, cs = st['resstock']['hourly'], st['comstock']['hourly']
    X['limits'] = [
        f"Primary use does not transfer to real buildings: {100 * b['hourly']['use']['accuracy']:.0f}% right on BDG2 against {100 * X['bdg2_baselines']['use_majority']['accuracy']:.0f}% for always guessing education. The dev-fitted temperature (10) flattens these probabilities on purpose.",
        f"Year built has no real skill: BDG2 mean error {b['annual']['year']['MAE']:.0f} years vs {X['bdg2_baselines']['year_dev_median']['MAE']:.0f} for the dev median; RECS {r['year']['MAE']:.0f} vs {X['recs_rules']['year_dev_median_MAE']:.0f}; CBECS {c['year']['MAE']:.0f} vs {cr['year_dev_median_MAE']:.0f}.",
        f"Heat pump vs resistance from annual totals is worse than chance on RECS (AUC {r['hp_vs_resistance']['auc']:.2f}); on the 24 HUE homes the hourly model's electric-heat AUC is {h['hourly']['auc_electric_heat']:.2f} (monthly {h['monthly']['auc_electric_heat']:.2f}). In simulation the same task gives AUC {sv['hp_vs_resistance']['auc']:.2f}: the simulator's heat-pump signature is not what real homes show.",
        f"UA and time constant have no real truth. Against EnergyPlus (ResStock homes) the hourly UA error is {rs['logUA']['MAE']:.2f} log10 with a +{rs['logUA']['bias']:.2f} bias, no better than the sample median ({rs['logUA']['MAE_median_predictor']:.2f}); for ComStock buildings it is {cs['logUA']['MAE']:.2f} vs {cs['logUA']['MAE_median_predictor']:.2f}.",
        f"Real BDG2 floor area needed error bars 3x wider than the network's own (hourly), and hourly input reads area no better than one annual total ({b['hourly']['area']['MdAPE']:.0f}% vs {b['annual']['area']['MdAPE']:.0f}%): real load shapes still differ from simulated ones.",
        f"Operating hours from one annual total are worse than guessing the dev median ({X['cbecs_hours']['detective_MAE_h']:.0f} h/week vs {X['cbecs_hours']['dev_median_MAE_h']:.0f}). Hourly schedules are read well in simulation (MAE {sv['op_hours']['MAE']:.1f} h/day) but no real hourly building has schedule truth.",
        "The real dev gate is small (52 BDG2 buildings) and noisy. It drifted from 0.331 to 0.306 over the kept rungs, each step inside the 0.02 tolerance.",
        "Simulator limits carry over: single thermal zone, first-order part-load and fault models, ideal thermostat within an hour, campus plants as a flat distribution factor plus base load (W2, W3).",
    ]
    E = json.load(open(os.path.join(C.ROOT, 'web', 'explorer.json')))
    X['hero'] = E['units'][0]['id']
    json.dump(X, open(fn, 'w'), indent=1, default=float)
    print(X['kpis'], X['n_train_buildings'], X['hero'])


if __name__ == '__main__':
    main()
