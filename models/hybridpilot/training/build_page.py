"""Assemble web/hybridpilot.html from the template and result files."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from hp_env import ROOT

POL = 'hp'
rep = json.load(open(f'{ROOT}/data/exam_report.json'))
T = rep['table']
STRATA = ['ALL', 'power-split HEV', 'P2 HEV', 'series HEV', 'power-split PHEV', 'series PHEV (EREV)', 'bus (series)', 'bus (P2)', 'truck (P2)', 'truck (PHEV series)']


def ren(d):
    return {('hybridpilot' if k == POL else k): v for k, v in d.items()}


table = {s: ren(T[s]) for s in STRATA if s in T}
for s, r in table.items():
    r['rule'] = r.get('rule')
boot = {s: v for s, v in rep['boot_vs_aecms'][POL].items() if v and s in STRATA}
ff = {('hybridpilot' if k == POL else k): v for k, v in rep['fixed_vs_matched'].items()}
A = table['ALL']
n_twin = rep['groups']['twin']['dp']['n']; n_gen = rep['groups']['generated']['dp']['n']
gt, gg = rep['groups']['twin'], rep['groups']['generated']
bA = boot['ALL']
intro = (f"{A['dp']['n']} exam trips, never used for any choice: {n_twin} real VED exam routes (2 per exam hybrid or plug-in car, "
         f"fixed random draw) each driven by a RoadSim twin of the car that drove it, plus {n_gen} trips by 54 held-out generated "
         f"hybrids (3 per class and architecture) on exam routes, Zurich bus routes (ZTBus exam legs) and heavy-duty cycles. "
         f"Each cell: mean % of DP at matched battery energy · share of trips that held charge. "
         f"HybridPilot runs with no preview; navigation distance was available on a random half of the trips.")
note = (f"HybridPilot minus A-ECMS over all exam trips: {100 * bA['mean']:+.2f} points of DP (95 % interval {100 * bA['lo']:+.2f} to {100 * bA['hi']:+.2f}, "
        f"bootstrap over {bA['n_veh']} vehicles). On the real-car twins: HybridPilot {100 * gt['hybridpilot' if 'hybridpilot' in gt else POL]['pct_dp_mean']:.2f} %, "
        f"A-ECMS {100 * gt['aecms']['pct_dp_mean']:.2f} %. On generated vehicles: {100 * gg[POL]['pct_dp_mean']:.2f} % vs {100 * gg['aecms']['pct_dp_mean']:.2f} %. "
        f"* MPC sees the next 15 s exactly and DP the whole trip: both are benchmarks, not causal controllers. "
        f"Means above 100 % happen where DP's assumptions (warm engine, no catalyst warm-up rule, coarse grid) leave room; the frozen-set score gives no credit above 100 %.")
ffnote = ("Phase 1 corrected fuel with a fixed factor: battery energy × 1/(0.33 × 0.90) of fuel. On exam HEV trips the DP's own marginal value of charge "
          "is lower than that factor, so the fixed factor over-credits strategies that end with extra charge and penalises those that end low. "
          "Matched SOC re-solves DP to the strategy's own end charge, which removes the question.")

# reality check
rows = []
for split in ('dev', 'exam'):
    fn = f'{ROOT}/data/simcheck_{split}.parquet'
    if not os.path.exists(fn): continue
    d = pd.read_parquet(fn)
    d['ratio'] = d.fuel_sim_g / d.fuel_log_g
    for (arch, strat), g in d.groupby(['arch', 'strategy']):
        if strat not in ('aecms', 'hybridpilot', 'powersplit_rule', 'phev_cdcs', 'parallel_rule'): continue
        rows.append(dict(label=f"{split} · {arch} · {'HybridPilot' if strat == 'hybridpilot' else strat}", n=int(len(g)),
                         ratio=float(g.fuel_sim_g.sum() / g.fuel_log_g.sum()) if g.fuel_log_g.sum() > 50 else None,
                         dsoc_log=float(g.dsoc_log.mean()) if 'dsoc_log' in g and g.dsoc_log.notna().any() else None, dsoc_sim=float(g.dsoc_sim.mean()), on_log=float(g.onfrac_log.mean()), on_sim=float(g.onfrac_sim.mean()),
                         st_log=float(g.starts_km_log.mean()), st_sim=float(g.starts_km_sim.mean())))
ex = [r for r in rows if r['label'].startswith('exam · hev_powersplit · aecms')]
rd = ex[0] if ex else rows[0]
real_intro = ("RoadSim cannot be graded on a real car's controller, but it can be checked against real hybrids on their own trips. Each VED hybrid trip "
              "(warm fragment, logged speed, lidar grade, airport weather) was driven by the car's twin with RoadSim's strategies and compared with what the car logged: "
              "fuel over valid seconds (VED's MAF-based estimate), engine-on share (rpm above 300) and engine starts per km. Dev cars were looked at first; exam cars only for this report. Nothing was tuned on either.")
real_note = (f"The simulated hybrids are much thriftier than the real ones: on exam power-split HEV trips the twins burn {rd['ratio']:.2f}× the logged fuel in total, "
             f"runs its engine {100 * rd['on_sim']:.0f} % of the time against {100 * rd['on_log']:.0f} % logged, and starts it {rd['st_sim']:.1f} times per km against {rd['st_log']:.1f}. "
             "Real Toyota hybrids keep the engine turning above about 70 km/h (100 % of dev seconds above 72 km/h), run it at stops in cold weather, and start it less often. "
             "Part of the fuel gap may be the MAF-based fuel estimate (it counts air through an engine that is spinning unfuelled). "
             "So the absolute fuel numbers on this page are simulator numbers, and the ranking of strategies is a ranking inside RoadSim's physics. "
             "Plug-ins are worse: the twins burn 2-2.5× the logged fuel and draw about half the logged battery energy, because the simulated plug-ins start their engines for cabin heat and warm-up where the real cars (Volt, Energi) mostly heat electrically; logged and simulated SOC change still correlate (r = 0.83). "
             "The strategies are separated by 1-3 % of fuel; the simulator's real-world fuel bias is about 20 % for HEVs and larger for plug-ins. A calibration rung on dev (engine spinning at speed, electric cabin heat for plug-ins, start penalty) is the first next step.")

rj = json.load(open(f'{ROOT}/ratchet.json'))
ratchet = [dict(rung=r['rung'], note=r['note'], fs=r['frozen']['skill'], fp=r['frozen']['pct_dp_mean'], fc=r['frozen']['cs_rate'],
                dp=r['dev']['pct_dp_mean'], dc=r['dev']['cs_rate'], kept=r['kept']) for r in rj]

limits = [
    "<b>Graded in the simulator, not on cars.</b> The exam uses real routes and weather, but every fuel number comes from RoadSim. On real VED hybrid trips RoadSim's twins burn about 30 % less fuel than logged and run the engine half as often (sanity check above).",
    f"<b>The margin is small.</b> All causal strategies except the rules sit within 1-2 % of DP. HybridPilot's exam gain over A-ECMS is {100 * bA['mean']:+.2f} points of DP; most of the remaining gap is in P2 and heavy-duty P2 vehicles.",
    "<b>The on/off head did not work as a switch.</b> Using the network's P(on) directly (R1, R2) made more engine-on mistakes than letting the learned s decide on/off through the cost. The kept policy uses the head only as an input to nothing; the price of charge does the work.",
    "<b>Preview and a second DAgger round did not pass the ratchet</b> (R4, R6, R7: +0.007 or worse). With RoadSim's 1 Hz physics and small HEV batteries, knowing the next 30 s barely changes the best decision.",
    "<b>DP is not a perfect bound.</b> It assumes a warm engine and ignores the catalyst warm-up rule the causal strategies obey; on cold starts and some plug-in trips causal strategies beat it (means above 100 %).",
    "<b>Plug-ins without navigation</b> cannot know whether to spend the battery early; with distance-to-go they blend. Charge-depleting rules are already near-optimal on short trips.",
    "<b>Real-car behaviour missing in the simulator</b>: engine kept spinning at speed, cabin heat by engine in a Volt-like car at 0 °C (the real Volt used its electric heater; the twin burns fuel), production drivability calibrations. These shift absolute fuel and starts per km.",
    "<b>Heavy duty is thin</b>: buses on real Zurich routes but trucks only on regulatory cycles and city routes; the hybrid bus is known to be too good in RoadSim."]

policy = json.load(open(f'{ROOT}/data/final_policy.json'))
DATA = dict(exam=dict(table=table, strata=STRATA, boot=boot, ff=ff, intro=intro, note=note, ffnote=ffnote),
            real=dict(rows=rows, intro=real_intro, note=real_note), ratchet=ratchet, limits=limits, policy=policy)
lab = json.load(open(f'{ROOT}/web/lab.json')); net = json.load(open(f'{ROOT}/web/net.json'))
h = open(f'{ROOT}/web/page_template.html').read()
h = h.replace('/*DATA*/null', json.dumps(DATA, default=float)).replace('/*LAB*/null', json.dumps(lab)).replace('/*NET*/null', json.dumps(net))
open(f'{ROOT}/web/hybridpilot.html', 'w').write(h)
print('page bytes', len(h))
