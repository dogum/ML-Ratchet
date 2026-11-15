"""ratchet.yaml from the raw logs: the Phase D ladder (work/ratchet2.json, copied to training/ratchet2.json) first, then
the v1 ladder as it was (its released rung now marked kept: v1.0 shipped it, v2 supersedes it).
python write_ratchet_yaml.py <released rung name>"""
import _paths as _P  # portable paths (see _paths.py)
import os, sys, json, re

rel = sys.argv[1]
L = json.load(open(os.path.join(_P.SRC, 'ratchet2.json')))
y = open(os.path.join(_P.MODEL, 'ratchet.yaml'), encoding='utf-8').read()
v1 = y[y.index('  - name: "v1.0 — RoadSim physics v1'):]   # the v1 ladder as committed (name fixed 14 Nov 2026)
v1 = v1.replace('decision: released, why: "+0.015 over rung 4; dev D 0.129."', 'decision: kept, why: "+0.015 over rung 4; dev D 0.129. Released as v1.0; superseded by v2."')


def q(s):
    return '"' + str(s).replace('"', "'") + '"'


rows = []
for r in L['rungs']:
    fv = r['frozen']; v = r['val'][fv]
    dec = 'released' if r['name'] == rel else r['decision']
    why = r.get('why', '')
    m = re.search(r'val2\[(\w)\] skill ([\d.]+) vs ([\d.]+) \(d ([+-][\d.]+)\); dev D2 ([\d.]+) vs ([\d.]+) \(d ([+-][\d.]+)\)', why)
    short = (f"skill {m.group(4)} on val2[{fv}] ({m.group(2)} vs {m.group(3)}); dev D2 {m.group(5)} vs {m.group(6)} ({m.group(7)})"
             + ('; gate-only rung' if r.get('gate_only') else '')) if m else why
    if 'physics rung' in why:
        short += f"; physics rung, judged on D2 (needs < -0.01); first rule: {r.get('rule1')}"
    if r.get('note'): short += '. ' + r['note']
    rows.append(f"      - {{ id: v2-{r['name'].split('_')[0]}, change: {q(r['change'])}, sim: {v['skill']:.3f}, decision: {dec}, "
                f"why: {q(short)}, metrics: [\"{100 * v['MdAPE']:.1f}%\", {r['dev']['D']:.4f}] }}")
sg = json.load(open(os.path.join(_P.SRC, 'sigma_rung.json')))
rows.append(f"      - {{ id: v2-r12, change: {q('error bars: sigma x a per-architecture factor that brings the simulated val2 set to 68% (simulation only), then the dev factor per group')}, "
            f"decision: kept, why: {q('leave-one-car-out on the 16 dev VED cars, mean |held-out coverage - 68%%|: %.3f vs %.3f with v1 factors alone, %.3f with a car-level term; no effect on the energy' % (sg['sim+v1']['err'], sg['v1']['err'], sg['car']['err']))} }}")
head = f'''# The ratchet ladders, normalised. Raw logs: training/ratchet2.json + rungs2.json (Phase D), training/ratchet.json + rungs.json (v1).
model: energynet
ladders:
  - name: "v2.0 — RoadSim physics v2 (Phase D, 14 Nov 2026)"
    objective: "Skill = 1 - median |log error| / median |log error| of the class-median guess, on the frozen simulated set val2 (400 unseen generated vehicles x 10 trips on RoadSim physics v2)"
    rule: "Keep if the val2 skill beats the incumbent by more than 0.01 AND the dev score D2 (mean of 18 dev median errors: VED 4 fleets, EPA 9 tests, full-ZTBus dev legs and missions, Mersin, ICCT) gets no more than 0.01 worse. A rung that changes the simulated physics is judged on its own physics version of val2, the incumbent re-scored on it. Gate-only rungs (inputs only real data carries) must improve D2 by more than 0.01."
    margin: 0.01
    dev_gate: "D2 (mean of 18 dev MdAPEs) may not rise by more than 0.01"
    frozen_sets: [val2_a, val2_b, val2_c, val2_d, val2_e, val2_f, val2_eL]
    scored_on: val2
    note: "Versions of val2: a RoadSim v2 raw, b + OBD label model, c + v1's on-road wrapper, d c + v1's plug-in patch, e b + plug-in patch, f a + plug-in patch (rung v2-r5), eL e with labels from RoadSim's EPA procedure emulation (rung v2-r10, 'e+L'). Each rung is scored on the version it was judged on."
    columns_source: Frozen-set median error and the dev score D2 (lower is better)
    columns: ["Val2 MdAPE", "Dev D2"]
    rungs:
'''
open(os.path.join(_P.MODEL, 'ratchet.yaml'), 'w', encoding='utf-8', newline='\n').write(head + '\n'.join(rows) + '\n' + v1)
print(open(os.path.join(_P.MODEL, 'ratchet.yaml'), encoding='utf-8').read()[:3000])
