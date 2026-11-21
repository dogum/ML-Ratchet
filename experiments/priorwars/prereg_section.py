"""Print a model's PREREG section from its adapter and its dev / frozen-sim tuning results (no exam data).
python prereg_section.py <model> <section number>"""
import json, os, sys
import harness as h

name, num = sys.argv[1], sys.argv[2]
a = h.A(name)
td = json.load(open(os.path.join(h.ROOT, name, 'tune_dev.json')))
ts = json.load(open(os.path.join(h.ROOT, name, 'tune_sim.json')))
Xd, Yd, ud = a.dev()
Xe, ue = a.exam_X()
units = getattr(a, 'exam_units', lambda: ue)()
n_units = len(set(units.tolist()))
lcks = getattr(a, 'LC_KS', h.lc_ks(len(Xd)))
L = []
L.append(f'## {num}. {a.TITLE}  (`{name}`)\n')
L.append((a.__doc__ or '').strip() + '\n')
L.append(f'- Feature table: {Xd.shape[1]} columns. Dev rows: {len(Xd)}. Exam rows: {len(Xe)} ({n_units} bootstrap units).')
L.append('- Fitted targets: ' + ', '.join(f"`{t['name']}` ({t['kind']})" for t in a.TARGETS) + '.')
if hasattr(a, 'items'):
    import numpy as np
    it = a.items(Yd, {t['name']: np.zeros(len(Xd)) for t in a.TARGETS})
    L.append('- Scored items (metric): ' + '; '.join(f'{k} ({", ".join(v[2])})' for k, v in it.items()) + '.')
else:
    L.append('- Scored items (metric): ' + '; '.join(f"{t['name']} ({', '.join(a.metrics(t['name']))})" for t in a.TARGETS) + '.')
L.append(f'- Headline: {a.HEADLINE}.')
L.append(f'- Bootstrap unit: per {getattr(a, "UNIT", "exam unit")}; {n_units} units.')
L.append(f"- Card baselines re-scored beside the arms: {', '.join(a.baselines_exam().keys()) if hasattr(a, 'baselines_exam') else 'none'}.")
L.append(f"- C-dev: {'yes (ridge / logistic, section 4)' if hasattr(a, 'C_dev_predict') else 'no (the card has no dev-fitted classical learner)'}.")
L.append(f'- LC k values: {lcks if lcks else "none (dev too small)"}.')
L.append('\n**Tuning results (dev and frozen simulated set only):**\n')
L.append(f"- P on dev: headline {td['P_on_dev_headline']:.4f}. P on the 400 frozen-sim rows: {ts['P_on_simval_headline']:.4f}.")
L.append('- G-dev cross-validation inside dev: ' + '; '.join(f"{k} {v['headline']:.4f}" for k, v in td['grid'].items())
         + f" -> **{td['chosen_prep']}**.")
if 'C_dev_cv' in td:
    L.append(f"- C-dev cross-validation inside dev: headline {td['C_dev_cv']['headline']:.4f}.")
L.append('- G-sim on the frozen simulated set: ' + '; '.join(f"{k} {v['headline']:.4f}" for k, v in ts['grid'].items())
         + f" -> **{ts['chosen_prep']} @ {ts['chosen_ctx']} rows**.")
L.append('')
print('\n'.join(L))
