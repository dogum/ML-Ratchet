"""Markdown table of every conformal wrapper: dev coverage / CRPS raw -> in-sample -> LOBO (medians over basins)."""
import glob, json, os
rows = []
for f in sorted(glob.glob('/home/claude/bakeoff/river/arms_filt/conformal/*_conformal/work/fit_report.json')):
    R = json.load(open(f))
    for j, v in R['jobs'].items():
        m = v['median']; mm = m.get('mean_over_basins', {})
        rows.append(f"| {R['base_id']} | {j} | {m['raw_cov50']:.2f} / {m['raw_cov90']:.2f} | {m['loo_cov50']:.2f} / {m['loo_cov90']:.2f} | "
                    f"{mm.get('raw_cov50', float('nan')):.2f} / {mm.get('raw_cov90', float('nan')):.2f} -> {mm.get('loo_cov50', float('nan')):.2f} / {mm.get('loo_cov90', float('nan')):.2f} | "
                    f"{m['raw_crps_rel']:.3f} -> {m['loo_crps_rel']:.3f} |")
print('| Base arm | Job | Raw cov50 / cov90 (median) | Conformal LOBO (median) | Mean over basins raw -> LOBO | CRPS/Q raw -> LOBO |')
print('|---|---|---|---|---|---|')
print('\n'.join(rows))
