"""Print a compact board of every evaluated rung (sim val + dev) from <arm>/work/eval/*_summary.json, plus v2.1."""
import glob, json, os
ARMS = ['tab_mlp', 'tab_gbm', 'tab_gp', 'tab_tabpfn']
ref = json.load(open('/home/claude/bakeoff/vehicle/harness/results/dev/enet_v21_table_summary.json')) if os.path.exists('/home/claude/bakeoff/vehicle/harness/results/dev/enet_v21_table_summary.json') else None
A = ['conv', 'hev_parallel_p2', 'hev_series', 'hev_powersplit', 'phev_powersplit', 'phev_series', 'bev']
print(f"{'arm':12s} {'rung':5s} {'skill':>6s} {'MdAPE':>6s} {'c50':>5s} {'c90':>5s} {'crps':>6s} {'D16':>6s} | sim by arch (MdAPE) | dev VED ICE/HEV/EV/PHEV  EPA cFTP/cHWY/US06/hFTP/bevC  Mer d/c ICCT | dev cov50/90 VED_ICE")
rows = [('enet_v21', ref)] if ref else []
for arm in ARMS:
    for p in sorted(glob.glob(f'/home/claude/bakeoff/vehicle/arms_table/{arm}/work/eval/*_table_summary.json')):
        rows.append((arm, json.load(open(p))))
for arm, s in rows:
    sv = s.get('simval', {}); dv = s.get('dev', {})
    st = dv.get('sets', {})
    g = lambda k, f='MdAPE': st.get(k, {}).get(f, float('nan'))
    ba = sv.get('by_arch', {})
    print(f"{arm:12s} {s.get('tag','')[:5]:5s} {sv.get('skill', float('nan')):6.3f} {sv.get('MdAPE', float('nan')):6.4f} {sv.get('cov50', float('nan')):5.2f} {sv.get('cov90', float('nan')):5.2f} {sv.get('crps_log', float('nan')):6.4f} {dv.get('D16', float('nan')):6.4f} | "
          + ' '.join(f"{ba.get(a, float('nan')):.3f}" for a in A) + ' | '
          + '/'.join(f"{g(k):.3f}" for k in ['VED_ICE', 'VED_HEV', 'VED_EV', 'VED_PHEV']) + '  '
          + '/'.join(f"{g(k):.3f}" for k in ['EPA_conv_FTP', 'EPA_conv_HWY', 'EPA_conv_US06', 'EPA_hev_FTP', 'EPA_bev_FTP']) + '  '
          + '/'.join(f"{g(k):.3f}" for k in ['Mersin_diesel', 'Mersin_cng', 'ICCT'])
          + f" | {g('VED_ICE','cov50'):.2f}/{g('VED_ICE','cov90'):.2f}")
