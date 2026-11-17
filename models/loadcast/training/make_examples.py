"""release/examples: 3 real EXAM building-meters (the first electricity, chilled water and steam units of the page's
fixed draw), their BDG2 meter 2016-2017 (kWh per hour, UTC) and site weather, plus metadata. BDG2 is CC BY-SA 4.0."""
import json, os
import numpy as np, pandas as pd
import lc_common as C
from webdata import pick_units

OUT = f'{C.ROOT}/release/examples'


def main():
    os.makedirs(OUT, exist_ok=True)
    picks = pick_units(); chosen = []
    for m in ('electricity', 'chilledwater', 'steam'):
        chosen.append(next((b, mm) for b, mm in picks if mm == m))
    meta = C.bd.bdg2_meta().set_index('id'); info = []
    for b, m in chosen:
        r = meta.loc[b]; site = r['site_id']
        y = C.real_meter(m, [b])[b]
        f = f'{b}_{m}.csv.gz'; y.rename('kWh').to_frame().to_csv(f'{OUT}/{f}', float_format='%.3f')
        wf = f'weather_{site}.csv.gz'
        if not os.path.exists(f'{OUT}/{wf}'):
            w, wm = C.site_weather(site)
            w[['T_out', 'dewpoint', 'GHI', 'wind_speed']].to_csv(f'{OUT}/{wf}', float_format='%.2f')
        w, wm = C.site_weather(site)
        yb = r['year_built']
        info.append(dict(building=b, meter=m, site=site, tz=wm['tz'], lat=wm['lat'], use=r['primary_use'],
                         area_m2=float(r['floor_area_m2']), year_built=None if not np.isfinite(yb) else float(yb),
                         meters_present=C.meters_of(r['meters']), meter_file=f, weather_file=wf))
    bl = C.bd.baselines('load_bdg2').set_index(['building_id', 'meter'])
    for i in info:
        r = bl.loc[(i['building'], i['meter'])]
        i['baselines_2017'] = {k: float(r[f'{k}_cvrmse']) for k in ('towt', 'g14_changepoint', 'persistence_24h')}
    json.dump(info, open(f'{OUT}/examples.json', 'w'), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
