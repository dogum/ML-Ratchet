"""Generator: 156 random buildings (13 per use type, balanced), one year at 1 h with the climate-zone TMYx hint.
Checks: every ledger closes, unmet hours, autosizing sanity, and site EUI by use type against REAL survey
respondents of the 'prior' split (CBECS 2018 / RECS 2020; annual, weighted medians). Generated weather is a typical
year of a representative city, so this compares distributions, not buildings."""
import numpy as np
import pandas as pd

from _common import write
from buildingsim import simulate, read_epw, epw_path, generate_buildings, USE_TYPES
from buildingsim.generator import _load


def wmed(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])


def main():
    specs = []
    for i, u in enumerate(USE_TYPES):
        specs += generate_buildings(13, seed=100 + i, primary_use=u)
    ws = {}
    rows = []
    for s in specs:
        c = s['weather_hint']
        if c not in ws:
            ws[c] = read_epw(epw_path(c))
        try:
            out = simulate(s, ws[c], '2019-01-01', '2020-01-01', dt='1h')
        except Exception as e:
            rows.append(dict(id=s['id'], use=s['primary_use'], error=repr(e)))
            print('ERR', s['id'], s['primary_use'], s['system'], e)
            continue
        S = out['summary']
        rows.append(dict(id=s['id'], use=s['primary_use'], system=out['spec']['system'], cz=s.get('climate_zone'),
                         area=s['floor_area_m2'], EUI=S['EUI_kWh_m2_yr'],
                         elec_share=S['kWh_by_meter']['electricity'] / max(S['site_kWh'], 1),
                         unmet_h=S['unmet_heating_h'] + S['unmet_cooling_h'], closure=out['ledger']['closure_rel'],
                         lag=out['ledger']['system_lag_share'], wall_s=S['wall_s']))
    d = pd.DataFrame(rows)
    ok = d[d.get('error').isna()] if 'error' in d else d
    cb = _load('cbecs_prior.parquet')
    rc = _load('recs_prior.parquet')
    ref = {}
    for u, g in cb.groupby('primary_use'):
        e = (g['site_kWh'] / g['floor_area_m2']).values
        m = np.isfinite(e) & (e > 0)
        ref[u] = dict(n=int(m.sum()), weighted_median_EUI=wmed(e[m], g['weight'].values[m]))
    for u, ht in (('residential_single', ('single_family_detached', 'single_family_attached', 'mobile_home')),
                  ('residential_multi', ('apartment_2_4', 'apartment_5plus'))):
        g = rc[rc['housing_type'].astype(str).isin(ht)] if rc['housing_type'].astype(str).isin(ht).any() else \
            rc[rc['primary_use'] == u]
        e = (g['site_kWh'] / g['floor_area_m2']).values
        m = np.isfinite(e) & (e > 0)
        ref[u] = dict(n=int(m.sum()), weighted_median_EUI=wmed(e[m], g['weight'].values[m]) if m.any() else None)
    by_use = {}
    for u, g in ok.groupby('use'):
        by_use[u] = dict(n=len(g), EUI_median=float(g.EUI.median()), EUI_p10=float(g.EUI.quantile(0.1)),
                         EUI_p90=float(g.EUI.quantile(0.9)), survey=ref.get(u),
                         ratio_to_survey_median=float(g.EUI.median() / ref[u]['weighted_median_EUI'])
                         if ref.get(u, {}).get('weighted_median_EUI') else None,
                         systems=g.system.value_counts().to_dict(), unmet_h_median=float(g.unmet_h.median()))
        print(u, by_use[u])
    write('generator.json', dict(
        what='156 generated buildings (13 per use type), TMYx of the climate-zone city, 1 h, one year',
        n_ok=len(ok), n_errors=int(d['error'].notna().sum()) if 'error' in d else 0,
        max_ledger_closure=float(ok.closure.max()), max_system_lag_share=float(ok.lag.max()),
        unmet_hours_median=float(ok.unmet_h.median()), unmet_hours_p90=float(ok.unmet_h.quantile(0.9)),
        by_use=by_use, buildings=rows,
        survey_reference='CBECS 2018 / RECS 2020 respondents in the prior split (real, weighted site EUI medians)'))


if __name__ == '__main__':
    main()
