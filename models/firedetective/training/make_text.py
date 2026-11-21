"""Page text and tables from out/exam.json (writes out/page_text.json)."""
from common import *

E = json.load(open(latest('out/exam.json')))
A = E['all']
sev = A['forest']; sevall = A['all_fuels']; wind = A['wind']; lf = A['lfmc']


def rc(r):
    return '—' if not r or not np.isfinite(r.get('rho', np.nan)) else f"{r['rho']:.2f} <span class='muted'>[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]</span>"


def mc(r, d=3):
    return '—' if not r or not np.isfinite(r.get('mae', np.nan)) else f"{r['mae']:.{d}f} <span class='muted'>[{r['ci'][0]:.{d}f}, {r['ci'][1]:.{d}f}]</span>"


def cov(x):
    return '—' if x is None or not np.isfinite(x) else f'{100 * x:.0f} %'


tables = [
    dict(title=f"Burn severity, forest fires (n = {sev['n']})",
         head=['Method', 'Spearman ρ', 'MAE (fraction)', '80 % coverage'],
         rows=[dict(cls='net', cells=['FireDetective', rc(sev['rho']['network']), mc(sev['mae']['network']), cov(sev['cov80_network'])]),
               dict(cells=['ERC max (gridMET)', rc(sev['rho']['erc_max']), mc(sev['mae']['erc_max']), '—']),
               dict(cells=['BI max', rc(sev['rho']['bi_max']), '—', '—']),
               dict(cells=['FWI max', rc(sev['rho']['fwi_max']), '—', '—']),
               dict(cells=['Van Wagner crown potential', rc(sev['rho']['cfp']), mc(sev['mae']['cfp']), '—']),
               dict(cells=['Dev median (dumb)', '—', mc(sev['mae']['dumb']), '—'])],
         note=f"Truth: share of the burned area MTBS maps as high severity (median {sev['truth_median']:.3f}). All fuels (n = {sevall['n']}): "
              f"FireDetective ρ {sevall['rho']['network']['rho']:.2f}, ERC ρ {sevall['rho']['erc_max']['rho']:.2f}. "
              "MAE uses a 2-parameter map to MTBS units fitted on dev fires, for every method."),
    dict(title=f"Wind on the fire days, read blind (n = {wind['n']})",
         head=['Method', 'Spearman ρ', 'MAE (m/s)', '80 % coverage'],
         rows=[dict(cls='net', cells=['FireDetective (wind inputs removed)', rc(wind['rho']['network']), mc(wind['mae']['network'], 2), cov(wind['cov80_network'])]),
               dict(cells=['Rothermel back-calculation', rc(wind['rho']['rothermel_bc']), mc(wind['mae']['rothermel_bc'], 2), '—']),
               dict(cells=['Dev median (dumb)', '—', mc(wind['mae']['dumb'], 2), '—'])],
         note='Truth: gridMET daily mean 10 m wind at the fire, averaged over the observed days.'),
    dict(title=f"Live fuel moisture (n = {lf['n']})",
         head=['Method', 'Spearman ρ', 'MAE (% dry mass)', '80 % coverage'],
         rows=[dict(cls='net', cells=['FireDetective', rc(lf['rho']['network']), mc(lf['mae']['network'], 1), cov(lf['cov80_network'])]),
               dict(cells=['Climatology: type × month (dev sites)', rc(lf['rho']['clim']), mc(lf['mae']['clim'], 1), '—']),
               dict(cells=['Dev-site median (dumb)', '—', mc(lf['mae']['dumb'], 1), '—'])],
         note='Truth: median Globe-LFMC sample within 25 km and ±10 days of the fire.'),
]
per = []
for p in ('main', 'recent'):
    S = E[p]
    per.append(dict(cells=[p, str(S['forest']['n']), rc(S['forest']['rho']['network']), rc(S['forest']['rho']['erc_max']),
                           str(S['wind']['n']), rc(S['wind']['rho']['network']), str(S['lfmc']['n']), rc(S['lfmc']['rho']['network'])]))
tables.append(dict(title='By exam period', head=['Period', 'n sev.', 'ρ severity', 'ρ ERC', 'n wind', 'ρ wind', 'n LFMC', 'ρ LFMC'], rows=per,
                   note='Main = 2017-21 exam fires; recent = 2022-23 (whole seasons held out). MTBS severity is coded in LANDFIRE mostly up to 2020, so recent fires have little severity truth.'))
json.dump(dict(exam_tables=tables), open(f'{ROOT}/out/page_tables.json', 'w'), indent=1)
print(json.dumps(dict(sev_rho=sev['rho']['network'], erc=sev['rho']['erc_max'], wind=wind['rho']['network'], lf=lf['rho']['network']), indent=0))
