"""Assemble web/lawfinder.html from web/template.html + web/data.json + per-law page metadata."""
import json, os, sys
import numpy as np
import lawratchet as lr

ROOT = lr.ROOT
READING = json.load(open(f'{ROOT}/web/readings.json')) if os.path.exists(f'{ROOT}/web/readings.json') else {}

META = {
    'water': dict(exam_key='mae', exam_cols=[['mae', 'MAE'], ['r2', 'R²'], ['rho', 'ρ'], ['bias', 'bias']],
                  exam_order=['law', 'sim_law', 'fu_dev', 'fu_2.6', 'fu_sim', 'flowcast', 'dumb'], yrange=[0, 1.3],
                  dev_metric='mean |error| of Q/P on the 52 dev basins', exam_metric='mean |error| of Q/P',
                  exam_desc='619 CAMELS-US basins, water years 1990-99', exam_unit='basins',
                  exam_note='Truth: gauged Q over ERA5-Land P for the decade. FlowCast row = its decade-mean flow / P '
                            '(KGE β × observed ratio). All rows use the same ERA5-Land forcing.',
                  textbook_rows=('fu_dev',), network='flowcast', short={'flowcast': 'FlowCast'}, target_ml='<mi>Q</mi><mo>/</mo><mi>P</mi>'),
    'fire': dict(exam_key='mae_log10', exam_cols=[['mae_log10', 'all days |log₁₀|'], ['bias_log10', 'all days bias'],
                                                   ['g_mae', 'growth days |log₁₀|'], ['g_rho', 'growth days ρ']],
                 exam_order=['law', 'sim_law', 'rothermel_dev', 'rothermel_sim', 'fl_prev', 'firecast', 'clim', 'persist'],
                 yrange=[-2.3, 3.0], dev_metric='mean |log₁₀ ratio| on 791 dev fire-days',
                 exam_metric='mean |log₁₀ ratio|', exam_desc='FEDS fire-days, 2012-2024', exam_unit='fire-days',
                 exam_note='Score: log₁₀ of (predicted + 1 cell) / (observed + 1 cell) new area, 1 cell = 0.0081 km². '
                           'fl_prev, clim and persist are FireData\'s own baseline rows; FireCast is its thresholded map.',
                 textbook_rows=('rothermel_dev',), network='firecast', dumb_row='clim', short={'fl_prev': "yesterday's growth", 'firecast': 'FireCast'},
                 target_ml='<msub><mi>log</mi><mn>10</mn></msub><mi>G</mi>',
                 decode={'label': 'new area G', 'f': 'Math.pow(10,y)-0.0081', 'unit': 'km²/day'}),
    'grid_inertia': dict(exam_key='mdape', exam_cols=[['mdape', 'MdAPE', '%'], ['mae', 'MAE'], ['rho', 'ρ'], ['bias_median', 'bias']],
                         exam_order=['law', 'sim_law', 'k_x_demand', 'gridpulse', 'dumb'], yrange=[0, 480],
                         dev_metric='mean |error| / truth on the 4 dev events', exam_metric='MdAPE %',
                         exam_desc='410 GB events, 2018-2025', exam_unit='events',
                         exam_note='Truth: NESO outturn inertia of the settlement period. Demand and IBR share from the NESO generation mix.',
                         textbook_rows=('k_x_demand',), network='gridpulse', target_ml='<mi>E</mi>'),
    'grid_nadir': dict(exam_key='mae', exam_cols=[['mae', 'MAE', ' Hz'], ['mdape', 'MdAPE', '%'], ['rho', 'ρ'], ['bias_median', 'bias']],
                       exam_order=['law', 'sim_law', 'swing_ramp', 'gridpulse_5s', 'dumb'], yrange=[0, 1.0],
                       dev_metric='median |error| (Hz) on the 4 dev events', exam_metric='MAE (Hz)',
                       exam_desc='GB events with a loss label, 2018-2025', exam_unit='events',
                       exam_note='Truth: f0 minus the lowest 1 s value within 90 s. Loss labels are GridData\'s REMIT/FUELINST '
                                 'matches (medium or low confidence) and are noisy. GridPulse reads 5 s of the trace, which the law does not.',
                       textbook_rows=('swing_ramp',), network='gridpulse_5s', target_ml='<mi>Δ</mi><mi>f</mi>'),
    'bldg': dict(exam_key='mdape', exam_cols=[['mdape', 'MdAPE', '%'], ['bias_median_pct', 'bias', '%'], ['rho', 'ρ'], ['r2', 'R²']],
                 exam_order=['law', 'sim_law', 'signature_dev', 'signature_sim', 'loglin_dev', 'dumb'], yrange=[0, 450],
                 dev_metric='median |error| / truth on the dev homes', exam_metric='MdAPE %',
                 exam_desc='RECS 2020 single-family homes', exam_unit='homes',
                 exam_note='Truth: EIA\'s billing-based annual site energy / floor area. No network in the collection predicts it.',
                 textbook_rows=('signature_dev',), network=None, target_ml='<mi>EUI</mi>'),
}


def textbook(L, E):
    i = L['id']
    if i == 'water':
        w = E['w_dev']
        return dict(js='(Math.pow(1 + Math.pow(phi, TB), 1 / TB) - phi)', args=w,
                    mathml=f'<mi>Q</mi><mo>/</mo><mi>P</mi><mo>=</mo><msup><mrow><mo>(</mo><mn>1</mn><mo>+</mo><msup><mi>φ</mi><mi>ω</mi></msup><mo>)</mo></mrow><mrow><mn>1</mn><mo>/</mo><mi>ω</mi></mrow></msup><mo>−</mo><mi>φ</mi><mo>,</mo><mspace width="0.5em"/><mi>ω</mi><mo>=</mo><mn>{w:.2f}</mn>',
                    note=f'Budyko with Fu\'s curve: one free shape ω, fitted on the 52 dev basins ({w:.2f}). '
                         f'WaterData\'s baseline uses ω = 2.6; fitted on WaterSim it is {E["w_sim"]:.2f}.')
    if i == 'fire':
        return dict(js='Math.log10(TB.k * TB.ros(R0, W, M) * P * 1.44 + 0.0081)', args={'k': E['k_dev']},
                    mathml=f'<mi>G</mi><mo>=</mo><mi>k</mi><mo>·</mo><msub><mi>R</mi><mtext>head</mtext></msub><mo>(</mo><mi>U</mi><mo>,</mo><mi>M</mi><mo>,</mo><mtext>fuel</mtext><mo>)</mo><mo>·</mo><mi>P</mi><mo>·</mo><mn>24</mn><mtext> h</mtext><mo>,</mo><mspace width="0.5em"/><mi>k</mi><mo>=</mo><mn>{E["k_dev"]:.3g}</mn>',
                    note=f'Rothermel head rate of spread (the fuel group\'s most common model, the day\'s wind and 100-h '
                         f'moisture) times the perimeter for 24 h, scaled by k fitted on dev ({E["k_dev"]:.3g}; on FireSim {E["k_sim"]:.3g}). '
                         f'A k far below 1 says real fires grow far slower than a head fire running all day.')
    if i == 'grid_inertia':
        k = E['k_demand_GVAs_per_GW']
        return dict(js='TB * D', args=k, mathml=f'<mi>E</mi><mo>=</mo><mn>{k:.2f}</mn><mtext> s</mtext><mo>·</mo><mi>D</mi>',
                    note=f'GridPulse\'s best baseline: inertia proportional to demand, k = median E/D of the 4 dev events ({k:.2f} s).')
    if i == 'grid_nadir':
        k = E['k_swing']
        return dict(js='TB * L * L / E', args=k,
                    mathml=f'<mi>Δ</mi><mi>f</mi><mo>=</mo><mi>k</mi><mfrac><msup><mi>ΔP</mi><mn>2</mn></msup><mi>E</mi></mfrac><mo>,</mo><mspace width="0.5em"/><mi>k</mi><mo>=</mo><mn>{k:.3g}</mn>',
                    note='Swing equation with a primary response that ramps linearly to full: nadir depth = f0 ΔP² T / (4 E R). '
                         'T/R is unknown on real events, so it is one constant fitted on the 4 dev events.')
    if i == 'bldg':
        c = E['coef_dev']
        return dict(js='TB[0] + TB[1] * H + TB[2] * C', args=c,
                    mathml=f'<mi>EUI</mi><mo>=</mo><mn>{c[0]:.0f}</mn><mo>+</mo><mn>{c[1]:.1f}</mn><mtext> HDD</mtext><mo>+</mo><mn>{c[2]:.1f}</mn><mtext> CDD</mtext>',
                    note='The degree-day energy signature (ASHRAE inverse-model style, balance point fixed at 18 °C), '
                         'fitted on the dev homes.')


def verdict(L, E, M):
    k = M['exam_key']
    rows = E['rows']
    law = rows['law'][k]
    rivals = {n: rows[n][k] for n in rows if n not in ('law', 'sim_law') and rows[n].get(k) is not None and np.isfinite(rows[n][k])}
    best = min(rivals.values())
    if law < best * 0.98:
        return {'cls': 'win', 'text': 'beats every rival'}
    if law <= best * 1.02:
        return {'cls': 'tie', 'text': 'ties the best rival'}
    if rows.get(M.get('dumb_row', 'dumb'), {}).get(k, np.inf) <= best * 1.0001:
        return {'cls': 'loss', 'text': 'loses to the dumb guess'}
    name = min(rivals, key=rivals.get)
    return {'cls': 'loss', 'text': f"loses to {M.get('short', {}).get(name, name.replace('_', ' '))}"}


def main():
    D = json.load(open(f'{ROOT}/web/data.json'))
    for L in D['laws']:
        M = META[L['id']]
        L.update({k: v for k, v in M.items() if k not in ('textbook_rows', 'network', 'target_ml')})
        L['target']['symbol_ml'] = None
        L['released']['mathml_eq'] = M['target_ml'] + '<mo>=</mo>' + L['released']['mathml']
        L['released']['mathml'] = L['released']['mathml']
        L['reading'] = READING.get(L['id'], '')
        if L['id'] == 'fire':
            r0 = json.load(open(f'{ROOT}/web/fire_ros.json'))
            for v in L['vars']:
                if v['id'] == 'R0':
                    v['choices'] = [{'name': g, 'value': r0['r0'][g]} for g in r0['groups']]
                    v['default_choice'] = r0['groups'].index('TU')
            L['ros_table'] = r0
        E = L.get('exam')
        if E and 'sim_law' in E['rows']:
            E['rows']['sim_law']['name'] = f"{L['platform']}'s own law ({E.get('knee', '')}, PySR-style pick on simulations only)"
        if E:
            tb = textbook(L, E)
            L['textbook'] = {'js': tb['js'], 'mathml': tb['mathml'], 'note': tb['note']}
            L['textbookArgs'] = tb['args']
            L['verdict'] = verdict(L, E, M)
            k = M['exam_key']
            L['score'] = {'exam': f"{E['rows']['law']['n']} {M['exam_unit']}", 'metric': M['exam_metric'],
                          'law': E['rows']['law'][k], 'textbook': E['rows'][M['textbook_rows'][0]][k],
                          'network': E['rows'][M['network']][k] if M['network'] else None, 'dumb': E['rows'][M.get('dumb_row', 'dumb')][k], 'sim_law': E['rows']['sim_law'][k]}
        if E and L['id'] == 'fire':
            for kk, row in E['rows'].items():
                g = E['growth_days'].get(kk, {})
                row['g_mae'] = g.get('mae_log10'); row['g_rho'] = g.get('rho')
                if kk == 'firecast':
                    row['mae_log10'] = None; row['bias_log10'] = None     # FireCast rows exist on growth days only
            gd = E['growth_days']
            L['score'].update(metric='mean |log₁₀ ratio|, growth days / all days',
                              law=f"{gd['law']['mae_log10']:.3f} / {E['rows']['law']['mae_log10']:.3f}",
                              sim_law=f"{gd['sim_law']['mae_log10']:.3f} / {E['rows']['sim_law']['mae_log10']:.3f}",
                              textbook=f"{gd['rothermel_dev']['mae_log10']:.3f} / {E['rows']['rothermel_dev']['mae_log10']:.3f}",
                              network=f"{gd['firecast']['mae_log10']:.3f} / –",
                              dumb=f"{gd['clim']['mae_log10']:.3f} / {E['rows']['clim']['mae_log10']:.3f}")
            L['verdict'] = {'cls': 'mixed', 'text': "best on growth days, loses on all days"}
        # compact the exam block: rows + rungs only
        if E:
            L['exam'] = {'rows': E['rows'], 'rungs': E.get('rungs')}
        for r in L['ladder']:
            r['exam'] = (E or {}).get('rungs', {}).get(r['id']) if E else None
    D['limits'] = json.load(open(f'{ROOT}/web/limits.json')) if os.path.exists(f'{ROOT}/web/limits.json') else []
    D['credits'] = open(f'{ROOT}/web/credits.html').read() if os.path.exists(f'{ROOT}/web/credits.html') else ''
    page = open(f'{ROOT}/web/template.html').read()
    js_extra = ''
    page = page.replace('/*DATA*/null', json.dumps(D, separators=(',', ':'), default=float))
    open(f'{ROOT}/web/lawfinder.html', 'w').write(page)
    print('page KB', os.path.getsize(f'{ROOT}/web/lawfinder.html') / 1e3)


if __name__ == '__main__':
    main()
