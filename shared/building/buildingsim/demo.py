"""BuildingSim demo: an office (VAV, chiller, boiler) and a house (air-source heat pump) in Chicago, one winter and
one summer week each at 15 min; ledgers and comfort printed; one fault injected (office OA damper stuck open in
winter); one custom controller (house pre-cooling under a time-of-use tariff); plot saved to demo_weeks.png.
Runs in ~15 s single-threaded."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '1')
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from buildingsim import simulate, read_epw, epw_path, ledger  # noqa: E402

W = read_epw(epw_path('Chicago'))           # TMYx typical year, rebased to 2019
OFFICE = dict(id='demo_office', primary_use='office', floor_area_m2=5000, n_floors=3, year_built=2005,
              climate_zone='5A', system='vav_chiller_boiler', hvac=dict(chiller_condenser='air'))
HOUSE = dict(id='demo_house', primary_use='residential_single', floor_area_m2=180, n_floors=2, year_built=1995,
             climate_zone='5A', system='heat_pump_air', setpoints=dict(heat=20.5, cool=24.0, setback_heat=17.0,
                                                                         schedule='night_setback'))
WEEKS = {'winter': ('2019-01-14 06:00', '2019-01-21 06:00'), 'summer': ('2019-07-15 05:00', '2019-07-22 05:00')}
TOU = dict(base=0.13, periods=[dict(months=[6, 7, 8, 9], hours=(16, 21), weekdays_only=True, price=0.42)])


def show(name, out):
    L = ledger(out)
    z = L['zone_heat_kWh']
    S = out['summary']
    print(f"\n== {name}: {S['steps']} steps in {S['wall_s']:.2f} s")
    print('  zone heat ledger [kWh]: ' + ', '.join(f'{k} {v:+.0f}' for k, v in z.items()
                                                  if k not in ('residual', 'closure_rel')))
    print(f"  closures: zone {z['closure_rel']:.1e}, moisture {L['zone_moisture_kg']['closure_rel']:.1e}, air side "
          f"{L['air_side_kWh']['closure_rel']:.1e}, equipment {L['equipment_kWh']['closure_rel']:.1e}, meters "
          f"{L['meters_kWh']['closure_rel']:.1e} -> ok={L['ok']}")
    print('  meters [kWh]: ' + ', '.join(f'{k} {v:.0f}' for k, v in S['kWh_by_meter'].items() if abs(v) > 0.5))
    ts = out['ts']
    occ = ts['occupancy'] > 0.05
    print(f"  comfort: mean PPD occupied {S['mean_PPD_occupied']:.1f} %, operative T occupied "
          f"{ts['T_op'][occ].min():.1f}..{ts['T_op'][occ].max():.1f} C, unmet heating {S['unmet_heating_h']:.1f} h, "
          f"cooling {S['unmet_cooling_h']:.1f} h")


def precool(obs):
    """Custom controller: pre-cool from noon to 16:00 on weekdays, then coast through the 16-21 peak."""
    if obs['daytype'] == 0 and 12 <= obs['hour'] < 16:
        return dict(cool_sp=22.0)
    if obs['daytype'] == 0 and 16 <= obs['hour'] < 21:
        return dict(cool_sp=27.0)
    return None


def main():
    t0 = time.time()
    runs = {}
    for bname, spec in (('office', OFFICE), ('house', HOUSE)):
        for season, (a, b) in WEEKS.items():
            out = simulate(spec, W, a, b, dt='15min', prices=TOU if bname == 'house' else None)
            runs[(bname, season)] = out
            show(f'{bname} {season}', out)
    # one fault: office OA damper stuck fully open in the winter week from Wednesday
    a, b = WEEKS['winter']
    fault = dict(type='oa_damper_stuck', position=1.0, start='2019-01-16 06:00')
    f_out = simulate(OFFICE, W, a, b, dt='15min', faults=[fault])
    show('office winter, OA damper stuck open from Wed', f_out)
    tr, trh = f_out['trend'], runs[('office', 'winter')]['trend']
    m = (tr['fault'] != '') & (tr['occupied'] == 1)
    print('  trend signature (faulted vs healthy, occupied hours): OA damper %.0f%% vs %.0f%%, MAT %.1f vs %.1f C, '
          'gas %.0f vs %.0f kWh' % (tr['OA_damper_pct'][m].mean(), trh['OA_damper_pct'][m].mean(), tr['MAT'][m].mean(),
                                    trh['MAT'][m].mean(), tr['gas_kW'][m].sum() / 4, trh['gas_kW'][m].sum() / 4))
    # one controller: house pre-cooling under TOU
    a, b = WEEKS['summer']
    c_out = simulate(HOUSE, W, a, b, dt='15min', prices=TOU, controller=precool)
    show('house summer, pre-cooling controller', c_out)
    for nm, o in (('default thermostat', runs[('house', 'summer')]), ('pre-cooling', c_out)):
        ts = o['ts']
        pk = ts['price'] > 0.3
        print(f"  {nm:18s}: {ts['meter_electricity_kW'].sum() / 4:.0f} kWh, cost ${(ts['meter_electricity_kW'] * ts['price']).sum() / 4:.2f}, "
              f"peak-period {ts['meter_electricity_kW'][pk].sum() / 4:.0f} kWh")
    # plot
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex='col')
    for j, season in enumerate(WEEKS):
        for i, bname in enumerate(('house', 'office')):
            ts = runs[(bname, season)]['ts']
            ax = axs[i, j]
            ax.plot(ts.index, ts['T_zone'], 'k', lw=1, label='zone air')
            ax.plot(ts.index, ts['heat_sp'], 'r--', lw=0.7, label='heat setpoint')
            ax.plot(ts.index, ts['cool_sp'], 'b--', lw=0.7, label='cool setpoint')
            ax.plot(ts.index, ts['T_out'], 'g', lw=0.7, label='outdoor')
            ax.set_ylabel('deg C')
            ax2 = ax.twinx()
            ax2.fill_between(ts.index, 0, ts['meter_electricity_kW'], color='orange', alpha=0.3, label='electricity kW')
            if ts['meter_gas_kW'].max() > 0:
                ax2.plot(ts.index, ts['meter_gas_kW'], color='brown', lw=0.6, label='gas kW')
            ax2.set_ylabel('kW')
            ax.set_title(f'{bname}, {season} week (Chicago TMYx)')
            if i == 0 and j == 0:
                ax.legend(loc='upper left', fontsize=7)
                ax2.legend(loc='upper right', fontsize=7)
    ax = axs[2, 0]
    tsf = f_out['ts']
    ax.plot(tr.index, tr['OA_damper_pct'], 'm', lw=0.8, label='OA damper % (faulted)')
    ax.plot(trh.index, trh['OA_damper_pct'], 'm:', lw=0.8, label='OA damper % (healthy)')
    ax.plot(tr.index, tr['MAT'], 'c', lw=0.8, label='mixed air C (faulted)')
    ax.plot(trh.index, trh['MAT'], 'c:', lw=0.8, label='mixed air C (healthy)')
    ax.axvline(np.datetime64('2019-01-16T12:00'), color='k', lw=0.5)
    ax.set_title('office: OA damper stuck open from Wednesday (trend log)')
    ax.legend(fontsize=7)
    ax = axs[2, 1]
    for nm, o, st in (('default', runs[('house', 'summer')], ':'), ('pre-cooling', c_out, '-')):
        ax.plot(o['ts'].index, o['ts']['meter_electricity_kW'], st, lw=0.9, label=f'electricity kW, {nm}')
    ax.plot(c_out['ts'].index, c_out['ts']['price'] * 10, 'r', lw=0.6, label='price x10 $/kWh')
    ax.set_title('house: pre-cooling under a 16-21 h time-of-use peak')
    ax.legend(fontsize=7)
    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_weeks.png')
    plt.savefig(path, dpi=80)
    print(f'\nplot -> {path}; total {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
