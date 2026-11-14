"""Inline pd.js, data.js and the text blocks into web/plantdetective.html."""
from common import *
import json
T = json.load(open(f'{ROOT}/out/exam_tables.json'))
yr = {r['kind']: r for r in T['ctx']}; yo = {r['kind']: r for r in T['out']}
lede = (f"Graded on {yr['pooled']['units']} exam units with enough full-load hours, against truth computed from each unit's own "
        f"CAMPD record over 2018-2025. With every year pooled, the Detective puts minimum stable load within "
        f"{100 * yr['pooled']['pmin_frac']:.1f} points (median) and the 99th-percentile ramp within {100 * yr['pooled']['ramp_up']:.0f} %; "
        f"the simple output statistics are off by {100 * yr['pooled']['pmin_frac|b_stats']:.0f} points and {100 * yr['pooled']['ramp_up|b_stats']:.0f} %. "
        f"From output alone it names the technology right {100 * yo['pooled']['tech_acc']:.0f} % of the time. Its heat rates are good for coal "
        f"and gas steam units (about 5 %) but biased high for gas turbines and combined cycles, where EIA-923 or a plain technology default does better. "
        f"Counting starts and run lengths needs no network: with a year or more of history the plain counts win.")
limits = [
    "Combined-cycle heat rate is 21 % too high (median). The simulator assumed, from the 3 dev CC plants (two cogeneration plants, one reporting only its gas turbine), that many CC units report gas-turbine-only output and burn steam for industry. Most exam CC units report the whole block at about 7 MMBtu/MWh. A plain 7.3 default beats the Detective here (7 % vs 21 %).",
    "Gas-turbine heat rate is 15 % high (median). The simulated CT range (8.8-13.5) sits above the Texas fleet (median 9.9), and the network does not lean on the EIA-923 number enough; EIA-923 alone gets 6.5 %. A rejected rung (R3) tried a skip connection to EIA-923 and failed the dev gate.",
    "Starts, run lengths and start types are counted, not inferred: for a year or more of history the plain counts of the output series are as good as it gets (they are the truth's own definition). The network only helps for a month or a quarter of history, and only for starts per year.",
    "The minimum-up and minimum-down times (5th percentile of spell lengths) are noisy targets: one trip or test changes them. Error bars cover them worst (48-53 % inside the 68 % band).",
    "Dev calibration rests on 10 plants. Their mix (many cogeneration units) is not the exam's. The ramp truth is what a unit did in an hour, not its rated capability.",
    "Prices are ERCOT hub day-ahead prices and Henry Hub gas, not the unit's own node or fuel contract. Units outside ERCOT (SPP, MISO, El Paso) were graded with ERCOT prices.",
]
txt = ("document.getElementById('examLede').textContent = " + json.dumps(lede) + ";\n"
       "document.getElementById('limitList').innerHTML = " + json.dumps(''.join(f'<li>{l}</li>' for l in limits)) + ";")
html = open(f'{ROOT}/web/page.html').read()
html = html.replace('<script>/*PDJS*/</script>', '<script>' + open(f'{ROOT}/web/pd.js').read() + '</script>')
html = html.replace('<script>/*PDDATA*/</script>', '<script>' + open(f'{ROOT}/web/data.js').read() + '</script>')
html = html.replace('/*TEXT*/', txt)
open(f'{ROOT}/web/plantdetective.html', 'w').write(html)
json.dump(dict(lede=lede, limits=limits), open(f'{ROOT}/out/page_text.json', 'w'), indent=1)
print('html', len(html) / 1e6, 'MB')
