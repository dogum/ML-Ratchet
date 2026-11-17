"""Assemble web/loadcast.html from page.html + lcnet.js + data_<run>.json + weights_<run>.json + summaries.

    python3 build_page.py <run>
"""
import sys, os, json
import lc_common as C

ROWS = [dict(group='LoadCast (forecast 2017 from 2016 and actual 2017 weather)',
             rows=['LoadCast cold', 'LoadCast warm (last 2 weeks)', 'LoadCast warm (last 8 weeks)', 'LoadCast warm (2016)']),
        dict(group='Baselines fitted on 2016 (BuildingData)', rows=['TOWT (B)', 'change-point hourly', 'dumb cold (dev median)']),
        dict(group='Nowcasts that read the real 2017 meter (not forecasts)', rows=['persistence 24 h*', 'same hour last week*', 'persistence 1 h*']),
        dict(group='Daily and monthly totals', rows=['LoadCast warm (2016), daily', 'change-point daily', 'LoadCast warm (2016), monthly', 'change-point monthly'])]


def main(run):
    R = f'{C.ROOT}'
    ex = json.load(open(f'{R}/exam/exam_{run}.json'))
    rat = json.load(open(f'{R}/ratchet.json'))
    cal = json.load(open(f'{R}/data/calib_summary.json'))
    extra = json.load(open(f'{R}/web/page_text.json'))
    summ = dict(exam=ex, exam_rows=ROWS, calib=cal,
                ratchet=[dict(run=r['run'], note=r['note'].split(': ', 1)[-1], accepted=r['accepted'],
                              obj=r['frozen']['objective'], dev=r['dev']['score'], why=r['why']) for r in rat],
                **extra)
    page = open(f'{R}/web/page.html').read()
    page = page.replace('/*__LCNET__*/', open(f'{R}/web/lcnet.js').read())
    page = page.replace('/*__DATA__*/null', open(f'{R}/web/data_{run}.json').read())
    page = page.replace('/*__WEIGHTS__*/null', open(f'{R}/web/weights_{run}.json').read())
    page = page.replace('/*__SUMMARY__*/null', json.dumps(summ, default=float).replace('NaN', 'null'))
    out = f'{R}/web/loadcast.html'
    open(out, 'w').write(page)
    print(out, os.path.getsize(out) / 1e6, 'MB')


if __name__ == '__main__':
    main(sys.argv[1])
