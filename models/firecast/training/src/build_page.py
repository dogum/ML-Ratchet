"""Assemble web/firecast.html from web/page_tpl.html + fcnet.js + data (examples, weights, exam, ratchet, twin).
Usage: python3 build_page.py <rung>"""
from common import *  # noqa


def main(rung):
    tpl = open(f'{ROOT}/web/page_tpl.html').read()
    js = open(f'{ROOT}/web/fcnet.js').read().replace('if (typeof module', '//').replace("!== 'undefined') module.exports = FCNet;", '')
    ex = json.load(open(f'{ROOT}/web/examples.json'))
    w = json.load(open(f'{ROOT}/web/weights.json'))
    S = json.load(open(f'{OUT}/exam/summary_{rung}.json' if os.path.exists(f'{OUT}/exam/summary_{rung}.json') else shipped(f'exam_summary_{rung}.json')))
    R = json.load(open(f'{ROOT}/ratchet.json' if os.path.exists(f'{ROOT}/ratchet.json') else f'{TRAIN}/ratchet.json'))
    tw = json.load(open(shipped('twin_compare_p0_p1.json')))
    cal = json.load(open(shipped(f'cal_{rung}_F.json')))
    T = json.load(open(f'{ROOT}/web/text.json'))
    E0 = json.load(open(shipped('eval_r0.json')))
    rungs = [dict(rung='r0', change='Base: p0 physics bank (668 fires), U-Net 374k, 1,800 steps', sim=E0['sim_p1_F']['ap_mean'],
                  dev_ap=E0['dev_F']['ap_mean'], dev_f1=E0['dev_F']['f1_mean'], kept=True)]
    for r in R['rungs'][1:]:
        rungs.append(dict(rung=r['rung'], change=r['change'], sim=r['sim_ap_F'], dev_ap=r['dev_ap_F'], dev_f1=r['dev_f1_F'], kept=r['kept']))
    # hero: the large-stratum example with the median FireCast AP among the large ones (rule, not taste)
    large = [i for i, e in enumerate(ex) if e['meta']['stratum'] == 'large']
    large.sort(key=lambda i: ex[i]['meta']['firecast']['ap'])
    hero = large[len(large) // 2]
    D = dict(ex=ex, w=w, gamma=[e['meta']['gamma'] for e in ex], T=cal['T'], b=cal['b'], thr=cal['thr'],
             exam={k: S[k] for k in ('all', 'early', 'main', 'recent')}, replay=S['replay_subset'],
             rel=S['reliability']['F'], bysize=S['by_size'], rungs=rungs, final=rung, frozen='p1',
             twin=tw, text=T, hero=hero)
    html = tpl.replace('/*FCNET*/', js).replace('/*DATA*/null', json.dumps(D, separators=(',', ':')))
    open(f'{ROOT}/web/firecast.html', 'w').write(html)
    print('firecast.html', len(html) / 1e6, 'MB; hero', hero)


if __name__ == '__main__':
    main(sys.argv[1])
