"""Page data: noise.json (lab noise slices), events.json (dev + exam GB windows with truths). results.json is
written by results.py from exam.json + ratchet.json."""
from common import *
import dataset as DS, evaluate as EV
import griddata as gd


def noise():
    nz = DS.noise_split('val'); rng = np.random.default_rng(1)
    sel = nz[rng.choice(len(nz), 40, replace=False)]
    sel = sel - sel[:, :60].mean(1, keepdims=True)
    json.dump(np.round(sel * 1000).astype(int).tolist(), open(f'{ROOT}/web/noise.json', 'w'), separators=(',', ':'))


def events():
    gm = gd.gb_genmix().set_index('time_utc')['GENERATION']
    ev = gd.events('GB'); ev = ev[ev.split.isin(['dev', 'exam'])]
    out = []
    for _, r in ev.iterrows():
        try:
            m = EV.real_gb(r.event_id)
        except Exception:
            continue
        dem = EV.DEV_DEMAND.get(r.event_id, float(gm.get(pd.Timestamp(r.t0_utc).floor('30min'), np.nan)))
        loss = EV.DEV_LOSS.get(r.event_id, r.loss_MW)
        tier = 'dev' if r.split == 'dev' else (r.loss_confidence if isinstance(r.loss_confidence, str) else None)
        out.append(dict(id=r.event_id, t=pd.Timestamp(r.t0_utc).strftime('%Y-%m-%d %H:%M'), year=int(r.year), split=r.split,
                        w=np.round(m['w'][0], 3).tolist(), f0=round(float(m['f0'][0]), 4),
                        demand=None if not np.isfinite(dem) else round(dem), E=None if pd.isna(r.inertia_outturn_GVAs) else float(r.inertia_outturn_GVAs),
                        loss=None if pd.isna(loss) else round(float(loss)), tier=tier,
                        nadir=round(float(m['nadir'][0]), 4), settle=round(float(m['settle'][0]), 4)))
    out.sort(key=lambda e: (e['split'] != 'dev', e['t']))
    json.dump(out, open(f'{ROOT}/web/events.json', 'w'), separators=(',', ':'))
    print(len(out), 'events')


if __name__ == '__main__':
    noise(); events()
