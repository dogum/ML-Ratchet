"""Real earthquakes and stations for QuakeNet: the USGS catalogue of M5.5+ events (2012-2025) and broadband stations of
the Global Seismographic Network (networks IU and II) with 1-sample-per-second (LH) channels."""
import _paths as _P  # portable paths (see _paths.py)
import json, urllib.request, numpy as np
from obspy.clients.fdsn import Client
D = _P.WORK + '/data/'
def events():
    ev = []
    for y in range(2012, 2026):
        u = f'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&minmagnitude=5.5&starttime={y}-01-01&endtime={y + 1}-01-01&orderby=time-asc'
        js = json.load(urllib.request.urlopen(u, timeout=120))
        for f in js['features']:
            p = f['properties']; lon, lat, dep = f['geometry']['coordinates']
            ev.append(dict(id=f['id'], time=p['time'] / 1000, lat=lat, lon=lon, depth=dep, mag=p['mag'], magType=p['magType'], place=p['place']))
        print(y, len(ev), flush=True)
    json.dump(ev, open(D + 'events.json', 'w'))
def stations():
    inv = Client('EARTHSCOPE').get_stations(network='IU,II', channel='LHZ', level='channel', starttime='2012-01-01')
    out = {}
    for net in inv:
        for sta in net:
            chans = sorted({c.location_code for c in sta.channels if c.code == 'LHZ'})
            out[f'{net.code}.{sta.code}'] = dict(lat=sta.latitude, lon=sta.longitude, locs=chans, start=str(sta.start_date), end=str(sta.end_date))
    json.dump(out, open(D + 'stations.json', 'w')); print(len(out), 'stations')
if __name__ == '__main__':
    events(); stations()
