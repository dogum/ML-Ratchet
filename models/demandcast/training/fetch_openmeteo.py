"""Day-ahead forecast weather from the Open-Meteo Previous Runs API (documented, free non-commercial, CC BY 4.0).

For each weather point we fetch the latest-run value (lead ~0 h, 'day0') and the value forecast one day earlier
('previous_day1', lead ~24-48 h). DemandCast uses the difference (day1 - day0) as a real forecast error and adds it to
the NASA POWER / ASOS actuals, so forecast-mode inputs keep the actual-weather climate but carry real forecast error.
Coverage found on 13 Nov 2026: temperature_2m_previous_day1 from 2022; humidity, radiation and wind from 2024.
  2022 (training-input year): pop-point temperature only -> real temperature forecast-error library for the simulator
  2023: pop-point temperature only; 2024-2025: everything -> exam in forecast mode
Cache: data/openmeteo/<pid>_<year>.json (resumable).
"""
import os, sys, json, time, urllib.request, urllib.parse
from common import *

OUT = f'{DATA}/openmeteo'
URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'


def vars_for(kind, year):
    if year <= 2023:
        return ['temperature_2m', 'temperature_2m_previous_day1'] if kind == 'pop' else []
    v = {'pop': ['temperature_2m', 'relative_humidity_2m', 'shortwave_radiation'],
         'wind': ['wind_speed_10m', 'wind_speed_80m', 'temperature_2m'],
         'solar': ['shortwave_radiation', 'temperature_2m']}[kind]
    return v + [x + '_previous_day1' for x in v]


def fetch(p, year):
    f = f'{OUT}/{p["pid"]}_{year}.json'
    if os.path.exists(f): return 'cached'
    v = vars_for(p['kind'], year)
    if not v: return 'skip'
    q = dict(latitude=p['lat'], longitude=p['lon'], hourly=','.join(v), start_date=f'{year}-01-01',
             end_date=f'{year}-12-31', timezone='GMT', wind_speed_unit='ms')
    req = urllib.request.Request(URL + '?' + urllib.parse.urlencode(q), headers={'User-Agent': 'research-client'})
    for attempt in range(4):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=120))
            if 'hourly' in d:
                json.dump(d['hourly'], open(f, 'w')); return 'ok'
            return 'err ' + str(d)[:200]
        except Exception as e:
            msg = str(e)
            if '429' in msg or 'limit' in msg.lower():
                return 'limit ' + msg[:120]
            time.sleep(20 * (attempt + 1))
    return 'fail'


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    # rate limits on the shared proxy IP allow ~50 year-requests an hour: fetch the heaviest-weighted points only
    TOPK = {'pop': 4, 'wind': 3, 'solar': 3}
    for kind, year in (('pop', 2024), ('wind', 2024), ('solar', 2024), ('pop', 2025), ('wind', 2025), ('solar', 2025), ('pop', 2022)):
        if True:
            for ba in BAS:
                pts = sorted([p for p in gd.weather_points(ba)['points'] if p['kind'] == kind], key=lambda p: -p['w'])
                for p in pts[:TOPK[kind]]: jobs.append((p, year))
    waits = 0
    for p, year in jobs:
        while True:
            r = fetch(p, year)
            print(time.strftime('%H:%M'), p['pid'], year, r, flush=True)
            if r.startswith('limit') and waits < 80:
                waits += 1; time.sleep(900); continue
            break
        if r == 'ok': time.sleep(30)
