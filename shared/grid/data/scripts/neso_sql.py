"""Gentle helper for the NESO CKAN datastore_search_sql API (documented: api.neso.energy/api/3/action/...)."""
import requests, time, json
API = 'https://api.neso.energy/api/3/action/datastore_search_sql'
def sql(q, tries=4, pause=2.0, timeout=600):
    last = None
    for a in range(tries):
        try:
            r = requests.post(API, json={'sql': q}, timeout=timeout)
            if r.status_code == 200:
                d = r.json()
                if d.get('success'): time.sleep(pause); return d['result']['records']
                last = d
            else:
                last = (r.status_code, r.text[:300])
                if r.status_code in (400, 409): break   # query error: do not retry
        except Exception as e:
            last = repr(e)
        time.sleep(20 * (a + 1))
    raise RuntimeError(f'NESO SQL failed: {last}')
