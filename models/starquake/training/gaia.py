"""Gaia DR3 radii for the exam and dev stars (independent of asteroseismology): positional match (2 arcsec, brightest)
via the ESA Gaia TAP service with an uploaded table; radius_flame and radius_gspphot from astrophysical_parameters."""
import _paths as _P  # portable paths (see _paths.py)
import json, numpy as np
from astropy.table import Table
from astroquery.gaia import Gaia
D = _P.WORK + '/data/'
sp = json.load(open(D + 'split.json')); rows = []
for k in sp['dev'] + sp['exam']:
    h = json.loads(str(np.load(f'{D}lc/{k}.npz')['hdr'])); rows.append((k, h['RA_OBJ'], h['DEC_OBJ']))
t = Table(rows=rows, names=('kic', 'ra', 'dec'))
q = """SELECT u.kic, g.source_id, g.phot_g_mean_mag, g.parallax, g.parallax_error, a.radius_flame, a.radius_flame_lower, a.radius_flame_upper, a.radius_gspphot, a.teff_gspphot,
DISTANCE(POINT(u.ra, u.dec), POINT(g.ra, g.dec)) AS dist
FROM tap_upload.kics AS u JOIN gaiadr3.gaia_source AS g ON 1 = CONTAINS(POINT(u.ra, u.dec), CIRCLE(g.ra, g.dec, 2.0/3600))
LEFT JOIN gaiadr3.astrophysical_parameters AS a ON a.source_id = g.source_id"""
r = Gaia.launch_job_async(q, upload_resource=t, upload_table_name='kics').get_results()
best = {}
for x in r:
    k = int(x['kic'])
    if k not in best or x['phot_g_mean_mag'] < best[k]['G']:
        f = lambda v: None if np.ma.is_masked(v) else float(v)
        best[k] = dict(source_id=int(x['source_id']), G=f(x['phot_g_mean_mag']), plx=f(x['parallax']), e_plx=f(x['parallax_error']), R_flame=f(x['radius_flame']), R_flame_lo=f(x['radius_flame_lower']), R_flame_hi=f(x['radius_flame_upper']), R_gsp=f(x['radius_gspphot']), teff_gsp=f(x['teff_gspphot']))
json.dump(best, open(D + 'gaia.json', 'w')); print('matched', len(best), 'of', len(rows), 'with R_flame', sum(v['R_flame'] is not None for v in best.values()))
