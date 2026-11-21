"""LANDFIRE ImageServer access (lfps.usgs.gov ArcGIS REST exportImage; public, documented). EPSG:5070, 30 m grid.
Grid alignment: LANDFIRE CONUS 30 m cell edges sit at x = 15 + 30k, y = 15 + 30k (service extent -2362425, 3177435)."""
import io, time, requests, numpy as np, rasterio
BASE = 'https://lfps.usgs.gov/arcgis/rest/services'
S = requests.Session()
VERS = {2016: 'Landfire_LF2016/LF2016_{}_CONUS', 2022: 'Landfire_LF2022/LF2022_{}_CONUS',
        2023: 'Landfire_LF2023/LF2023_{}_CONUS', 2024: 'Landfire_LF2024/LF2024_{}_CONUS'}
TOPO = {'Elev': 'Landfire_Topo/LF2020_Elev_CONUS', 'SlpD': 'Landfire_Topo/LF2020_SlpD_CONUS',
        'Asp': 'Landfire_Topo/LF2020_Asp_CONUS'}


def lf_version(fire_year):
    """Latest LANDFIRE version on the service whose disturbance year is before the fire year (pre-fire), and a flag
    when only a post-fire version exists (fires 2012-2016: LF2016 includes their own burn)."""
    if fire_year <= 2016: return 2016, True
    if fire_year <= 2022: return 2016, False
    if fire_year == 2023: return 2022, False
    if fire_year == 2024: return 2023, False
    return 2024, False


def service(layer, ver):
    return TOPO[layer] if layer in TOPO else VERS[ver].format(layer)


def export(layer, ver, x0, y1, nx, ny, res=30.0, retries=6):
    """Return int16/float array for bbox [x0, x0+nx*res] x [y1-ny*res, y1] (EPSG:5070), nearest neighbour."""
    url = f'{BASE}/{service(layer, ver)}/ImageServer/exportImage'
    p = dict(bbox=f'{x0},{y1 - ny * res},{x0 + nx * res},{y1}', bboxSR=5070, imageSR=5070, size=f'{nx},{ny}',
             format='tiff', interpolation='RSP_NearestNeighbor', f='image')
    for i in range(retries):
        try:
            r = S.get(url, params=p, timeout=(30, 300))
            if r.status_code == 200 and r.content[:2] in (b'II', b'MM'):
                with rasterio.MemoryFile(r.content) as mf, mf.open() as s:
                    a = s.read(1)
                    assert a.shape == (ny, nx), a.shape
                    return a
            err = r.text[:200]
        except Exception as e:
            err = repr(e)
        time.sleep(5 * (i + 1))
    raise RuntimeError(f'LANDFIRE export failed {layer} {ver}: {err}')


FUEL_GROUPS = {'NB': range(91, 100), 'GR': list(range(101, 110)) + [1, 2, 3], 'GS': range(121, 125),
               'SH': list(range(141, 150)) + [4, 5, 6, 7], 'TU': range(161, 166),
               'TL': list(range(181, 190)) + [8, 9, 10], 'SB': list(range(201, 205)) + [11, 12, 13]}


def fuel_group_codes(a):
    g = np.full(a.shape, '', object)
    for k, v in FUEL_GROUPS.items(): g[np.isin(a, list(v))] = k
    return g
