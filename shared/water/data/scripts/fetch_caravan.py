"""Stream the Caravan v1.6 netCDF tarball from Zenodo and keep only what we need.

Zenodo record 15529786 (doi:10.5281/zenodo.15529786), Caravan-nc.tar.gz, 24.8 GB, CC-BY 4.0.
The tarball is never written to disk: it is read once as a gzip stream (a reconnecting HTTP reader resumes with a
Range request from the current byte offset if the connection drops) and each wanted member is converted on the fly.

Kept:
  attributes/*, licenses/*, VERSION  -> data/raw/caravan/...                    (as published)
  timeseries/netcdf/<region>/*.nc    -> data/proc/caravan/<region>/<gauge_id>.parquet  (contract columns, float32)
Regions: camels (US, all 671, 1951-2023), camelsgb, camelsaus, camelscl, camelsbr (1981-2023, fewer columns).

Restart-safe: members whose parquet already exists are skipped (the stream still has to be read up to them).
The archive is untrusted data: nothing from it is executed; member names are sanitised before writing.
"""
import io, json, os, sys, time, tarfile, re
import numpy as np, pandas as pd, requests
import pyarrow as pa, pyarrow.parquet as pq
from scipy.io import netcdf_file
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root

URL = 'https://zenodo.org/api/records/15529786/files/Caravan-nc.tar.gz/content'
ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
RAW = f'{ROOT}/raw/caravan'
PROC = f'{ROOT}/proc/caravan'
PROG = f'{ROOT}/logs/caravan_progress.json'
US = 'camels'
OTHER = ['camelsgb', 'camelsaus', 'camelscl', 'camelsbr']
WANT = [US] + OTHER
EXPECT = {'camels': 671}


class ReconnectingReader(io.RawIOBase):
    def __init__(self, url, chunk=1 << 20):
        self.url, self.pos, self.chunk = url, 0, chunk
        self.resp, self.it, self.buf = None, None, b''
        self.total = None
        self._open()

    def _open(self):
        for attempt in range(30):
            try:
                h = {'Range': f'bytes={self.pos}-'} if self.pos else {}
                r = requests.get(self.url, stream=True, headers=h, timeout=(30, 120), allow_redirects=True)
                if self.pos and r.status_code != 206:
                    raise IOError(f'range not honoured: {r.status_code}')
                r.raise_for_status()
                if self.total is None:
                    self.total = int(r.headers.get('Content-Length', 0)) + self.pos
                self.resp, self.it = r, r.iter_content(self.chunk)
                return
            except Exception as e:
                print(f'[reader] open failed at {self.pos}: {e}; retry {attempt}', flush=True)
                time.sleep(min(300, 10 * (attempt + 1)))
        raise IOError('could not reopen stream')

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            try:
                self.buf = next(self.it)
            except StopIteration:
                return 0
            except Exception as e:
                print(f'[reader] stream error at {self.pos}: {e}; reconnecting', flush=True)
                try:
                    self.resp.close()
                except Exception:
                    pass
                time.sleep(5)
                self._open()
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        self.pos += n
        return n


def g(v, k):
    return np.asarray(v[k][:], dtype=np.float32)


class _H5Vars:
    """Minimal netCDF4/HDF5 reader with the scipy netcdf_file .variables interface (Caravan v1.6 CAMELS-AUS files
    are netCDF4, the others netCDF3)."""
    def __init__(self, data):
        import h5py
        self.f = h5py.File(io.BytesIO(data), 'r')
        self.variables = {k: self.f[k] for k in self.f.keys()}

    def close(self):
        self.f.close()


def _dates(v):
    d = v['date']
    units = d.attrs.get('units', b'days since 1951-01-01') if hasattr(d, 'attrs') else getattr(d, 'units', b'days since 1951-01-01')
    units = units.decode() if isinstance(units, bytes) else str(units)
    origin = pd.Timestamp(units.split('since')[1].strip()[:10])
    step = units.split('since')[0].strip()
    vals = np.asarray(d[:], dtype=np.int64)
    return origin + pd.to_timedelta(vals, unit={'days': 'D', 'hours': 'h'}.get(step, 'D'))


def convert(data, region):
    if data[:4] == b'\x89HDF':
        f = _H5Vars(data)
    else:
        f = netcdf_file(io.BytesIO(data), 'r', mmap=False)
    v = f.variables
    idx = _dates(v)
    u, w = g(v, 'u_component_of_wind_10m_mean'), g(v, 'v_component_of_wind_10m_mean')
    Td = g(v, 'dewpoint_temperature_2m_mean')
    cols = {
        'P': g(v, 'total_precipitation_sum'),                      # mm/day (ERA5-Land)
        'T_mean': g(v, 'temperature_2m_mean'),                     # degC
        'T_min': g(v, 'temperature_2m_min'),
        'T_max': g(v, 'temperature_2m_max'),
        'SRAD': g(v, 'surface_net_solar_radiation_mean'),          # W/m2, NET shortwave (ERA5-Land)
        'LRAD_net': g(v, 'surface_net_thermal_radiation_mean'),    # W/m2, net longwave (negative = loss)
        'vp': (610.94 * np.exp(17.625 * Td / (Td + 243.04))),      # Pa, from mean dewpoint (Alduchov-Eskridge)
        'wind': np.hypot(u, w),                                    # m/s, |mean vector| (lower bound of mean speed)
        'PET': g(v, 'potential_evaporation_sum_FAO_PENMAN_MONTEITH'),  # mm/day, FAO-56 PM (Caravan v1.5+)
        'pressure': g(v, 'surface_pressure_mean'),                 # kPa
        'SWE_era5': g(v, 'snow_depth_water_equivalent_mean'),      # mm (ERA5-Land model state, not observed)
        'sm1': g(v, 'volumetric_soil_water_layer_1_mean'),         # m3/m3, 0-7 cm (ERA5-Land)
    }
    if region == US:
        for i in (2, 3, 4):
            cols[f'sm{i}'] = g(v, f'volumetric_soil_water_layer_{i}_mean')
    cols['Q_mm'] = g(v, 'streamflow')                              # mm/day (observed, from source dataset)
    df = pd.DataFrame({k: np.asarray(x, dtype=np.float32) for k, x in cols.items()}, index=idx)
    df.index.name = 'date'
    if region != US:
        df = df.loc['1981-01-01':]
    f.close()
    return df


def main():
    os.makedirs(RAW, exist_ok=True)
    for r in WANT:
        os.makedirs(f'{PROC}/{r}', exist_ok=True)
    rd = ReconnectingReader(URL)
    stream = io.BufferedReader(rd, buffer_size=1 << 20)
    counts = {r: 0 for r in WANT}
    t0 = time.time(); last = 0
    seen_regions = []
    with tarfile.open(fileobj=stream, mode='r|gz') as tar:
        for m in tar:
            name = m.name
            if '..' in name or name.startswith('/'):
                continue
            parts = name.split('/')
            keep_shape = (len(parts) >= 4 and parts[1] == 'shapefiles' and parts[2] in WANT)
            if m.isfile() and len(parts) >= 3 and (parts[1] in ('attributes', 'licenses') or keep_shape):
                out = os.path.join(RAW, *[re.sub(r'[^\w.\-]', '_', p) for p in parts[1:]])
                os.makedirs(os.path.dirname(out), exist_ok=True)
                if not os.path.exists(out):
                    open(out, 'wb').write(tar.extractfile(m).read())
            elif m.isfile() and len(parts) == 2 and parts[1] in ('VERSION', 'README.md'):
                open(os.path.join(RAW, parts[1]), 'wb').write(tar.extractfile(m).read())
            elif m.isfile() and len(parts) == 5 and parts[1] == 'timeseries' and parts[2] == 'netcdf':
                region = parts[3]
                if region not in seen_regions:
                    seen_regions.append(region)
                    print(f'[tar] region {region} starts at byte {rd.pos/1e9:.2f} GB', flush=True)
                if region in WANT and name.endswith('.nc'):
                    gid = re.sub(r'[^\w\-]', '_', parts[4][:-3])
                    out = f'{PROC}/{region}/{gid}.parquet'
                    if not os.path.exists(out):
                        raw = tar.extractfile(m).read()
                        try:
                            df = convert(raw, region)
                        except Exception as e:
                            print(f'[convert] FAILED {name}: {e}', flush=True)
                            fd = f'{RAW}/failed'; os.makedirs(fd, exist_ok=True)
                            if len(os.listdir(fd)) < 3:
                                open(f'{fd}/{gid}.nc', 'wb').write(raw)
                            continue
                        tmp = out + '.tmp'
                        pq.write_table(pa.Table.from_pandas(df), tmp, compression='zstd', compression_level=9,
                                       use_byte_stream_split=True)
                        os.replace(tmp, out)
                    counts[region] += 1
            elif m.isfile() and parts[1] not in ('timeseries',):
                print(f'[tar] skipped member {name} ({m.size} B)', flush=True)
            if time.time() - last > 60:
                last = time.time()
                prog = {'bytes': rd.pos, 'total': rd.total, 'frac': rd.pos / max(1, rd.total or 1),
                        'elapsed_s': round(time.time() - t0), 'counts': counts, 'regions_seen': seen_regions,
                        'last_member': name}
                json.dump(prog, open(PROG, 'w'), indent=1)
                print(json.dumps(prog), flush=True)
    json.dump({'done': True, 'bytes': rd.pos, 'counts': counts, 'regions_seen': seen_regions,
               'elapsed_s': round(time.time() - t0)}, open(PROG, 'w'), indent=1)
    print('DONE', counts, flush=True)


if __name__ == '__main__':
    main()
