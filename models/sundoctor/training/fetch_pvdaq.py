"""Download the two real PVDAQ systems the exams read (added 15 Nov 2026 to rebuild work/ from scratch): the 2023
Solar Data Prize copies of systems 2107 and 9068 on the OEDI data lake (CC BY 4.0), about 1.9 GB, into work/data/.
Then run prep_2107.py, prep_9068.py, realdata.py and real9068.py as before."""
import _paths as _P  # portable paths (see _paths.py)
import os, subprocess
BASE = 'https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize/'
FILES = ['2107_OEDI/data/2107_electrical_data.csv', '2107_OEDI/data/2107_irradiance_data.csv', '2107_OEDI/data/2107_environment_data.csv',
         '9068_OEDI/data/9068_ac_power_data.csv', '9068_OEDI/data/9068_tracker_data.csv', '9068_OEDI/data/9068_irradiance_data.csv',
         '9068_OEDI/data/9068_environment_data.csv', '9068_OEDI/data/9068_dc_combiner_data.csv']

if __name__ == '__main__':
    D = _P.WORK + '/data/'; os.makedirs(D, exist_ok=True)
    for f in FILES:
        out = D + os.path.basename(f)
        if os.path.exists(out): print('have', out); continue
        r = subprocess.run(['curl', '-sS', '-f', '-L', '-m', '3600', '-o', out + '.part', BASE + f], capture_output=True, text=True)
        if r.returncode: print('FAIL', f, r.stderr.strip()); continue
        os.replace(out + '.part', out); print('ok', out, os.path.getsize(out) // 2 ** 20, 'MB', flush=True)
