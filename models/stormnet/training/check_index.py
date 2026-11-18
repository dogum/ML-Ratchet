"""Cross-check one final month from the WDC publication against NASA OMNI2 word 41."""
import hashlib
from urllib.request import urlopen

import numpy as np

from noise import RESULTS, read_year, write_json

URL = "https://wdc.kugi.kyoto-u.ac.jp/dst_final/201103/dst1103.for.request"


def main():
    with urlopen(URL, timeout=30) as response:
        data = response.read()
    days = data.decode("ascii").rstrip().splitlines()
    values = []
    for day, row in enumerate(days, 1):
        if row[:7] != "DST1103" or int(row[8:10]) != day or int(row[16:20]) != 0:
            raise ValueError("Unexpected Kyoto date or base value")
        values.extend(int(row[i:i + 4]) for i in range(20, 116, 4))
    time, _, index, _ = read_year(2011, index=True)
    month = index[(time >= np.datetime64("2011-03-01")) & (time < np.datetime64("2011-04-01"))]
    if len(values) != 744 or not np.array_equal(values, month):
        raise ValueError("Final Kyoto Dst differs from OMNI on the hourly UTC grid")
    write_json(RESULTS / "index_check.json", dict(url=URL, sha256=hashlib.sha256(data).hexdigest(),
               hours=len(values), max_abs_nT=0, source_clock="UTC interval start; issue is interval end"))


if __name__ == "__main__":
    main()
