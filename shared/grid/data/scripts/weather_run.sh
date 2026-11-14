#!/bin/bash
cd "${ML_RATCHET_GRID:-$(cd "$(dirname "$0")/../.." && pwd)}/data"
for s in power asos build; do nice -n 5 python3 -I scripts/weather.py $s; done
