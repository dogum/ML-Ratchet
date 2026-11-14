#!/bin/bash
cd "${ML_RATCHET_GRID:-$(cd "$(dirname "$0")/../.." && pwd)}/data"
for s in inertia genmix scan; do nice -n 5 python3 -I scripts/neso_fetch.py $s; done
