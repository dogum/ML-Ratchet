#!/bin/bash
# Resumable download of Building Data Genome 2 (CC BY-SA 4.0) from GitHub LFS media URLs.
M=https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master
R=https://raw.githubusercontent.com/buds-lab/building-data-genome-project-2/master
D="${ML_RATCHET_BUILDING:-$(cd "$(dirname "$0")/../.." && pwd)}/data/raw/bdg2"   # data root: $ML_RATCHET_BUILDING or shared/building
mkdir -p "$D/raw" "$D/cleaned"
get() { # url dest
  if [ -f "$2.ok" ]; then echo "skip $2"; return; fi
  curl -sS -L --retry 5 -C - -o "$2" "$1" && touch "$2.ok" && echo "got $2 $(stat -c %s $2)"
}
get $R/LICENSE $D/LICENSE.txt
get $R/README.md $D/README.md
get $M/data/metadata/metadata.csv $D/metadata.csv
get $M/data/weather/weather.csv $D/weather.csv
for m in electricity chilledwater steam hotwater gas water irrigation solar; do
  get $M/data/meters/raw/$m.csv $D/raw/$m.csv
  get $M/data/meters/cleaned/${m}_cleaned.csv $D/cleaned/${m}_cleaned.csv
done
echo DONE
