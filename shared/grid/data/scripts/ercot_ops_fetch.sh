#!/bin/bash
# download ERCOT monthly Operations Reports (public docx), gently
cd "${ML_RATCHET_GRID:-$(cd "$(dirname "$0")/../.." && pwd)}/data/raw/ercot_ops"
while IFS=$'\t' read -r m u; do
  [ -s "$m.docx" ] && file "$m.docx" | grep -q "Word" && continue
  sleep 8; curl -sS -L -m 120 -o "$m.docx" "$u"; echo "$m $(file -b $m.docx | cut -c1-25)"
done < urls.tsv
