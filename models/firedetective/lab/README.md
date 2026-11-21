# FireDetective offline lab

Open `index.html` in a browser (3.6 MB; fonts load from Google Fonts when online). It is the page released with v1.0
on 19 Nov 2026: the 3-member network (weights rounded to 5 significant figures) runs in your browser on all 944 exam
fires, from the input features the exam used. Pick a fire, see its burned area and FRP per half-day pass and the
three readings beside their truths, and move the lab sliders (faster growth, drier fuels, hotter fire, canopy base
height), which edit the fire's input features and rerun the network. Then the exam tables, the ratchet, the twin
test and the limits.

It was not rebuilt on import. `training/build_page.py` fills `training/template.html` with the data and
`training/fdnet.js`, but needs the exam input records that `exam.py build` writes under WORK (fire data rebuild and
LANDFIRE downloads) and `out/page_text.json` from `make_text.py` and `make_page_text.py`. The published copy is
exactly that output inside a host wrapper; `web/wrap_lab.py` gave it a plain doctype and head:

```
python training/build_page.py          # after the exam records exist under WORK
python web/wrap_lab.py                 # WORK/web/firedetective.html -> lab/index.html
```

The site's live instrument (`site/src/instruments/firedetective/`, data from `web/export_instrument.py`) is a
separate, smaller build: four fires from their raw observations, with the features recomputed in the browser.
Data credits are in the page footer and in `examples/README.md`.
