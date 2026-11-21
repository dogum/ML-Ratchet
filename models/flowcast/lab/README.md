# FlowCast offline lab

Open `index.html` in a browser. It holds the released ungauged and nowcast weights, the three real exam rivers of
`examples/`, embedded fonts and the exam and ratchet tables. Pick a river, click the chart to move the cursor, and
change the rain, temperature, PET, a storm, the basin attributes or the climate statics: the network reruns on all
731 days in the page. Nowcast mode reads the gauge up to the cursor (issue) day and shows the 1-7 day forecasts
against persistence.

The site uses the same numerical and UI source in `site/src/instruments/flowcast/`. Rebuild from the repository root:

```
python models/flowcast/web/export_instrument.py    # packs + reference.json -> site/public/instruments/flowcast/
node models/flowcast/web/build_lab.mjs             # -> models/flowcast/lab/index.html (needs site/node_modules)
```

Data credits: `examples/README.md` (Caravan CC BY 4.0, ERA5-Land Copernicus, USGS public domain).
