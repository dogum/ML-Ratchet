# FireCast offline lab

Open `index.html` in a browser. It is the page published with the release on 19 Nov 2026, self-contained: the
released weights (float16, base64), twelve seeded exam fire-days (four small, four medium, four large; the demo's
three are #0, #4 and #8), the exam, reliability, ratchet and twin-test tables. The explorer shows FireCast's map
against the fire-line baseline and the real next-day burn; the weather lab changes tomorrow's wind, humidity and
100-h fuel moisture and reruns the network in the page (a few seconds; its inputs are 8-bit, so its maps differ from
the Python forecast by 0.002-0.008 in probability on average, as the page says). Fonts come from Google Fonts when
online and fall back to system fonts offline.

Imported on 21 Nov 2026 from the published page. The host's wrapper (its `<html><head>` with a viewport style, and
a host-specific `<meta>`) was replaced by a plain document head (charset, viewport, description, title); the page's
own head links, `<style>`, body and scripts are byte for byte as published. The page's template, its JavaScript
port (`fcnet.js`) and its text (`text.json`) were not shipped in the zip; `training/src/build_page.py` and
`webdata.py` show how the page data was built.

The site instrument (`site/src/instruments/firecast/`) is a separate TypeScript port on the three fire-days of
`examples/`, with exact (not 8-bit) inputs. Rebuild its data from the repository root:

```
python models/firecast/web/export_instrument.py    # packs + reference.json -> site/public/instruments/firecast/
```

Data credits: `examples/README.md` (FEDS-MTBS CC BY 4.0; VIIRS, MTBS, gridMET public).
