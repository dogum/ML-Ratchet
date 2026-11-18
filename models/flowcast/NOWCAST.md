# FlowCast nowcast mode (optional, GAUGED protocol)

`flowcast_nowcast_weights.npz`: rung r1b fine-tuned for 2,500 steps on the same simulated bank with three extra
inputs: the gauge-like flow `lead` days earlier (log scale, in units of mean precipitation), an availability flag
(gaps simulated in 30 % of training windows) and lead/7. One network serves leads 1-7 days. It uses the basin's own
recent observed flow, so it is a gauged method; compare it with persistence (and NWM, which is a simulation, not a
forecast). Bands are the raw pinball quantiles (no dev widening was fitted for this mode).

```python
m = fc.FlowCast('flowcast_nowcast_weights.npz')
out = m.predict(forcing_df, attrs, '1995-10-01', '1996-09-30', qobs=observed_mm_per_day_series, lead=3)
```

| Lead | Dev NSE (WY2001-14) | Main exam NSE 1990-99 | Persistence | Recent exam NSE 2016-23 | Persistence |
|---|---|---|---|---|---|
| 1 day | 0.633 (persistence 0.549) | **0.569** | 0.475 | **0.591** | 0.480 |
| 3 days | 0.567 | **0.474** | −0.190 | **0.511** | −0.210 |
| 7 days | 0.520 | **0.426** | −0.465 | **0.466** | −0.458 |

Medians over 619 (main) and 603 (recent) exam basins, scored once. NWM v3.0 (no gauge input at run time): 0.550 main,
0.687 recent. KGE is lower than persistence at lead 1 (0.635 vs 0.738): persistence copies the variance exactly.
"Forecast" here means the observed weather of the target days is known (a hindcast); real forecasts would use
forecast weather.
