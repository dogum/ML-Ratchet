# BatteryPilot v1.3. What was known before the offer?

Independent first pass, 2026-11-25, on the released R8 policy at `61c70bfb2a27f3d592fab317ccd40a048e020b4b`.
BatteryPilot submits a five-point offer curve before each hour. The realized real-time price then clears that offer.
The decision tested below is the submitted curve in dollars per stored MWh. Dispatch and revenue can legitimately change when the clearing price changes. [Model card](../../../../models/batterypilot/MODEL_CARD.md#L19)

## Observed temperature enters the offer through planned wear

**What failed?** The explicit temperature feature reads the previous observation. The exam also computes the hourly wear-cost series from realized temperatures, supplies the current wear cost to the policy, and supplies the future portion to the day-ahead planner. The planner's value curve feeds the network. The rolling optimizer receives the same realized wear series. [Exam input path](../../../../models/batterypilot/training/evaluate.py#L11) · [Planner](../../../../models/batterypilot/training/backbone.py#L73) · [Released policy](../../../../models/batterypilot/batterypilot.py#L428)

**What changed when tested?** On ERCOT West at 14:00 local, January 2, 2024, raising the next 12 hours' temperatures by 5 °C moved the current offer by up to **$1.1877/MWh**. Adding $100/MWh to the same future real-time prices moved it by **$0**. Adding $100/MWh to the next local day's day-ahead prices first moved the offer at the scheduled 14:00 publication, by **$18.0569/MWh**. These controls distinguish access to future observations from legitimate access to published prices.

For an economic replay, the 1 MW / 2 MWh battery used the previous hour's observed temperature for current wear, starting at 25 °C. Each day-ahead plan held that estimate fixed over its planning horizon. Both strategies settled at the original real-time prices and paid wear at actual temperatures. This isolates the wear input while preserving the release's other inputs and initialization.

| Node-year | BatteryPilot released | BatteryPilot replay | Rolling optimizer released | Rolling optimizer replay |
|---|---:|---:|---:|---:|
| ERCOT Houston 2023 | 101.415 | 101.468 | 91.365 | 91.267 |
| ERCOT West 2024 | 38.725 | 38.762 | 37.595 | 37.512 |
| CAISO SP15 2025 | 10.850 | 10.848 | 11.937 | 12.077 |

Units are $/kW-year net of wear, annualized to 8,760 hours. The strategy ranking held on each node-year. The largest absolute change to BatteryPilot was $0.053/kW-year.

**What consequence remains unmeasured?** The other 45 exam node-years, other battery durations, and the effect on training or rung selection remain open. The three examples establish an information-boundary failure and a small revenue effect in this replay. They do not bound the full-exam effect.

## First-hour real-time price enters its own offer

**What failed?** At the start of a path, the history indices for real-time prices and spreads are clipped to row zero. The offer submitted for that row therefore reads its own clearing price through the missing history. [History padding](../../../../models/batterypilot/batterypilot.py#L271)

**What changed when tested?** Increasing only the first ERCOT West clearing price by $100/MWh moved the largest first-offer knot by **$3.0164/MWh** in absolute value. The original and perturbed paths had identical pre-hour information.

**What consequence remains unmeasured?** The wear replay preserved first-hour price padding in both arms. The revenue effect of replacing that padding with a pre-hour price or an explicit missing-history convention remains open.

## Rung arithmetic and reproduction

All nine candidate decisions matched the logged scores under the strict 0.01 frozen-score margin and the development gate. R0 initializes the ladder and R5 is a classical reference. R8 was the last promoted candidate. R9 and R11 gained 0.0060 and 0.0084 over R8 on the frozen score, respectively. Both improved the development score and both fell short of the frozen margin. [Rung log](../../../../models/batterypilot/training/ratchet_v1_3.json)

Run from the repository root with NumPy and SciPy. The probe loads the shipped weights and example CSVs, applies the perturbations in memory, and prints both revenue columns for each strategy.

```sh
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/batterypilot_v1_3_probe.py
```

**Review scope.** This pass covers protocol items 1, 4 and 8. The recorded discrepancies concern the pre-hour decision contract. Calibration, training reproducibility, simulator fidelity and the full market comparison have separate evidence requirements. A second independent pass and an overall model verdict are pending.
