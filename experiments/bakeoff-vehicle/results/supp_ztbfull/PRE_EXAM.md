# Full-ZTBus supplementary exam: state before the exam (25 Nov 2026, written before `exam_supp.py ztbfull --run`)

Why: the cloud machine had only RoadData v1's ZTBus subset (78 legs of 8 exam missions), so the main exam scored no
full-ZTBus rows (handoff item 21). This machine has the full set (proc/hd/ztbus_full, 7,567 legs of 1,409 missions;
splits from proc/hd/ztbus_full_splits.json: 294 dev, 1,115 exam missions).

**What it scores.** Every examined arm (21) plus the two TabPFN arms (after their own supplement), on all 5,905 exam
legs of the 1,115 exam missions and one standstill layover record per mission. Legs are built as build_real.hd builds
the v1 ZTBus legs (bus, BEV, the 19 t curb mass as reported mass); layovers as EnergyNet's realfeat.ztb. Three boards:
- legs: the 5,847 labelled legs with energy > 0 (EnergyNet's filter; clusters = missions);
- missions, the bake-off's rule: the sum of a mission's labelled legs against their summed label;
- missions, EnergyNet's rule (evaluate.ztb_missions): every leg plus the layover against the mission's metered kWh.
The dev-median reference uses the bake-off's rule on legs (pooled dev kWh/km x leg km) and EnergyNet's baseline on
missions (pooled dev kWh/km x mission driven km).

**Rehearsal on dev (no exam data; 1,641 labelled legs, 21 unlabelled, 294 missions).** EnergyNet v2.1 reproduces its
published dev terms: legs MdAPE 0.0699 (published 0.06985, ratchet2.json r17), missions by EnergyNet's rule 0.0557
(published 0.05567). Other rows on dev (legs / missions bake-off rule / missions EnergyNet rule): MLP 0.067 / 0.053 /
0.056; GRU 0.066 / 0.052 / 0.057; formula 0.095 / 0.079 / 0.073; climatology 0.150 / 0.124 / 0.114; dev median
0.128 / 0.094 / 0.088. Three rehearsal bugs were fixed before the exam: the record set is named 'hd' (the scorer and
the reference rows read ZTBus legs from it), the records carry empty fuel / cycle columns (the reference code reads
them), and the VED showcase is skipped.

**Target to check on the exam:** EnergyNet v2.1's published full-ZTBus exam (exam3.json v2): legs 0.0704 (n 5,847),
missions 0.0559 (n 1,115).
