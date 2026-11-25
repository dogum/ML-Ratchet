# r2_x_cross: Cross-family ensemble: transformer + GRU + MLP + GP, equal weights (round 2, X)

*Vehicle Bake-off round 2 (24-25 Nov 2026). Pre-registration: `round2/PREREG.md`. Track: trace.*

**What.** The Vehicle Bake-off's next-rung 4: one forecast from four learner families at once. The transformer and GRU ensembles read the trip trace; the MLP and the sparse GP read EnergyNet's 151 physics features, rebuilt from the same trace batch. Their 5 log-quantiles are averaged level by level with equal weights. It also produces the between-family spread that round 2's label-free gap detector reads (round2/gap/).

**Training.** none: the members are frozen round-2 / v1.0 arms (hashed with this folder through extra_files). The member list is in members.json.

**Rungs** (frozen sim val: 3,985 trips, frozen tiers; dev D16 = mean of the 16 dev MdAPEs):

| Rung | Change | Sim val skill | Sim val MdAPE | Dev D16 | Decision | Why |
|---|---|---|---|---|---|---|
| r0 | equal-weight cross-family ensemble of the four round-2 family entries (D2 transformer, D1 GRU, v1.0 tab_mlp, v1.0 tab_gp) | 0.6691 | 0.0776 | 0.1111 | released | registered as a base rung (PREREG section 3: D16 0.1111 is not worse than its best member's 0.1065 by more than 0.01). Pre-exam verdict: NOT a gain: sim val skill 0.6691 is below its best member's 0.6715 (the GP member, 0.596, pulls it down), D16 +0.0046 worse. |

**Compute.** no training of its own; members: D2 4.88 CPU-h / 5.21 GPU-h, D1 1.29 / 1.39, v1.0 tab_mlp 0.62, v1.0 tab_gp 0.81 single-thread CPU-h

**Inputs and features.** none of its own. Members: r2_d2_transformer and r2_d1_gru on the trace batch; v1.0 tab_mlp and tab_gp on EnergyNet's 151 table inputs rebuilt from the trace batch by round2/arms/common/xlib.py (route vector, physics vector and E_phys from the trace and rl with EnergyNet's own feature code; the vehicle vector from spec with the label entries made relative again; checked against the table batch: median |log difference| of the members' predictions 1e-6 to 8e-5, arms/check_xinputs.json). Combination fixed in PREREG section 3: each level's log-quantile is the mean of the 4 members' (weights 1/4), pred = exp(mean log q50).

**Limits.** Quantile averaging keeps each member's band width and adds no between-member spread. Equal weights let the weakest member (the GP, sim skill 0.596) pull the average below the best member on sim val. Predict cost is the sum of the four members' (two 5-network trace ensembles). The members share the simulator's misses (dev bias: plug-ins -9.7 %, Mersin CNG -18.4 %, ICCT +21.9 %), so averaging them cannot remove those.
