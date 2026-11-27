# FireCast v1.0 | Final-perimeter grid crop and Mode-F boundary sensitivity

**Cross-checked evaluation limit.** Reviewer A independently reran reviewer B's `dc30c02` probe against model base `61c70bf`, verified weight health across all exported convolutions (`0` dead filters, all finite), and ran an outward-padding counterfactual to test whether the Mode-F U-Net exploits the final-perimeter grid boundary. [Original report and probe](reviews/reviewer-b/firecast-v1.0.md) · [Real-day crop](../../models/firecast/training/src/realdays.py#L77) · [Scoring rule](../../models/firecast/firecast.py#L345)

**What reproduced.** On the three shipped growth fire-days (`examples/demo_days.npz`), the archived grids exclude **44.8%** (`Hwy 113`: 22,853 of 41,375), **59.0%** (`Coon`: 20,228 of 49,375), and **0.0%** (`Pole Creek`: 162,129 of 162,129) of the nominal 10 km candidate window. All three archived AP and F1 scores reproduced (`<= 8e-8`), folded-Conv Torch↔NumPy logit parity held within `2.86e-6`, repeating day-0 weather shifted AP (`0.2963 -> 0.2838`, `0.0556 -> 0.0192`, `0.5752 -> 0.5764`), and all `p1` rung decisions reproduced.

**Boundary-exploitation test (Mode F).** To test whether the network exploits the proximity of the final-perimeter grid edge, the second-pass probe pads each shipped day outward by `+112` unburned 90 m cells on all four sides (covering 100% of the nominal 10 km window and shifting `pix_pm` / `pix_am` coordinates accordingly), runs `FireCastNP` in Mode F (no LANDFIRE), and scores both the original interior and the full 10 km window (taking exterior cells outside the final-perimeter grid as unburned):

| Fire-day (grid) | Interior `max \|dp\|` | Cells crossing `0.10` | Exterior cells added (`max p`, `>= 0.10`) | AP: orig → padded interior → full 10 km | F1: orig → full 10 km |
|---|---:|---:|---|---:|---:|
| Hwy 113 (`163×145`) | 0.0397 | 1 | 18,522 (`0.0220`, `0`) | 0.2963 → 0.3014 → 0.3012 | 0.1852 → 0.1851 |
| Coon (`134×156`) | 0.0077 | 0 | 29,147 (`0.0005`, `0`) | 0.0556 → 0.0526 → 0.0526 | 0.0000 → 0.0000 |
| Pole Creek (`559×531`) | 0.0002 | 0 | 0 (`0.0000`, `0`) | 0.5752 → 0.5752 → 0.5752 | 0.4297 → 0.4297 |

On all three packaged Mode-F days, moving the grid boundary outside the 10 km window puts **zero** exterior cells above the `0.10` threshold and changes AP by only `+0.0049`, `−0.0029`, and `0.0000`. The U-Net does not materially exploit the final-perimeter edge on these Mode-F examples.

**What remains open.** Mode-FL evaluation (with real exterior LANDFIRE fuels and terrain), full-exam replay across all exam fire-days on prospectively drawn issue-time grids, archived issue-time weather forecasts (rather than persistence), and pre-fold BatchNorm checkpoint verification remain open.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/firecast_v1_0_probe.py
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_dc30c02_probe.py firecast
```

The second-pass probe uses NumPy, SciPy, and CPU PyTorch. [Second-pass probe](reviews/reviewer-a/peer_dc30c02_probe.py)

## Resolution (27 Nov 2026)

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| Final-perimeter crop excludes 45–59 % of the 10 km window | Evaluation limit | **Documented, no code change.** The second pass already tested the concern: outward padding moved AP by +0.0049 / −0.0029 / 0.0000 with no exterior cell above 0.10. The card's limits now say so and name what remains untested (Mode FL exterior fuels, issue-time grids, archived forecasts). | Card limits. |
