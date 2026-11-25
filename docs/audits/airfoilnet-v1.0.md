# AirfoilNet v1.0 | Input ordering and reflection affect lift predictions

**Confirmed public coordinate-array defect.** The same geometry at the same Reynolds number should give the same polar in either documented point order. Reviewer B serialized the upper and lower surfaces of each of the four shipped UIUC airfoils into a counts-header Lednicer file, then supplied the coordinates through both public paths. On the common −6° to +6° grid, parsing with `read_dat` reproduced the Selig lift curve exactly in all four cases. Passing the same two surface blocks directly to `polar(coords=...)` changed lift coefficient by maxima of **4.09–5.44**. The file parser rearranges the surfaces. Direct normalization locates the trailing edge using the first and last points. [File parser and normalization](../../models/airfoilnet/airfoilnet.py#L25) · [Public polar API](../../models/airfoilnet/airfoilnet.py#L193)

**Reflection error persists inside the common angle range.** Reflecting the geometry vertically and reversing the angle should reverse the sign of lift. Each airfoil's maximum absolute sum of original and reflected lift was 0.195–1.133 on that restricted grid. The first pass measured 0.579–1.526 over the full measured angle sets, some of whose reflected angles extend beyond the default polar grid. This is an independent symmetry check on the learned surrogate. It does not measure wind-tunnel error for inverted airfoils. The split contained 7 dev and 75 exam identifiers with zero overlap. The shipped examples use canonical coordinates, so an effect on the published exam from raw Lednicer ordering remains unestablished.

The archived weights identify a three-member MLP. The module header describes the rejected DeepONet architecture. All three point estimates replayed against Torch within 1.1e−6 here. Both eight-row ladders reproduced their recorded decisions. The first-pass parity check covered means only. [Shipped reader](../../models/airfoilnet/airfoilnet.py#L10)

Model snapshot `61c70bf`, cross-check at `b53c7c7`. Both probes ran with NumPy, pandas and CPU PyTorch.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-a/airfoilnet_v1_0_probe.py
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/peer_b53c7c7_probe.py
```

[First-pass evidence](reviews/reviewer-a/airfoilnet-v1.0.md) · [Independent probe](reviews/reviewer-b/peer_b53c7c7_probe.py)
