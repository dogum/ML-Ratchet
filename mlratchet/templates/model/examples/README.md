# Examples

A few real exam examples for `demo.py` (small `.npz` files with a JSON `meta` entry holding the truth).

- Draw them with a fixed seed within strata (`mlratchet.splits.fixed_draw`), never by how well the model does.
- Never use dev examples here; dev units are for calibration.
- Keep them small (well under 1 MB each) and check the data licence allows redistribution.
  If it doesn't, ship a fetch script instead.
