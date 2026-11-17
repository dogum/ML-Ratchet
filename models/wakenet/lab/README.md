# WakeNet interactive lab

`index.html` is a self-contained browser page built by `../training/build_lab.py`. It embeds WakeNet's `float16` weights and the six real exam snapshots from `../examples/`, runs the 3-D wake anchor and 246k-parameter `EdgeWakeGNN` live in JavaScript, and presents the ratchet ladder and 24,035-snapshot exam.

On the main website, the live instrument is served through `site/src/instruments/wakenet/` (exported by `../web/export_instrument.py` and parity-tested against Python to `1.9e-7`).
