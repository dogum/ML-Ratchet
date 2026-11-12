"""Check the simulator against something independent, and record the numbers.

Examples from the collection: QuakeNet's local synthesis vs Syngine full synthetics (r = 0.9999);
TransitNet's transit model vs batman (≤ 0.94 ppm); SunDoctor's shading vs pvlib (1e-15);
SwellCast's ridge slope vs Munk's g/(4πD) (within 1.4%). One good check beats ten vague ones.
"""
import json

import numpy as np

import _paths as _P  # noqa: F401  (portable paths)
import simulator as S


def main():
    rng = np.random.default_rng(0)
    out = {"physics_version": S.PHYSICS_VERSION, "checks": []}
    # TODO: compare S.simulate(...) with a reference and append
    # {"name": ..., "reference": ..., "metric": ..., "value": ..., "pass": bool}
    with open("physics_check.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
