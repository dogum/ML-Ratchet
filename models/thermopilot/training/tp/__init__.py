import os, sys
_t = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # training/, for _paths
if _t not in sys.path:
    sys.path.insert(0, _t)
import _paths  # noqa: E402,F401  (puts BuildingSim / BuildingData from shared/building on sys.path)
