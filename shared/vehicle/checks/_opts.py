"""RoadSim physics options for the check chain (one place, so every check runs with v1 or v2 alike).

    import _opts; _opts.init()          # reads --options NAME|JSON from sys.argv (removes it), else $ROADSIM_OPTIONS
    simulate(..., options=_opts.get())  # None = physics v1
    _opts.out('cycles_vs_epa.json')     # v1 -> cycles_vs_epa.json; other presets -> cycles_vs_epa__<name>.json
    _opts.merged({'hvac_T_set': 22.2})  # the current options plus per-test procedure settings

run_all.py --options roadsim-physics-v2 passes the choice on through the environment.
"""
import os, sys, json

ENV = 'ROADSIM_OPTIONS'
_OPT = None


def _parse(s):
    if s is None or s == '' or s == 'roadsim-physics-v1':
        return None
    s = s.strip()
    return json.loads(s) if s.startswith('{') else s


def init(argv=None):
    """Take --options from argv (default sys.argv; the pair is removed so scripts with their own argparse are
    unaffected), else from the environment. Returns the options (None = v1)."""
    global _OPT
    argv = sys.argv if argv is None else argv
    val = os.environ.get(ENV)
    for i, a in enumerate(list(argv)):
        if a == '--options' and i + 1 < len(argv):
            val = argv[i + 1]; del argv[i:i + 2]; break
        if a.startswith('--options='):
            val = a.split('=', 1)[1]; del argv[i]; break
    _OPT = _parse(val)
    if _OPT is not None:
        os.environ[ENV] = val if isinstance(val, str) else json.dumps(val)
    return _OPT


def set(options):
    global _OPT
    _OPT = _parse(options) if isinstance(options, str) or options is None else options


def get():
    return _OPT


def name(options=None):
    o = _OPT if options is None else options
    if o is None:
        return 'roadsim-physics-v1'
    if isinstance(o, str):
        return o
    if isinstance(o, dict) and isinstance(o.get('preset'), str) and len(o) == 1:
        return o['preset']
    return 'custom'


def out(fname, options=None):
    n = name(options)
    if n == 'roadsim-physics-v1':
        return fname
    b, e = os.path.splitext(fname)
    return f'{b}__{n}{e}'


def merged(extra=None, options=None):
    """current options plus `extra` (procedure settings), as an options argument for simulate()."""
    o = _OPT if options is None else options
    if not extra:
        return o
    if o is None:
        return dict(extra)
    if isinstance(o, str):
        return dict(extra, preset=o)
    return dict(o, **extra)


def resolved(spec=None, options=None):
    """the full resolved option dict (for code that needs a value, e.g. thermal parameters)."""
    from roadsim.options import resolve
    return resolve(_OPT if options is None else options, spec)
