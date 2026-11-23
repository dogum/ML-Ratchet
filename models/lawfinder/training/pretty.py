"""Readable forms of a law: sympy-simplified text and MathML (for the page) with rounded constants."""
import sympy as sp
import lawsearch as ls

SYM = {'phi': 'φ', 'fs': 'f_s', 'sync': 'δ', 'S': 'S', 'P': 'P', 'W': 'U', 'M': 'M', 'R0': 'R_0',
       'D': 'D', 's': 's', 'L': 'ΔP', 'E': 'E'}
UNI = {'phi': 'φ', 'fs': 'f_s', 'sync': 'δ', 'S': 'S', 'P': 'P', 'W': 'U', 'M': 'M', 'R0': 'R₀', 'D': 'D', 's': 's',
       'L': 'ΔP', 'E': 'E'}


def _round(v, sig=4):
    if v == 0:
        return sp.Integer(0)
    r = float(f'{v:.{sig}g}')
    if abs(r - round(r)) < 1e-12 and abs(r) < 1e6:
        return sp.Integer(int(round(r)))
    return sp.Float(r, sig)


def to_sympy(tree, consts, names, sig=4):
    syms = {n: sp.Symbol(SYM.get(n, n), positive=True) for n in names}

    def rec(t):
        k = t[0]
        if k == 'x': return syms[names[t[1]]]
        if k == 'c': return _round(float(consts[t[1]]), sig)
        a = rec(t[1])
        if k == '+': return a + rec(t[2])
        if k == '-': return a - rec(t[2])
        if k == '*': return a * rec(t[2])
        if k == '/': return a / rec(t[2])
        if k == '^': return a ** rec(t[2])
        if k == 'exp': return sp.exp(a)
        if k == 'log': return sp.log(a)
        if k == 'sqrt': return sp.sqrt(a)
        if k == 'inv': return 1 / a
        if k == 'sq': return a ** 2
    e = rec(tree)
    try:
        e2 = sp.nsimplify(e, rational=False) if False else e
        e2 = sp.simplify(e2)
        if sp.count_ops(e2) <= sp.count_ops(e):
            e = e2
    except Exception:
        pass
    return e.xreplace({f: _round(float(f), sig) for f in e.atoms(sp.Float)})


def mathml(tree, consts, names):
    e = to_sympy(tree, consts, names)
    return sp.mathml(e, printer='presentation')


def latex(tree, consts, names):
    return sp.latex(to_sympy(tree, consts, names))


def text(tree, consts, names):
    s = str(to_sympy(tree, consts, names))
    s = s.replace('R_0', 'R₀')
    return s.replace('**', '^').replace('*', '·')
