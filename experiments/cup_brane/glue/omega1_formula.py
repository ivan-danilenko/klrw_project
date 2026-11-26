"""
Closed formula for the glued complex (I-brane) of sl_{n+1}, omega_1 -- pure Python.
Compares terms, signs and coefficients with the glued complexes computed by glue.py.

Coefficient: with tau_x = t_{x,x-1} / t_{x,x+1}  (t_{1,0} := u_1, t_{n,n+1} := u_n)
    internal, (i):  1
    (ii):           prod_{x = l+1..k} tau_x
    (iii), extra colour c <= l:   prod_{x = c..l} tau_x^{-1}
    (iii), extra colour c > k:    prod_{x = l+1..c} tau_x
Sign: exterior-algebra rule of omega1_build.rule_sign (re-implemented here), in the basis where
the envelope S_k is multiplied by (-1)^{k(k-1)/2} relative to the output of standard_ebranes.
"""
import sys, os
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omega1_table import load
from omega1_rule import targets, vertices
from omega1_koszul import monomial, contract, wedge, replace

def rule_sign(n, v, tag):
    k, A, C = v
    order = lambda y: y[1] if y[0] == "a" else 2 * n + 2 - y[1]
    gens = monomial(n, v)
    if tag[0] == "int":
        return contract(gens, order, ("g", tag[1]))[1]
    d = tag[1]; l = k - d; sw = list(range(l + 1, k + 1))
    if tag[0] == "i":
        ops = [("rho", c) for c in sw[:-1]] + [("ia", k)]; s = (-1) ** (d + 1)
    elif tag[0] == "ii":
        ops = [("ia", l + 1)] + [("rho", c) for c in sw[1:]]; s = 1
    else:
        c = tag[2]
        ops = [("ia", k), ("ig", k), ("eg", c)] if d == 1 else \
              [("ia", l + 1)] + [("rho", x) for x in sw[1:-1]] + [("ia", k), ("eg", c)]
        s = (-1) ** ((l + 1 - c) + d) if c <= l else (-1) ** (c - k)
    for op, c in ops:
        if op == "rho": gens, t = replace(gens, order, c)
        elif op == "ia": gens, t = contract(gens, order, ("a", c))
        elif op == "ig": gens, t = contract(gens, order, ("g", c))
        else: gens, t = wedge(gens, order, ("g", c))
        s *= t
    return s

def tau(n, x, e):
    c = Counter()
    c["u_1" if x == 1 else "t_%d_%d" % (x, x - 1)] += e
    c["u_%d" % n if x == n else "t_%d_%d" % (x, x + 1)] -= e
    return c

def coefficient(n, v, tag):
    k = v[0]; c = Counter()
    if tag[0] in ("int", "i"):
        return {}
    d = tag[1]; l = k - d
    if tag[0] == "ii":
        rng, e = range(l + 1, k + 1), 1
    elif tag[2] <= l:
        rng, e = range(tag[2], l + 1), -1
    else:
        rng, e = range(l + 1, tag[2] + 1), 1
    for x in rng:
        c.update(tau(n, x, e))
    return {key: val for key, val in c.items() if val}

if __name__ == "__main__":
    for n in (2, 3, 4, 5):
        d, P, V, terms = load(n)
        actual = {(t[0], t[1]): (t[3], {a: b for a, b in t[2].items() if b}) for t in terms}
        bad_sign = bad_coef = 0; count = 0
        for v in vertices(n):
            for w, tag in targets(n, v):
                count += 1
                k, l = v[0], w[0]
                gauge = (-1) ** (k * (k - 1) // 2 + l * (l - 1) // 2)
                s, par = actual[(v, w)]
                bad_sign += (s != gauge * rule_sign(n, v, tag))
                bad_coef += (par != coefficient(n, v, tag))
        print("sl_%d omega_1: %d terms in the glued complex, %d given by the formula; wrong sign %d, wrong coefficient %d" % (
            n + 1, len(actual), count, bad_sign, bad_coef))
