"""
Pure-Python enumeration of ALL homogeneous basis elements (dots allowed) of the components
of maps on the cube, for proofs/04_first_order.md section 5.

    python3 proofs/checks/04_degree.py TYPE RANK RED WORD [HDEG]

A basis element of e_I A e_J is  mu * x^alpha * beta_sigma  (F1); it is homogeneous of the
degree D(I) - D(J) of a component of a map of homological degree HDEG (|J| = |I| + HDEG) iff
(Proposition 4 and the coefficient comparison of section 5)

    n_c  = Shat_c  := (D(I)-D(J))_{s,c} - S_c(sigma)          (exponent of r_c)
    m_ab + m_ba = Ehat_ab / 2                                  (exponents of t_ab, t_ba)
    l_k  = What / 2                                            (exponent of u_k)
    a_c  = sum_{b ~ c} m_cb + l_c - n_c >= 0                   (number of dots of colour c)

``elements(model, I, J)`` yields (sigma, n, m, l, a) for all solutions (the dots are then any
monomial with a_c dots on the strands of colour c).
"""
import itertools
import sys
import os
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from degree_model import Model  # noqa: E402


def deficit(model, I, J, sigma):
    target = Counter(model.D(I))
    target.subtract(model.D(J))
    target.subtract(model.content(I, J, sigma))
    return target


def solutions(model, dfc):
    """All (n, m, l, a) for a deficit; n, l, a dicts over colours, m dict over ordered edges."""
    cols = sorted(set(model.col))
    n = {}
    for c in cols:
        v = dfc.get(("s", c), 0)
        if v < 0:
            return []
        n[c] = v
    l = {c: 0 for c in cols}
    for key, v in dfc.items():
        if key[0] == "w":
            if v < 0 or v % 2:
                return []
            l[key[1]] = v // 2
        if key[0] == "e" and (v < 0 or v % 2):
            return []
    edges = []
    for e in sorted(tuple(sorted(e)) for e in model.E):
        a, b = e
        if a in cols and b in cols:
            edges.append((a, b, dfc.get(("e", a, b), 0) // 2))
        elif dfc.get(("e", a, b), 0):
            return []
    out = []
    for split in itertools.product(*[range(h + 1) for _, _, h in edges]):
        m = {}
        for (a, b, h), k in zip(edges, split):
            m[a, b] = k
            m[b, a] = h - k
        a_ = {}
        good = True
        for c in cols:
            a_[c] = sum(v for (x, y), v in m.items() if x == c) + l[c] - n[c]
            if a_[c] < 0:
                good = False
                break
        if good:
            out.append((dict(n), {k: v for k, v in m.items() if v}, {k: v for k, v in l.items() if v}, a_))
    return out


def elements(model, I, J, bij=None):
    if bij is None:
        bij = list(model.bijections())
    for sigma in bij:
        dfc = deficit(model, I, J, sigma)
        for sol in solutions(model, dfc):
            yield (tuple(sigma[s] for s in range(1, model.m + 1)),) + sol


def all_elements(model, hdeg=-1):
    slots = range(1, model.m + 1)
    bij = list(model.bijections())
    out = []
    for h in range(0, model.m + 1):
        for I in itertools.combinations(slots, h):
            I = frozenset(I)
            if not 0 <= h + hdeg <= model.m:
                continue
            for J in itertools.combinations(slots, h + hdeg):
                J = frozenset(J)
                for el in elements(model, I, J, bij):
                    out.append((I, J) + el)
    return out


def describe(model, I, J, sigma, n, m, l, a):
    mv = {s: sigma[s - 1] for s in range(1, model.m + 1) if sigma[s - 1] != s}
    mono = []
    for c, v in sorted(n.items()):
        if v:
            mono.append("r%d^%d" % (c, v) if v > 1 else "r%d" % c)
    for (x, y), v in sorted(m.items()):
        mono.append("t%d%d^%d" % (x, y, v) if v > 1 else "t%d%d" % (x, y))
    for c, v in sorted(l.items()):
        mono.append("u%d^%d" % (c, v) if v > 1 else "u%d" % c)
    dots = ["x%d^%d" % (c, v) for c, v in sorted(a.items()) if v]
    return "I=%-14s J=%-14s sigma %-22s %-18s dots %s" % (
        sorted(I), sorted(J), mv, "*".join(mono) or "1", " ".join(dots) or "-")


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3])
    word = tuple(int(c) for c in sys.argv[4])
    hdeg = int(sys.argv[5]) if len(sys.argv) > 5 else -1
    model = Model(ct, [(red, word)])
    els = all_elements(model, hdeg)
    byorder = defaultdict(list)
    for e in els:
        byorder[sum(e[3].values())].append(e)
    for k in sorted(byorder):
        print("r-order", k, ":", len(byorder[k]), "families")
        for e in byorder[k][:400]:
            print("   ", describe(model, *e))
