"""Fit sign rules (linear over GF(2) in simple counts) for the terms of the omega_1 I-brane, per term type."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omega1_table import load, fmt
from omega1_rule import targets, vertices

def features(n, v, tag):
    k, A, C = v
    B = frozenset(range(1, n + 1)) - A - C
    BL = frozenset(c for c in B if c <= k); BR = B - BL
    f = {"1": 1, "k": k, "n": n, "|A|": len(A), "|BL|": len(BL), "|BR|": len(BR), "|C|": len(C)}
    if tag[0] == "int":
        c = tag[1]
    elif tag[0] == "iii":
        c = tag[2]; f["d"] = tag[1]
    else:
        c = None; f["d"] = tag[1]
    if tag[0] != "int":
        d = tag[1]; l = k - d
        f["l"] = l
        f["A<=l"] = sum(1 for x in A if x <= l); f["BL<=l"] = sum(1 for x in BL if x <= l)
        f["d(d-1)/2"] = d * (d - 1) // 2
    if c is not None:
        f["c"] = c
        for name, S in (("A", A), ("BL", BL), ("BR", BR), ("C", C)):
            f[name + ">c"] = sum(1 for x in S if x > c)
            f[name + "<c"] = sum(1 for x in S if x < c)
    return f

def solve_gf2(rows, rhs, names):
    """Find a 0/1 combination of features reproducing rhs mod 2 (Gaussian elimination)."""
    m = [[r[nm] % 2 for nm in names] + [b] for r, b in zip(rows, rhs)]
    piv = []; r = 0
    for col in range(len(names)):
        p = next((i for i in range(r, len(m)) if m[i][col]), None)
        if p is None: continue
        m[r], m[p] = m[p], m[r]
        for i in range(len(m)):
            if i != r and m[i][col]:
                m[i] = [a ^ b for a, b in zip(m[i], m[r])]
        piv.append(col); r += 1
    if any(row[-1] and not any(row[:-1]) for row in m):
        return None
    return [names[col] for i, col in enumerate(piv) if m[i][-1]]

if __name__ == "__main__":
    data = {}
    for n in (2, 3, 4, 5):
        d, P, V, terms = load(n)
        data[n] = {(t[0], t[1]): t[3] for t in terms}
    groups = {}
    for n in data:
        for v in vertices(n):
            for w, tag in targets(n, v):
                key = tag[0] if tag[0] != "iii" else ("iiiL" if tag[2] <= v[0] - tag[1] else "iiiR")
                if tag[0] == "int":
                    key = "intBA" if tag[1] <= v[0] else "intCB"
                groups.setdefault(key, []).append((features(n, v, tag), 0 if data[n][(v, w)] == 1 else 1))
    for key, lst in sorted(groups.items()):
        names = sorted(set.intersection(*[set(f) for f, _ in lst]))
        sol = solve_gf2([f for f, _ in lst], [b for _, b in lst], names)
        print("%-6s %4d terms: sign = (-1)^( %s )" % (key, len(lst), " + ".join(sol) if sol is not None else "NO LINEAR RULE"))
