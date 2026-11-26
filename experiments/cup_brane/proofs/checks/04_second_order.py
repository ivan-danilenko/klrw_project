"""
Second order (proofs/04_first_order.md, section 10): where do the two groups of (*) live?

    sage -python proofs/checks/04_second_order.py TYPE RANK MAXLEN [MINLEN]

For every word (length MINLEN..MAXLEN, every red colour) and vertex I it computes, in the klrw
basis, the r-degree-2 parts of
    E  = delta_0^2 + delta_0 delta_1 + delta_1 delta_0        (the "lower products"),
    S  = delta_0 delta_2 + delta_1 delta_1 + delta_2 delta_0,
checks E + S = 0 at r-degree 2 (i.e. (*) of section 10 for the tree formula), and sorts the
basis terms of E and of S by the *sector* of their bijection: the multiset of colours of the
non-trivial cycles of sigma, e.g. "(c1)(c2)" = two transpositions of different colours.
"""
import sys, os, itertools, importlib, time
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L = importlib.import_module("04_lib")


def sector(C, sigma):
    seen = set(); cyc = []
    for s in range(1, C.m + 1):
        if s in seen or sigma[s - 1] == s:
            continue
        c = []; x = s
        while x not in seen:
            seen.add(x); c.append(x); x = sigma[x - 1]
        cyc.append("%d-cycle c%d" % (len(c), C.col(s)))
    return " ".join(sorted(cyc)) or "id"


def run(ct, red, word, stats):
    C = L.Ctx(ct, red, word)
    V = list(L.subsets(C.m))
    d = {n: {I: (C.delta0(I) if n == 0 else C.delta_n(I, n)) for I in V} for n in (0, 1, 2)}
    ok = True
    for I in V:
        E = defaultdict(lambda: C.K.zero()); S = defaultdict(lambda: C.K.zero())
        parts = {nm: defaultdict(lambda: C.K.zero()) for nm in ("d0d0", "d0d1+d1d0", "d1d1", "d0d2+d2d0")}
        for (a, b, tgt, nm) in ((0, 0, E, "d0d0"), (0, 1, E, "d0d1+d1d0"), (1, 0, E, "d0d1+d1d0"),
                                (0, 2, S, "d0d2+d2d0"), (1, 1, S, "d1d1"), (2, 0, S, "d0d2+d2d0")):
            for J, x in d[a][I].items():
                for K, y in d[b].get(J, {}).items():
                    tgt[K] += x * y
                    parts[nm][K] += x * y
        for nm, dct in parts.items():
            for K, el in dct.items():
                for braid, e, c in C.terms(C.r_part(el, 2)):
                    stats["part " + nm, sector(C, C.bijection(I, K, braid)), False] += 1
        for K in set(E) | set(S):
            e2 = C.r_part(E.get(K, C.K.zero()), 2); s2 = C.r_part(S.get(K, C.K.zero()), 2)
            if not (e2 + s2).is_zero():
                ok = False; print("FAIL (*)", word, red, sorted(I), sorted(K))
            for name, el in (("E", e2), ("S", s2)):
                for braid, e, c in C.terms(el):
                    sec = sector(C, C.bijection(I, K, braid))
                    dots = any(e[p] for p in range(len(C.names)) if C.names[p].startswith("x_"))
                    stats[name, sec, dots] += 1
    return ok


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]; maxlen = int(sys.argv[3]); minlen = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    stats = Counter(); nf = 0; t0 = time.time(); nw = 0
    for w in L.all_words(ct[1], maxlen, minlen):
        for red in range(1, ct[1] + 1):
            nf += not run(ct, red, w, stats); nw += 1
    print("type", ct, "lengths", minlen, "..", maxlen, "word x red", nw, "failures of (*)", nf, "[%.0fs]" % (time.time() - t0))
    for k in sorted(stats):
        print("   %-2s sector %-40s dots %-5s basis terms %d" % (k[0], k[1], k[2], stats[k]))
