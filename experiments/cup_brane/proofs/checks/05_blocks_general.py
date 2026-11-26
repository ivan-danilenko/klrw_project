"""
Data for Section 10 of proofs/05_main_sliding.md: the chain map equation for ARBITRARY non-cyclic words.

For w = w'.c (last slot m; the colour c occurs N times in w', at the slots a_1 < ... < a_N) the chain map
equation  (d' (x) 1) F = F (1 (x) d')  is, by the Sliding Lemma, equivalent to

    sum_{t', chi}  coef'(t') * gamma(chi) * E(t', chi)   =   F_1 (1 (x) d')  -  (d' (x) 1) F_1          (*)

(t' a term of d', chi a bad letter of its diagram, F_1 = the tree terms containing a pair (a_i, m)).
The script expands every E(t', chi) and every product on the right in the basis and reports, for the
components (I', K') grouped by the status ('+' = on the right of the red strand) of a_1, ..., a_N in I'
and in K':

   clean  = every E and every product is a single basis diagram with a monomial coefficient, and the basis
            diagrams cancel in pairs;
   dirty  = anything else (a product that vanishes or has several basis terms or dots; larger groups).

It also checks that (*) holds.

    sage -python 05_blocks_general.py TYPE RANK N N_WORDS MAXLEN [SEED] [a1]

With the last argument "a1" everything is restricted to the components with a_1 not in K', the ones
needed for the Reduction Lemma 4.5 with a = a_1.
"""
import sys, importlib, itertools, random
from collections import defaultdict
L = importlib.import_module("05_lib")
from cyclefree import find_cycle
from formula import Stack


def analyse(ct, w, res, stats, red=1, restrict=False):
    S = L.Setup(ct, red, w)
    m = S.m
    c = w[-1]
    occ = [s for s in range(1, m) if w[s - 1] == c]
    r_c = S.mono((("r_%d" % c, 1),))
    for h in range(0, m):
        for Ip in itertools.combinations(range(1, m), h):
            I = frozenset(Ip) | {m}
            ids = L.strand_ids(S, I)
            groups = defaultdict(list)
            dirty = defaultdict(bool)

            def record(K, kind, element):
                d = S.describe(I, K, element)
                if restrict and occ[0] in K:
                    for k, v in d.items():
                        groups[K, k].append(v)
                    return
                stats["%s: %s" % (kind, "0" if not d else "1 diagram" if len(d) == 1 else ">1 diagrams")] += 1
                if len(d) != 1:
                    dirty[K] = True
                for k, v in d.items():
                    if len(v.monomials()) > 1:
                        dirty[K] = True
                        stats["%s: coefficient with dots" % kind] += 1
                    groups[K, k].append(v)

            for t in S.terms(I):
                if t.j == m:
                    continue
                x = S.el(I, t)
                isF1 = any(m in p for p in t.F)
                for t2 in S.terms(t.J):
                    if t2.j == m or m in t2.J:
                        continue
                    record(t2.J, "product F1*d'" if isF1 else "product d'*F1", x * S.el(t.J, t2))
                if not isF1:
                    K = t.J - {m}
                    bw = L.word_of(S, S.beta(I, t.J, t.sigma))
                    cr, _ = L.crossings(ids, bw)
                    nb = 0
                    for kk, (s, s2) in enumerate(cr):
                        if s != "W" and w[s - 1] == c and s2 != "W" and S.st.adjacent(s, s2):
                            nb += 1
                            i = bw[kk]
                            Ew = bw[:kk] + tuple(range(m, i + 1, -1)) + tuple(range(i - 1, 0, -1)) \
                                + tuple(y + 1 for y in bw[kk + 1:])
                            coef = r_c * S.mono((("t_%d_%d" % (c, w[s2 - 1]), 1),)) * S.mono(t.mono) * S.D(t.sign)
                            record(K, "E", coef * S.from_word(I, Ew))
                    if not (restrict and occ[0] in K):
                        stats["terms of d' with %d bad letters" % nb] += 1
            for (K, k), g in groups.items():
                assert sum(g) == 0, "equation (*) fails"
                if len(g) != 2:
                    dirty[K] = True
                if not (restrict and occ[0] in K):
                    stats["cancellation groups of size %d" % len(g)] += 1
            for K in set(K for K, _ in groups) | set(dirty):
                if restrict and occ[0] in K:
                    continue
                key = ("".join("+" if s in I else "-" for s in occ), "".join("+" if s in K else "-" for s in occ))
                res[key][dirty[K]] += 1


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    N = int(sys.argv[3])
    n_words, maxlen = int(sys.argv[4]), int(sys.argv[5])
    rng = random.Random(int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    restrict = len(sys.argv) > 7 and sys.argv[7] == "a1"
    res = defaultdict(lambda: defaultdict(int))
    stats = defaultdict(int)
    seen = set()
    tries = 0
    while len(seen) < n_words and tries < 200000:
        tries += 1
        n = rng.randint(N + 2, maxlen)
        w = tuple(rng.randint(1, ct[1]) for _ in range(n))
        if w in seen or w.count(w[-1]) != N + 1 or find_cycle(Stack(ct, w), max_pairs=4) is not None:
            continue
        seen.add(w)
        analyse(ct, w, res, stats, restrict=restrict)
    print("type %s, N = %d, %d non-cyclic words of length <= %d: equation (*) holds in all components" % (ct, N, len(seen), maxlen))
    if restrict:
        print("   (statistics restricted to the components with a_1 not in K')")
    for k, v in sorted(stats.items()):
        print("   %-45s %d" % (k, v))
    print("   status of a_1..a_N in I' / in K' :  clean components, dirty components")
    for k, v in sorted(res.items()):
        print("      %s / %s : %6d %6d" % (k[0], k[1], v[False], v[True]))
