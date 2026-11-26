"""
Checks for Theorems A, B and C of proofs/04_first_order.md (order 0 and order 1).

    sage -python proofs/checks/04_order01.py TYPE RANK MAXLEN [MINLEN] [SAMPLE SEED]

For every word of length MINLEN..MAXLEN over the nodes of the Dynkin diagram (or a random
sample of SAMPLE of them) and every colour of the red strand, and every vertex I:

 (T0)  delta_0^2 e_I has no term of r-degree 0                       (Theorem A)
 (T1)  the r-degree-1 part of delta_0^2 e_I equals the predicted sum over bad triples
       sum_{i<j in I} eps(I;i,j) r_{c(j)} t_{c(j)c(i)} sum_{y<i, c(y)=c(j)~c(i)} beta(tau_y)
                                                                      (Theorem B)
 (T2)  every composite m_x * (one-pair term) and (one-pair term) * m_x is, modulo r, the
       predicted element: the basis element of the composite bijection, times the bigon
       polynomial t_{c(x)c(p)} x_x + t_{c(p)c(x)} x_{a or b} if a < x < b (0 if c(x) = c(p),
       1 if c(x), c(p) are not adjacent); the concatenated word is reduced unless a < x < b,
       and then exactly the pair (x, strand of a) resp. (x, strand of b) crosses twice
                                                                      (Lemma 6 / Prop. 7)
 (T3)  (delta_0^2 + delta_0 delta_1 + delta_1 delta_0) e_I has no term of r-degree <= 1
                                                                      (Theorem C)
"""
import sys, os, time, random, itertools, importlib
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L = importlib.import_module("04_lib")


def predicted_first(C, I):
    res = defaultdict(lambda: C.K.zero())
    for i, j in itertools.combinations(sorted(I), 2):
        e1 = (-1) ** (C.n_above(I, i) + C.n_above(I, j))
        K = I - {i, j}
        if not C.adj(C.col(i), C.col(j)):
            continue
        for y in range(1, i):
            if C.col(y) == C.col(j):
                tau = list(C.ident)
                tau[y - 1], tau[j - 1] = j, y
                res[K] += (C.D(e1) * C.r(C.col(j)) * C.t(C.col(j), C.col(i))) * C.basis(I, K, tuple(tau))
    return res


def mover_and_pair(J0, J, sigma):
    moved = [s for s in range(1, len(sigma) + 1) if sigma[s - 1] != s]
    assert len(moved) == 2
    a, b = sorted(moved)
    (j,) = [s for s in J0 - J if s != b]
    return j, (a, b)


def predicted_composite(C, I, x, j, p, side, K, sig):
    """Prediction (mod r) for the composite; side 'A' = m_x first, 'B' = m_x last."""
    a, b = p
    base = C.basis(I, K, sig)
    if x == a:
        return base, "bad-triple"
    if not (a < x < b):
        return base, "reduced"
    cx, cp = C.col(x), C.col(a)
    if cx == cp:
        return C.K.zero(), "bigon-same-colour"
    if not C.adj(cx, cp):
        return base, "bigon-far"
    other = a if side == "A" else b
    Q = C.t(cx, cp) * C.dot(I, x) + C.t(cp, cx) * C.dot(I, other)
    return Q * base, "bigon-adjacent"


def compose_sigma(s1, s2):
    return tuple(s2[s1[s - 1] - 1] for s in range(1, len(s1) + 1))


def check_word(ct, red, word, stats):
    C = L.Ctx(ct, red, word)
    m = C.m
    V = list(L.subsets(m))
    d0 = {I: C.delta0(I) for I in V}
    d1 = {I: C.delta_n(I, 1) for I in V}
    ok = True
    for I in V:
        if len(I) < 2:
            continue
        sq0 = defaultdict(lambda: C.K.zero())
        for J, x in d0[I].items():
            for K, y in d0[J].items():
                sq0[K] += x * y
        pred = predicted_first(C, I)
        for K in set(sq0) | set(pred):
            v = sq0.get(K, C.K.zero())
            if not C.r_part(v, 0).is_zero():
                print("T0 FAIL", ct, red, word, sorted(I), sorted(K)); ok = False
            if C.r_part(v, 1) != pred.get(K, C.K.zero()):
                print("T1 FAIL", ct, red, word, sorted(I), sorted(K), C.r_part(v, 1), pred.get(K)); ok = False
            if not C.max_r_part(v, 1).is_zero():
                stats["delta0^2 nonzero mod r^2 (components)"] += 1
            if not (v - C.max_r_part(v, 1)).is_zero():
                stats["delta0^2 has r-degree>=2 terms (components)"] += 1
        # composites
        tot = defaultdict(lambda: C.K.zero())
        for K, v in sq0.items():
            tot[K] += v
        for x in I:
            J0 = I - {x}
            mx = C.move(I, x) * C.D((-1) ** C.n_above(I, x))
            for (J, sigma, mono, sign) in C.tree_terms(J0, 1):
                j, p = mover_and_pair(J0, J, sigma)
                term = (C.D(sign) * C.mono(mono)) * C.basis(J0, J, sigma)
                prod = C.move(I, x) * C.basis(J0, J, sigma)
                tot[J] += mx * term
                pr, kind = predicted_composite(C, I, x, j, p, "A", J, sigma)
                stats["A:" + kind] += 1
                if C.r_part(prod, 0) != pr:
                    print("T2 FAIL A", ct, red, word, sorted(I), x, j, p, kind, C.r_part(prod, 0), pr); ok = False
                w = tuple(C.move(I, x).leading_support().word()) + tuple(C.basis(J0, J, sigma).leading_support().word())
                cnt, _ = L.crossing_counts(C, I, w)
                twice = sorted(tuple(sorted(map(str, k))) for k, v in cnt.items() if v >= 2)
                exp_twice = [tuple(sorted(map(str, (x, p[0]))))] if kind.startswith("bigon") else []
                if twice != exp_twice or max(cnt.values()) > 2:
                    print("T2 FAIL A crossings", word, sorted(I), x, j, p, twice, exp_twice); ok = False
        for (J, sigma, mono, sign) in C.tree_terms(I, 1):
            j, p = mover_and_pair(I, J, sigma)
            term = (C.D(sign) * C.mono(mono)) * C.basis(I, J, sigma)
            for x in J:
                K = J - {x}
                mx = C.move(J, x)
                prod = C.basis(I, J, sigma) * mx
                tot[K] += term * (C.D((-1) ** C.n_above(J, x)) * mx)
                pr, kind = predicted_composite(C, I, x, j, p, "B", K, sigma)
                stats["B:" + kind] += 1
                if C.r_part(prod, 0) != pr:
                    print("T2 FAIL B", ct, red, word, sorted(I), x, j, p, kind, C.r_part(prod, 0), pr); ok = False
                w = tuple(C.basis(I, J, sigma).leading_support().word()) + tuple(mx.leading_support().word())
                cnt, _ = L.crossing_counts(C, I, w)
                twice = sorted(tuple(sorted(map(str, k))) for k, v in cnt.items() if v >= 2)
                exp_twice = [tuple(sorted(map(str, (x, p[1]))))] if kind.startswith("bigon") else []
                if twice != exp_twice or max(cnt.values()) > 2:
                    print("T2 FAIL B crossings", word, sorted(I), x, j, p, twice, exp_twice); ok = False
        for K, v in tot.items():
            if not C.max_r_part(v, 1).is_zero():
                print("T3 FAIL", ct, red, word, sorted(I), sorted(K), C.max_r_part(v, 1)); ok = False
        stats["vertices"] += 1
    return ok


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    maxlen = int(sys.argv[3])
    minlen = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    sample = int(sys.argv[5]) if len(sys.argv) > 5 else None
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    rank = ct[1]
    words = list(L.all_words(rank, maxlen, minlen))
    if sample is not None and sample < len(words):
        words = random.Random(seed).sample(words, sample)
    stats = Counter()
    t0 = time.time()
    nfail = 0
    for w in words:
        for red in range(1, rank + 1):
            if not check_word(ct, red, w, stats):
                nfail += 1
            stats["words x reds"] += 1
    print("type", ct, "lengths", minlen, "..", maxlen, "words", len(words), "failures", nfail,
          "[%.0fs]" % (time.time() - t0))
    for k, v in sorted(stats.items()):
        print("   %-50s %d" % (k, v))
