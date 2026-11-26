"""
Referee's own Hecke-algebra test of the Deodhar property (C'_w = C'_{s_1}...C'_{s_l}) for the
minuscule elements, to check Prop. 7.1 (type A cross-check), the input of Prop. 7.3 (13-slot D_6
element non-Deodhar) and the type-E statements of section 7.

Soergel normalisation: H_s^2 = 1 + (v^-1 - v) H_s, C'_s = H_s + v.  The product P of the C'_s
along a reduced word of w is bar-invariant with H_w-coefficient 1; it equals the KL element iff
every other coefficient lies in v Z[v].   python3 R03_deodhar.py [E7]
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R03_core import *


def weyl(E, n):
    A = cartan(E, n)
    e = tuple(tuple(1 if a == b else 0 for a in range(1, n + 1)) for b in range(1, n + 1))  # x(alpha_j), j=1..n

    def mul_s(x, i):
        xi = x[i - 1]
        return tuple(tuple(x[j - 1][t] - A[i][j] * xi[t] for t in range(n)) for j in range(1, n + 1))

    def up(x, i):              # l(x s_i) > l(x)  iff  x(alpha_i) > 0
        return any(c > 0 for c in x[i - 1])
    return e, mul_s, up


def padd(p, q, sh=0, c=1):
    for k, v in q.items():
        p[k + sh] = p.get(k + sh, 0) + c * v


def product(word, E, n):
    e, mul_s, up = weyl(E, n)
    P = {e: {0: 1}}
    for i in word:
        Q = {}
        for x, p in P.items():
            xs = mul_s(x, i)
            # v * H_x
            padd(Q.setdefault(x, {}), p, 1)
            if up(x, i):
                padd(Q.setdefault(xs, {}), p)
            else:
                padd(Q.setdefault(xs, {}), p)
                padd(Q.setdefault(x, {}), p, -1)      # v^-1 H_x
                padd(Q.setdefault(x, {}), p, 1, -1)   # -v H_x
        P = {}
        for x, p in Q.items():
            p = {k: c for k, c in p.items() if c}
            if p:
                P[x] = p
        w = None
    # w = element of the full word
    w = e
    for i in word:
        w = mul_s(w, i)
    return P, w


def is_deodhar(word, E, n):
    P, w = product(word, E, n)
    assert P[w] == {0: 1}
    return all(min(p) >= 1 for x, p in P.items() if x != w), len(P)


def run(typ, n, k, label, crit=None):
    t0 = time.time()
    E = edges(typ, n); adj = make_adj(E)
    heaps, covers, order = all_weights_with_heaps(E, n, k)
    nd = []
    cyc = []
    mism = []
    for mu in order:
        (h,) = heaps[mu]
        w = list(h)
        d, sz = is_deodhar(w, E, n)
        c = conn_search_fast(Tables(w, adj))[0] is not None
        if not d:
            nd.append(len(w))
        if c:
            cyc.append(len(w))
        if (not d) and c:
            pass
        if (not d) != c:
            mism.append((len(w), "".join(map(str, w)), "non-Deodhar" if not d else "Deodhar", "cyclic" if c else "not cyclic"))
    print("%s: %d weights; non-Deodhar ideal sizes %s; cyclic ideal sizes %s; weights where the two differ: %s  (%.1fs)" % (
        label, len(order), sorted(nd), sorted(cyc), mism, time.time() - t0), flush=True)


if __name__ == "__main__":
    run("A", 7, 4, "A_7 omega_4")
    run("A", 7, 3, "A_7 omega_3")
    run("D", 6, 6, "D_6 half-spin")
    run("D", 6, 1, "D_6 vector")
    run("E", 6, 1, "E_6 omega_1")
    run("D", 7, 7, "D_7 half-spin")
    if len(sys.argv) > 1:
        run("E", 7, 7, "E_7 omega_7")
