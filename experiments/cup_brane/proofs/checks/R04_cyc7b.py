"""
Referee: the dot-free differential of 1221122 (A_2, red 2), made airtight.

    sage -python proofs/checks/R04_cyc7b.py build     construct d_F = d_T + kappa* + eps_4, save JSON
    sage -python proofs/checks/R04_cyc7b.py verify    reload the JSON ALONE and re-check everything
    sage -python proofs/checks/R04_cyc7b.py gauge     is d_F gauge-equivalent to d_X = d_T + X ?

JSON: proofs/checks/R04_data/cyclic7_1221122_W2_dotfree.json (and ..._dotted.json for d_X).
Basis: klrw basis braid (fast.braid_for / Definition B.10 of 01_foundations.md), dots at the left
end, klrw dot names x_c_k (k-th strand of colour c from the left of the source state).
"""
import sys, os, json, time, itertools
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from R04_cyc7 import Ctx, candidates, compose, anticomm, vectorize, dot_monomials, X_elements  # noqa: E402
from cupbrane import Cube  # noqa: E402
from sage.all import matrix, QQ, vector, lcm  # noqa: E402

DATA = os.path.join(HERE, "R04_data")
F_DOTFREE = os.path.join(DATA, "cyclic7_1221122_W2_dotfree.json")
F_DOTTED = os.path.join(DATA, "cyclic7_1221122_W2_dotted.json")


def solve_order(C, d0, mp, k, cands):
    """dot-free eps of r-degree k with  [delta_0, eps]_k = -(mp)_k ; returns dict or None"""
    idx = {}
    obs = vectorize(C, {(I, K): x for I, row in mp.items() for K, x in row.items()}, lambda e: C.rdeg(e) == k, idx)
    elk = [(I, J, C.mono(n, m) * C.basis(I, J, s)) for (I, J, s, n, m, a) in cands]
    cols = [vectorize(C, anticomm(C, d0, I, J, el), lambda e: C.rdeg(e) == k, idx) for I, J, el in elk]
    nr = max(len(idx), 1)
    M = matrix(QQ, nr, max(len(cols), 1), {(r_, c_): v for c_, col in enumerate(cols) for r_, v in col.items()})
    ob = vector(QQ, nr, obs)
    if M.augment(ob.column()).rank() > M.rank():
        return None, M.rank(), len(cols)
    sol = M.solve_right(-ob)
    assert all(QQ(c).denominator() == 1 for c in sol), sol
    eps = defaultdict(lambda: defaultdict(lambda: C.K.zero()))
    for (I, J, el), c in zip(elk, sol):
        if c:
            eps[I][J] += C.D(int(c)) * el
    return eps, M.rank(), len(cols)


def add(C, d, e):
    out = {I: dict(row) for I, row in d.items()}
    for I, row in e.items():
        for J, x in row.items():
            out.setdefault(I, {})
            out[I][J] = out[I].get(J, C.K.zero()) + x
    return out


def build():
    C = Ctx()
    t0 = time.time()
    d0, d01, dT = C.tree(0), C.tree(1), C.tree()
    sq = compose(C, dT, dT)
    sqd = {(I, K): x for I, row in sq.items() for K, x in row.items()}
    cands2 = candidates(C, -1, 2, True)
    els = [(I, J, C.mono(n, m) * C.basis(I, J, s)) for (I, J, s, n, m, a) in cands2]
    idx2 = {}
    cols = [vectorize(C, anticomm(C, d0, I, J, el), lambda e: C.rdeg(e) == 2, idx2) for I, J, el in els]
    M = matrix(QQ, max(len(idx2), 1), len(cols), {(r_, c_): v for c_, col in enumerate(cols) for r_, v in col.items()})
    ker = [v * lcm([QQ(c).denominator() for c in v.list()]) for v in M.right_kernel().basis()]
    # r^3 equation with ALL products (delta_0 + delta_1) kappa and kappa (delta_0 + delta_1)
    idx3 = {}
    obs = vectorize(C, sqd, lambda e: C.rdeg(e) == 3, idx3)
    kaps, vecs = [], []
    for v in ker:
        k = defaultdict(lambda: defaultdict(lambda: C.K.zero()))
        for i, c in v.dict().items():
            I, J, el = els[i]
            k[I][J] += C.D(int(c)) * el
        kaps.append(k)
        tot = defaultdict(lambda: C.K.zero())
        for I, row in k.items():
            for J, el in row.items():
                for (a, b), x in anticomm(C, d01, I, J, el).items():
                    tot[a, b] += x
        vecs.append(vectorize(C, tot, lambda e: C.rdeg(e) == 3, idx3))
    n = len(idx3)
    Mk = matrix(QQ, n, len(vecs), {(r_, c_): v for c_, col in enumerate(vecs) for r_, v in col.items()})
    ob = vector(QQ, n, obs)
    print("kernel dim %d; r^3 rank %d, with obstruction %d" % (len(ker), Mk.rank(), Mk.augment(ob.column()).rank()))
    sol = Mk.solve_right(-ob)
    print("kappa* = combination", list(sol), "of the kernel basis; kernel of the r^3 map:", Mk.right_kernel().dimension())
    kstar = defaultdict(lambda: defaultdict(lambda: C.K.zero()))
    for k, c in zip(kaps, sol):
        assert QQ(c).denominator() == 1
        for I, row in k.items():
            for J, el in row.items():
                kstar[I][J] += C.D(int(c)) * el
    d = add(C, dT, kstar)
    sq = compose(C, d, d)
    c4 = candidates(C, -1, 4, True)
    eps4, rk, nc = solve_order(C, d0, sq, 4, c4)
    print("r^4: %d dot-free candidates, rank %d, solvable: %s" % (nc, rk, eps4 is not None))
    dF = add(C, d, eps4)
    sq = compose(C, dF, dF)
    print("d_F^2 = 0 exactly:", all(x.is_zero() for row in sq.values() for x in row.values()))
    show(C, "kappa*", kstar, dT)
    show(C, "eps_4", eps4, dT)
    os.makedirs(DATA, exist_ok=True)
    save(C, dF, F_DOTFREE, "d_F = d_T + kappa* + eps_4 (dot-free completion of the tree formula, r-degree-2 part changed by kappa*)")
    I0, J0, Xdot, Xr = X_elements(C)
    dX = add(C, dT, {I0: {J0: Xdot + Xr}})
    save(C, dX, F_DOTTED, "d_X = d_T + X (dotted completion of 04_first_order.md, Thm 9.1(d))")
    print("[%.0fs]" % (time.time() - t0))


def sigma_of(C, I, J, braid):
    arrive = C.cube.strand_map(C.eps(I), braid)
    labels = C.cube.labels(C.eps(J))
    sig = {}
    for a, t in zip(arrive, labels):
        if not isinstance(a, tuple):
            sig[a + 1] = t + 1
    return [sig[s] for s in range(1, C.Wd.m + 1)]


def show(C, name, mp, dT):
    tree_keys = set()
    for I, row in dT.items():
        for J, x in row.items():
            for braid, coeff in x:
                tree_keys.add((I, J, tuple(sigma_of(C, I, J, braid))))
    print(name + ":")
    for I, row in sorted(mp.items(), key=lambda kv: sorted(kv[0])):
        for J, x in row.items():
            for braid, coeff in x:
                sg = sigma_of(C, I, J, braid)
                LR = [s for s in C.Wd.slots if s not in I and sg[s - 1] in J]
                print("   %s -> %s  sigma %s  coeff %s  monotone %s  bijection of a tree term: %s" % (
                    sorted(I), sorted(J), {s: sg[s - 1] for s in C.Wd.slots if sg[s - 1] != s}, coeff,
                    not LR, (I, J, tuple(sg)) in tree_keys))


def save(C, d, path, what):
    terms = []
    for I, row in d.items():
        for J, x in row.items():
            for braid, coeff in x:
                sg = sigma_of(C, I, J, braid)
                for e, c in coeff.dict().items():
                    mono = {C.names[p]: int(v) for p, v in enumerate(e) if v}
                    terms.append([sorted(I), sorted(J), sg, mono, int(c)])
    json.dump(dict(cartan_type=["A", 2], red=2, word=[1, 2, 2, 1, 1, 2, 2], what=what,
                   basis="klrw basis braid of sigma (fast.braid_for), dots at the left end, klrw names x_c_k",
                   n_terms=len(terms), terms=terms), open(path, "w"), indent=0)
    print("saved", path, len(terms), "terms")


def load(C, path):
    data = json.load(open(path))
    d = defaultdict(lambda: defaultdict(lambda: C.K.zero()))
    for src, tgt, sg, mono, c in data["terms"]:
        I, J = frozenset(src), frozenset(tgt)
        m = C.D.one()
        for nm, p in mono.items():
            m *= C.D(nm) ** p
        d[I][J] += (C.D(c) * m) * C.basis(I, J, {s: sg[s - 1] for s in range(1, len(sg) + 1)})
    return {I: dict(row) for I, row in d.items()}, data


def verify():
    C = Ctx()
    t0 = time.time()
    cube = Cube(C.K, [(2, (1, 2, 2, 1, 1, 2, 2))], build=True)      # klrw's own grading of the cube
    d0 = C.tree(0)
    d01 = C.tree(1)
    for path in (F_DOTFREE, F_DOTTED):
        d, data = load(C, path)
        sq = compose(C, d, d)
        zero = all(x.is_zero() for row in sq.values() for x in row.values())
        # homogeneity with klrw's grading: deg(component I->J) = eq_degree[J] - eq_degree[I]
        hom = True
        for I, row in d.items():
            for J, x in row.items():
                if x.is_zero():
                    continue
                if x.degree(check_if_homogeneous=True) != cube.eq_degree[C.eps(J)] - cube.eq_degree[C.eps(I)]:
                    hom = False
        dots = any(nm.startswith("x_") for t in data["terms"] for nm in t[3])
        us = any(nm.startswith("u_") for t in data["terms"] for nm in t[3])
        # r-degree <= 1 part equals the tree formula's delta_0 + delta_1
        low = True
        allI = set(d) | set(d01)
        for I in allI:
            for J in set(d.get(I, {})) | set(d01.get(I, {})):
                a = d.get(I, {}).get(J, C.K.zero())
                b = d01.get(I, {}).get(J, C.K.zero())
                diff = a - b
                for braid, coeff in diff:
                    for e in coeff.dict():
                        if C.rdeg(e) <= 1:
                            low = False
        # r-free part is exactly delta_0
        print("%s: %d terms; d^2 = 0: %s; homogeneous (klrw grading): %s; dots: %s; u: %s; "
              "r-degree<=1 part = delta_0 + delta_1: %s" % (os.path.basename(path), data["n_terms"], zero, hom, dots, us, low))
    print("[%.0fs]" % (time.time() - t0))


def enum_h(C, rdeg):
    """homogeneous basis elements (dots allowed) of homological degree 0 and r-degree rdeg"""
    out = []
    for (I, J, s, n, m, a) in candidates(C, 0, rdeg, False):
        if I == J and all(s[x] == x for x in C.Wd.slots):
            if rdeg == 0:
                continue
        for dm in dot_monomials(C, I, a):
            out.append((I, J, dm * C.mono(n, m) * C.basis(I, J, s)))
    return out


def comm(C, h_I, h_J, h, d):
    """h d - d h for h in one component h_I -> h_J"""
    out = defaultdict(lambda: C.K.zero())
    for K, y in d.get(h_J, {}).items():
        out[h_I, K] += h * y
    for Ip, row in d.items():
        x = row.get(h_I)
        if x is not None:
            out[Ip, h_J] -= x * h
    return out


def gauge():
    """Necessary condition for g d_X g^{-1} = d_F with g = 1 + h_1 + h_2 + ..., h_k of r-degree k,
    homogeneous of homological degree 0 (Thm 8.1(c) gauges): the r^1 and r^2 parts of
    g d_X - d_F g vanish, i.e.
        [h_1, delta_0]_1 = 0,      [h_1, delta_0 + delta_1]_2 + [h_2, delta_0]_2 = kappa*  (r^2 part of d_F - d_X).
    Also lists the r-free homogeneous elements of homological degree 0 (other possible gauges)."""
    C = Ctx()
    t0 = time.time()
    dF, _ = load(C, F_DOTFREE)
    dX, _ = load(C, F_DOTTED)
    d0, d01 = C.tree(0), C.tree(1)
    h0 = enum_h(C, 0)
    print("r-free homogeneous elements of hom. degree 0, other than the identities:", len(h0),
          "; of them with a parameter t:", sum(1 for I, J, el in h0 if any(
              any(e[p] for p in C.ppos) for b, cf in el for e in cf.dict())))
    h1 = enum_h(C, 1)
    h2 = enum_h(C, 2)
    print("hom. degree 0: r-degree 1: %d elements, r-degree 2: %d elements" % (len(h1), len(h2)))
    # rhs: r^2 part of d_F - d_X
    diff = {}
    for I in set(dF) | set(dX):
        for J in set(dF.get(I, {})) | set(dX.get(I, {})):
            diff[I, J] = dF.get(I, {}).get(J, C.K.zero()) - dX.get(I, {}).get(J, C.K.zero())
    idx1, idx2 = {}, {}
    rhs = vectorize(C, diff, lambda e: C.rdeg(e) == 2, idx2)
    low = vectorize(C, diff, lambda e: C.rdeg(e) <= 1, {})
    assert not low
    cols1, cols2 = [], []
    for I, J, el in h1:
        cm = comm(C, I, J, el, d01)
        cols1.append(vectorize(C, cm, lambda e: C.rdeg(e) == 1, idx1))
        cols2.append(vectorize(C, cm, lambda e: C.rdeg(e) == 2, idx2))
    for I, J, el in h2:
        cm = comm(C, I, J, el, d0)
        cols1.append({})
        cols2.append(vectorize(C, cm, lambda e: C.rdeg(e) == 2, idx2))
    n1, n2 = len(idx1), len(idx2)
    ncol = len(cols1)
    entries = {}
    for c_, (a, b) in enumerate(zip(cols1, cols2)):
        for r_, v in a.items():
            entries[r_, c_] = v
        for r_, v in b.items():
            entries[n1 + r_, c_] = v
    M = matrix(QQ, n1 + n2, ncol, entries)
    bvec = vector(QQ, n1 + n2, {n1 + r_: v for r_, v in rhs.items()})
    r1, r2 = M.rank(), M.augment(bvec.column()).rank()
    print("order-r^2 gauge system: %d unknowns, rank %d, with right-hand side %d -> %s" % (
        ncol, r1, r2, "NOT gauge-equivalent (no g = 1 + O(r) exists)" if r2 > r1 else "solvable at r^2"))
    print("[%.0fs]" % (time.time() - t0))


if __name__ == "__main__":
    if sys.argv[1] in ("build", "verify", "gauge"):
        {"build": build, "verify": verify, "gauge": gauge}[sys.argv[1]]()


def gauge_full():
    """Construct g = 1 + h_1 + h_2 + ... (h_k homogeneous, hom. degree 0, r-degree k, dots allowed)
    with  g d_X = d_F g  order by order; stop when exact or when an order is not solvable."""
    C = Ctx()
    t0 = time.time()
    dF, _ = load(C, F_DOTFREE)
    dX, _ = load(C, F_DOTTED)
    d0, d01 = C.tree(0), C.tree(1)
    one = {I: {I: C.K.idempotent(C.cube.state(C.eps(I)))} for I in C.V}

    def E_of(g):
        a = compose(C, g, dX)
        b = compose(C, dF, g)
        out = {}
        for I in set(a) | set(b):
            for K in set(a.get(I, {})) | set(b.get(I, {})):
                x = a.get(I, {}).get(K, C.K.zero()) - b.get(I, {}).get(K, C.K.zero())
                if not x.is_zero():
                    out[I, K] = x
        return out

    def rdegs(E):
        c = Counter()
        for x in E.values():
            for braid, coeff in x:
                for e in coeff.dict():
                    c[C.rdeg(e)] += 1
        return c

    # orders 1 and 2 jointly (as in gauge()), then one order at a time
    h1, h2 = enum_h(C, 1), enum_h(C, 2)
    E = E_of(one)
    idx1, idx2 = {}, {}
    rhs2 = vectorize(C, E, lambda e: C.rdeg(e) == 2, idx2)
    assert not vectorize(C, E, lambda e: C.rdeg(e) <= 1, {})
    cols1, cols2 = [], []
    for I, J, el in h1:
        cm = comm(C, I, J, el, d01)
        cols1.append(vectorize(C, cm, lambda e: C.rdeg(e) == 1, idx1))
        cols2.append(vectorize(C, cm, lambda e: C.rdeg(e) == 2, idx2))
    for I, J, el in h2:
        cm = comm(C, I, J, el, d0)
        cols1.append({})
        cols2.append(vectorize(C, cm, lambda e: C.rdeg(e) == 2, idx2))
    n1, n2 = len(idx1), len(idx2)
    ent = {}
    for c_, (a, b) in enumerate(zip(cols1, cols2)):
        for r_, v in a.items():
            ent[r_, c_] = v
        for r_, v in b.items():
            ent[n1 + r_, c_] = v
    M = matrix(QQ, n1 + n2, len(cols1), ent)
    bvec = vector(QQ, n1 + n2, {n1 + r_: -v for r_, v in rhs2.items()})
    sol = M.solve_right(bvec)
    print("orders 1-2: unique solution (rank %d of %d); non-zero h-coefficients: %d; integral: %s" % (
        M.rank(), len(cols1), sum(1 for c in sol if c), all(QQ(c).denominator() == 1 for c in sol)))
    g = {I: dict(row) for I, row in one.items()}
    hs = h1 + h2
    for (I, J, el), c in zip(hs, sol):
        if c:
            g.setdefault(I, {})
            g[I][J] = g[I].get(J, C.K.zero()) + C.D(QQ(c)) * el
    for (I, J, el), c in zip(hs, sol):
        if c:
            print("   h: %s -> %s  coefficient %s  element %s" % (sorted(I), sorted(J), c, el))
    for k in range(3, 12):
        E = E_of(g)
        rd = rdegs(E)
        print("after orders < %d: g d_X - d_F g has coefficients by r-degree %s" % (k, dict(sorted(rd.items()))))
        if not rd:
            print("==> g d_X = d_F g EXACTLY: d_F and d_X are gauge-equivalent (g = 1 + h, h in (r), nilpotent)")
            break
        k0 = min(rd)
        hk = enum_h(C, k0)
        idx = {}
        rhs = vectorize(C, E, lambda e: C.rdeg(e) == k0, idx)
        cols = [vectorize(C, comm(C, I, J, el, d0), lambda e: C.rdeg(e) == k0, idx) for I, J, el in hk]
        nr = max(len(idx), 1)
        Mk = matrix(QQ, nr, max(len(cols), 1), {(r_, c_): v for c_, col in enumerate(cols) for r_, v in col.items()})
        bk = vector(QQ, nr, {r_: -v for r_, v in rhs.items()})
        r1, r2 = Mk.rank(), Mk.augment(bk.column()).rank()
        print("order r^%d: %d candidates h, rank %d, with right-hand side %d" % (k0, len(hk), r1, r2))
        if r2 > r1:
            print("==> NOT solvable at r^%d: no gauge g = 1 + O(r) with g d_X = d_F g (given the forced lower orders)" % k0)
            break
        s = Mk.solve_right(bk)
        for (I, J, el), c in zip(hk, s):
            if c:
                g.setdefault(I, {})
                g[I][J] = g[I].get(J, C.K.zero()) + C.D(QQ(c)) * el
    print("[%.0fs]" % (time.time() - t0))


if __name__ == "__main__" and sys.argv[1] == "gauge_full":
    gauge_full()
