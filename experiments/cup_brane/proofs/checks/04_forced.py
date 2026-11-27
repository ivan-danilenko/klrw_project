"""
Forcedness at first order (proofs/04_first_order.md, section 8, Computation 8.6): the linear algebra.

    sage -python proofs/checks/04_forced.py TYPE RANK RED WORD [WORD ...]
    sage -python proofs/checks/04_forced.py TYPE RANK all MAXLEN [MINLEN]

For each word it enumerates (with 04_degree.py) ALL homogeneous basis elements, dots
included, of the components of maps of homological degree -1 and 0 on the cube, sorted by
their r-degree, and computes over QQ, modulo (r)^2:

  Z1  = { eta in V^{-1}_1 : [delta_0, eta] = delta_0 eta + eta delta_0 = 0 mod r^2 }
  B1  = { [delta_0, h] = delta_0 h - h delta_0 mod r^2 : h in V^0_1 }
        (the first-order gauge directions; B1 is contained in Z1)
  Z0t = { eta in V^{-1}_0 with all terms containing some t : [delta_0, eta] = 0 mod r }
  B0t = { [delta_0, h] mod r : h in V^0_0, all terms containing some t }

where V^h_k = span of the homogeneous basis elements of homological degree h and r-degree
k.  Also: the element delta_1 - (tree formula) and the inhomogeneous equation
delta_0^2 + [delta_0, eps_1] = 0 mod r^2 are solved for eps_1 in V^{-1}_1, and the
solution set is compared with delta_1 + Z1.
"""
import sys, os, itertools, importlib, time
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L = importlib.import_module("04_lib")
Dg = importlib.import_module("04_degree")
from sage.all import matrix, QQ, vector


def dot_monomials(C, I, a):
    """All dot monomials (as base ring elements) with a[c] dots on the strands of colour c
    (dot variables named after the left state e_I)."""
    per = []
    for c, k in sorted(a.items()):
        nstr = sum(1 for s in range(1, C.m + 1) if C.col(s) == c)
        opts = []
        for comp in itertools.product(range(k + 1), repeat=nstr):
            if sum(comp) == k:
                mon = C.D.one()
                for idx, e in enumerate(comp, 1):
                    if e:
                        mon *= C.D("x_%d_%d" % (c, idx)) ** e
                opts.append(mon)
        per.append(opts)
    for combo in itertools.product(*per):
        mon = C.D.one()
        for x in combo:
            mon *= x
        yield mon


def param_monomial(C, n, m, l):
    mon = C.D.one()
    for c, v in n.items():
        if v:
            mon *= C.D("r_%d" % c) ** v
    for (x, y), v in m.items():
        if v:
            mon *= C.D("t_%d_%d" % (x, y)) ** v
    for c, v in l.items():
        if v:
            mon *= C.D("u_%d" % c) ** v
    return mon


MONOTONE = False      # set by the command line flag -M


def is_monotone(I, J, sigma):
    """No strand crosses the red strand from the left to the right."""
    return not any(s not in I and sigma[s - 1] in J for s in range(1, len(sigma) + 1))


def space(C, hdeg, rorder, need_t=False):
    """list of (I, J, element, description)"""
    out = []
    for (I, J, sigma, n, m, l, a) in Dg.all_elements(C.model, hdeg):
        if sum(n.values()) != rorder:
            continue
        if MONOTONE and not is_monotone(I, J, sigma):
            continue
        if need_t and not any(m.values()):
            continue
        base = C.basis(I, J, sigma)
        pm = param_monomial(C, n, m, l)
        for dm in dot_monomials(C, I, a):
            out.append((I, J, (pm * dm) * base, (sigma, n, m, l, a)))
    return out


def commutator(C, d0, I, J, el, sign):
    """[delta_0, el] = delta_0 el + sign * el delta_0, as dict (src, tgt) -> element."""
    res = defaultdict(lambda: C.K.zero())
    for x in range(1, C.m + 1):
        if x not in I:
            Ip = I | {x}
            res[Ip, J] += d0[Ip][I] * el
    for x in J:
        res[I, J - {x}] += C.D(sign) * (el * d0[J][J - {x}])
    return res


def vectorize(C, dct, rdeg, index):
    v = {}
    for (S, T), el in dct.items():
        part = C.r_part(el, rdeg)
        for braid, e, c in C.terms(part):
            key = (S, T, braid, e)
            if key not in index:
                index[key] = len(index)
            v[index[key]] = v.get(index[key], 0) + c
    return {k: c for k, c in v.items() if c}


def mat(cols, nrows):
    M = matrix(QQ, nrows, len(cols), sparse=True)
    for j, col in enumerate(cols):
        for i, c in col.items():
            M[i, j] = c
    return M


def analyse(ct, red, word, verbose=True):
    t0 = time.time()
    C = L.Ctx(ct, red, word)
    V = list(L.subsets(C.m))
    d0 = {I: C.delta0(I) for I in V}
    res = {}
    # ---- first order
    Vm1 = space(C, -1, 1)
    V0 = space(C, 0, 1)
    idx = {}
    colsZ = [vectorize(C, commutator(C, d0, I, J, el, +1), 1, idx) for I, J, el, _ in Vm1]
    nrows = len(idx)
    MZ = mat(colsZ, max(nrows, 1))
    rankZ = MZ.rank()
    res["dim V^-1_1"] = len(Vm1)
    res["dim Z1"] = len(Vm1) - rankZ
    # gauge: [delta_0, h] in V^{-1}_1 coordinates
    idx2 = {}
    vm1_cols = [vectorize(C, {(I, J): el}, 1, idx2) for I, J, el, _ in Vm1]
    colsB = [vectorize(C, commutator(C, d0, I, J, el, -1), 1, idx2) for I, J, el, _ in V0]
    n2 = len(idx2)
    MV = mat(vm1_cols, max(n2, 1))
    MB = mat(colsB, max(n2, 1))
    res["dim V^0_1"] = len(V0)
    res["dim B1"] = MB.rank() if len(colsB) else 0
    if len(colsB):
        # B1 inside span(V^{-1}_1)?
        res["B1 inside V^-1_1"] = (MV.augment(MB).rank() == MV.rank())
    # inhomogeneous equation: delta_0^2 + [delta_0, eps1] = 0 mod r^2
    rhs = defaultdict(lambda: C.K.zero())
    for I in V:
        for J, x in d0[I].items():
            for K, y in d0[J].items():
                rhs[I, K] += x * y
    b = vectorize(C, rhs, 1, idx)
    nrows = len(idx)
    MZ = mat(colsZ, max(nrows, 1))
    bv = vector(QQ, max(nrows, 1), {i: -c for i, c in b.items()})
    try:
        sol = MZ.solve_right(bv)
        res["eps1 solvable"] = True
        # compare with delta_1
        d1vec = {}
        idx3 = {}
        d1 = {}
        for I in V:
            for J, el in C.delta_n(I, 1).items():
                d1[I, J] = el
        dv = vectorize(C, d1, 1, idx3)
        sv = defaultdict(int)
        for k, (I, J, el, _) in enumerate(Vm1):
            if sol[k]:
                for key, c in vectorize(C, {(I, J): el}, 1, idx3).items():
                    sv[key] += sol[k] * c
        diff = {k: sv.get(k, 0) - dv.get(k, 0) for k in set(sv) | set(dv)}
        res["particular solution == delta_1"] = all(v == 0 for v in diff.values())
    except ValueError:
        res["eps1 solvable"] = False
    # ---- order 0 with t
    W0 = space(C, -1, 0, need_t=True)
    H0 = space(C, 0, 0, need_t=True)
    idx4 = {}
    cols0 = [vectorize(C, commutator(C, d0, I, J, el, +1), 0, idx4) for I, J, el, _ in W0]
    res["dim V^-1_0 (with t)"] = len(W0)
    res["dim Z0t"] = len(W0) - (mat(cols0, max(len(idx4), 1)).rank() if cols0 else 0)
    idx5 = {}
    w0cols = [vectorize(C, {(I, J): el}, 0, idx5) for I, J, el, _ in W0]
    colsB0 = [vectorize(C, commutator(C, d0, I, J, el, -1), 0, idx5) for I, J, el, _ in H0]
    res["dim V^0_0 (with t)"] = len(H0)
    res["dim B0t"] = mat(colsB0, max(len(idx5), 1)).rank() if colsB0 else 0
    res["seconds"] = round(time.time() - t0, 1)
    if verbose:
        print("%s red %d word %s: %s" % ("".join(map(str, ct)), red, "".join(map(str, word)),
                                         ", ".join("%s %s" % kv for kv in res.items())), flush=True)
    return C, res, Vm1, V0, MZ


if __name__ == "__main__":
    if "-M" in sys.argv:
        MONOTONE = True
        sys.argv.remove("-M")
        print("restricted to MONOTONE elements (no strand crosses W from left to right)")
    ct = [sys.argv[1], int(sys.argv[2])]
    if sys.argv[3] == "all":
        maxlen = int(sys.argv[4]); minlen = int(sys.argv[5]) if len(sys.argv) > 5 else 1
        summary = Counter()
        for w in L.all_words(ct[1], maxlen, minlen):
            for red in range(1, ct[1] + 1):
                _, res, _, _, _ = analyse(ct, red, w)
                summary["Z1=0" if res["dim Z1"] == 0 else "Z1!=0"] += 1
                summary["B1=0" if res["dim B1"] == 0 else "B1!=0"] += 1
                summary["Z1=B1" if res["dim Z1"] == res["dim B1"] else "Z1!=B1"] += 1
                # the inhomogeneous equation is solvable (Theorem C: delta_1 is a solution); if
                # Z1 = 0 the solution is unique and must be delta_1:
                summary["solvable" if res.get("eps1 solvable") else "NOT solvable"] += 1
                if res["dim Z1"] == 0:
                    summary["Z1=0 and unique solution == delta_1" if res.get("particular solution == delta_1")
                            else "Z1=0 but solution != delta_1 (ERROR)"] += 1
                mult = max(w.count(c) for c in set(w))
                summary["mult<=2: " + ("Z1=0" if res["dim Z1"] == 0 else "Z1!=0") if mult <= 2 else "mult>=3: " + ("Z1=0" if res["dim Z1"] == 0 else "Z1!=0")] += 1
                summary["Z0t=0" if res["dim Z0t"] == 0 else "Z0t!=0"] += 1
                summary["Z0t=B0t" if res["dim Z0t"] == res["dim B0t"] else "Z0t!=B0t"] += 1
        print("SUMMARY", dict(summary))
    else:
        red = int(sys.argv[3])
        for spec in sys.argv[4:]:
            analyse(ct, red, tuple(int(c) for c in spec))
