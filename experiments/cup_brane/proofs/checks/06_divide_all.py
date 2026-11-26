"""
The differential defined by iterated division (06_main_intertwiner.md, section 8).

Start with d_0 = d_Phi.  For k = 1, ..., m let Gamma^(k) = zeta_k + h_k be the closed formula
(L.step_formula_row; with "plus" the formula H+ in which a hanging pair may start at a chain
slot), and define d_k from d_{k-1} by changing only the components I -> K with k in I, k not
in K:

     zeta_k(I) * d_k(I -> K)  =  sum_J d_{k-1}(I -> J) Gamma(J -> K)  -  sum_{J != I} Gamma(I -> J) d_{k-1}(J -> K).

The script solves this for d_k(I -> K) in the complete ansatz of the right degree (it reports
the components for which the right side is not divisible), and at the end compares d_m with
the tree formula and tests d_m^2 = 0.  No information about the tree formula or the true
differential is used.

    sage -python proofs/checks/06_divide_all.py TYPE RANK RED WORD [plus] [print]
"""
import sys, os, importlib, time
from collections import defaultdict
L = importlib.import_module("06_lib")
from cupbrane import case_name
from cyclefree import find_cycle


def deg_k(S, k):
    return L.deg_mixed(S, k)


def divide_step(S, dprev, k):
    from sage.matrix.constructor import matrix
    from sage.modules.free_module_element import vector
    from sage.rings.rational_field import QQ
    gam = {}
    def Gam(I):
        if I not in gam:
            gam[I] = L.step_formula_row(S, I, k)
        return gam[I]
    degA = L.deg_mixed(S, k)          # shifts of C^(k): tree-type on slots <= k
    new = {}
    failures = []
    for I in L.subsets(S.m):
        row = dict(dprev(I))
        if k in I:
            zeta = Gam(I)[I]
            R = defaultdict(lambda: S.KLRW.zero())
            for J, a in dprev(I).items():
                for K, g in Gam(J).items():
                    if k not in K:
                        R[K] += a * g
            for J, h in Gam(I).items():
                if J == I:
                    continue
                for K, b in dprev(J).items():
                    R[K] -= h * b
            for K in [K for K in row if k not in K]:
                del row[K]
            for K, rhs in R.items():
                if rhs.is_zero():
                    continue
                basis = S.basis(I, K, degA(I) - degA(K))
                rows, ent, rv = {}, {}, {}
                for col, b in enumerate(basis):
                    for c, v in L.coords(zeta * b).items():
                        ent[rows.setdefault(c, len(rows)), col] = int(v)
                for c, v in L.coords(rhs).items():
                    rv[rows.setdefault(c, len(rows))] = int(v)
                M = matrix(QQ, len(rows), len(basis), ent, sparse=True)
                try:
                    x = M.solve_right(vector(QQ, len(rows), rv))
                except ValueError:
                    failures.append((I, K))
                    continue
                el = S.KLRW.zero()
                for c, b in zip(x, basis):
                    if c:
                        assert c.denominator() == 1
                        el += S.D(int(c)) * b
                assert (zeta * el - rhs).is_zero()
                row[K] = el
        new[I] = row
    return (lambda I: new[frozenset(I)]), failures


def run(ct, red, word, show=False):
    S = L.Setup(ct, red, word)
    t0 = time.time()
    d = S.dPhi
    bad = []
    for k in range(1, S.m + 1):
        d, failures = divide_step(S, d, k)
        if failures:
            bad.append((k, len(failures)))
    cyc = find_cycle(S.stack, max_pairs=4) is not None
    dd = S.full(d)
    sq = S.compose(dd, dd)
    d2 = all(not r for r in sq.values())
    diff = []
    for I in L.subsets(S.m):
        t = S.d(I)
        for K in set(t) | set(dd[I]):
            if not (t.get(K, S.KLRW.zero()) - dd[I].get(K, S.KLRW.zero())).is_zero():
                diff.append((I, K))
    print("{} red {} word {}: cyclic {}; steps with non-divisible components: {}; d^2 = 0: {}; components different from the tree formula: {}  [{:.0f}s]".format(
        "".join(map(str, ct)), red, "".join(map(str, word)), cyc, bad or "none", d2, len(diff), time.time() - t0), flush=True)
    path = os.path.join(L.HERE, "06_data", case_name(ct, [(red, tuple(word))]) + ".json")
    if os.path.exists(path):
        dt = L.d_from_json(S, path)
        n = sum(1 for I in L.subsets(S.m) for K in set(dt(I)) | set(dd[I])
                if not (dt(I).get(K, S.KLRW.zero()) - dd[I].get(K, S.KLRW.zero())).is_zero())
        print("    components different from the stored corrected differential:", n)
    if show:
        for I, K in diff:
            print("  d[{} -> {}] - tree:".format(L.fmt_set(I), L.fmt_set(K)))
            for sigma, w, coeff in L.terms_of(S, I, K, dd[I].get(K, S.KLRW.zero()) - S.d(I).get(K, S.KLRW.zero())):
                print("      sigma {:<22} len {:2d}  {}".format(L.cycles_str(sigma), len(w), coeff.factor()))
    return not bad and d2


if __name__ == "__main__":
    if "plus" in sys.argv:
        sys.argv.remove("plus"); L.SHARED = True
    show = "print" in sys.argv
    if show:
        sys.argv.remove("print")
    ct = [sys.argv[1], int(sys.argv[2])]
    args = sys.argv[3:]
    for red, word in zip(args[0::2], args[1::2]):
        run(ct, int(red), tuple(int(c) for c in word), show)
