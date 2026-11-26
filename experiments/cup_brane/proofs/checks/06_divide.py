"""
The differential DEFINED by the one-step formula.

For the last slot m, let Gamma = z + h be the closed formula of Conjecture H (L.step_formula_row)
and d' the tree differential of the prefix.  Chain map condition  d^(m-1) Gamma = Gamma d  forces,
for I containing m and K not containing m, |K| = |I| - 1:

     zeta_I * F_{I,K} = R_{I,K} :=  Phi_m(I) [K = I - m]  +  sum_J d'_R(I -> J) h(J -> K)  -  sum_J h(I -> J) d'_L(J -> K)

(zeta_I = the diagonal entry of Gamma at I; d'_R, d'_L = components of the tree differential
inside C_R, C_L, with their signs).  This script solves  zeta_I * F = R  for F in the complete
ansatz of the right degree, for every (I, K), and compares F with the tree formula and, if a
file proofs/checks/06_data/<case>.json exists, with the stored true differential.

    sage -python proofs/checks/06_divide.py TYPE RANK RED WORD [print]
"""
import sys, os, importlib, time
from collections import defaultdict
L = importlib.import_module("06_lib")
from cupbrane import case_name


def divide(S, verbose=False):
    m = S.m
    dA = L.d_mixed(S, m - 1)
    from sage.matrix.constructor import matrix
    from sage.modules.free_module_element import vector
    from sage.rings.rational_field import QQ
    F = {}
    failures = []
    for I in L.subsets(m):
        if m not in I:
            continue
        row = L.step_formula_row(S, I, m)
        zeta = row[I]
        R = defaultdict(lambda: S.KLRW.zero())
        for J, a in dA(I).items():
            if m in J:
                for K, h in L.step_formula_row(S, J, m).items():
                    if m not in K:
                        R[K] += a * h              # d'_R then h
            else:
                R[J] += a                           # Phi_m(I) * identity
        for J, h in row.items():
            if J == I:
                continue
            for K, b in dA(J).items():              # J does not contain m: tree differential of the prefix
                R[K] -= h * b
        F[I] = {}
        for K, rhs in R.items():
            if rhs.is_zero():
                continue
            basis = S.basis(I, K, S.deg(I) - S.deg(K))
            rows = {}
            ent = {}
            for col, b in enumerate(basis):
                for c, v in L.coords(zeta * b).items():
                    ent[rows.setdefault(c, len(rows)), col] = int(v)
            rv = {}
            for c, v in L.coords(rhs).items():
                rv[rows.setdefault(c, len(rows))] = int(v)
            M = matrix(QQ, len(rows), len(basis), ent, sparse=True)
            vec = vector(QQ, len(rows), rv)
            try:
                x = M.solve_right(vec)
            except ValueError:
                failures.append((I, K))
                continue
            assert all(c.denominator() == 1 for c in x)
            el = S.KLRW.zero()
            for c, b in zip(x, basis):
                if c:
                    el += S.D(int(c)) * b
            assert (zeta * el - rhs).is_zero()
            F[I][K] = el
    return F, failures


if __name__ == "__main__":
    if "plus" in sys.argv:
        sys.argv.remove("plus"); L.SHARED = True          # formula H+ of section 8
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3]); word = tuple(int(c) for c in sys.argv[4])
    S = L.Setup(ct, red, word)
    m = S.m
    t0 = time.time()
    F, failures = divide(S)
    print("{} red {} word {}: zeta * F = R solvable for all components: {}  ({} components, {} not divisible)  [{:.0f}s]".format(
        "".join(map(str, ct)), red, sys.argv[4], not failures, sum(len(r) for r in F.values()), len(failures), time.time() - t0), flush=True)
    for I, K in failures[:10]:
        print("   NOT divisible:", L.fmt_set(I), "->", L.fmt_set(K))
    # compare with the tree formula
    diff_tree = []
    for I in F:
        t = {K: v for K, v in S.d(I).items() if m not in K}
        for K in set(t) | set(F[I]):
            if not (t.get(K, S.KLRW.zero()) - F[I].get(K, S.KLRW.zero())).is_zero():
                diff_tree.append((I, K))
    print("components of F different from the tree formula:", len(diff_tree))
    path = os.path.join(L.HERE, "06_data", case_name(ct, [(red, word)]) + ".json")
    if os.path.exists(path):
        d = L.d_from_json(S, path)
        diff_true = []
        for I in F:
            t = {K: v for K, v in d(I).items() if m not in K}
            for K in set(t) | set(F[I]):
                if not (t.get(K, S.KLRW.zero()) - F[I].get(K, S.KLRW.zero())).is_zero():
                    diff_true.append((I, K))
        print("components of F different from the stored true differential:", len(diff_true))
    for I, K in diff_tree:
        print("  F[{} -> {}] - tree:".format(L.fmt_set(I), L.fmt_set(K)))
        for sigma, w, coeff in L.terms_of(S, I, K, F[I].get(K, S.KLRW.zero()) - S.d(I).get(K, S.KLRW.zero())):
            print("      sigma {:<22} len {:2d}  {}".format(L.cycles_str(sigma), len(w), coeff.factor()))
    # is the resulting cube a complex?
    def dnew(I):
        if m not in I:
            return S.d(I)
        out = {K: v for K, v in S.d(I).items() if m in K}
        out.update(F[I])
        return out
    if not failures:
        dd = S.full(dnew)
        sq = S.compose(dd, dd)
        print("the cube with F as connecting map satisfies d^2 = 0:", all(not r for r in sq.values()))
