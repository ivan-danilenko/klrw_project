"""
Revision of 06_main_intertwiner.md (point D4 of R06_referee.md, finding G0 of R04_referee.md).

Which normal form does the division with h+ produce for the cyclic word 1221122, A_2, red 2?

  F = zeta_7^{-1} ( Phi_7 - (d'(x)1) h+ - h+ (1(x)d') )         (06_divide.divide, H+ = SHARED)

is compared, in every component, with the three stored differentials of this word:

  d_X   04_data/cyclic7_1221122_W2_tree.json      (dotted completion keeping the tree formula's
                                                    r^2-part, 04_first_order.md Thm 9.1(d))
  d_X'  R04_data/cyclic7_1221122_W2_dotted.json   (the referee's independent copy of d_X)
  d_F   R04_data/cyclic7_1221122_W2_dotfree.json  (dot-free completion, Thm 9.1(e))

and with the differential of the iterated division from d_Phi (06_divide_all.divide_step,
k = 1..7).  Then, with the referee's gauge h_g (R04_data/cyclic7_1221122_W2_gauge_h.json,
d_F = (1 + h_g) d_X (1 - h_g)), the division is repeated with the modified homotopy

  h' = h+ - zeta_7 . h_g        (Gamma' = Gamma (1 - h_g) = zeta_7 + h+ - zeta_7 h_g,
                                 since h+ h_g = 0: h+ ends in C_L, h_g starts in C_R)

and its quotient is compared with d_F.  Finally: (1 + h_g) is a chain isomorphism between the
cubes with the quotient differentials, and the h-components of both comparison maps are listed
with their slot sums (triangularity, point D1).  Part (6) does the same for red 1 (the example
of section 8.1, solver differential 06_data/A2__W1_1221122.json): the same terms h_g conjugate
it into a dot-free differential d_F1, which is the quotient of the division with
h+ - zeta_7 h_g, and whose terms outside the tree formula are those of d_F.

    cd proofs/checks && sage -python 06r_gauge_1221122.py
"""
import os, sys, json, importlib, time
from collections import defaultdict
L = importlib.import_module("06_lib")
DV = importlib.import_module("06_divide")
DA = importlib.import_module("06_divide_all")

CT, RED, WORD = ["A", 2], 2, (1, 2, 2, 1, 1, 2, 2)


def load_04(S, path):
    """04_data format: [src, tgt, sigma, params (list of [name, power]), dots (list), coeff]."""
    data = json.load(open(path))
    assert tuple(data["word"]) == S.word and data["red"] == RED
    tab = defaultdict(lambda: defaultdict(lambda: S.KLRW.zero()))
    for src, tgt, sigma, params, dots, coeff in data["terms"]:
        c = S.D(coeff)
        for name, power in list(params) + list(dots):
            c *= S.D(name) ** power
        tab[frozenset(src)][frozenset(tgt)] += c * S.term_element(src, tgt, tuple(sigma))
    return {I: dict(tab.get(I, {})) for I in L.subsets(S.m)}, len(data["terms"])


def load_R04(S, path):
    """R04_data format: [src, tgt, sigma, {name: power}, coeff]."""
    data = json.load(open(path))
    if "word" in data:
        assert tuple(data["word"]) == S.word and data["red"] == RED
    tab = defaultdict(lambda: defaultdict(lambda: S.KLRW.zero()))
    for src, tgt, sigma, mono, coeff in data["terms"]:
        c = S.D(coeff)
        for name, power in mono.items():
            c *= S.D(name) ** power
        tab[frozenset(src)][frozenset(tgt)] += c * S.term_element(src, tgt, tuple(sigma))
    return {I: dict(tab.get(I, {})) for I in L.subsets(S.m)}, len(data["terms"])


def diff_components(S, a, b):
    out = []
    for I in L.subsets(S.m):
        ra, rb = a.get(I, {}), b.get(I, {})
        for K in set(ra) | set(rb):
            if not (ra.get(K, S.KLRW.zero()) - rb.get(K, S.KLRW.zero())).is_zero():
                out.append((I, K))
    return out


def show_diff(S, a, b, comps, na="a", nb="b", limit=12):
    for I, K in comps[:limit]:
        print("    {} -> {}:".format(L.fmt_set(I), L.fmt_set(K)))
        for name, mp in ((na, a), (nb, b)):
            for sigma, w, coeff in L.terms_of(S, I, K, mp.get(I, {}).get(K, S.KLRW.zero())):
                print("        {:<9} sigma {:<22} {}".format(name, L.cycles_str(sigma), coeff.factor()))


def squares_to_zero(S, d):
    return all(not r for r in S.compose(d, d).values())


def n_dotted(S, d):
    """number of (component, basis braid) terms whose coefficient contains a dot variable."""
    n = 0
    for I, row in d.items():
        for K, el in row.items():
            for braid, coeff in el:
                if any(str(v).startswith("x") for v in coeff.variables()):
                    n += 1
    return n


def cube_from_quotient(S, F):
    m = S.m
    def dnew(I):
        I = frozenset(I)
        if m not in I:
            return S.d(I)
        out = {K: v for K, v in S.d(I).items() if m in K}
        out.update(F[I])
        return out
    return S.full(dnew)


def divide_with(S, hrow):
    """zeta_I F_{I,K} = Phi_m [K = I - m] + sum_J d'_R(I->J) h(J->K) - sum_J h(I->J) d'_L(J->K);
    hrow(I) = dict J -> element (the off-diagonal part), zeta = diagonal of step_formula_row."""
    from sage.matrix.constructor import matrix
    from sage.modules.free_module_element import vector
    from sage.rings.rational_field import QQ
    m = S.m
    dA = L.d_mixed(S, m - 1)
    F, failures = {}, []
    for I in L.subsets(m):
        if m not in I:
            continue
        zeta = L.step_formula_row(S, I, m)[I]
        R = defaultdict(lambda: S.KLRW.zero())
        for J, a in dA(I).items():
            if m in J:
                for K, h in hrow(J).items():
                    R[K] += a * h
            else:
                R[J] += a
        for J, h in hrow(I).items():
            for K, b in dA(J).items():
                R[K] -= h * b
        F[I] = {}
        for K, rhs in R.items():
            if rhs.is_zero():
                continue
            basis = S.basis(I, K, S.deg(I) - S.deg(K))
            rows, ent, rv = {}, {}, {}
            for col, b in enumerate(basis):
                for c, v in L.coords(zeta * b).items():
                    ent[rows.setdefault(c, len(rows)), col] = int(v)
            for c, v in L.coords(rhs).items():
                rv[rows.setdefault(c, len(rows))] = int(v)
            try:
                x = matrix(QQ, len(rows), len(basis), ent, sparse=True).solve_right(vector(QQ, len(rows), rv))
            except ValueError:
                failures.append((I, K))
                continue
            el = S.KLRW.zero()
            for c, b in zip(x, basis):
                if c:
                    assert c.denominator() == 1
                    el += S.D(int(c)) * b
            assert (zeta * el - rhs).is_zero()
            F[I][K] = el
    return F, failures


if __name__ == "__main__":
    t0 = time.time()
    L.SHARED = True                                   # formula H+ (Definition 8.1)
    S = L.Setup(CT, RED, WORD)
    m = S.m
    dX, nX = load_04(S, os.path.join(L.HERE, "04_data", "cyclic7_1221122_W2_tree.json"))
    dX2, nX2 = load_R04(S, os.path.join(L.HERE, "R04_data", "cyclic7_1221122_W2_dotted.json"))
    dF, nF = load_R04(S, os.path.join(L.HERE, "R04_data", "cyclic7_1221122_W2_dotfree.json"))
    hg, _ = load_R04(S, os.path.join(L.HERE, "R04_data", "cyclic7_1221122_W2_gauge_h.json"))
    print("stored: d_X {} terms (d^2=0: {}), d_X' {} terms (d^2=0: {}), d_F {} terms (d^2=0: {})".format(
        nX, squares_to_zero(S, dX), nX2, squares_to_zero(S, dX2), nF, squares_to_zero(S, dF)), flush=True)
    print("d_X = d_X' in all components:", not diff_components(S, dX, dX2))
    print("terms with dots: d_X {}, d_F {}".format(n_dotted(S, dX), n_dotted(S, dF)))
    print("d_X vs d_F: {} components differ".format(len(diff_components(S, dX, dF))), flush=True)

    # (1) one division with h+
    F, failures = DV.divide(S)
    print("(1) division with h+: divisible in all components: {} ({} components, {} not divisible)  [{:.0f}s]".format(
        not failures, sum(len(r) for r in F.values()), len(failures), time.time() - t0), flush=True)
    dq = cube_from_quotient(S, F)
    print("    quotient cube d^2 = 0:", squares_to_zero(S, dq))
    print("    terms with dots in the quotient:", n_dotted(S, dq))
    for name, ref in (("d_X (04_data)", dX), ("d_X' (R04_data dotted)", dX2), ("d_F (R04_data dot-free)", dF)):
        comps = diff_components(S, dq, ref)
        print("    quotient vs {}: {} components differ".format(name, len(comps)), flush=True)
        if comps and name.startswith("d_F"):
            show_diff(S, dq, ref, comps, "quotient", "d_F")
    tree = S.full(S.d)
    comps = diff_components(S, dq, tree)
    print("    quotient vs the tree formula: {} components differ".format(len(comps)))
    show_diff(S, dq, tree, comps, "quotient", "tree")

    # (2) iterated division from d_Phi
    d = S.dPhi
    bad = []
    for k in range(1, m + 1):
        d, fl = DA.divide_step(S, d, k)
        if fl:
            bad.append((k, len(fl)))
    dit = S.full(d)
    print("(2) iterated division (k = 1..7): non-divisible steps: {}; equal to the one-step quotient: {}  [{:.0f}s]".format(
        bad or "none", not diff_components(S, dit, dq), time.time() - t0), flush=True)

    # (3) division with h' = h+ - zeta_7 h_g
    def hprime(I):
        I = frozenset(I)
        row = L.step_formula_row(S, I, m)
        out = {J: v for J, v in row.items() if J != I}
        if m in I:
            zeta = row[I]
            for J, v in hg.get(I, {}).items():
                out[J] = out.get(J, S.KLRW.zero()) - zeta * v
        return {J: v for J, v in out.items() if not v.is_zero()}
    F2, failures2 = divide_with(S, hprime)
    dq2 = cube_from_quotient(S, F2)
    print("(3) division with h' = h+ - zeta_7 h_g: divisible: {}; d^2 = 0: {}; vs d_F: {} components differ; vs d_X: {}  [{:.0f}s]".format(
        not failures2, squares_to_zero(S, dq2), len(diff_components(S, dq2, dF)), len(diff_components(S, dq2, dX)),
        time.time() - t0), flush=True)
    # sanity: divide_with(h+) reproduces (1)
    F1b, _ = divide_with(S, lambda I: {J: v for J, v in L.step_formula_row(S, I, m).items() if J != frozenset(I)})
    print("    (divide_with(h+) = 06_divide.divide: {})".format(not diff_components(S, cube_from_quotient(S, F1b), dq)))

    # (4) the gauge as a chain isomorphism between the two quotient cubes
    one = {I: {I: S.eI(I)} for I in L.subsets(m)}
    def plus(a, b, s=1):
        out = {}
        for I in L.subsets(m):
            acc = defaultdict(lambda: S.KLRW.zero())
            for K, v in a.get(I, {}).items():
                acc[K] += v
            for K, v in b.get(I, {}).items():
                acc[K] += s * v
            out[I] = {K: v for K, v in acc.items() if not v.is_zero()}
        return out
    g, gi = plus(one, hg), plus(one, hg, -1)
    conj = S.compose(S.compose(g, dq), gi)
    print("(4) (1 + h_g) d_quotient (1 - h_g) = d_quotient': {}   (h_g^2 = 0: {})".format(
        not diff_components(S, conj, dq2), all(not r for r in S.compose(hg, hg).values())))

    # (5) slot sums of the h-components of the two comparison maps (point D1)
    for name, hr in (("h+", lambda I: {J: v for J, v in L.step_formula_row(S, I, m).items() if J != frozenset(I)}),
                     ("h'", hprime)):
        n = n_inc = n_colex = 0
        for I in L.subsets(m):
            for J in hr(I):
                n += 1
                n_inc += sum(J) >= sum(I)
                n_colex += max(set(I) ^ set(J)) not in I
        print("(5) {}: {} off-diagonal components; with sum(J) >= sum(I): {}; not colex-smaller: {}".format(name, n, n_inc, n_colex))

    # (6) red 1: the same word with the red strand of colour 1 (the example of section 8.1).
    #     d_s = the solver's differential 06_data/A2__W1_1221122.json (= the quotient of the division
    #     with h+, 06_divide.py A 2 1 1221122 plus).  The SAME terms h_g (built in the algebra for
    #     red 1) are homogeneous of degree 0 there too; is (1 + h_g) d_s (1 - h_g) dot-free?  Is it the
    #     quotient of the division with h+ - zeta_7 h_g?  Are its terms outside the tree formula the
    #     same as those of d_F (red 2)?
    def extra_terms(S_, d_):
        tree_ = S_.full(S_.d)
        out = set()
        for I, K in diff_components(S_, d_, tree_):
            el = d_[I].get(K, S_.KLRW.zero()) - tree_[I].get(K, S_.KLRW.zero())
            for sigma, w, coeff in L.terms_of(S_, I, K, el):
                out.add((tuple(sorted(I)), tuple(sorted(K)), sigma, str(coeff)))
        return out
    extra_F2 = extra_terms(S, dF)
    S1 = L.Setup(CT, 1, WORD)
    ds = S1.full(L.d_from_json(S1, os.path.join(L.HERE, "06_data", "A2__W1_1221122.json")))
    F1, fl1 = DV.divide(S1)
    print("(6) red 1: stored solver differential d_s: d^2 = 0: {}; terms with dots: {}; = quotient of the division with h+: {}".format(
        squares_to_zero(S1, ds), n_dotted(S1, ds), not fl1 and not diff_components(S1, ds, cube_from_quotient(S1, F1))), flush=True)
    hj = json.load(open(os.path.join(L.HERE, "R04_data", "cyclic7_1221122_W2_gauge_h.json")))
    hg1 = {I: {} for I in L.subsets(m)}
    homog = True
    for src, tgt, sigma, mono, coeff in hj["terms"]:
        c = S1.D(coeff)
        for name, power in mono.items():
            c *= S1.D(name) ** power
        el = c * S1.term_element(src, tgt, tuple(sigma))
        I, J = frozenset(src), frozenset(tgt)
        homog &= el.degree(check_if_homogeneous=True) == S1.deg(I) - S1.deg(J)
        hg1[I][J] = el
    one1 = {I: {I: S1.eI(I)} for I in L.subsets(m)}
    def plus1(a, b, s=1):
        out = {}
        for I in L.subsets(m):
            acc = defaultdict(lambda: S1.KLRW.zero())
            for K, v in a.get(I, {}).items():
                acc[K] += v
            for K, v in b.get(I, {}).items():
                acc[K] += s * v
            out[I] = {K: v for K, v in acc.items() if not v.is_zero()}
        return out
    dF1 = S1.compose(S1.compose(plus1(one1, hg1), ds), plus1(one1, hg1, -1))
    dF1 = {I: {K: v for K, v in row.items() if not v.is_zero()} for I, row in dF1.items()}
    print("    h_g homogeneous of degree 0 for red 1: {}; d_F1 := (1+h_g) d_s (1-h_g): d^2 = 0: {}; terms with dots: {}".format(
        homog, squares_to_zero(S1, dF1), n_dotted(S1, dF1)))
    def hprime1(I):
        I = frozenset(I)
        row = L.step_formula_row(S1, I, m)
        out = {J: v for J, v in row.items() if J != I}
        if m in I:
            for J, v in hg1.get(I, {}).items():
                out[J] = out.get(J, S1.KLRW.zero()) - row[I] * v
        return {J: v for J, v in out.items() if not v.is_zero()}
    F3, fl3 = divide_with(S1, hprime1)
    print("    division with h+ - zeta_7 h_g: divisible: {}; quotient = d_F1: {}".format(
        not fl3, not fl3 and not diff_components(S1, cube_from_quotient(S1, F3), dF1)))
    extra_F1 = extra_terms(S1, dF1)
    print("    terms of d_F1 outside the tree formula: {}; the same as those of d_F (red 2): {}".format(
        len(extra_F1), extra_F1 == extra_F2))
    print("[{:.0f}s]".format(time.time() - t0))
