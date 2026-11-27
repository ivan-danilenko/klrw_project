"""
Revision of 06_main_intertwiner.md, point D1 of R06_referee.md: for which order is each G
triangular?  (Lemma 5.0, Proposition 5.1, Remark 5.3 of the revised text.)

    J <_c I  (colex)  iff |J| = |I| and sum_{x in J} 2^x < sum_{x in I} 2^x
                      (iff the largest element of the symmetric difference lies in I);
    J  <  I  (sum)    iff |J| = |I| and sum(J) < sum(I).

(0) pure combinatorics, all pairs of subsets of {1..9}: Lemma 5.0 (a) "a bijection
    I\\J -> J\\I lowering every slot  =>  J < I and J <_c I", (b) "k in I\\J and I, J agree
    above k  =>  J <_c I"; and a pair as in (b) with sum(J) >= sum(I) exists ({1,5} -> {3,4}).
(1) G of Definition 6.2 (06_lib.G_formula_row): every off-diagonal component is < and <_c.
(2) factorised G = Gamma^(1) ... Gamma^(m) with the one-step maps of the formula of
    Conjecture H (and H+, 06_lib.SHARED): every component of every h_k satisfies (a) [so it
    is < and <_c]; the composite (06_check_factorisation.composite) is <_c- and <-triangular
    with diagonal prod_k diag(Gamma^(k)) = prod_{x in I} zeta_x.
(3) degree test: one-step shapes I -> J (k in I\\J, agreeing above k, sum(J) >= sum(I)) for
    which the complete ansatz of the degree of a component of Gamma^(k) is non-zero, i.e.
    for which the cone form alone (Proposition 5.2) does not exclude a component that is not
    <-smaller.

    cd proofs/checks && sage -python 06r_triangular.py
"""
import sys, itertools, importlib, time
L = importlib.import_module("06_lib")
FA = importlib.import_module("06_check_factorisation")


def colex_less(J, I):
    return len(J) == len(I) and sum(2 ** x for x in J) < sum(2 ** x for x in I)


def sum_less(J, I):
    return len(J) == len(I) and sum(J) < sum(I)


def lowering_bijection(I, J):
    """is there a bijection I\\J -> J\\I with beta(x) < x for all x?  (Hall: sorted comparison)"""
    a, b = sorted(set(I) - set(J)), sorted(set(J) - set(I))
    return len(a) == len(b) and all(y < x for x, y in zip(a, b))


def part0(n=9):
    ok_a = ok_b = True
    n_a = n_b = 0
    example = None
    for size in range(1, n + 1):
        subs = [frozenset(c) for c in itertools.combinations(range(1, n + 1), size)]
        for I in subs:
            for J in subs:
                if I == J:
                    continue
                if lowering_bijection(I, J):
                    n_a += 1
                    ok_a &= sum_less(J, I) and colex_less(J, I)
                for k in I - J:
                    if all((x in I) == (x in J) for x in range(k + 1, n + 1)):
                        n_b += 1
                        ok_b &= colex_less(J, I)
                        if sum(J) >= sum(I) and example is None and size == 2:
                            example = (sorted(I), sorted(J), k)
    # colex is "largest element of the symmetric difference lies in I"
    ok_c = all(colex_less(J, I) == (max(set(I) ^ set(J)) in I)
               for size in range(1, 6) for I in itertools.combinations(range(1, 7), size)
               for J in itertools.combinations(range(1, 7), size) if I != J)
    print("(0) Lemma 5.0(a) on {} pairs: {};  (b) on {} pairs: {};  colex = max of symmetric difference in I: {};"
          "  a pair as in (b) that is not sum-smaller: {}".format(n_a, ok_a, n_b, ok_b, ok_c, example), flush=True)
    return ok_a and ok_b and ok_c


def part1(ct, red, word):
    S = L.Setup(ct, red, word)
    n = bad = 0
    for I in L.subsets(S.m):
        for J in L.G_formula_row(S, I):
            if J == I:
                continue
            n += 1
            bad += not (sum_less(J, I) and colex_less(J, I) and lowering_bijection(I, J))
    print("(1) Def. 6.2, {} red {} word {}: {} off-diagonal components, not (< and <_c): {}".format(
        "".join(map(str, ct)), red, "".join(map(str, word)), n, bad), flush=True)
    return bad == 0


def zeta_x(S, I, x):
    """zeta_x = prod_{a<x, c(a)=c(x)} (x_a - x_x) at the vertex I."""
    lab = S.labels(I)
    el = S.eI(I)
    for a in range(1, x):
        if S.col(a) == S.col(x):
            el = el * (S.dot_of(lab, a) - S.dot_of(lab, x))
    return el


def part2(ct, red, word, shared=False):
    L.SHARED = shared
    S = L.Setup(ct, red, word)
    t0 = time.time()
    n_h = bad_h = 0
    for k in range(1, S.m + 1):
        for I in L.subsets(S.m):
            for J in L.step_formula_row(S, I, k):
                if J == I:
                    continue
                n_h += 1
                bad_h += not (k in I and k not in J and lowering_bijection(I, J)
                              and all((x in I) == (x in J) for x in range(k + 1, S.m + 1)))
    G = FA.composite(S)
    n_G = bad_c = bad_s = bad_d = 0
    for I in L.subsets(S.m):
        diag = S.eI(I)
        for x in sorted(I):
            diag = diag * zeta_x(S, I, x)
        bad_d += not (G[I].get(I, S.KLRW.zero()) - diag).is_zero()
        for J in G[I]:
            if J == I:
                continue
            n_G += 1
            bad_c += not colex_less(J, I)
            bad_s += not sum_less(J, I)
    L.SHARED = False
    print("(2) {} {} red {} word {}: h-components {}, not (cone shape and lowering): {};  factorised G: {} off-diagonal"
          " components, not <_c: {}, not <: {};  diagonal != prod zeta_x: {}  [{:.0f}s]".format(
              "H+" if shared else "H ", "".join(map(str, ct)), red, "".join(map(str, word)), n_h, bad_h, n_G, bad_c, bad_s,
              bad_d, time.time() - t0), flush=True)
    return bad_h == bad_c == bad_s == bad_d == 0


def part3(ct, red, word):
    S = L.Setup(ct, red, word)
    found = []
    n = 0
    for k in range(2, S.m + 1):
        dA, dB = L.deg_mixed(S, k - 1), L.deg_mixed(S, k)
        for I in L.subsets(S.m):
            if k not in I:
                continue
            for J in L.subsets(S.m, len(I)):
                if k in J or any((x in I) != (x in J) for x in range(k + 1, S.m + 1)) or sum(J) < sum(I):
                    continue
                n += 1
                b = S.basis(I, J, dA(I) - dB(J))
                if b:
                    found.append((k, sorted(I), sorted(J), len(b)))
    print("(3) {} red {} word {}: {} one-step shapes with sum(J) >= sum(I); with a non-zero hom space of the right degree: {}{}".format(
        "".join(map(str, ct)), red, "".join(map(str, word)), n, len(found), "  e.g. " + str(found[:4]) if found else ""), flush=True)
    return found


if __name__ == "__main__":
    ok = part0()
    for ct, red, word in ((["A", 2], 1, (1, 2, 1, 2)), (["A", 3], 2, (2, 1, 3, 2, 1, 3)), (["D", 4], 1, (1, 2, 3, 4, 2, 1)),
                          (["A", 4], 2, (2, 1, 3, 2, 4, 3))):
        ok &= part1(ct, red, word)
    for ct, red, word in ((["A", 3], 2, (2, 1, 3, 2, 1, 3)), (["A", 2], 1, (1, 2, 1, 2, 1)), (["A", 2], 1, (1, 1, 2, 1, 1)),
                          (["A", 3], 1, (1, 2, 3, 1, 3, 1)), (["A", 2], 1, (1, 2, 1, 1, 2, 1)), (["A", 2], 2, (2, 2, 1, 1, 2, 2))):
        ok &= part2(ct, red, word)
    for ct, red, word in ((["A", 2], 1, (1, 2, 1, 1, 2, 1)), (["A", 2], 2, (2, 2, 1, 1, 2, 2)), (["A", 2], 2, (1, 2, 2, 1, 1, 2, 2))):
        ok &= part2(ct, red, word, shared=True)
    for ct, red, word in ((["A", 2], 1, (1, 2, 1, 2, 1)), (["A", 2], 1, (1, 1, 2, 1, 1)), (["A", 3], 2, (2, 1, 3, 2, 1)),
                          (["A", 2], 2, (1, 2, 1, 2)), (["A", 3], 3, (3, 1, 2, 1, 3, 1))):
        part3(ct, red, word)
    print("all triangularity checks (0)-(2):", "OK" if ok else "FAIL")
