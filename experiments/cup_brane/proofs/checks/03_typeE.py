"""
Theorem E of proofs/03_cyclic_combinatorics.md: computer proof for E6 (omega_1, omega_6) and
E7 (omega_7).  Pure Python, about a minute.

    python3 proofs/checks/03_typeE.py

For every order ideal I of the minuscule poset (27 resp. 56 ideals) cyclicity is decided by
THREE independent methods, which must agree:
  M1  the exhaustive tree search 03_lib.Word.search          (relies on Lemma 2.3)
  M2  the literal pattern detector 03_patterns.patterns     (relies on Theorem 2.8)
  M3  brute force over ALL configurations: every mover and every set of pairwise disjoint
      equal-coloured pairs avoiding it (03_lib.Word.brute)   (relies on nothing but the definitions)
Then it checks that the cyclic ideals are exactly the ideals containing the unique minimal
one, prints it (weight, root content, word), lists all its cyclic configurations with three
pairs and all its inclusion-minimal cyclic configurations, and checks that the order on
weights (mu <= nu iff nu - mu is a sum of simple roots) is reverse inclusion of ideals.
"""
import importlib, os, sys, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L = importlib.import_module("03_lib")
M = importlib.import_module("03_minuscule")
Pt = importlib.import_module("03_patterns")


def content(word, I, n):
    c = [0] * (n + 1)
    for s in I:
        c[word[s]] += 1
    return tuple(c[1:])


def three_methods(typ, n, k, brute=True):
    word, adj, less, ideals, wt = M.setup(typ, n, k)
    cyc = {}
    t0 = time.time()
    for I in ideals:
        sub = [word[s] for s in I]
        W = L.Word(sub, adj)
        m1 = W.search() is not None
        m2 = bool(Pt.patterns(sub, adj))
        res = [m1, m2]
        if brute:
            res.append(W.brute() is not None)
        assert len(set(res)) == 1, (I, res)
        cyc[I] = m1
    minimal = [I for I in ideals if cyc[I] and not any(cyc[J] for J in ideals if set(J) < set(I))]
    assert len(minimal) == 1
    I0 = minimal[0]
    for I in ideals:
        assert cyc[I] == set(I0).issubset(I)
    # weight order = reverse inclusion (contents: mu = lambda - content)
    for I, J in itertools.product(ideals, repeat=2):
        ci, cj = content(word, I, n), content(word, J, n)
        assert (all(a <= b for a, b in zip(ci, cj))) == set(I).issubset(J)
    print("%s%d omega_%d: %d ideals, %d cyclic; methods M1 = M2%s agree on every ideal [%.1fs]" % (
        typ, n, k, len(ideals), sum(cyc.values()), " = M3" if brute else "", time.time() - t0))
    print("   cyclic  <=>  I contains I_c ;  weight order = reverse inclusion of ideals: checked")
    sub = [word[s] for s in I0]
    print("   I_c: %d slots, weight mu_c = %s (fundamental-weight coordinates), lambda - mu_c = %s" % (
        len(I0), list(wt[I0][1:]), list(content(word, I0, n))))
    print("   word of I_c: %s" % "".join(map(str, sub)))
    W = L.Word(sub, adj)
    # all three-pair cyclic configurations of I_c
    sq = []
    for j in range(W.m):
        av = [p for p in W.pairs if j not in p]
        for F in itertools.combinations(av, 3):
            if len({x for p in F for x in p}) == 6 and W.is_cyclic_config(j, F):
                sq.append((j, F))
    mins = W.minimal_cyclic()
    print("   cyclic configurations of I_c with 3 pairs: %d ; inclusion-minimal cyclic configurations: %d" % (
        len(sq), len(mins)))
    for j, F in sorted(mins):
        sh = L.shape(W, j, F)
        print("      mover %d (colour %d), pairs %s : cycle length %d, mover at distance %d" % (
            j + 1, sub[j], ["(%d,%d)c%d" % (a + 1, b + 1, sub[a]) for a, b in F], sh["cycle_len"], sh["mover_dist"]))
    occ = [s + 1 for s in range(len(sub)) if sub[s] == sub[sq[0][0]]] if sq else []
    print("   slots of the mover's colour in I_c: %s" % occ)
    return cyc, I0


if __name__ == "__main__":
    for typ, n, k in [("E", 6, 1), ("E", 6, 6), ("E", 7, 7)]:
        three_methods(typ, n, k)
