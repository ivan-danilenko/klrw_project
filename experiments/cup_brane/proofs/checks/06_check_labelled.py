"""
Checks of the labelled calculus (06_main_intertwiner.md, section 3):

 (a) D(s -> s') does not depend on the reduced word (random reduced words);
 (b) product rule  D(s -> s') D(s' -> s'') = prod_{pairs inverted twice} q_{ij} * D(s -> s'');
 (c) dots slide:  x_i D(s -> s') = D(s -> s') x_i  (dot on the labelled strand i);
 (d) for words with every colour at most twice: every tree term beta(I; j, F) satisfies
        beta = [sigma_F]                                  if the mover has no earlier twin,
        (x_a - x_j) beta = [sigma_F] + r [sigma_F (a j)]  if a < j is the twin of the mover j.

    sage -python proofs/checks/06_check_labelled.py TYPE RANK RED WORD [N_SAMPLES]
"""
import sys, importlib, random, itertools
L = importlib.import_module("06_lib")

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3]); word = tuple(int(c) for c in sys.argv[4])
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    S = L.Setup(ct, red, word)
    rng = random.Random(1)
    labels = list(range(1, S.m + 1)) + ["W"]
    na = nb = nc = 0
    for _ in range(n):
        s = tuple(rng.sample(labels, len(labels)))
        s1 = tuple(rng.sample(labels, len(labels)))
        s2 = tuple(rng.sample(labels, len(labels)))
        d1 = L.Dlab(S, s, s1)
        assert (d1 - L.Dlab(S, s, s1, rng)).is_zero() and (d1 - L.Dlab(S, s, s1, rng)).is_zero(); na += 1
        d2 = L.Dlab(S, s1, s2)
        d12 = L.Dlab(S, s, s2)
        q = S.D.one()
        for pair in L.inverted_pairs(s, s1) & L.inverted_pairs(s1, s2):
            i, j = tuple(pair)
            q *= L.qfactor(S, s, i, j)
        assert (d1 * d2 - q * d12).is_zero(), (s, s1, s2); nb += 1
        for i in range(1, S.m + 1):
            assert (S.dot_of(s, i) * d1 - d1 * S.dot_of(s1, i)).is_zero(); nc += 1
    print("(a) independence of the reduced word: {} OK; (b) product rule: {} OK; (c) dots slide: {} OK".format(na, nb, nc), flush=True)
    if max(word.count(c) for c in set(word)) <= 2:
        twin = {b: a for a, b in L.pairs_of(S)}
        nd = 0
        for I in L.subsets(S.m):
            lab = S.labels(I)
            for term in S.stack.terms_from(I):
                J, sigma, mono, sign = term
                (j,) = [s for s in I if s not in J and sigma[s - 1] == s]
                beta = S.term_element(I, J, sigma)
                br = L.bracket(S, I, J, sigma)
                if j in twin:
                    a = twin[j]
                    tau = list(sigma); tau[a - 1], tau[j - 1] = j, a      # sigma o (a j): a -> j, j -> a
                    lhs = (S.dot_of(lab, a) - S.dot_of(lab, j)) * beta
                    rhs = br + S.r(S.col(j)) * L.bracket(S, I, J, tuple(tau))
                    assert (lhs - rhs).is_zero(), (sorted(I), j, sigma)
                else:
                    assert (beta - br).is_zero(), (sorted(I), j, sigma)
                nd += 1
        print("(d) psi-expansion of the tree terms: {} terms OK".format(nd))
