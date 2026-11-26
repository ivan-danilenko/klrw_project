"""
Check of the cone decomposition (Lemma 3.2 / Proposition 3.3 of proofs/05_main_sliding.md).

For a word w = w'.c (last slot m) and every vertex I:
  * m not in I : terms of d(w) at I  =  terms of d(w') at I, same signs; diagram = iota_L(beta')
  * m in I     : (alpha) terms with j < m and no pair containing m  =  terms of d(w') at I' = I - m
                         with the OPPOSITE sign; diagram = iota_R(beta')
                 (beta)  mover m: exactly one term, F empty, sign +1
                 (gamma) the remaining terms contain a pair (a, m)
The equality of diagrams is checked in the algebra of w: the basis braid of the term of w' (a word in
the crossings of the algebra of w'), re-read with the extra strand on the far left / right, gives the
same element as the basis braid of the term of w.

    sage -python 05_cone.py TYPE RANK MAXLEN N_WORDS [SEED]       (arbitrary words, also cyclic ones)
"""
import sys, importlib, itertools, random
L = importlib.import_module("05_lib")


def check_word(ct, w, red=1, elements=True):
    m = len(w)
    S = L.Setup(ct, red, w)
    Sp = L.Setup(ct, red, w[:-1])
    n_terms = 0
    for h in range(0, m + 1):
        for I in itertools.combinations(range(1, m + 1), h):
            I = frozenset(I)
            terms = S.terms(I)
            Ip = I - {m}
            tp = {(t.j, t.F): t for t in Sp.terms(Ip)}
            if m not in I:
                assert {(t.j, t.F) for t in terms} == set(tp), (w, I)
                for t in terms:
                    t0 = tp[t.j, t.F]
                    assert t.sign == t0.sign and t.mono == t0.mono and t.J == t0.J
                    if elements:
                        (b0, _), = list(Sp.beta(Ip, t0.J, t0.sigma))
                        assert S.from_word(I, tuple(x + 1 for x in b0.word())) == S.beta(I, t.J, t.sigma)
            else:
                alpha = [t for t in terms if t.j < m and all(m not in p for p in t.F)]
                beta = [t for t in terms if t.j == m]
                gamma = [t for t in terms if t.j < m and any(m in p for p in t.F)]
                assert len(alpha) + len(beta) + len(gamma) == len(terms)
                assert {(t.j, t.F) for t in alpha} == set(tp), (w, I)
                for t in alpha:
                    t0 = tp[t.j, t.F]
                    assert t.sign == -t0.sign and t.mono == t0.mono and t.J == t0.J | {m}
                    if elements:
                        (b0, _), = list(Sp.beta(Ip, t0.J, t0.sigma))
                        assert S.from_word(I, tuple(b0.word())) == S.beta(I, t.J, t.sigma)
                assert len(beta) == 1 and beta[0].F == () and beta[0].sign == 1 and beta[0].J == Ip
                if elements:   # X_m is the word (m, m-1, ..., 1): the last strand crosses everything
                    assert S.from_word(I, tuple(range(m, 0, -1))) == S.beta(I, Ip, beta[0].sigma)
                for t in gamma:
                    (a,) = [p[0] for p in t.F if p[1] == m]
                    assert w[a - 1] == w[-1] and a not in I and m not in t.J and a in t.J
            n_terms += len(terms)
    return n_terms


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    maxlen, n_words = int(sys.argv[3]), int(sys.argv[4])
    rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    total = 0
    for k in range(n_words):
        n = rng.randint(2, maxlen)
        w = tuple(rng.randint(1, ct[1]) for _ in range(n))
        total += check_word(ct, w, red=rng.randint(1, ct[1]))
    print("cone decomposition: %d words of type %s, %d terms, all checks passed" % (n_words, ct, total))
