"""
Check of Proposition 10.2 of proofs/05_main_sliding.md (the four-term bigon identity), for ARBITRARY words.

Slots a < j < j' < b,  c(a) = c(b) = c,  c(j) = e ~ c,  c(j') = e' ~ c,  vertex I with j, j', b in I, a not in I.
Hypotheses:  (H1) no slot of colour c strictly between a and b;
             (H2) no slot < j of colour e, no slot < j' of colour e'   (so e != e').
Then, with beta_0 the reduced diagram  j -> -j, j' -> -j', A: -a -> -b, M: +b -> +a  and dots at e_I,

   P1 = m_j * beta(j'; (a,b))  = (t_{ce} x_A + t_{ec} x_j)    beta_0
   P4 = beta(j'; (a,b)) * m_j  = (t_{ce} x_M + t_{ec} x_j)    beta_0
   P2 = beta(j; (a,b)) * m_j'  = (t_{ce'} x_M + t_{e'c} x_j') beta_0
   P3 = m_j' * beta(j; (a,b))  = (t_{ce'} x_A + t_{e'c} x_j') beta_0

and the four products, with the coefficients of the tree formula, add up to zero.

    sage -python 05_fourterm.py TYPE RANK N_WORDS MAXLEN [SEED]
"""
import sys, importlib, itertools, random
L = importlib.import_module("05_lib")

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    n_words, maxlen = int(sys.argv[3]), int(sys.argv[4])
    rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    n_cases = n_words_used = n_N2 = 0
    tries = 0
    while n_words_used < n_words and tries < 20000:
        tries += 1
        n = rng.randint(4, maxlen)
        w = tuple(rng.randint(1, ct[1]) for _ in range(n))
        quads = []
        for a, j, jp, b in itertools.combinations(range(1, n + 1), 4):
            c, e, ep = w[a - 1], w[j - 1], w[jp - 1]
            if w[b - 1] != c or c in w[a:b - 1]:
                continue
            if e in w[:j - 1] or ep in w[:jp - 1]:
                continue
            quads.append((a, j, jp, b))
        if not quads:
            continue
        S = L.Setup(ct, rng.randint(1, ct[1]), w)
        if not all(S.st.adjacent(a, j) and S.st.adjacent(a, jp) for a, j, jp, b in quads):
            quads = [q for q in quads if S.st.adjacent(q[0], q[1]) and S.st.adjacent(q[0], q[2])]
            if not quads:
                continue
        n_words_used += 1
        for a, j, jp, b in quads:
            c, e, ep = w[a - 1], w[j - 1], w[jp - 1]
            others = [x for x in range(1, n + 1) if x not in (a, j, jp, b)]
            for _ in range(4):
                I = frozenset([j, jp, b] + [x for x in others if rng.random() < 0.5])
                K = (I - {j, jp, b}) | {a}
                sig = list(range(1, n + 1)); sig[a - 1], sig[b - 1] = b, a
                beta0 = S.beta(I, K, tuple(sig))
                xA, xM, xj, xjp = (L.dot(S, I, x) for x in (a, b, j, jp))
                tce = S.mono((("t_%d_%d" % (c, e), 1),)); tec = S.mono((("t_%d_%d" % (e, c), 1),))
                tcep = S.mono((("t_%d_%d" % (c, ep), 1),)); tepc = S.mono((("t_%d_%d" % (ep, c), 1),))
                pred = {(j, (), jp, ((a, b),)): (tce * xA + tec * xj) * beta0,
                        (jp, ((a, b),), j, ()): (tce * xM + tec * xj) * beta0,
                        (j, ((a, b),), jp, ()): (tcep * xM + tepc * xjp) * beta0,
                        (jp, (), j, ((a, b),)): (tcep * xA + tepc * xjp) * beta0}
                total = S.K.zero()
                found = 0
                for t1 in S.terms(I):
                    for t2 in S.terms(t1.J):
                        if t2.J != K:
                            continue
                        key = (t1.j, t1.F, t2.j, t2.F)
                        assert key in pred, (w, sorted(I), key)      # no other product into K
                        assert S.beta(I, t1.J, t1.sigma) * S.beta(t1.J, t2.J, t2.sigma) == pred[key]
                        total += S.el(I, t1) * S.el(t1.J, t2)
                        found += 1
                assert found == 4 and total.is_zero()
                n_cases += 1
                n_N2 += (c in w[:a - 1])
    print("four-term identity (Prop. 10.2), type %s: %d words, %d cases (%d of them with a further slot of colour c "
          "below a), all four products as predicted, sum zero" % (ct, n_words_used, n_cases, n_N2))
