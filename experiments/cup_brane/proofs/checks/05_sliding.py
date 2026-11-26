"""
Check of the Sliding Lemma (Lemma 2.1 of proofs/05_main_sliding.md), with exact signs.

Setting: a state s of n strands (black strands of arbitrary colours and one red strand W_k, in
any order), one further black strand of colour c at the far right (position n+1).
X = psi_n psi_{n-1} ... psi_1 moves the extra strand to the far left.  For ANY word
beta = psi_{i_1} ... psi_{i_l} in the crossings of the first n positions (reduced or not):

    (beta (x) 1) X  -  X (1 (x) beta)  =  sum over bad letters t of   coeff_t * E_t

where the letter t (crossing the strands s at position i_t and s' at position i_t + 1) is bad if
   (1) s is black of colour c and s' is black of a colour adjacent to c:  coeff = r_c t_{c, c(s')},
   (2) s is black of colour c = k and s' is the red strand:                coeff = r_k u_k,
and  E_t = (letters before t) (n, n-1, ..., i_t+2) (i_t-1, ..., 1) (letters after t, shifted by +1).

    sage -python 05_sliding.py [N_SAMPLES] [SEED]
"""
import sys, random
from alg import Alg
from degree_model import edges as dynkin_edges


def check(ct, seq, c, beta, verbose=False):
    """seq: list of tokens ('1', '2', 'W1', ...) of the n strands; c: colour of the extra strand."""
    n = len(seq)
    full = list(seq) + [str(c)]
    A = Alg(ct, full)
    E = dynkin_edges(ct)
    X = tuple(range(n, 0, -1))
    lhs = A.word(full, tuple(beta) + X) - A.word(full, X + tuple(i + 1 for i in beta))
    rhs = A.K.zero()
    cur = list(seq)
    n_bad = 0
    for t, i in enumerate(beta):
        s, s2 = cur[i - 1], cur[i]
        coeff = None
        if s == str(c):
            if s2[0] == "W":
                if int(s2[1:]) == c:
                    coeff = A.r(c) * A.u(c)
            elif frozenset((int(s2), c)) in E:
                coeff = A.r(c) * A.t(c, int(s2))
        if coeff is not None:
            n_bad += 1
            word = tuple(beta[:t]) + tuple(range(n, i + 1, -1)) + tuple(range(i - 1, 0, -1)) \
                + tuple(j + 1 for j in beta[t + 1:])
            rhs += coeff * A.word(full, word)
        cur[i - 1], cur[i] = cur[i], cur[i - 1]
    ok = (lhs - rhs).is_zero()
    if verbose or not ok:
        print(ct, seq, "extra", c, "beta", beta, "bad letters", n_bad, "OK" if ok else "FAIL")
    return ok, n_bad


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    n_ok = n_fail = total_bad = 0
    # the three elementary cases, explicitly
    for ct, seq, c, beta in [(["A", 2], ["1", "2"], 1, (1,)), (["A", 2], ["2", "1"], 1, (1,)),
                             (["A", 2], ["1", "W1"], 1, (1,)), (["A", 2], ["W1", "1"], 1, (1,)),
                             (["A", 2], ["1", "1"], 1, (1,)), (["A", 2], ["1", "1"], 2, (1,)),
                             (["A", 3], ["1", "3"], 1, (1,)), (["A", 2], ["1", "W2"], 1, (1,))]:
        ok, nb = check(ct, seq, c, beta, verbose=True)
        n_ok += ok; n_fail += not ok
    for _ in range(N):
        ct = rng.choice([["A", 2], ["A", 3], ["D", 4]])
        rank = ct[1]
        n = rng.randint(2, 6)
        seq = [str(rng.randint(1, rank)) for _ in range(n - 1)]
        seq.insert(rng.randint(0, n - 1), "W%d" % rng.randint(1, rank))
        c = rng.randint(1, rank)
        # random word; never cross... (red-red crossings do not exist: there is one red strand)
        beta = tuple(rng.randint(1, n - 1) for _ in range(rng.randint(1, 7)))
        ok, nb = check(ct, seq, c, beta)
        n_ok += ok; n_fail += not ok; total_bad += nb
    print("sliding lemma: %d cases OK, %d FAIL, %d bad letters in total" % (n_ok, n_fail, total_bad))
