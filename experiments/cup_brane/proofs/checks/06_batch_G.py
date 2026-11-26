"""
Existence and uniqueness of the comparison map G : C_Phi -> C (tree differential) in the
complete ansatz, for a list of words or random words.

    sage -python proofs/checks/06_batch_G.py TYPE RANK RED WORD [RED WORD ...]
    sage -python proofs/checks/06_batch_G.py TYPE RANK random N MAXLEN MAXMULT [SEED]   (first letter = red colour)

For each word: is the word cyclic (cyclefree.find_cycle), does the tree formula square to zero,
dimension of the space of chain maps, whether a chain map with G_emptyset = 1 exists, and whether
its diagonal is  G_II = prod_{x in I} prod_{i<x, c(i)=c(x)} (x_i - x_x).
"""
import sys, importlib, random, time
import numpy as np, scipy.sparse as sp
L = importlib.import_module("06_lib")
from cupbrane import solve_integer, rank_mod_p
from cyclefree import find_cycle


def zminus(S, I):
    el = S.eI(I)
    for x in sorted(I):
        el = el * (-S.z(I, x)) if sum(1 for i in range(1, x) if S.col(i) == S.col(x)) % 2 else el * S.z(I, x)
    return el


def run(ct, red, word):
    t0 = time.time()
    S = L.Setup(ct, red, word)
    cyc = find_cycle(S.stack, max_pairs=4)
    d = S.full(S.d)
    sq = S.compose(d, d)
    d2 = all(not row for row in sq.values())
    sys_ = L.ChainMapSystem(S, S.dPhi, S.d, S.degPhi, S.deg)
    A = sys_.scipy()
    dim = len(sys_.vars) - rank_mod_p(A)
    empty = frozenset()
    k0 = [k for k, (I, J, b) in enumerate(sys_.vars) if I == empty and J == empty]
    row = sp.csr_matrix(([1], ([0], [k0[0]])), shape=(1, A.shape[1]), dtype=np.int64)
    b = sp.csc_matrix(([1], ([A.shape[0]], [0])), shape=(A.shape[0] + 1, 1), dtype=np.int64)
    x = solve_integer(sp.vstack([A, row]).tocsr(), b)
    diag = None
    if x is not None:
        G = sys_.assemble(x)
        diag = all((G.get(I, {}).get(I, S.KLRW.zero()) - zminus(S, I)).is_zero() for I in L.subsets(S.m))
    print("{} red {} word {}: cyclic {}, tree d^2=0 {}, unknowns {}, chain maps dim {}, G with G_0=1 {}, diagonal = z_I {}, max|c| {}  [{:.0f}s]".format(
        "".join(map(str, ct)), red, "".join(map(str, word)), cyc is not None, d2, len(sys_.vars), dim,
        "exists" if x is not None else "NONE", diag, None if x is None else int(np.abs(x).max()), time.time() - t0), flush=True)
    return x is not None


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    if sys.argv[3] == "random":
        n, maxlen, maxmult = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
        rng = random.Random(int(sys.argv[7]) if len(sys.argv) > 7 else 0)
        seen = set()
        while len(seen) < n:
            Lw = rng.randint(3, maxlen)
            word = tuple(rng.randint(1, ct[1]) for _ in range(Lw))
            red = word[0]
            mult = max(word.count(c) for c in set(word))
            if mult > maxmult or mult < 3 or word in seen:
                continue
            seen.add(word)
            run(ct, red, word)
    else:
        args = sys.argv[3:]
        for red, word in zip(args[0::2], args[1::2]):
            run(ct, int(red), tuple(int(c) for c in word))
