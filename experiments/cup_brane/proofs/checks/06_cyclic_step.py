"""
Cyclic case, one-step version: chain maps
     Gamma : C^(m-1) = Cone(Phi_m on the tree cube of the prefix)  ->  (C, d_true),
d_true = the corrected differential stored by 06_cyclic_true_d.py.  Complete ansatz unless
"restricted" is given (then only the components I -> I and (m in I) -> (m not in J)).

    sage -python proofs/checks/06_cyclic_step.py TYPE RANK RED WORD [restricted] [print]
"""
import sys, os, importlib, time
import numpy as np, scipy.sparse as sp
L = importlib.import_module("06_lib")
from cupbrane import solve_integer, rank_mod_p, case_name

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3]); word = tuple(int(c) for c in sys.argv[4])
    S = L.Setup(ct, red, word)
    m = S.m
    d = L.d_from_json(S, os.path.join(L.HERE, "06_data", case_name(ct, [(red, word)]) + ".json"))
    dA = L.d_mixed(S, m - 1)
    pairs = None
    if "restricted" in sys.argv:
        pairs = lambda I, J: I == J or (m in I and m not in J)
    t0 = time.time()
    sys_ = L.ChainMapSystem(S, dA, d, L.deg_mixed(S, m - 1), S.deg, pairs=pairs)
    A = sys_.scipy()
    rk = rank_mod_p(A)
    print("unknowns {}, equations {}, space of chain maps: dim {}  [{:.0f}s]".format(len(sys_.vars), A.shape[0], len(sys_.vars) - rk, time.time() - t0), flush=True)
    empty = frozenset()
    k0 = [k for k, (I, J, b) in enumerate(sys_.vars) if I == empty and J == empty]
    row = sp.csr_matrix(([1], ([0], [k0[0]])), shape=(1, A.shape[1]), dtype=np.int64)
    b = sp.csc_matrix(([1], ([A.shape[0]], [0])), shape=(A.shape[0] + 1, 1), dtype=np.int64)
    x = solve_integer(sp.vstack([A, row]).tocsr(), b)
    print("chain map with Gamma_emptyset = 1:", "NONE" if x is None else "exists, {} terms, max |c| {}".format(int(np.count_nonzero(x)), int(np.abs(x).max())), flush=True)
    if x is not None:
        G = sys_.assemble(x)
        # compare with the closed formula of Conjecture H
        n_diff = 0
        for I in L.subsets(m):
            F = L.step_formula_row(S, I, m)
            for J in set(F) | set(G.get(I, {})):
                diff = G.get(I, {}).get(J, S.KLRW.zero()) - F.get(J, S.KLRW.zero())
                if not diff.is_zero():
                    n_diff += 1
                    print("  differs from the formula of Conjecture H at {} -> {}:".format(L.fmt_set(I), L.fmt_set(J)))
                    for sigma, w, coeff in L.terms_of(S, I, J, diff):
                        print("      sigma {:<22} len {:2d}  {}".format(L.cycles_str(sigma), len(w), coeff.factor()))
        print("components differing from Conjecture H:", n_diff)
