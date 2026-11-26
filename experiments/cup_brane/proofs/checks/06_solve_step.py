"""
One-step comparison map  Gamma : C^(m-1) = Cone(Phi_m on the tree cube of w') -> C = tree cube of w.

    sage -python proofs/checks/06_solve_step.py TYPE RANK RED WORD [print]

Complete ansatz; reports existence / uniqueness, and prints the components I -> J, I != J
(these are h_m) when "print" is given.
"""
import sys, importlib
import numpy as np, scipy.sparse as sp
L = importlib.import_module("06_lib")
from cupbrane import solve_integer, rank_mod_p


def solve_step(S, k=None, show=False, verbose=True):
    """chain maps C^(k-1) -> C^(k) (k = m by default) normalised by G_empty = 1."""
    k = S.m if k is None else k
    dA, dB = L.d_mixed(S, k - 1), L.d_mixed(S, k)
    gA, gB = L.deg_mixed(S, k - 1), L.deg_mixed(S, k)
    sys_ = L.ChainMapSystem(S, dA, dB, gA, gB)
    A = sys_.scipy()
    rk = rank_mod_p(A)
    hs = L.ChainMapSystem(S, dA, dB, gA, gB, hdeg=1, sgn=-1)
    empty = frozenset()
    k0 = [i for i, (I, J, b) in enumerate(sys_.vars) if I == empty and J == empty]
    row = sp.csr_matrix(([1], ([0], [k0[0]])), shape=(1, A.shape[1]), dtype=np.int64)
    A2 = sp.vstack([A, row]).tocsr()
    b = sp.csc_matrix(([1], ([A.shape[0]], [0])), shape=(A.shape[0] + 1, 1), dtype=np.int64)
    x = solve_integer(A2, b)
    if verbose:
        print("step {}: unknowns {}, equations {}, solution space dim {} (mod p), homotopy unknowns {}, solution with G_empty=1: {}".format(
            k, len(sys_.vars), A.shape[0], len(sys_.vars) - rk, len(hs.vars),
            "NONE" if x is None else "{} terms, max |c| {}".format(int(np.count_nonzero(x)), int(np.abs(x).max()))), flush=True)
    if x is None:
        return None
    G = sys_.assemble(x)
    if show:
        L.print_map(S, {I: {J: v for J, v in row.items() if k in I} for I, row in G.items()}, skip_diagonal=True)
    return G


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3]); word = tuple(int(c) for c in sys.argv[4])
    S = L.Setup(ct, red, word)
    print("type", ct, "red", red, "word", sys.argv[4], flush=True)
    show = len(sys.argv) > 5
    Gs = {}
    ks = [int(a) for a in sys.argv[6:]] or range(1, S.m + 1)
    for k in ks:
        Gs[k] = solve_step(S, k, show=show)
