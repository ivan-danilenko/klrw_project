"""
Solve  d_Phi G = G d  for a chain map G : C_Phi -> C (tree differential d), complete ansatz.

    sage -python proofs/checks/06_solve_G.py TYPE RANK RED WORD [print]

Reports: number of unknowns, dimension of the solution space, dimension of the space of
null-homotopic solutions (d_Phi s + s d), whether a solution with G_emptyset = 1 exists,
and an l1-minimal integer solution.
"""
import sys, importlib
import numpy as np, scipy.sparse as sp
L = importlib.import_module("06_lib")
from cupbrane import solve_integer


def solve(S, verbose=True, d=None, show=False):
    d = S.d if d is None else d
    sys_ = L.ChainMapSystem(S, S.dPhi, d, S.degPhi, S.deg)
    M = sys_.matrix()
    ker = M.right_kernel_matrix()
    # homotopies s : C_Phi -> C, |J| = |I| + 1 ; G = d_Phi s + s d
    hs = L.ChainMapSystem(S, S.dPhi, d, S.degPhi, S.deg, hdeg=1, sgn=-1)
    # image of homotopies inside the G-variables
    index = {}
    for k, (I, J, b) in enumerate(sys_.vars):
        (c,) = L.coords(b).keys()
        index[(I, J), c] = k
    from sage.matrix.constructor import matrix
    from sage.rings.rational_field import QQ
    ent = {}
    for (row, k), v in hs.entries.items():
        key = [kk for kk, vv in hs.rows.items() if vv == row]
    inv_rows = {v: k for k, v in hs.rows.items()}
    for (row, k), v in hs.entries.items():
        if v:
            ent[index[inv_rows[row]], k] = v
    H = matrix(QQ, len(sys_.vars), len(hs.vars), ent, sparse=True)
    rkH = H.rank()
    empty = frozenset()
    k0 = [k for k, (I, J, b) in enumerate(sys_.vars) if I == empty and J == empty]
    if verbose:
        print("unknowns {}, equations {}, solution space dim {}, null-homotopic dim {}, homotopy unknowns {}".format(
            len(sys_.vars), M.nrows(), ker.nrows(), rkH, len(hs.vars)), flush=True)
    assert len(k0) == 1
    # integer l1-minimal solution with G_empty = 1
    A = sys_.scipy()
    row = sp.csr_matrix(([1], ([0], [k0[0]])), shape=(1, A.shape[1]), dtype=np.int64)
    A2 = sp.vstack([A, row]).tocsr()
    b = sp.csc_matrix(([1], ([A.shape[0]], [0])), shape=(A.shape[0] + 1, 1), dtype=np.int64)
    x = solve_integer(A2, b)
    if x is None:
        print("NO SOLUTION with G_empty = 1")
        return sys_, None
    G = sys_.assemble(x)
    if verbose:
        print("l1-minimal integer solution: {} terms, max |coeff| {}".format(int(np.count_nonzero(x)), int(np.abs(x).max())))
    if show:
        L.print_map(S, G, skip_diagonal=True)
    return sys_, G


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3]); word = tuple(int(c) for c in sys.argv[4])
    S = L.Setup(ct, red, word)
    print("type", ct, "red", red, "word", sys.argv[4], flush=True)
    solve(S, show=len(sys.argv) > 5)
