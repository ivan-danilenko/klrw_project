"""
Cyclic case: dimension of the space of chain maps  C^(m-1) -> (C, d_true)  and of its subspace
of null-homotopic maps  d^(m-1) s + s d_true  (complete ansatz; ranks mod p).

    sage -python proofs/checks/06_cyclic_homotopy.py TYPE RANK RED WORD
"""
import sys, os, importlib, time
L = importlib.import_module("06_lib")
from cupbrane import rank_mod_p, case_name
import scipy.sparse as sp, numpy as np

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3]); word = tuple(int(c) for c in sys.argv[4])
    S = L.Setup(ct, red, word)
    m = S.m
    d = L.d_from_json(S, os.path.join(L.HERE, "06_data", case_name(ct, [(red, word)]) + ".json"))
    dA = L.d_mixed(S, m - 1)
    gA = L.deg_mixed(S, m - 1)
    t0 = time.time()
    sys_ = L.ChainMapSystem(S, dA, d, gA, S.deg)
    A = sys_.scipy()
    dim = len(sys_.vars) - rank_mod_p(A)
    index = {}
    for k, (I, J, b) in enumerate(sys_.vars):
        (c,) = L.coords(b).keys()
        index[(I, J), c] = k
    hs = L.ChainMapSystem(S, dA, d, gA, S.deg, hdeg=1, sgn=-1)
    inv_rows = {v: k for k, v in hs.rows.items()}
    rows, cols, data = [], [], []
    for (row, k), v in hs.entries.items():
        if v:
            rows.append(index[inv_rows[row]]); cols.append(k); data.append(v)
    H = sp.csr_matrix((data, (rows, cols)), shape=(len(sys_.vars), max(1, len(hs.vars))), dtype=np.int64)
    rk = rank_mod_p(H) if len(hs.vars) else 0
    print("chain maps: dim {}; homotopies: {} unknowns, null-homotopic chain maps: dim {}; chain maps modulo homotopy: dim {}  [{:.0f}s]".format(
        dim, len(hs.vars), rk, dim - rk, time.time() - t0), flush=True)
