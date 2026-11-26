"""
Cyclic case: does a comparison map G : C_Phi -> (C, d_true) exist, where d_true is the
order-by-order corrected differential stored by 06_cyclic_true_d.py ?

    sage -python proofs/checks/06_cyclic_G.py TYPE RANK RED WORD [triangular]

Complete ansatz (all homogeneous elements of the right degree); with "triangular" only
components I -> J with J <= I in the colourwise dominance order and J != I off the diagonal,
and the diagonal fixed to z_I (see the md file).
"""
import sys, os, importlib, time
import numpy as np, scipy.sparse as sp
L = importlib.import_module("06_lib")
from cupbrane import solve_integer, rank_mod_p, case_name

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3]); word = tuple(int(c) for c in sys.argv[4])
    S = L.Setup(ct, red, word)
    path = os.path.join(L.HERE, "06_data", case_name(ct, [(red, word)]) + ".json")
    d = L.d_from_json(S, path)
    # sanity: d^2 = 0, and tree part
    dd = S.full(d)
    sq = S.compose(dd, dd)
    print("d_true^2 = 0:", all(not row for row in sq.values()), flush=True)
    tree = S.full(S.d)
    n_same = sum(1 for I in tree for J in tree[I] if (tree[I][J] - dd[I].get(J, S.KLRW.zero())).is_zero())
    n_tree = sum(len(r) for r in tree.values()); n_true = sum(len(r) for r in dd.values())
    print("components: tree {}, true {}, equal {}".format(n_tree, n_true, n_same), flush=True)
    t0 = time.time()
    sys_ = L.ChainMapSystem(S, S.dPhi, d, S.degPhi, S.deg)
    A = sys_.scipy()
    print("unknowns {}, equations {}  [{:.0f}s]".format(len(sys_.vars), A.shape[0], time.time() - t0), flush=True)
    rk = rank_mod_p(A)
    print("space of chain maps: dim {}  [{:.0f}s]".format(len(sys_.vars) - rk, time.time() - t0), flush=True)
    empty = frozenset()
    k0 = [k for k, (I, J, b) in enumerate(sys_.vars) if I == empty and J == empty]
    row = sp.csr_matrix(([1], ([0], [k0[0]])), shape=(1, A.shape[1]), dtype=np.int64)
    b = sp.csc_matrix(([1], ([A.shape[0]], [0])), shape=(A.shape[0] + 1, 1), dtype=np.int64)
    x = solve_integer(sp.vstack([A, row]).tocsr(), b)
    print("chain map with G_emptyset = 1:", "NONE" if x is None else "exists, {} terms, max |c| {}".format(int(np.count_nonzero(x)), int(np.abs(x).max())), flush=True)
    if x is not None:
        G = sys_.assemble(x)
        import pickle
        zm = importlib.import_module("06_batch_G").zminus
        diag = [I for I in L.subsets(S.m) if not (G.get(I, {}).get(I, S.KLRW.zero()) - zm(S, I)).is_zero()]
        print("diagonal G_II = z_I for all I:", not diag, "" if not diag else "fails at {} vertices, e.g. {}".format(len(diag), sorted(diag[0])))
        if "print" in sys.argv:
            L.print_map(S, G)
