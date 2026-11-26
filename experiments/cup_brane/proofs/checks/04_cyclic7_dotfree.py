"""
Theorem 8.1 of proofs/04_first_order.md: the tree formula of the cyclic word 1221122
(A_2, red strand of colour 2) cannot be completed by dot-free terms.

    sage -python proofs/checks/04_cyclic7_dotfree.py

Computes
  (1) the number of dot-free homogeneous basis elements of homological degree -1 (all
      components) by r-degree (pure Python enumeration, degree_model.Model.candidates);
  (2) the r-degree-3 part of d_tree^2 (exact, in the KLRW algebra over Z);
  (3) the space Z2 of dot-free elements kappa of r-degree 2 with [delta_0, kappa] = 0 mod r^3,
      and whether (d_tree^2)_3 lies in the span of the r^3-parts of
      [delta_0 + delta_1, kappa] (kappa in Z2): it does not;
  (4) for information: the dimension of the corresponding space Z1 at r-degree 1.
"""
import sys, os, importlib, time
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
L = importlib.import_module("04_lib")
F = importlib.import_module("04_forced")
from sage.all import matrix, QQ, vector

WORD = (1, 2, 2, 1, 1, 2, 2)
RED = 2

if __name__ == "__main__":
    t0 = time.time()
    C = L.Ctx(["A", 2], RED, WORD)
    V = list(L.subsets(C.m))
    cands = C.model.candidates()
    byr = Counter(sum(p for n, p in mono if n.startswith("r")) for I, J, s, mono in cands)
    print("(1) dot-free homogeneous candidates (hom. degree -1) by r-degree:", sorted(byr.items()))
    d0 = {I: C.delta0(I) for I in V}
    d1 = {I: C.delta_n(I, 1) for I in V}
    d2 = {I: C.delta_n(I, 2) for I in V}
    dt = {}
    for I in V:
        row = defaultdict(lambda: C.K.zero())
        for dd in (d0, d1, d2):
            for J, el in dd[I].items():
                row[J] += el
        dt[I] = dict(row)
    sq = L.compose(C, dt, dt)
    sq = {(I, K): v for I, row in sq.items() for K, v in row.items()}
    idx = {}
    obs = F.vectorize(C, sq, 3, idx)
    print("(2) (d_tree^2)_3 has", len(obs), "non-zero coefficients; r-degrees <= 2 vanish:",
          all(len(F.vectorize(C, sq, k, {})) == 0 for k in (0, 1, 2)))
    for k in (1, 2):
        ck = [(I, J, s, mono) for I, J, s, mono in cands if sum(p for n, p in mono if n.startswith("r")) == k]
        idk = {}
        cols = [F.vectorize(C, F.commutator(C, d0, I, J, C.mono(mono) * C.basis(I, J, s), +1), k, idk) for I, J, s, mono in ck]
        M = F.mat(cols, max(len(idk), 1))
        K = M.right_kernel().basis()
        print("(%d) r-degree %d: %d dot-free candidates, kernel of [delta_0, -] mod r^%d: dim %d" % (5 - k if k == 1 else 3, k, len(ck), k + 1, len(K)))
        if k == 2:
            # r^3 part of [delta_0 + delta_1, kappa] for kappa in the kernel
            vecs = []
            for v in K:
                kap = defaultdict(lambda: C.K.zero())
                for i, c in v.dict().items():
                    I, J, s, mono = ck[i]
                    kap[I, J] += C.D(int(c) if c == int(c) else c) * (C.mono(mono) * C.basis(I, J, s))
                tot = defaultdict(lambda: C.K.zero())
                for (I, J), el in kap.items():
                    for dd in (d0, d1):
                        for x in range(1, C.m + 1):
                            if x not in I:
                                Ip = I | {x}
                                if I in dd[Ip]:
                                    tot[Ip, J] += dd[Ip][I] * el
                        for Jp, el2 in dd[J].items():
                            tot[I, Jp] += el * el2
                vecs.append(F.vectorize(C, tot, 3, idx))
            n = len(idx)
            Mk = F.mat(vecs, n) if vecs else matrix(QQ, n, 0)
            ob = vector(QQ, n, obs)
            r1 = Mk.rank(); r2 = Mk.augment(ob.column()).rank()
            print("    span of the r^3-parts of [delta_0 + delta_1, kappa]: rank %d; with (d_tree^2)_3 added: rank %d -> %s" % (
                r1, r2, "obstruction NOT removable" if r2 > r1 else "removable"))
    print("[%.0fs]" % (time.time() - t0))
