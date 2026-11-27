"""
Referee: the explicit gauge h with d_F = (1 + h) d_X (1 - h) for 1221122 (A_2, red 2).

    sage -python proofs/checks/R04_gauge_h.py

Reads ONLY the two JSON files of R04_data/ and the two basis words below (found by
R04_cyc7b.py gauge_full), saves h as R04_data/cyclic7_1221122_W2_gauge_h.json, checks
h^2 = 0 and (1 + h) d_X (1 - h) = d_F exactly, and reports the bijections of h.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from R04_cyc7 import Ctx, compose  # noqa: E402
from R04_cyc7b import load, sigma_of, F_DOTFREE, F_DOTTED, DATA  # noqa: E402

C = Ctx()
dF, _ = load(C, F_DOTFREE)
dX, _ = load(C, F_DOTTED)
H = [((4, 6, 7), (1, 2, 3), (1, 2, 1, 3, 7)), ((4, 5, 6, 7), (1, 2, 3, 5), (1, 5, 6, 7, 6))]
mono = C.D("r_2") ** 2 * C.D("t_2_1") ** 2
h = {}
rec = []
for I, J, w in H:
    I, J = frozenset(I), frozenset(J)
    el = C.K.element_from_braid(w, left_state=C.cube.state(C.eps(I)))
    (braid, coeff), = list(el)
    sg = sigma_of(C, I, J, braid)
    # the same element from the klrw basis braid of sigma
    assert C.basis(I, J, {s: sg[s - 1] for s in range(1, 8)}) == el
    h[I] = {J: -mono * el}
    LR = [s for s in range(1, 8) if s not in I and sg[s - 1] in J]
    RL = [s for s in range(1, 8) if s in I and sg[s - 1] not in J]
    rec.append([sorted(I), sorted(J), sg, {"r_2": 2, "t_2_1": 2}, -1])
    print("h component %s -> %s: sigma %s; crosses W left->right: %s, right->left: %s" % (
        sorted(I), sorted(J), {s: sg[s - 1] for s in range(1, 8) if sg[s - 1] != s}, LR, RL))
json.dump(dict(what="gauge h with d_F = (1+h) d_X (1-h); basis = klrw basis braid", terms=rec),
          open(os.path.join(DATA, "cyclic7_1221122_W2_gauge_h.json"), "w"), indent=0)
hh = compose(C, h, h)
print("h^2 = 0:", all(x.is_zero() for row in hh.values() for x in row.values()))
one = {I: {I: C.K.idempotent(C.cube.state(C.eps(I)))} for I in C.V}
g = {I: dict(row) for I, row in one.items()}
gi = {I: dict(row) for I, row in one.items()}
for I, row in h.items():
    for J, x in row.items():
        g[I][J] = x
        gi[I][J] = -x
lhs = compose(C, compose(C, g, dX), gi)
keys = {(I, J) for I, row in lhs.items() for J in row} | {(I, J) for I, row in dF.items() for J in row}
ok = all((lhs.get(I, {}).get(J, C.K.zero()) - dF.get(I, {}).get(J, C.K.zero())).is_zero() for I, J in keys)
print("(1 + h) d_X (1 - h) = d_F exactly:", ok)
