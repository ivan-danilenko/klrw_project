"""
Decode the dotted terms of the true differential for a configuration whose link graph has a cycle.

    sage -python decode.py data_local/FILE.json "V" "U"

Finds integers c with   truth(V -> U) - (tree formula) = sum c * M * (dot on strand s at end) * beta
where beta is the reduced exchange diagram, M a parameter monomial, s a slot (strand at the
source) and end in {src, tgt}.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cupbrane import *
from fast import braid_for, exp_for
from formula import Stack
from interval import interval_terms


def term_keys(cube, V, U, element):
    """KLRW element (V -> U) -> dict key -> coeff in the (sigma, mono, dots) format."""
    n = cube.n
    eps_V = tuple(1 if s + 1 in V else 0 for s in range(n))
    eps_U = tuple(1 if s + 1 in U else 0 for s in range(n))
    labels = cube.labels(eps_U)
    out = {}
    for braid, params, dots, c in entry_terms(cube.KLRW, element):
        arrive = cube.strand_map(eps_V, braid)
        sigma = {a + 1: t + 1 for a, t in zip(arrive, labels) if not isinstance(a, tuple)}
        key = (tuple(sigma[s] for s in range(1, n + 1)), tuple(sorted(params)), tuple(sorted(dots)))
        out[key] = out.get(key, 0) + c
    return out


def dot_variable(cube, V, slot):
    """Dot variable of the strand sitting in ``slot`` in the state of vertex V."""
    n = cube.n
    eps = tuple(1 if s + 1 in V else 0 for s in range(n))
    labels = cube.labels(eps)
    colour = cube.strand_colour[slot - 1]
    k = 0
    for lab in labels:
        if not isinstance(lab, tuple) and cube.strand_colour[lab] == colour:
            k += 1
            if lab == slot - 1:
                break
    return cube.KLRW.base()("x_%d_%d" % (colour, k))


def decode(path, V, U, verbose=True):
    data = json.load(open(path))
    ct, red, word = data["cartan_type"], data["red"], tuple(data["word"])
    P = [(red, word)]
    KLRW = make_algebra(ct, P); cube = Cube(KLRW, P, build=False); stack = Stack(ct, word)
    D = KLRW.base()
    V, U = frozenset(V), frozenset(U)
    truth = {}
    for src, tgt, sigma, mono, dots, c in data["terms"]:
        if frozenset(src) == V and frozenset(tgt) == U:
            truth[(tuple(sigma), tuple(tuple(x) for x in mono), tuple(tuple(x) for x in dots))] = c
    for J, sigma, mono, sign in stack.terms_from(V):
        if J == U:
            key = (sigma, mono, ())
            truth[key] = truth.get(key, 0) - sign
    truth = {k: v for k, v in truth.items() if v}
    if verbose:
        print("V =", sorted(V), " U =", sorted(U), " out", sorted(V - U), " in", sorted(U - V))
        print("extra terms beyond the tree formula:", len(truth))
    # the exchange diagram: sigma of the dotted terms
    sigmas = set(k[0] for k in truth if k[2])
    assert len(sigmas) == 1, sigmas
    (sigma,) = sigmas
    _, _, braid = braid_for(cube, V, U, sigma)
    beta = KLRW.monomial(braid)
    idem_U = KLRW.idempotent(cube.state(tuple(1 if s + 1 in U else 0 for s in range(cube.n))))
    inv = {sigma[s - 1]: s for s in range(1, cube.n + 1)}
    # candidates
    cands = {}
    for (sg, mono, dots) in truth:
        if not dots:
            continue
        (dname, power), = dots
        assert power == 1
        colour = int(dname.split("_")[1])
        M = D.monomial(*exp_for(KLRW, mono))
        for slot in range(1, cube.n + 1):
            if cube.strand_colour[slot - 1] != colour:
                continue
            el_src = M * dot_variable(cube, V, slot) * beta
            # at the target the strand sits in slot sigma(slot)
            el_tgt = M * (beta * (dot_variable(cube, U, sigma[slot - 1]) * idem_U))
            cands[(mono, slot, "src")] = term_keys(cube, V, U, el_src)
            if el_tgt != el_src:
                cands[(mono, slot, "tgt")] = term_keys(cube, V, U, el_tgt)
    names = sorted(cands)
    rows = sorted(set(k for c in cands.values() for k in c) | set(truth))
    A = sp.lil_matrix((len(rows), len(names)), dtype=np.int64)
    ridx = {k: i for i, k in enumerate(rows)}
    for jx, nm in enumerate(names):
        for k, v in cands[nm].items():
            A[ridx[k], jx] = v
    b = sp.lil_matrix((len(rows), 1), dtype=np.int64)
    for k, v in truth.items():
        b[ridx[k], 0] = v
    x = solve_integer(A.tocsr(), b.tocsc())
    if x is None:
        print("  NO decomposition into dotted reduced diagrams")
        return None
    rank = rank_mod_p(A)
    res = {nm: int(c) for nm, c in zip(names, x) if c}
    if verbose:
        print("  decomposition (unique: %s):" % (rank == len(names)))
        for (mono, slot, end), c in sorted(res.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            side = "R" if slot in V else "L"
            print("    %+d * %-36s dot on strand from slot %2d (c%d, %s) at %s" % (
                c, "*".join(n_ if p == 1 else "%s^%d" % (n_, p) for n_, p in mono), slot, cube.strand_colour[slot - 1], side, end))
    return res


if __name__ == "__main__":
    decode(sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]))
