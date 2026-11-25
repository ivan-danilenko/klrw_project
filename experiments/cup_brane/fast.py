"""
Fast correction loop: the ansatz at each order consists only of the
degree-allowed dot-free diagrams predicted by ``degree_model.Model``.

    sage -python fast.py TYPE RANK RED [min|max|WORD] [LENGTH]
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cupbrane import *
from cupbrane import _to_csc
from degree_model import Model
from klrw.cython_exts.perfect_complex_corrections_ext import CorrectionsMatrix


def braid_for(cube, I, J, sigma):
    """KLRW braid from vertex I to vertex J realising the slot bijection sigma (1-based tuple)."""
    n = cube.n
    eps_I = tuple(1 if s + 1 in I else 0 for s in range(n))
    eps_J = tuple(1 if s + 1 in J else 0 for s in range(n))
    left = list(cube.labels(eps_I))
    inv = {sigma[s] - 1: s for s in range(n)}  # target slot -> source slot (0-based)
    right = [x if isinstance(x, tuple) else inv[x] for x in cube.labels(eps_J)]
    word = braid_from_marked_state(left, right)
    Braid = cube.KLRW.KLRWBraid
    braid = Braid._element_constructor_(cube.state(eps_J), word)
    assert braid.left_state() == cube.state(eps_I)
    return eps_I, eps_J, braid


def exp_for(KLRW, mono):
    D = KLRW.base()
    el = D.one()
    for name, power in mono:
        el *= D(name) ** power
    (e,) = el.exponents()
    return e


def corrections_matrix(n_rows, n_cols, entries):
    """entries: dict (row, col) -> list of (braid, exp)."""
    braids, exps, entry_ptrs, indices = [], [], [0], []
    indptrs = np.zeros(n_cols + 1, dtype=np.intc)
    by_col = defaultdict(list)
    for (row, col), lst in entries.items():
        by_col[col].append((row, lst))
    count = 0
    for col in range(n_cols):
        for row, lst in sorted(by_col.get(col, []), key=lambda x: x[0]):
            for braid, e in lst:
                braids.append(braid)
                exps.append(e)
            entry_ptrs.append(len(braids))
            indices.append(row)
            count += 1
        indptrs[col + 1] = count
    b_arr = np.empty(len(braids), dtype=object)
    e_arr = np.empty(len(exps), dtype=object)
    for i in range(len(braids)):
        b_arr[i] = braids[i]
        e_arr[i] = exps[i]
    return CorrectionsMatrix(
        b_arr, e_arr, np.array(entry_ptrs, dtype=np.intc), np.array(indices, dtype=np.intc), indptrs, n_rows
    )


def fast_correct(cube, candidates, verbose=True, max_iterations=200):
    """
    ``candidates``: list of (I, J, sigma, mono) from the degree model.
    Returns (d_csc, log) or (None, log) if some order is infeasible within the candidates.
    """
    KLRW = cube.KLRW
    dot_algebra = KLRW.base()
    end_algebra = KLRW.opposite
    degree = -1
    projectives = cube.projectives
    by_exp = defaultdict(list)
    for I, J, sigma, mono in candidates:
        if not mono:
            continue
        e = dot_algebra.etuple_ignoring_dots(exp_for(KLRW, mono))
        by_exp[e].append((I, J, sigma, mono))

    d_csc = _to_csc(KLRW, projectives, cube.d0, degree)
    d_csr = {h: m.to_csr() for h, m in d_csc.items()}
    d_geom_csc = MappingProxyType(dict(d_csc))
    d_geom_csr = MappingProxyType(dict(d_csr))
    log = []
    used = set()
    for it in range(max_iterations):
        exp = min_exp_in_square(d_csr, d_csc, degree=degree)
        if exp is None:
            if verbose:
                print("differential closes; candidate orders never used:",
                      sorted(str(dot_algebra.monomial(*e)) for e in by_exp if e not in used), flush=True)
            return d_csc, log
        exp = dot_algebra.etuple_ignoring_dots(exp)
        monomial = dot_algebra.monomial(*exp)
        if exp in used:
            print("order {} appears again: candidates insufficient".format(monomial))
            return None, log
        used.add(exp)
        t0 = time.time()
        entries = {h: defaultdict(list) for h in d_csc}
        full_exp = None
        for I, J, sigma, mono in by_exp.get(exp, []):
            eps_I, eps_J, braid = braid_for(cube, I, J, sigma)
            h, col = cube.index[eps_I]
            _, row = cube.index[eps_J]
            entries[h][row, col].append((braid, exp_for(KLRW, mono)))
        corrections = {
            h: corrections_matrix(len(projectives[h - 1]), len(projectives[h]), entries[h])
            for h in d_csc
        }
        basis_in_product = {}
        A = system_on_corrections_lhs(
            klrw_algebra=KLRW, d_geom_csc=d_geom_csc, d_geom_csr=d_geom_csr,
            projectives=projectives, corrections=corrections,
            basis_appearing_in_product=basis_in_product, degree=degree,
            relevant_parameter_part=exp, verbose=False,
        )
        try:
            b = system_on_corrections_rhs(
                dot_algebra=dot_algebra, grading_group=KLRW.grading_group,
                d_csc=d_csc, d_csr=d_csr, projectives=projectives,
                basis_appearing_in_product=basis_in_product, degree=degree,
                relevant_parameter_part=exp, verbose=False,
            )
        except AssertionError as err:
            print("order {}: right-hand side not reachable by candidates: {}".format(monomial, err))
            return None, log
        A = sp.csr_matrix(A)
        b = sp.csc_matrix(b)
        if b.shape[0] < A.shape[0]:
            b.resize((A.shape[0], 1))
        x = solve_integer(A, b)
        rec = dict(monomial=str(monomial), shape=A.shape, x=x, rank=rank_mod_p(A), A=A, b=b,
                   variables=[(h, row, col, braid, e) for h in sorted(d_csc) for (row, col), lst in
                              sorted(entries[h].items(), key=lambda kv: (kv[0][1], kv[0][0])) for braid, e in lst])
        log.append(rec)
        if x is None:
            print("order {}: {}x{} INFEASIBLE within candidates".format(monomial, *A.shape))
            return None, log
        if verbose:
            print("order {}: {} eqs x {} unknowns, rank {}, solution {} terms, max |c| {} [{:.1f}s]".format(
                monomial, A.shape[0], A.shape[1], rec["rank"], int(np.count_nonzero(x)),
                int(np.abs(x).max()) if len(x) else 0, time.time() - t0), flush=True)
        x_csc = sp.csc_matrix(x.reshape(-1, 1).astype(np.intc))
        x_csc.sort_indices()
        d_csc = update_differential(KLRW, d_csc, corrections, x_csc, projectives, degree)
        d_csr = {h: m.to_csr() for h, m in d_csc.items()}
    raise RuntimeError("no closure")


def run_fast(ct, punctures, out_dir=None, processes=12, verbose=True):
    t0 = time.time()
    model = Model(ct, punctures)
    cands = model.candidates(processes)
    if verbose:
        print("candidates: {} ({} non-geometric) [{:.1f}s]".format(
            len(cands), sum(1 for c in cands if c[3]), time.time() - t0), flush=True)
    KLRW = make_algebra(ct, punctures)
    cube = Cube(KLRW, punctures)
    if verbose:
        print("cube built [{:.1f}s]".format(time.time() - t0), flush=True)
    d, log = fast_correct(cube, cands, verbose=verbose)
    if d is None:
        return cube, None, log, None
    terms = [term_record(cube, t) for t in differential_terms(cube, d)]
    cand_set = set((I, J, s, m) for I, J, s, m in cands)
    actual = set()
    for t in terms:
        assert not t["dots"]
        actual.add((frozenset(t["src"]), frozenset(t["tgt"]), tuple(t["sigma"]),
                    tuple(sorted((n, p) for n, p in t["params"]))))
    coeffs = sorted(set(abs(t["coeff"]) for t in terms))
    print("terms {}, candidates {}, terms not candidates {}, candidates unused {}, |coeff| values {}".format(
        len(actual), len(cand_set), len(actual - cand_set), len(cand_set - actual), coeffs), flush=True)
    data = dict(
        cartan_type=list(ct), punctures=[[r, list(w)] for r, w in punctures],
        strand_colour=list(cube.strand_colour), strand_puncture=list(cube.strand_puncture),
        orders=[dict(monomial=r["monomial"], equations=int(r["shape"][0]), unknowns=int(r["shape"][1]), rank=int(r["rank"]),
                     solution_terms=int(np.count_nonzero(r["x"]))) for r in log],
        terms=terms, method="restricted to degree-allowed candidates", seconds=time.time() - t0,
    )
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, case_name(ct, punctures) + ".json")
        json.dump(data, open(path, "w"))
        print("saved", path, flush=True)
    return cube, d, log, data


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3])
    spec = sys.argv[4] if len(sys.argv) > 4 else "min"
    length = int(sys.argv[5]) if len(sys.argv) > 5 else None
    if spec in ("min", "max"):
        word = minuscule_word(ct, red, {"min": min, "max": max}[spec], length=length)
    else:
        word = tuple(int(c) for c in spec)
    print("type", ct, "red", red, "word", word, flush=True)
    run_fast(ct, [(red, word)], out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_fast"))
