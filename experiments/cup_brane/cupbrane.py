"""
Tools for studying the corrected differential of stacks of nested U-branes
(stable envelopes) over the deformed KLRW algebra.

Run with ``sage -python`` from the repository root (or with the repository
root on ``sys.path``).

Layout convention (one puncture, red colour ``k``, word ``(i_1, ..., i_m)``):
the vertex ``I`` of the cube (``I`` = set of strands to the right of the red
strand) is the idempotent

    [strands not in I, reversed order]  W_k  [strands in I, in order]

so strand 1 is the innermost U.  Several punctures are placed side by side.
The differential has homological degree -1 and moves one strand from the
right of its puncture to the left.
"""

import sys
import os
import itertools
import json
import time
from collections import defaultdict
from types import MappingProxyType

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import scipy.sparse as sp

import sage.all  # noqa: F401
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.combinat.root_system.cartan_type import CartanType

from klrw.framed_dynkin import (
    FramedDynkinDiagram_with_dimensions,
    NodeInFramedQuiver,
    NodeInFramedQuiverWithMarks,
)
from klrw.klrw_algebra import KLRWAlgebra
from klrw.perfect_complex import KLRWIrreducibleProjectiveModule, KLRWPerfectComplex
from klrw.cython_exts.sparse_csc import CSC_Mat
from klrw.perfect_complex_corrections import (
    possible_corrections,
    system_on_corrections_lhs,
    system_on_corrections_rhs,
    min_exp_in_square,
)
from klrw.cython_exts.perfect_complex_corrections_ext import update_differential
from klrw.stable_envelopes import braid_from_marked_state


# ---------------------------------------------------------------------------
# algebra
# ---------------------------------------------------------------------------
def make_algebra(cartan_type, punctures, base=ZZ, invertible=False, **extra):
    """
    ``punctures`` is a list of ``(red_colour, word)``.
    """
    ct = CartanType(cartan_type)
    DD = FramedDynkinDiagram_with_dimensions(ct)
    for red, word in punctures:
        DD[NodeInFramedQuiver(red, framing=True)] += 1
        for c in word:
            DD[NodeInFramedQuiver(c)] += 1
    options = dict(
        dot_scaling=True,
        edge_scaling=True,
        vertex_scaling=True,
        invertible_parameters=invertible,
        warnings=True,
    )
    options.update(extra)
    return KLRWAlgebra(base, DD, **options)


# ---------------------------------------------------------------------------
# the cube
# ---------------------------------------------------------------------------
class Cube:
    """
    Cube of T-branes for a list of punctures with nested U-branes.

    Strands are numbered globally 0..n-1 (puncture after puncture, innermost
    first).  A vertex is a 0/1 tuple ``eps``; ``eps[s] == 1`` means strand
    ``s`` is to the right of its puncture.
    """

    def __init__(self, KLRW, punctures):
        self.KLRW = KLRW
        self.punctures = [(red, tuple(word)) for red, word in punctures]
        self.strand_colour = []
        self.strand_puncture = []
        for p, (red, word) in enumerate(self.punctures):
            for c in word:
                self.strand_colour.append(c)
                self.strand_puncture.append(p)
        self.n = len(self.strand_colour)
        self.marked_strand = tuple(
            NodeInFramedQuiverWithMarks(NodeInFramedQuiver(c), s)
            for s, c in enumerate(self.strand_colour)
        )
        self.marked_red = tuple(
            NodeInFramedQuiverWithMarks(NodeInFramedQuiver(red, framing=True), p)
            for p, (red, _) in enumerate(self.punctures)
        )
        self._build()

    # -- states -------------------------------------------------------------
    def marked_state(self, eps):
        state = []
        for p in range(len(self.punctures)):
            strands = [s for s in range(self.n) if self.strand_puncture[s] == p]
            state += [self.marked_strand[s] for s in reversed(strands) if eps[s] == 0]
            state.append(self.marked_red[p])
            state += [self.marked_strand[s] for s in strands if eps[s] == 1]
        return tuple(state)

    def labels(self, eps):
        """Positions -> strand number, or ('W', p) for red strands."""
        out = []
        for node in self.marked_state(eps):
            if node.unmark().is_framing():
                out.append(("W", node.mark))
            else:
                out.append(node.mark)
        return tuple(out)

    def state(self, eps):
        State = self.KLRW.KLRWBraid.KLRWstate_set
        return State._element_constructor_(
            tuple(x.unmark() for x in self.marked_state(eps))
        )

    # -- complex ------------------------------------------------------------
    def _build(self):
        KLRW = self.KLRW
        Braid = KLRW.KLRWBraid
        n = self.n
        self.vertices = defaultdict(list)  # hom_deg -> list of eps
        self.index = {}  # eps -> (hom_deg, position)
        for eps in itertools.product(range(2), repeat=n):
            h = sum(eps)
            self.index[eps] = (h, len(self.vertices[h]))
            self.vertices[h].append(eps)
        self.vertices = dict(self.vertices)

        zero = (0,) * n
        eq_degree = {zero: KLRW.grading_group.zero()}
        d0 = {h: {} for h in range(1, n + 1)}
        for h in range(1, n + 1):
            for eps in self.vertices[h]:
                src_marked = self.marked_state(eps)
                for s in range(n):
                    if eps[s] != 1:
                        continue
                    new = eps[:s] + (0,) + eps[s + 1 :]
                    sign = -1 if sum(eps[s + 1 :]) % 2 else 1
                    tgt_marked = self.marked_state(new)
                    word = braid_from_marked_state(src_marked, tgt_marked)
                    braid = Braid._element_constructor_(self.state(new), word)
                    assert braid.left_state() == self.state(eps)
                    entry = KLRW.base()(sign) * KLRW.monomial(braid)
                    deg = eq_degree[new] - entry.degree(check_if_homogeneous=True)
                    if eps in eq_degree:
                        assert eq_degree[eps] == deg
                    else:
                        eq_degree[eps] = deg
                    d0[h][self.index[new][1], self.index[eps][1]] = entry
        self.eq_degree = eq_degree
        self.projectives = {
            h: [
                KLRWIrreducibleProjectiveModule(self.state(eps), eq_degree[eps])
                for eps in self.vertices[h]
            ]
            for h in range(n + 1)
        }
        self.d0 = d0

    # -- helpers ------------------------------------------------------------
    def subset(self, eps):
        return tuple(s + 1 for s in range(self.n) if eps[s])

    def strand_map(self, src_eps, braid):
        """
        For a braid from the idempotent of ``src_eps``: tuple giving, for every
        position of the *right* state, the label of the strand ending there.
        """
        lab = list(self.labels(src_eps))
        for letter in braid.word():
            lab[letter - 1], lab[letter] = lab[letter], lab[letter - 1]
        return tuple(lab)


# ---------------------------------------------------------------------------
# integer solver
# ---------------------------------------------------------------------------
def solve_integer(A, b, objective="l1", weights=None):
    """
    Integer solution of ``A x = b`` (scipy sparse ``A``, sparse column ``b``).

    ``objective="l1"`` minimises the (weighted) l1 norm with HiGHS;
    the result is verified exactly in integer arithmetic.
    Returns a dense numpy int64 vector, or ``None`` if infeasible.
    """
    from scipy.optimize import linprog, milp, LinearConstraint, Bounds

    A = sp.csr_matrix(A).astype(np.int64)
    bvec = np.asarray(sp.csr_matrix(b).todense()).reshape(-1).astype(np.int64)
    p, j = A.shape
    if j == 0:
        return np.zeros(0, dtype=np.int64) if not bvec.any() else None
    w = np.ones(j) if weights is None else np.asarray(weights, dtype=float)
    c = np.concatenate([w, w])
    Aeq = sp.hstack([A, -A]).tocsr().astype(float)
    res = linprog(c, A_eq=Aeq, b_eq=bvec.astype(float), bounds=(0, None), method="highs")
    if res.status == 2:
        return None
    if res.status != 0:
        raise RuntimeError("linprog failed: " + res.message)
    x = res.x[:j] - res.x[j:]
    xr = np.rint(x).astype(np.int64)
    if np.abs(x - xr).max() > 1e-6 or (A @ xr != bvec).any():
        res = milp(
            c,
            constraints=LinearConstraint(Aeq, bvec.astype(float), bvec.astype(float)),
            integrality=np.ones(2 * j),
            bounds=Bounds(0, np.inf),
        )
        if res.status == 2:
            return None
        if res.x is None:
            raise RuntimeError("milp failed: " + res.message)
        x = res.x[:j] - res.x[j:]
        xr = np.rint(x).astype(np.int64)
    assert (A @ xr == bvec).all(), "rounded solution is not exact"
    return xr


def rank_mod_p(A, p=32003):
    """Rank of a scipy sparse integer matrix over GF(p) (a lower bound over QQ)."""
    from sage.matrix.constructor import matrix
    from sage.rings.finite_rings.finite_field_constructor import GF

    A = sp.coo_matrix(A)
    if A.shape[0] == 0 or A.shape[1] == 0:
        return 0
    entries = defaultdict(int)
    for i, jj, v in zip(A.row, A.col, A.data):
        entries[int(i), int(jj)] += int(v)
    M = matrix(GF(p), A.shape[0], A.shape[1], entries, sparse=True)
    return M.rank()


# ---------------------------------------------------------------------------
# the correction loop
# ---------------------------------------------------------------------------
def _to_csc(KLRW, projectives, differential, degree=-1):
    d_csc = {}
    for h, mat in differential.items():
        if isinstance(mat, CSC_Mat):
            d_csc[h] = mat
        else:
            if not isinstance(mat, dict):
                mat = mat.dict()
            d_csc[h] = CSC_Mat.from_dict(
                dict(mat),
                number_of_rows=len(projectives[h + degree]),
                number_of_columns=len(projectives[h]),
            )
        d_csc[h].change_ring(KLRW.opposite)
    return d_csc


def correct_differential(
    KLRW,
    projectives,
    differential,
    degree=-1,
    max_iterations=40,
    verbose=True,
    objective="l1",
    weight_function=None,
    restrict=None,
    record_rank=False,
    parallel_processes=1,
):
    """
    Order-by-order correction of ``differential`` (as in
    ``klrw.perfect_complex_corrections.corrected_diffirential_csc``), with an
    l1-minimal integer solution at every order and a log of every step.

    ``restrict(hom_deg, row, col, braid, exp) -> bool`` optionally filters the
    ansatz (variables for which it returns ``False`` are forced to zero).
    ``weight_function(hom_deg, row, col, braid, exp) -> float`` weights the l1 norm.

    Returns ``(d_csc, log)``.
    """
    end_algebra = KLRW.opposite
    dot_algebra = KLRW.base()
    d_csc = _to_csc(KLRW, projectives, differential, degree)
    d_csr = {h: m.to_csr() for h, m in d_csc.items()}
    log = []

    exp = min_exp_in_square(d_csr, d_csc, degree=degree)
    if exp is None:
        if verbose:
            print("differential closes without corrections")
        return d_csc, log

    def geometric_part(entry):
        return end_algebra(entry.value.geometric_part())

    d_geom_csc = {
        h: m.apply_entrywise(geometric_part, inplace=False).eliminate_zeros()
        for h, m in d_csc.items()
    }
    d_geom_csr = {h: m.to_csr() for h, m in d_geom_csc.items()}
    d_geom_csc = MappingProxyType(d_geom_csc)
    d_geom_csr = MappingProxyType(d_geom_csr)

    exp = dot_algebra.etuple_ignoring_dots(exp)
    assert not exp.is_constant(), "differential does not square to zero geometrically"

    for it in range(max_iterations):
        t0 = time.time()
        monomial = dot_algebra.monomial(*exp)
        corrections = possible_corrections(
            KLRW,
            projectives,
            degree=degree,
            relevant_parameter_part=exp,
            parallel_processes=parallel_processes,
            verbose=False,
        )
        basis_in_product = {}
        A = system_on_corrections_lhs(
            klrw_algebra=KLRW,
            d_geom_csc=d_geom_csc,
            d_geom_csr=d_geom_csr,
            projectives=projectives,
            corrections=corrections,
            basis_appearing_in_product=basis_in_product,
            degree=degree,
            relevant_parameter_part=exp,
            verbose=False,
        )
        b = system_on_corrections_rhs(
            dot_algebra=dot_algebra,
            grading_group=KLRW.grading_group,
            d_csc=d_csc,
            d_csr=d_csr,
            projectives=projectives,
            basis_appearing_in_product=basis_in_product,
            degree=degree,
            relevant_parameter_part=exp,
            verbose=False,
        )
        A = sp.csr_matrix(A)
        b = sp.csc_matrix(b)
        if b.shape[0] < A.shape[0]:
            b.resize((A.shape[0], 1))

        # variable table: (hom_deg, row, col, braid, exp), in the column order of A
        variables = []
        for h in sorted(d_csc.keys()):
            cm = corrections[h]
            cd = cm.dict()
            # the column order follows the storage order of the CorrectionsMatrix
            indptrs = np.asarray(cm._indptrs())
            indices = np.asarray(cm._indices())
            entry_ptrs = np.asarray(cm._entry_ptrs())
            braids = np.asarray(cm._corrections_braid())
            exps = np.asarray(cm._corrections_exp())
            for col in range(len(indptrs) - 1):
                for a in range(indptrs[col], indptrs[col + 1]):
                    row = int(indices[a])
                    for k in range(entry_ptrs[a], entry_ptrs[a + 1]):
                        variables.append((h, row, col, braids[k], exps[k]))
        assert len(variables) == A.shape[1], (len(variables), A.shape)

        keep = None
        weights = None
        if restrict is not None:
            keep = np.array([bool(restrict(*v)) for v in variables])
        if weight_function is not None:
            weights = np.array([float(weight_function(*v)) for v in variables])

        if keep is not None:
            cols = np.nonzero(keep)[0]
            x_small = solve_integer(
                A[:, cols], b, objective, None if weights is None else weights[cols]
            )
            if x_small is None:
                x = None
            else:
                x = np.zeros(A.shape[1], dtype=np.int64)
                x[cols] = x_small
        else:
            x = solve_integer(A, b, objective, weights)

        record = dict(
            iteration=it,
            exp=exp,
            monomial=str(monomial),
            shape=A.shape,
            nnz_rhs=int(b.nnz),
            variables=variables,
            A=A,
            b=b,
            x=x,
        )
        if record_rank:
            record["rank"] = rank_mod_p(A)
        log.append(record)
        if x is None:
            if verbose:
                print("order {}: system {}x{} is INFEASIBLE".format(monomial, *A.shape))
            return None, log

        if verbose:
            extra = ""
            if record_rank:
                extra = ", rank {} (kernel {})".format(
                    record["rank"], A.shape[1] - record["rank"]
                )
            print(
                "order {}: {} eqs x {} unknowns{}, rhs nnz {}, solution: {} terms, max |c| {}  [{:.1f}s]".format(
                    monomial,
                    A.shape[0],
                    A.shape[1],
                    extra,
                    b.nnz,
                    int(np.count_nonzero(x)),
                    int(np.abs(x).max()) if len(x) else 0,
                    time.time() - t0,
                )
            )

        x_csc = sp.csc_matrix(x.reshape(-1, 1).astype(np.intc))
        x_csc.sort_indices()
        d_csc = update_differential(KLRW, d_csc, corrections, x_csc, projectives, degree)
        d_csr = {h: m.to_csr() for h, m in d_csc.items()}

        exp = min_exp_in_square(d_csr, d_csc, degree=degree)
        if exp is None:
            if verbose:
                print("differential closes")
            return d_csc, log
        exp = dot_algebra.etuple_ignoring_dots(exp)

    raise RuntimeError("no closure after {} iterations".format(max_iterations))


# ---------------------------------------------------------------------------
# reading entries
# ---------------------------------------------------------------------------
def split_exponent(KLRW, exp):
    """ETuple -> (parameter monomial as tuple of (name, power), dots as same)."""
    D = KLRW.base()
    names = D.variable_names()
    params, dots = [], []
    for pos, power in exp.sparse_iter():
        name = names[pos]
        (dots if name.startswith("x_") else params).append((name, int(power)))
    return tuple(params), tuple(dots)


def entry_terms(KLRW, entry):
    """Opposite-algebra (or KLRW) element -> list of (braid, params, dots, coeff)."""
    value = getattr(entry, "value", entry)
    out = []
    for braid, coeff in value:
        for exp, c in coeff.iterator_exp_coeff():
            params, dots = split_exponent(KLRW, exp)
            out.append((braid, params, dots, int(c)))
    return out


def differential_terms(cube, d_csc):
    """
    All terms of a differential on a :class:`Cube`:
    list of dicts with source/target vertices, braid word, strand map,
    parameter monomial, dots and coefficient.
    """
    KLRW = cube.KLRW
    out = []
    for h in sorted(d_csc.keys()):
        for (row, col), entry in d_csc[h].dict().items():
            src = cube.vertices[h][col]
            tgt = cube.vertices[h - 1][row]
            for braid, params, dots, c in entry_terms(KLRW, entry):
                out.append(
                    dict(
                        h=h,
                        src=src,
                        tgt=tgt,
                        word=tuple(int(a) for a in braid.word()),
                        src_labels=cube.labels(src),
                        tgt_labels=cube.labels(tgt),
                        arrive=cube.strand_map(src, braid),
                        params=params,
                        dots=dots,
                        coeff=c,
                    )
                )
    return out


def fmt_mono(m):
    return "*".join(n if p == 1 else "{}^{}".format(n, p) for n, p in m) or "1"


def fmt_labels(labels):
    return "(" + " ".join("|" if isinstance(x, tuple) else str(x + 1) for x in labels) + ")"


def describe(cube, term):
    s = "{{{}}} -> {{{}}}".format(
        ",".join(map(str, cube.subset(term["src"]))),
        ",".join(map(str, cube.subset(term["tgt"]))),
    )
    perm = ""
    if term["arrive"] != term["tgt_labels"]:
        perm = "  RELABEL arrive=" + fmt_labels(term["arrive"])
    return "{:>3} * {:<22} {:<16} {:<28} {} => {}  len {}{}".format(
        term["coeff"],
        fmt_mono(term["params"]),
        fmt_mono(term["dots"]),
        s,
        fmt_labels(term["src_labels"]),
        fmt_labels(term["tgt_labels"]),
        len(term["word"]),
        perm,
    )


def check_square_zero(KLRW, projectives, d_csc, degree=-1):
    d_csr = {h: m.to_csr() for h, m in d_csc.items()}
    return min_exp_in_square(d_csr, d_csc, degree=degree) is None


# ---------------------------------------------------------------------------
# minuscule combinatorics
# ---------------------------------------------------------------------------
def minuscule_nodes(cartan_type):
    """Dynkin nodes k such that the fundamental weight omega_k is minuscule."""
    from sage.combinat.root_system.root_system import RootSystem

    Phi = RootSystem(CartanType(cartan_type))
    coroots = Phi.coroot_lattice().positive_roots()
    out = []
    for k in Phi.cartan_type().index_set():
        w = Phi.weight_lattice().fundamental_weight(k)
        if all(w.scalar(a) in (0, 1) for a in coroots):
            out.append(k)
    return out


def minuscule_word(cartan_type, k, choice=min, length=None):
    """
    A word (i_1, ..., i_d) with omega_k - alpha_{i_1} - ... a chain of weights of
    the minuscule representation, from the highest to the lowest weight.
    ``choice`` picks among the available descents (``min``, ``max`` or a callable).
    """
    from sage.combinat.root_system.root_system import RootSystem

    Phi = RootSystem(CartanType(cartan_type))
    L = Phi.weight_lattice()
    alpha = L.simple_roots()
    acheck = L.simple_coroots()
    mu = L.fundamental_weight(k)
    word = []
    while True:
        down = [i for i in Phi.cartan_type().index_set() if mu.scalar(acheck[i]) == 1]
        if not down or (length is not None and len(word) == length):
            break
        i = choice(down)
        word.append(i)
        mu = mu - alpha[i]
    return tuple(word)


def all_minuscule_words(cartan_type, k, length=None):
    """All words as in :func:`minuscule_word` (all linear extensions)."""
    from sage.combinat.root_system.root_system import RootSystem

    Phi = RootSystem(CartanType(cartan_type))
    L = Phi.weight_lattice()
    alpha = L.simple_roots()
    acheck = L.simple_coroots()
    index_set = Phi.cartan_type().index_set()
    out = []

    def rec(mu, word):
        down = [i for i in index_set if mu.scalar(acheck[i]) == 1]
        if not down or (length is not None and len(word) == length):
            out.append(tuple(word))
            return
        for i in down:
            rec(mu - alpha[i], word + [i])

    rec(L.fundamental_weight(k), [])
    return out


def heap_relations(cartan_type, word):
    """
    Covering-generating relations of the heap of ``word``: pairs (p, q), p < q
    (0-based positions) whose colours are equal or adjacent.
    """
    ct = CartanType(cartan_type)
    C = ct.cartan_matrix()
    idx = {v: a for a, v in enumerate(ct.index_set())}
    rel = []
    for q in range(len(word)):
        for p in range(q):
            if C[idx[word[p]], idx[word[q]]] != 0:
                rel.append((p, q))
    return rel


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------
def term_record(cube, term):
    """
    JSON-friendly description of one term.  ``sigma[s] = s'`` means that the
    strand sitting in slot ``s`` of the source vertex arrives in slot ``s'`` of
    the target vertex (slots are 1-based strand numbers of the cube).
    """
    sigma = {}
    for a, t in zip(term["arrive"], term["tgt_labels"]):
        if isinstance(a, tuple):
            assert a == t
            continue
        sigma[a + 1] = t + 1
    return dict(
        h=term["h"],
        src=list(cube.subset(term["src"])),
        tgt=list(cube.subset(term["tgt"])),
        sigma=[sigma[s] for s in range(1, cube.n + 1)],
        word=list(term["word"]),
        params=[list(x) for x in term["params"]],
        dots=[list(x) for x in term["dots"]],
        coeff=term["coeff"],
    )


def case_name(cartan_type, punctures):
    ct = CartanType(cartan_type)
    return "{}{}__{}".format(
        ct.type(),
        ct.rank(),
        "__".join(
            "W{}_{}".format(red, "".join(str(c) for c in word)) for red, word in punctures
        ),
    )


def run_case(cartan_type, punctures, out_dir=None, verbose=True, **kwargs):
    """Build the cube, correct the differential, optionally save JSON."""
    t0 = time.time()
    KLRW = make_algebra(cartan_type, punctures)
    cube = Cube(KLRW, punctures)
    d, log = correct_differential(
        KLRW, cube.projectives, cube.d0, verbose=verbose, record_rank=True, **kwargs
    )
    terms = [term_record(cube, t) for t in differential_terms(cube, d)]
    ct = CartanType(cartan_type)
    data = dict(
        cartan_type=[ct.type(), ct.rank()],
        punctures=[[red, list(word)] for red, word in punctures],
        strand_colour=list(cube.strand_colour),
        strand_puncture=list(cube.strand_puncture),
        variables=list(KLRW.base().variable_names()),
        orders=[
            dict(
                monomial=r["monomial"],
                equations=int(r["shape"][0]),
                unknowns=int(r["shape"][1]),
                rank=int(r.get("rank", -1)),
                solution_terms=int(np.count_nonzero(r["x"])),
            )
            for r in log
        ],
        terms=terms,
        seconds=time.time() - t0,
    )
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, case_name(cartan_type, punctures) + ".json")
        with open(path, "w") as f:
            json.dump(data, f)
        if verbose:
            print("saved", path)
    return cube, d, log, data
