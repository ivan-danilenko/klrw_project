"""
Gurobi-free replacements for the three solver calls in klrw/, installed by
monkeypatching (klrw/ itself is not modified).

    import patches; patches.install()

1. klrw.perfect_complex_corrections.solve_system  (integer x with A x = b)
2. klrw.homotopy.solve_system                     (same, None if infeasible)
3. klrw.dot_algebra.KLRWDotsAlgebra._solve_ilp_on_gradings_
       (enumerate ALL integer solutions of a tiny bounded system; exact, via
        sage Polyhedron.integral_points)
4. klrw.stable_envelopes.PerfectComplex is wrapped to force parallel_processes=1
   (spawned children would not see the patches).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CUP = os.path.dirname(HERE)
if CUP not in sys.path:
    sys.path.insert(0, CUP)

import cupbrane  # noqa: E402  (puts the repo root on sys.path, imports sage)

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from sage.misc.cachefunc import cached_method  # noqa: E402

STATS = {"ilp_calls": 0, "ilp_solutions": 0, "corr_solves": 0, "homotopy_solves": 0}


def _solve_or_none(A, b):
    A = sp.csr_matrix(A)
    b = sp.csc_matrix(b)
    if b.shape[0] < A.shape[0]:
        b.resize((A.shape[0], 1))
    x = cupbrane.solve_integer(A, b)
    if x is None:
        return None
    x_csc = sp.csc_matrix(x.reshape(-1, 1).astype(np.intc))
    x_csc.sort_indices()
    # exact check (the original asserts the same)
    assert (A @ x_csc - b).nnz == 0
    return x_csc


def corrections_solve_system(A, b, verbose=True):
    STATS["corr_solves"] += 1
    x = _solve_or_none(A, b)
    if x is None:
        raise RuntimeError("correction system infeasible (shape {})".format(A.shape))
    return x


def homotopy_solve_system(A, b, verbose=True):
    STATS["homotopy_solves"] += 1
    return _solve_or_none(A, b)


def _enumerate_integer_solutions(A_inv, A_non, y):
    """
    All integer (x_inv, x_non) with A_inv x_inv + A_non x_non = y, x_non >= 0.
    Returns an (n, k_inv + k_non) intc array.  Raises if the set is unbounded.
    """
    from sage.geometry.polyhedron.constructor import Polyhedron
    from sage.rings.rational_field import QQ

    A_inv = np.asarray(sp.csr_matrix(A_inv).todense(), dtype=np.int64)
    A_non = np.asarray(sp.csr_matrix(A_non).todense(), dtype=np.int64)
    k1, k2 = A_inv.shape[1], A_non.shape[1]
    k = k1 + k2
    if k == 0:
        if not np.any(y):
            return np.zeros((1, 0), dtype=np.intc)
        return np.zeros((0, 0), dtype=np.intc)
    A = np.hstack([A_inv, A_non])
    # drop all-zero rows; if any of them has non-zero rhs: infeasible
    eqns = []
    for i in range(A.shape[0]):
        if not A[i].any():
            if y[i] != 0:
                return np.zeros((0, k), dtype=np.intc)
            continue
        eqns.append([-int(y[i])] + [int(v) for v in A[i]])
    ieqs = []
    for j in range(k2):
        row = [0] * (k + 1)
        row[1 + k1 + j] = 1
        ieqs.append(row)
    P = Polyhedron(eqns=eqns, ieqs=ieqs, base_ring=QQ)
    if P.is_empty():
        return np.zeros((0, k), dtype=np.intc)
    if not P.is_compact():
        raise RuntimeError("grading ILP has an unbounded solution set: {}".format(P))
    pts = P.integral_points()
    out = np.zeros((len(pts), k), dtype=np.intc)
    for n, p in enumerate(pts):
        out[n] = [int(v) for v in p]
    return out


def _solve_ilp_on_gradings_(self, degree):
    grading_group = degree.parent()
    A_inv, A_non_inv = self._ilp_on_gradings_(grading_group)
    y = np.zeros(grading_group.dimension(), dtype=np.int64)
    for i, v in degree._vector_(sparse=True).dict(copy=False).items():
        y[i] = int(v)
    sols = _enumerate_integer_solutions(A_inv, A_non_inv, y)
    STATS["ilp_calls"] += 1
    STATS["ilp_solutions"] += sols.shape[0]
    return sols


def _fix_laurent_dot_algebra():
    """
    Sage 10.7: LaurentPolynomialRing_generic no longer derives from
    CommutativeRing, so in the MRO of
    klrw.dot_algebra.KLRWUpstairsDotsAlgebra_invertible_parameters the old-style
    CommutativeRing/ParentWithGens come *before* LaurentPolynomialRing_mpair and
    ``super(KLRWDotsAlgebra, self).__init__(R)`` runs CommutativeRing.__init__
    (instance never gets ``_R`` -> AttributeError on construction).
    We rebuild the class with LaurentPolynomialRing_mpair placed before
    CommutativeRing (the MRO the code had under Sage <= 10.3) and swap it in.
    """
    import klrw.dot_algebra as da
    from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing_mpair
    from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
    from sage.rings.ring import CommutativeRing

    old = da.KLRWUpstairsDotsAlgebra_invertible_parameters
    mro = old.__mro__
    if mro.index(LaurentPolynomialRing_mpair) < mro.index(CommutativeRing):
        return  # nothing to fix

    class _LaurentThenCommutative(LaurentPolynomialRing_mpair, CommutativeRing):
        pass

    class KLRWUpstairsDotsAlgebra_invertible_parameters(da.KLRWDotsAlgebra, _LaurentThenCommutative):
        class Element(LaurentPolynomialRing_mpair.Element):
            def _matmul_(self, other):
                return self.parent()._self_action.act(self, other)

        def init_algebra(self, base_ring, nondot_names_list, dot_names_list, order):
            polynomial_ring = PolynomialRing(
                base_ring,
                len(nondot_names_list) + len(dot_names_list),
                nondot_names_list + dot_names_list,
                order=order,
            )
            LaurentPolynomialRing_mpair.__init__(self, polynomial_ring)

    KLRWUpstairsDotsAlgebra_invertible_parameters.__module__ = "klrw.dot_algebra"
    KLRWUpstairsDotsAlgebra_invertible_parameters.__qualname__ = (
        "KLRWUpstairsDotsAlgebra_invertible_parameters"
    )
    da.KLRWUpstairsDotsAlgebra_invertible_parameters = KLRWUpstairsDotsAlgebra_invertible_parameters


_installed = False


def install():
    global _installed
    if _installed:
        return
    _fix_laurent_dot_algebra()
    import klrw.perfect_complex_corrections as pcc
    import klrw.homotopy as hmt
    import klrw.dot_algebra as da
    import klrw.stable_envelopes as se

    pcc.solve_system = corrections_solve_system
    hmt.solve_system = homotopy_solve_system
    f = _solve_ilp_on_gradings_
    f.__name__ = "_solve_ilp_on_gradings_"
    da.KLRWDotsAlgebra._solve_ilp_on_gradings_ = cached_method(f)

    original = pcc.PerfectComplex

    def PerfectComplex_serial(*args, **kwargs):
        kwargs["parallel_processes"] = 1
        return original(*args, **kwargs)

    se.PerfectComplex = PerfectComplex_serial
    _installed = True


def selftest():
    """Compare the enumerator with brute force on random small systems."""
    import itertools
    rng = np.random.default_rng(1)
    for trial in range(200):
        k1, k2, m = rng.integers(0, 3), rng.integers(1, 4), rng.integers(1, 4)
        A_inv = rng.integers(-2, 3, size=(m, k1))
        A_non = rng.integers(-2, 3, size=(m, k2))
        # make the dots bounded: a row with positive coefficients on x_non only
        A_non = np.vstack([A_non, np.ones((1, k2), dtype=int)])
        A_inv = np.vstack([A_inv, np.zeros((1, k1), dtype=int)])
        y = rng.integers(-3, 4, size=m + 1)
        y[-1] = abs(y[-1])
        try:
            sols = _enumerate_integer_solutions(A_inv, A_non, y)
        except RuntimeError:
            continue
        A = np.hstack([A_inv, A_non])
        brute = set()
        R = 12
        for x in itertools.product(*([range(-R, R + 1)] * k1 + [range(0, R + 1)] * k2)):
            if (A @ np.array(x, dtype=int) == y).all():
                brute.add(tuple(x))
        got = set(tuple(int(v) for v in s) for s in sols)
        # brute force is only valid inside the box; compare there
        got_in_box = set(x for x in got if all(abs(v) <= R for v in x))
        assert got_in_box == brute, (A_inv, A_non, y, got, brute)
    print("enumerator self-test passed")


if __name__ == "__main__":
    selftest()
