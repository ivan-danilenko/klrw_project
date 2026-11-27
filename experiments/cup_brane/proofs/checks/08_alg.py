"""
08 -- sage/klrw engine for the cyclic normal form (proofs/08_cyclic_normal_form.md).

    sage -python 08_alg.py           (self-test: d_T^2 for 1221122 red 2, the r-degrees of 04 Thm 9.1(b))

Maps on the cube are dicts  I -> {J: element of e_I A e_J}  (I, J frozensets of slots), acting by
right multiplication; composition (f g)_{IK} = sum_J f_{IJ} g_{JK} (first f, then g).
Basis: the klrw basis braid of a bijection (fast.braid_for), dots at the left end, klrw dot names
x_c_k (k-th strand of colour c from the left of the source state).

Vectors: an element of a component (I, K) is vectorised as {(I, K, braid word, exponent tuple): int};
exponent tuples include parameters and dots (klrw variable order, self.names).
"""
import os
import sys
import time
import json
import importlib
import multiprocessing as mp
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from cupbrane import make_algebra, Cube  # noqa: E402
from fast import braid_for  # noqa: E402
from sage.all import matrix, QQ, ZZ, vector, lcm, GF  # noqa: E402

C8 = importlib.import_module("08_comb")
DATA = os.path.join(HERE, "08_data")


class Ctx:
    def __init__(self, ct, red, word, build_cube=False, basis_mode="klrw"):
        self.ct = [ct[0], int(ct[1])]
        self.red = int(red)
        self.word = tuple(int(c) for c in word)
        self.W = C8.WordData(self.ct, self.red, self.word)
        self.m = self.W.m
        P = [(self.red, self.word)]
        self.K = make_algebra(self.ct, P)
        self.cube = Cube(self.K, P, build=build_cube)
        self.D = self.K.base()
        self.names = list(self.D.variable_names())
        self.rpos = [i for i, n in enumerate(self.names) if n.startswith("r_")]
        self.xpos = [i for i, n in enumerate(self.names) if n.startswith("x_")]
        self.ppos = [i for i, n in enumerate(self.names) if not n.startswith("x_")]
        self._b = {}
        self._mono = {}
        self.basis_mode = basis_mode
        import itertools
        self.V = [frozenset(c) for k in range(self.m + 1) for c in itertools.combinations(range(1, self.m + 1), k)]

    def eps(self, I):
        return tuple(1 if s + 1 in I else 0 for s in range(self.m))

    def basis(self, I, J, sigma):
        """klrw basis element of the bijection sigma (tuple) from e_I to e_J."""
        key = (I, J, sigma)
        if key not in self._b:
            if self.basis_mode == "klrw":
                _, _, braid = braid_for(self.cube, I, J, tuple(sigma))
                self._b[key] = self.K.monomial(braid)
            else:
                w = self.W.reduced_word(I, J, sigma, mode=self.basis_mode)
                el = self.K.element_from_braid(w, left_state=self.cube.state(self.eps(I)))
                self._b[key] = el
        return self._b[key]

    def mono(self, mono):
        if mono not in self._mono:
            e = self.D.one()
            for nm, p in mono:
                e *= self.D(nm) ** p
            self._mono[mono] = e
        return self._mono[mono]

    def term(self, I, J, sigma, mono, coeff):
        return (self.D(coeff) * self.mono(mono)) * self.basis(I, J, sigma)

    def rdeg(self, e):
        return sum(e[p] for p in self.rpos)

    def ndots(self, e):
        return sum(e[p] for p in self.xpos)

    def param_part(self, e):
        return tuple((self.names[p], e[p]) for p in self.ppos if e[p])

    def zero(self):
        return self.K.zero()

    # ------------------------------------------------------------------ maps
    def map_from_terms(self, terms):
        """terms: iterable of (I, J, sigma, mono, coeff)."""
        d = defaultdict(lambda: defaultdict(lambda: self.K.zero()))
        for I, J, sigma, mono, coeff in terms:
            d[frozenset(I)][frozenset(J)] += self.term(frozenset(I), frozenset(J), tuple(sigma), mono, coeff)
        return {I: dict(r) for I, r in d.items()}

    def tree_terms(self, maxpairs=99):
        out = []
        for I in self.V:
            for J, sigma, mono, sign in self.W.tree_terms(I):
                if C8.rdeg(mono) <= maxpairs:
                    out.append((I, J, sigma, mono, sign))
        return out

    def tree(self, maxpairs=99):
        return self.map_from_terms(self.tree_terms(maxpairs))

    def terms_of(self, d):
        """decompose a dot-free map into (I, J, sigma, mono, coeff); raises if dots occur."""
        out = []
        for I, row in d.items():
            for J, x in row.items():
                for braid, coeff in x:
                    sg = self.sigma_of(I, J, braid)
                    for e, c in coeff.dict().items():
                        assert self.ndots(e) == 0, "dotted term"
                        out.append((I, J, sg, tuple(sorted((self.names[p], int(v)) for p, v in enumerate(e) if v)), int(c)))
        return out

    def sigma_of(self, I, J, braid):
        arrive = self.cube.strand_map(self.eps(I), braid)
        labels = self.cube.labels(self.eps(J))
        sig = {}
        for a, t in zip(arrive, labels):
            if not isinstance(a, tuple):
                sig[a + 1] = t + 1
        return tuple(sig[s] for s in range(1, self.m + 1))


def compose(C, f, g):
    res = defaultdict(lambda: defaultdict(lambda: C.K.zero()))
    for I, row in f.items():
        for J, x in row.items():
            for K, y in g.get(J, {}).items():
                res[I][K] += x * y
    return res


def add_maps(C, *ms, coeffs=None):
    out = defaultdict(lambda: defaultdict(lambda: C.K.zero()))
    for k, mp_ in enumerate(ms):
        c = 1 if coeffs is None else coeffs[k]
        for I, row in mp_.items():
            for J, x in row.items():
                out[I][J] += C.D(c) * x if c != 1 else x
    return {I: {J: x for J, x in row.items() if not x.is_zero()} for I, row in out.items()}


def vec_elem(C, I, K, x, out, keep=None, mult=1):
    for braid, coeff in x:
        w = tuple(int(a) for a in braid.word())
        for e, c in coeff.dict().items():
            et = tuple(int(v) for v in e)
            if keep is not None and not keep(et):
                continue
            key = (I, K, w, et)
            out[key] = out.get(key, 0) + mult * int(c)


def vec_map(C, mp_, keep=None):
    out = {}
    for I, row in mp_.items():
        for K, x in row.items():
            vec_elem(C, I, K, x, out, keep)
    return {k: v for k, v in out.items() if v}


def anticomm_vec(C, d, I, J, el, keep=None):
    """d . el + el . d for el in the single component I -> J (hom. degree -1), vectorised."""
    out = {}
    for Ip, row in d.items():
        x = row.get(I)
        if x is not None:
            vec_elem(C, Ip, J, x * el, out, keep)
    for K, y in d.get(J, {}).items():
        vec_elem(C, I, K, el * y, out, keep)
    return {k: v for k, v in out.items() if v}


def comm_vec(C, d, I, J, el, keep=None):
    """el . d - d . el for el in the single component I -> J (hom. degree 0), vectorised."""
    out = {}
    for K, y in d.get(J, {}).items():
        vec_elem(C, I, K, el * y, out, keep)
    for Ip, row in d.items():
        x = row.get(I)
        if x is not None:
            vec_elem(C, Ip, J, x * el, out, keep, mult=-1)
    return {k: v for k, v in out.items() if v}


def is_zero_map(mp_):
    return all(x.is_zero() for row in mp_.values() for x in row.values())


def rdeg_key(C, key):
    return C.rdeg(key[3])


# ---------------------------------------------------------------------- parallel helpers
_G = {}


def _worker(args):
    fn, chunk = args
    return [fn(_G, item) for item in chunk]


def par_map(fn, items, procs=12, ctx=None):
    """map fn(G, item) over items in forked processes; G = the global dict _G (set before)."""
    items = list(items)
    if procs <= 1 or len(items) < 2 * procs:
        return [fn(_G, it) for it in items]
    chunks = [items[k::procs * 3] for k in range(procs * 3)]
    with mp.get_context("fork").Pool(procs) as pool:
        parts = pool.map(_worker, [(fn, ch) for ch in chunks])
    # reassemble in the original order
    res = [None] * len(items)
    for k, part in enumerate(parts):
        for i, r in zip(range(k, len(items), procs * 3), part):
            res[i] = r
    return res


# ---------------------------------------------------------------------- linear algebra
def sparse_matrix(cols, rowidx):
    nr = max(len(rowidx), 1)
    ent = {}
    for c_, col in enumerate(cols):
        for key, v in col.items():
            ent[rowidx[key], c_] = v
    return matrix(QQ, nr, max(len(cols), 1), ent, sparse=True)


def _solve_block(cols, rhs):
    rowidx = {}
    for col in cols:
        for k in col:
            rowidx.setdefault(k, len(rowidx))
    for k in rhs:
        rowidx.setdefault(k, len(rowidx))
    M = sparse_matrix(cols, rowidx)
    b = vector(QQ, max(len(rowidx), 1), {rowidx[k]: v for k, v in rhs.items()}, sparse=True)
    try:
        y0 = M.solve_right(b)
    except ValueError:
        return None
    ker = M.right_kernel_matrix(basis="echelon")
    return list(y0)[:len(cols)], [list(r) for r in ker.rows()]


def solve_affine(cols, rhs):
    """Solve sum_c y_c cols[c] = rhs (dicts key -> int).  Returns (y0, kernel basis) or None.
    Exact over QQ.  The system is split into the connected blocks of its incidence graph
    (columns sharing a row), which are solved separately."""
    n = len(cols)
    if n == 0:
        return ([], []) if not any(rhs.values()) else None
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for c_, col in enumerate(cols):
        for k in col:
            if k in owner:
                a, b = find(owner[k]), find(c_)
                if a != b:
                    parent[a] = b
            else:
                owner[k] = c_
    blocks = defaultdict(list)
    for c_ in range(n):
        blocks[find(c_)].append(c_)
    # rows of the rhs not met by any column must vanish
    brhs = defaultdict(dict)
    for k, v in rhs.items():
        if not v:
            continue
        if k not in owner:
            return None
        brhs[find(owner[k])][k] = v
    y0 = [QQ(0)] * n
    kb = []
    for root, members in blocks.items():
        res = _solve_block([cols[c_] for c_ in members], brhs.get(root, {}))
        if res is None:
            return None
        yb, kerb = res
        for c_, v in zip(members, yb):
            y0[c_] = v
        for kv in kerb:
            full = [QQ(0)] * n
            for c_, v in zip(members, kv):
                full[c_] = v
            kb.append(full)
    return y0, kb


if __name__ == "__main__":
    t0 = time.time()
    C = Ctx(["A", 2], 2, (1, 2, 2, 1, 1, 2, 2))
    dT = C.tree()
    sq = compose(C, dT, dT)
    v = vec_map(C, sq)
    byr = Counter()
    for k in v:
        byr[C.rdeg(k[3])] += 1
    print("1221122 red 2: d_T^2 coefficients by r-degree", dict(sorted(byr.items())), "[%.0fs]" % (time.time() - t0))
