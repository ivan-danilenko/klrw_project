"""
Referee's own helpers for checking proofs/05_main_sliding.md.  Independent of 05_lib.py.

Everything is phrased in terms of *arrangements*: a list of labels, left to right; labels are
slot numbers (1-based ints) or 'W' (red) -- or arbitrary hashables for free-standing states.
A word is a tuple of 1-based positions; letter i exchanges positions i, i+1.

    S = Setup(["A", 3], red, word)
    S.arr(I)                  arrangement of the vertex I
    S.el(arr, word)           element of the algebra of the word, read from arr
    S.rand_reduced(arr, target_arr)   a random reduced word from arr to target_arr
    S.my_terms(I)             the tree terms at I, from MY implementation of NOTES.md section 2
    S.coef(term)              coefficient as element of the dot algebra
"""
import os, sys, itertools, random
from collections import defaultdict, namedtuple
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from cupbrane import make_algebra, Cube                     # noqa
from klrw.framed_dynkin import NodeInFramedQuiver            # noqa
from sage.combinat.root_system.cartan_type import CartanType  # noqa

Term = namedtuple("Term", "I j F J parent sign mono")   # parent: pair -> (parent node, connector); mono: list of (c, c(connector))


def dynkin_edges(ct):
    """Set of frozenset({c, c'}) for adjacent colours, computed from the Cartan matrix (own code)."""
    C = CartanType(ct).cartan_matrix()
    idx = list(CartanType(ct).index_set())
    E = set()
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if C[a, b] != 0:
                assert C[a, b] == -1 and C[b, a] == -1
                E.add(frozenset((idx[a], idx[b])))
    return E


def apply_word(arr, word):
    arr = list(arr)
    for i in word:
        arr[i - 1], arr[i] = arr[i], arr[i - 1]
    return arr


def crossing_counts(arr, word):
    """n_D(A, B) for all pairs of labels, and for each letter (left strand, right strand)."""
    arr = list(arr)
    n = defaultdict(int)
    letters = []
    for i in word:
        a, b = arr[i - 1], arr[i]
        n[frozenset((a, b))] += 1
        letters.append((a, b))
        arr[i - 1], arr[i] = b, a
    return n, letters


def rand_reduced(arr, target, rng):
    """Random reduced word from arr to target (both lists of distinct labels)."""
    rank = {lab: p for p, lab in enumerate(target)}
    arr = list(arr)
    word = []
    while True:
        inv = [i for i in range(len(arr) - 1) if rank[arr[i]] > rank[arr[i + 1]]]
        if not inv:
            break
        i = rng.choice(inv)
        arr[i], arr[i + 1] = arr[i + 1], arr[i]
        word.append(i + 1)
    assert arr == list(target)
    return tuple(word)


def perm_sign(seq):
    s = 1
    seq = list(seq)
    for i in range(len(seq)):
        for k in range(i + 1, len(seq)):
            if seq[i] > seq[k]:
                s = -s
    return s


class Setup:
    def __init__(self, ct, red, word, build_cube=False):
        self.ct, self.red, self.word = ct, red, tuple(word)
        self.m = len(self.word)
        P = [(red, self.word)]
        self.K = make_algebra(ct, P)
        self.D = self.K.base()
        self.States = self.K.KLRWBraid.KLRWstate_set
        self.E = dynkin_edges(ct)
        self.cube = Cube(self.K, P, build=build_cube)
        self._stack = None

    # ---- colours -------------------------------------------------------------------------
    def col(self, lab):
        return "W" if lab == "W" else self.word[lab - 1]

    def adj(self, c, c2):
        return frozenset((c, c2)) in self.E

    # ---- states and elements ---------------------------------------------------------------
    def arr(self, I):
        I = set(I)
        return [s for s in range(self.m, 0, -1) if s not in I] + ["W"] + [s for s in range(1, self.m + 1) if s in I]

    def state(self, arr):
        nodes = tuple(NodeInFramedQuiver(self.red, framing=True) if lab == "W"
                      else NodeInFramedQuiver(self.word[lab - 1]) for lab in arr)
        return self.States._element_constructor_(nodes)

    def el(self, arr, word):
        return self.K.element_from_braid(tuple(word), left_state=self.state(arr))

    def idem(self, arr):
        return self.K.idempotent(self.state(arr))

    def dot(self, arr, lab):
        """Dot on the strand with label lab, at the state arr (times the idempotent)."""
        st = self.state(arr)
        pos = list(arr).index(lab)
        idx = st.index_among_same_color(pos)
        return self.D.dot_variable(st[pos], idx).monomial * self.K.idempotent(st)

    def dotvar(self, arr, lab):
        st = self.state(arr)
        pos = list(arr).index(lab)
        idx = st.index_among_same_color(pos)
        return self.D.dot_variable(st[pos], idx).monomial

    def r(self, c):
        return self.D.vertex_variable(NodeInFramedQuiver(c)).monomial

    def t(self, c, c2):
        return self.D.edge_variable(NodeInFramedQuiver(c), NodeInFramedQuiver(c2)).monomial

    def u(self, c):
        return self.D.edge_variable(NodeInFramedQuiver(c), NodeInFramedQuiver(c, framing=True)).monomial

    # ---- the tree formula, implemented from NOTES.md section 2 (brute force) ---------------
    def my_terms(self, I):
        """All admissible (j, F) at I by brute force over sets of disjoint pairs."""
        I = frozenset(I)
        m = self.m
        out = []
        for j in sorted(I):
            avail = [(a, b) for a in range(1, m + 1) for b in range(a + 1, m + 1)
                     if self.word[a - 1] == self.word[b - 1] and a not in I and b in I and b != j]
            # all sets of pairwise disjoint pairs
            def rec(k, chosen, used):
                if k == len(avail):
                    yield tuple(chosen)
                    return
                yield from rec(k + 1, chosen, used)
                a, b = avail[k]
                if a not in used and b not in used:
                    yield from rec(k + 1, chosen + [avail[k]], used | {a, b})
            for F in rec(0, [], set()):
                t = self.analyse(I, j, F)
                if t is not None:
                    out.append(t)
        return out

    def analyse(self, I, j, F):
        """Admissibility conditions 1-3 of NOTES.md section 2; returns a Term or None."""
        I = frozenset(I)
        cj = self.word[j - 1]
        cp = {p: self.word[p[0] - 1] for p in F}
        nodes = ["mover"] + list(F)
        nb = defaultdict(list)
        for p in F:
            a, b = p
            if a < j < b:
                if cp[p] == cj:
                    return None                                     # condition 1
                if self.adj(cp[p], cj):
                    nb["mover"].append(p)
                    nb[p].append("mover")
        for p, q in itertools.combinations(F, 2):
            (a, b), (a2, b2) = p, q
            inter = (a < a2 < b < b2) or (a2 < a < b2 < b)
            if inter:
                if cp[p] == cp[q]:
                    return None                                     # condition 2
                if self.adj(cp[p], cp[q]):
                    nb[p].append(q)
                    nb[q].append(p)
        n_edges = sum(len(v) for v in nb.values()) // 2
        # tree: connected and |edges| = |nodes| - 1
        parent = {}
        seen = {"mover"}
        todo = ["mover"]
        while todo:
            x = todo.pop()
            for y in nb[x]:
                if y not in seen:
                    seen.add(y)
                    parent[y] = x
                    todo.append(y)
        if len(seen) != len(nodes) or n_edges != len(F):
            return None
        # connectors, monomial, sign
        par = {}
        sign_conn = 1
        mono = []
        for p in F:
            q = parent[p]
            if q == "mover":
                x = j
            else:
                inside = [s for s in q if p[0] < s < p[1]]
                assert len(inside) == 1
                x = inside[0]
            par[p] = (q, x)
            if x in I:
                sign_conn = -sign_conn
            mono.append((cp[p], self.word[x - 1]))
        # Koszul sign: I increasing, replace b_p by a_p in place, delete j with (-1)^{# entries to its right},
        # times the sign of the permutation sorting the rest
        repl = {b: a for a, b in F}
        seq = [repl.get(s, s) for s in sorted(I)]
        k = seq.index(j)
        kappa = (-1) ** (len(seq) - 1 - k) * perm_sign(seq[:k] + seq[k + 1:])
        J = (I - {j} - {b for a, b in F}) | {a for a, b in F}
        return Term(I, j, tuple(sorted(F)), frozenset(J), par, kappa * sign_conn, tuple(sorted(mono)))

    def coef(self, term):
        c = self.D(term.sign)
        for cp, cx in term.mono:
            c = c * self.r(cp) * self.t(cp, cx)
        return c

    # ---- diagrams of configurations -------------------------------------------------------
    def target_arr(self, I, j, F):
        """Arrangement after the term (I; j, F): where every strand (labelled by its slot at I) ends."""
        I = frozenset(I)
        pos = {}
        for s in range(1, self.m + 1):
            pos[s] = s if s in I else -s
        if j is not None:
            pos[j] = -j
        for a, b in F:
            pos[a] = -b          # L_p: the strand at -a goes to -b
            pos[b] = +a          # R_p: the strand at +b goes to +a
        pos["W"] = 0
        return sorted(pos, key=lambda lab: pos[lab])

    def beta_word(self, I, j, F, rng):
        src = self.arr(I)
        return rand_reduced(src, self.target_arr(I, j, F), rng)

    def relabel_after(self, I, j, F):
        """Map: strand label at I  ->  slot label it carries at the target vertex J."""
        lab = {s: s for s in range(1, self.m + 1)}
        for a, b in F:
            lab[a], lab[b] = b, a
        lab["W"] = "W"
        return lab

    # ---- formula.Stack (for comparison) -----------------------------------------------------
    @property
    def stack(self):
        if self._stack is None:
            from formula import Stack
            self._stack = Stack(self.ct, self.word)
        return self._stack


def strand_colour_classes(S, labs):
    by = defaultdict(list)
    for lab in labs:
        if lab != "W":
            by[S.col(lab)].append(lab)
    return by


def check_S1_S2(S, arr, word, colour=None):
    """Hypotheses (S1), (S2) of Lemma 1.2 for the word from arr.  colour: label -> colour (default S.col)."""
    col = colour or S.col
    n, _ = crossing_counts(arr, word)
    labs = list(arr)
    s1 = True
    for pair, cnt in n.items():
        if cnt >= 2:
            a, b = tuple(pair)
            ca, cb = col(a), col(b)
            if "W" in (ca, cb):
                other = cb if ca == "W" else ca
                if other == S.red:
                    s1 = False
            else:
                if ca == cb or S.adj(ca, cb):
                    s1 = False
    s2 = True
    for a, c in itertools.combinations(labs, 2):
        if col(a) == "W" or col(a) != col(c) or n[frozenset((a, c))] < 1:
            continue
        for b in labs:
            if b in (a, c):
                continue
            if n[frozenset((a, b))] >= 1 and n[frozenset((b, c))] >= 1:
                cb = col(b)
                if cb == "W":
                    if col(a) == S.red:
                        s2 = False
                elif S.adj(cb, col(a)):
                    s2 = False
    return s1, s2
