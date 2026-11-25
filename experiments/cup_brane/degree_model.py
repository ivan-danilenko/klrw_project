"""
Pure-Python model of the multigrading on a one-puncture stack.

A "candidate" is (I, J, sigma, monomial): a dot-free basis diagram from the
idempotent of I to that of J (|J| = |I| - 1), determined by the colour
preserving bijection sigma of slots, times a monomial in r, t, u, such that
the total multidegree is that of a component of the differential.

    python3 degree_model.py data/CASE.json      # compare with stored terms
"""
import itertools
import json
import sys
from collections import Counter, defaultdict

CARTAN_EDGES = {
    "A": lambda n: [(i, i + 1) for i in range(1, n)],
    "D": lambda n: [(i, i + 1) for i in range(1, n - 1)] + [(n - 2, n)],
    "E": lambda n: [(1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)][: n - 2] + [(2, 4)],
}


def edges(ct):
    return set(frozenset(e) for e in CARTAN_EDGES[ct[0]](ct[1]))


class Model:
    """
    ``punctures`` = list of (red colour, word).  Slots are numbered globally
    1..m, puncture after puncture, innermost first.
    """

    BIG = 1000

    def __init__(self, ct, punctures):
        self.ct = ct
        self.punctures = [(r, list(w)) for r, w in punctures]
        self.reds = [r for r, _ in self.punctures]
        self.col = [c for _, w in self.punctures for c in w]
        self.punc = [p for p, (_, w) in enumerate(self.punctures) for _ in w]
        self.loc = [k + 1 for _, w in self.punctures for k in range(len(w))]
        self.m = len(self.col)
        self.E = edges(ct)

    def pos(self, s, I):
        """Position of slot s (red strand of puncture p sits at p * BIG)."""
        return self.punc[s - 1] * self.BIG + (self.loc[s - 1] if s in I else -self.loc[s - 1])

    # pair type: ('s', v) same colour, ('e', a, b) adjacent a<b, ('w',) crossing with own-colour red
    def pair_type(self, s, t):
        a, b = self.col[s - 1], self.col[t - 1]
        if a == b:
            return ("s", a)
        if frozenset((a, b)) in self.E:
            return ("e", min(a, b), max(a, b))
        return None

    def D(self, I):
        """Formal crossing content of the geometric path I -> empty set."""
        c = Counter()
        for j in I:
            if self.col[j - 1] == self.reds[self.punc[j - 1]]:
                c[("w", self.col[j - 1])] += 1
            for i in range(1, j):
                if self.punc[i - 1] != self.punc[j - 1]:
                    continue
                t = self.pair_type(i, j)
                if t:
                    c[t] += 1
        return c

    def content(self, I, J, sigma):
        """Crossing content of the reduced diagram of sigma from I to J."""
        c = Counter()
        for s in range(1, self.m + 1):
            a, b = self.pos(s, I), self.pos(sigma[s], J)
            for p, red in enumerate(self.reds):
                if red == self.col[s - 1] and (a - p * self.BIG) * (b - p * self.BIG) < 0:
                    c[("w", red)] += 1
            for t_ in range(s + 1, self.m + 1):
                if (a - self.pos(t_, I)) * (b - self.pos(sigma[t_], J)) < 0:
                    t = self.pair_type(s, t_)
                    if t:
                        c[t] += 1
        return c

    def monomials(self, deficit):
        """
        All monomials prod (r_v t_{v,w})^{n} (r_v u_v)^{n} whose content
        ( (v,v) + 2 (v,w) resp. (v,v) + 2 (v,W) ) equals ``deficit``.
        Returned as sorted tuples of (name, power).
        """
        if any(x < 0 for x in deficit.values()):
            return []
        gens = []
        for e in self.E:
            a, b = sorted(e)
            gens.append((("r_%d" % a, "t_%d_%d" % (a, b)), Counter({("s", a): 1, ("e", a, b): 2})))
            gens.append((("r_%d" % b, "t_%d_%d" % (b, a)), Counter({("s", b): 1, ("e", a, b): 2})))
        for red in set(self.reds):
            gens.append((("r_%d" % red, "u_%d" % red), Counter({("s", red): 1, ("w", red): 2})))
        out = []

        def rec(k, rest, acc):
            if all(x == 0 for x in rest.values()):
                out.append(tuple(sorted(acc.items())))
                return
            if k == len(gens):
                return
            names, cont = gens[k]
            n = 0
            cur = Counter(rest)
            acc2 = Counter(acc)
            while True:
                rec(k + 1, cur, acc2)
                cur = Counter(cur)
                cur.subtract(cont)
                if any(x < 0 for x in cur.values()):
                    break
                acc2 = Counter(acc2)
                for nm in names:
                    acc2[nm] += 1

        rec(0, Counter({k: v for k, v in deficit.items() if v}), Counter())
        return out

    def bijections(self):
        classes = defaultdict(list)
        for s in range(1, self.m + 1):
            classes[self.col[s - 1]].append(s)
        cls = list(classes.values())
        for perms in itertools.product(*[itertools.permutations(c) for c in cls]):
            sigma = {}
            for c, p in zip(cls, perms):
                for a, b in zip(c, p):
                    sigma[a] = b
            yield sigma

    def _typed_pairs(self):
        if not hasattr(self, "_tp"):
            tp = []
            for s in range(1, self.m + 1):
                for t in range(s + 1, self.m + 1):
                    ty = self.pair_type(s, t)
                    if ty:
                        tp.append((s, t, ty))
            self._tp = tp
            self._redslots = [
                (s, p, red) for s in range(1, self.m + 1)
                for p, red in enumerate(self.reds) if red == self.col[s - 1]
            ]
        return self._tp, self._redslots

    def content_fast(self, posI, posJ, sigma, target):
        """deficit = target - content, or None as soon as it becomes negative."""
        tp, redslots = self._typed_pairs()
        deficit = dict(target)
        for s, p, red in redslots:
            if (posI[s] - p * self.BIG) * (posJ[sigma[s]] - p * self.BIG) < 0:
                k = ("w", red)
                v = deficit.get(k, 0) - 1
                if v < 0:
                    return None
                deficit[k] = v
        for s, t, ty in tp:
            if (posI[s] - posI[t]) * (posJ[sigma[s]] - posJ[sigma[t]]) < 0:
                v = deficit.get(ty, 0) - 1
                if v < 0:
                    return None
                deficit[ty] = v
        return deficit

    def candidates_from(self, sources):
        """Candidates whose source vertex is in ``sources`` (iterable of frozensets)."""
        slots = range(1, self.m + 1)
        if not hasattr(self, "_bij"):
            self._bij = list(self.bijections())
        out = []
        for I in sources:
            DI = self.D(I)
            posI = {s: self.pos(s, I) for s in slots}
            for Jt in itertools.combinations(slots, len(I) - 1):
                J = frozenset(Jt)
                target = Counter(DI)
                target.subtract(self.D(J))
                if any(v < 0 for v in target.values()):
                    continue
                posJ = {s: self.pos(s, J) for s in slots}
                for sigma in self._bij:
                    deficit = self.content_fast(posI, posJ, sigma, target)
                    if deficit is None:
                        continue
                    for mono in self.monomials(Counter(deficit)):
                        out.append((I, J, tuple(sigma[s] for s in slots), mono))
        return out

    def candidates(self, processes=1):
        slots = range(1, self.m + 1)
        sources = [frozenset(c) for h in range(1, self.m + 1) for c in itertools.combinations(slots, h)]
        if processes == 1:
            return self.candidates_from(sources)
        import multiprocessing as mp
        chunks = [sources[k::4 * processes] for k in range(4 * processes)]
        with mp.get_context("fork").Pool(processes) as pool:
            parts = pool.map(self.candidates_from, chunks)
        return [c for part in parts for c in part]


def compare(path):
    data = json.load(open(path))
    ct = data["cartan_type"]
    model = Model(ct, data["punctures"])
    actual = {}
    for t in data["terms"]:
        if t["dots"]:
            print("term with dots:", t)
        key = (frozenset(t["src"]), frozenset(t["tgt"]), tuple(t["sigma"]),
               tuple(sorted((n, p) for n, p in t["params"])))
        assert key not in actual
        actual[key] = t["coeff"]
    cand = set(model.candidates())
    print(path)
    print("  actual terms %d, candidates %d, actual not candidate %d, candidate not actual %d" % (
        len(actual), len(cand), len(set(actual) - cand), len(cand - set(actual))))
    extra = Counter()
    for I, J, sigma, mono in cand - set(actual):
        extra[mono] += 1
    for mono, k in sorted(extra.items()):
        print("    unused candidates:", k, "x", "*".join("%s^%d" % x for x in mono))
    return model, actual, cand


if __name__ == "__main__":
    for p in sys.argv[1:]:
        compare(p)
