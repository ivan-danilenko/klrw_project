"""
Closed-form description of the corrected differential of a stack of nested
U-branes (conjectural; see NOTES.md).

Pure Python.  ``terms_from(I)`` generates all terms of d out of the vertex I.
"""
import itertools
from collections import defaultdict
from degree_model import edges as dynkin_edges


def perm_sign(seq):
    seq = list(seq)
    s = 1
    for i in range(len(seq)):
        for k in range(i + 1, len(seq)):
            if seq[i] > seq[k]:
                s = -s
    return s


class Stack:
    """
    One stack: ``Stack(ct, colours)``.  Several stacks side by side (a stable
    envelope with two punctures): ``Stack(ct, colours, puncture=[0, 0, ..., 1, 1])``;
    slots are numbered globally, stack after stack, innermost first, and
    exchanges only happen inside a stack.
    """

    def __init__(self, ct, colours, puncture=None):
        self.ct = ct
        self.col = list(colours)          # slot s (1-based) has colour col[s-1]
        self.m = len(self.col)
        self.punc = list(puncture) if puncture is not None else [0] * self.m
        self.E = dynkin_edges(ct)
        self.same = defaultdict(list)     # (puncture, colour) -> slots
        for s, c in enumerate(self.col, 1):
            self.same[self.punc[s - 1], c].append(s)
        self.pairs = [(a, b) for sl in self.same.values() for a, b in itertools.combinations(sl, 2)]

    def adjacent(self, s, t):
        return frozenset((self.col[s - 1], self.col[t - 1])) in self.E

    # -- link structure --------------------------------------------------------
    def analyse(self, j, F):
        """
        ``j`` mover, ``F`` tuple of disjoint pairs (a, b).  Returns None if the
        configuration is not admissible, else the dict child-pair -> (parent, connector slot),
        where parent is 'mover' or a pair and the connector is the slot of the parent lying
        strictly inside the child's interval.
        """
        cj = self.col[j - 1]
        links = defaultdict(list)
        for (a, b) in F:
            if a < j < b and self.punc[a - 1] == self.punc[j - 1]:
                if self.col[a - 1] == cj:
                    return None                       # same-colour mover inside a pair
                if self.adjacent(a, j):
                    links["mover"].append(((a, b), j))
                    links[(a, b)].append(("mover", None))
        for (a, b), (a2, b2) in itertools.combinations(F, 2):
            if a > a2:
                (a, b), (a2, b2) = (a2, b2), (a, b)
            if a < a2 < b < b2 and self.punc[a - 1] == self.punc[a2 - 1]:   # interlaced
                if self.col[a - 1] == self.col[a2 - 1]:
                    return None                       # interlaced pairs of equal colour
                if self.adjacent(a, a2):
                    # b lies inside (a2, b2);  a2 lies inside (a, b)
                    links[(a, b)].append(((a2, b2), b))
                    links[(a2, b2)].append(((a, b), a2))
        # tree rooted at the mover
        n_edges = sum(len(v) for v in links.values()) // 2
        if n_edges != len(F):
            return None
        parent = {}
        stack = ["mover"]
        seen = {"mover"}
        while stack:
            p = stack.pop()
            for child, connector in links.get(p, []):
                if child in seen or child == "mover":
                    continue
                seen.add(child)
                parent[child] = (p, connector)
                stack.append(child)
        if len(seen) != len(F) + 1:
            return None
        return parent

    def monomial(self, parent):
        mono = defaultdict(int)
        for (a, b), (p, connector) in parent.items():
            c = self.col[a - 1]
            cp = self.col[connector - 1]
            mono["r_%d" % c] += 1
            mono["t_%d_%d" % (c, cp)] += 1
        return tuple(sorted(mono.items()))

    def koszul(self, I, j, F):
        sigma = {}
        for a, b in F:
            sigma[b] = a
        img = [sigma.get(s, s) for s in sorted(I)]
        k = img.index(j)
        return (-1) ** (len(img) - 1 - k) * perm_sign(img[:k] + img[k + 1:])

    def sign(self, I, j, F, parent):
        s = self.koszul(I, j, F)
        for child, (p, connector) in parent.items():
            if connector in I:
                s = -s
        return s

    # -- enumeration -----------------------------------------------------------
    def terms_from(self, I):
        """All terms out of the vertex I: (J, sigma tuple, monomial, sign)."""
        I = frozenset(I)
        avail = [(a, b) for a, b in self.pairs if a not in I and b in I]
        out = []
        for j in sorted(I):
            # grow trees from the mover
            def grow(F, used):
                parent = self.analyse(j, F)
                if parent is not None:
                    yield F, parent
                for p in avail:
                    if F and p <= F[-1]:
                        continue
                    if p[0] in used or p[1] in used or j in p:
                        continue
                    yield from grow(F + (p,), used | set(p))
            for F, parent in grow((), frozenset()):
                J = (I - {j} - {b for a, b in F}) | {a for a, b in F}
                sigma = list(range(1, self.m + 1))
                for a, b in F:
                    sigma[a - 1], sigma[b - 1] = b, a
                out.append((frozenset(J), tuple(sigma), self.monomial(parent), self.sign(I, j, F, parent)))
        return out

    def all_terms(self):
        res = {}
        for h in range(1, self.m + 1):
            for I in itertools.combinations(range(1, self.m + 1), h):
                I = frozenset(I)
                for J, sigma, mono, sign in self.terms_from(I):
                    key = (I, J, sigma, mono)
                    assert key not in res
                    res[key] = sign
        return res


if __name__ == "__main__":
    import sys, json
    for path in sys.argv[1:]:
        data = json.load(open(path))
        st = Stack(data["cartan_type"], data["strand_colour"], data.get("strand_puncture"))
        actual = {}
        for t in data["terms"]:
            actual[(frozenset(t["src"]), frozenset(t["tgt"]), tuple(t["sigma"]),
                    tuple(sorted((n, p) for n, p in t["params"])))] = t["coeff"]
        pred = st.all_terms()
        common = set(pred) & set(actual)
        wrong = sum(1 for k in common if pred[k] != actual[k])
        print("%-40s actual %5d predicted %5d  missing %d  extra %d  wrong sign %d" % (
            path.split("/")[-1], len(actual), len(pred), len(set(actual) - set(pred)),
            len(set(pred) - set(actual)), wrong))
