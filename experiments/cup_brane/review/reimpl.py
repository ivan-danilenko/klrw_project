#!/usr/bin/env python3
"""Independent re-implementation of the closed formula of NOTES.md, section 2.

Written from the text of NOTES.md sections 1-2 (and the "Two punctures"
paragraph) only; none of the author's formula code was consulted.

Usage:
    python3 reimpl.py                 # compare against data/, data_fast/, data2/
    python3 reimpl.py FILE.json ...   # compare against the given files
    python3 reimpl.py -v FILE.json    # also list every discrepancy

Each generated term is a dict with keys
    src, tgt   sorted lists (vertices I, J),
    sigma      list, sigma[s-1] = slot where the strand of source slot s ends,
    params     sorted list of [variable, power],
    coeff      +1 / -1,
    mover, pairs, parent, connector   (bookkeeping, not compared).

Options controlling the points where the text had to be interpreted are
collected in the dict OPTIONS below; the defaults are the literal reading.
probe.py (same folder) reruns the comparison under the alternative readings.
"""

import glob
import json
import os
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Interpretation switches (defaults = most literal reading of NOTES.md 2).
# ---------------------------------------------------------------------------
OPTIONS = {
    # "x_p in I": I is the *source* vertex (literal).  Alternative: 'J'.
    "connector_set": "I",
    # Koszul sign: entries counted "to the right" of j are taken in the
    # sequence after the b_p -> a_p replacement (literal: the replacement
    # happens first).  Alternative 'before' would count in sorted(I); since
    # the replacement is in place the position of j does not change, so the
    # two agree -- kept only to document that it was checked.
    "kappa_count": "after",
    # Links (and pairs) only inside one stack (the "Two punctures" paragraph).
    "same_stack_only": True,
    # --- switches used only by probe.py to test alternative readings ---
    "cond1": True,              # admissibility 1 enforced
    "cond2": True,              # admissibility 2 enforced
    "graph": "tree",            # 'tree' | 'forest' | 'connected'
    "t_order": "pair_first",    # t_{c(p),c(x_p)}; 'parent_first' = t_{c(x_p),c(p)}
    "connector_sign": True,     # include (-1)^{#{p: x_p in I}}
    "kappa_replace": True,      # do the b_p -> a_p replacement in kappa
    "mover_link_needs_adjacent": True,
}


# ---------------------------------------------------------------------------
# Dynkin diagrams (SageMath conventions)
# ---------------------------------------------------------------------------
def dynkin_edges(cartan_type):
    letter, n = cartan_type
    edges = set()
    if letter == "A":
        for i in range(1, n):
            edges.add(frozenset((i, i + 1)))
    elif letter == "D":
        for i in range(1, n - 1):          # path 1 - 2 - ... - (n-1)
            edges.add(frozenset((i, i + 1)))
        edges.add(frozenset((n - 2, n)))   # n attached to n-2
    elif letter == "E":
        edges.add(frozenset((1, 3)))
        edges.add(frozenset((2, 4)))
        for i in range(3, n):
            edges.add(frozenset((i, i + 1)))
    else:
        raise ValueError("unsupported Cartan type %r" % (cartan_type,))
    return edges


# ---------------------------------------------------------------------------
# Permutation sign
# ---------------------------------------------------------------------------
def sort_sign(seq):
    inv = 0
    for x in range(len(seq)):
        for y in range(x + 1, len(seq)):
            if seq[x] > seq[y]:
                inv += 1
    return -1 if inv % 2 else 1


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------
class Stacks:
    """One or several stacks, slots numbered globally 1..m."""

    def __init__(self, cartan_type, colours, stack_of=None, options=None):
        self.edges = dynkin_edges(cartan_type)
        self.col = list(colours)                     # col[s-1] = c(s)
        self.m = len(colours)
        self.stack = list(stack_of) if stack_of is not None else [0] * self.m
        self.opt = dict(OPTIONS)
        if options:
            self.opt.update(options)

    # helpers -------------------------------------------------------------
    def c(self, s):
        return self.col[s - 1]

    def adj(self, c1, c2):
        return frozenset((c1, c2)) in self.edges

    def same_stack(self, *slots):
        if not self.opt["same_stack_only"]:
            return True
        return len({self.stack[s - 1] for s in slots}) == 1

    @staticmethod
    def interlaced(p, q):
        """a_p < a_q < b_p < b_q or the same with p, q swapped."""
        (ap, bp), (aq, bq) = p, q
        return ap < aq < bp < bq or aq < ap < bq < bp

    def linked_to_mover(self, j, p):
        a, b = p
        if not self.opt["mover_link_needs_adjacent"]:
            return a < j < b and self.c(j) != self.c(a) and self.same_stack(j, a)
        return a < j < b and self.adj(self.c(j), self.c(a)) and self.same_stack(j, a)

    def linked(self, p, q):
        return (self.interlaced(p, q) and self.adj(self.c(p[0]), self.c(q[0]))
                and self.same_stack(p[0], q[0]))

    # enumeration ---------------------------------------------------------
    def candidate_pairs(self, I, j):
        cj = self.c(j)
        out = []
        for a in range(1, self.m + 1):
            if a in I:
                continue
            for b in range(a + 1, self.m + 1):
                if b not in I or b == j:
                    continue
                if self.c(a) != self.c(b) or not self.same_stack(a, b):
                    continue
                # admissibility 1: no pair of the mover's colour around j
                if self.opt["cond1"] and self.c(a) == cj and a < j < b:
                    continue
                out.append((a, b))
        return out

    def pair_sets(self, I, j):
        """All sets F of pairwise disjoint candidate pairs obeying rule 2."""
        cand = self.candidate_pairs(I, j)
        res = []

        def rec(k, chosen, used):
            if k == len(cand):
                res.append(list(chosen))
                return
            rec(k + 1, chosen, used)
            p = cand[k]
            if p[0] in used or p[1] in used:
                return
            for q in chosen:
                # admissibility 2: no two interlaced pairs of the same colour
                if (self.opt["cond2"] and self.c(p[0]) == self.c(q[0])
                        and self.interlaced(p, q)):
                    return
            chosen.append(p)
            rec(k + 1, chosen, used | {p[0], p[1]})
            chosen.pop()

        rec(0, [], frozenset())
        return res

    def tree(self, j, F):
        """Admissibility 3. Returns (parent, connector) dicts or None."""
        nbrs = {"M": []}
        for p in F:
            nbrs[p] = []
        nedges = 0
        for p in F:
            if self.linked_to_mover(j, p):
                nbrs["M"].append(p)
                nbrs[p].append("M")
                nedges += 1
        for x in range(len(F)):
            for y in range(x + 1, len(F)):
                p, q = F[x], F[y]
                if self.linked(p, q):
                    nbrs[p].append(q)
                    nbrs[q].append(p)
                    nedges += 1
        if self.opt["graph"] == "tree" and nedges != len(F):
            return None
        parent = {}
        seen = {"M"}
        queue = ["M"]
        while queue:
            v = queue.pop(0)
            for w in nbrs[v]:
                if w not in seen:
                    seen.add(w)
                    parent[w] = v
                    queue.append(w)
        if self.opt["graph"] in ("tree", "connected") and len(seen) != len(F) + 1:
            return None
        if self.opt["graph"] == "forest":
            # acyclic: #edges = #vertices - #components
            comps = 0
            seen2 = set()
            for v in nbrs:
                if v in seen2:
                    continue
                comps += 1
                st = [v]
                seen2.add(v)
                while st:
                    u = st.pop()
                    for w in nbrs[u]:
                        if w not in seen2:
                            seen2.add(w)
                            st.append(w)
            if nedges != len(nbrs) - comps:
                return None
        connector = {}
        for p in F:
            if p not in parent:      # only for the 'forest' probe
                connector[p] = None
                continue
            q = parent[p]
            ap, bp = p
            if q == "M":
                x = j
            else:
                aq, bq = q
                if ap < aq < bp < bq:
                    x = aq
                elif aq < ap < bq < bp:
                    x = bq
                else:  # cannot happen: links are interlaced
                    raise AssertionError
            assert ap < x < bp
            connector[p] = x
        return parent, connector

    def kappa(self, I, j, F):
        seq = sorted(I)
        if self.opt["kappa_count"] == "before":
            right = len(seq) - 1 - seq.index(j)
        repl = {b: a for (a, b) in F} if self.opt["kappa_replace"] else {}
        seq = [repl.get(s, s) for s in seq]
        if self.opt["kappa_count"] == "after":
            right = len(seq) - 1 - seq.index(j)
        seq.remove(j)
        return (-1) ** right * sort_sign(seq)

    def term(self, I, j, F):
        t = self.tree(j, F)
        if t is None:
            return None
        parent, connector = t
        mono = Counter()
        for p in F:
            cp = self.c(p[0])
            mono["r_%d" % cp] += 1
            if connector[p] is None:
                continue
            cx = self.c(connector[p])
            if self.opt["t_order"] == "pair_first":
                mono["t_%d_%d" % (cp, cx)] += 1
            else:
                mono["t_%d_%d" % (cx, cp)] += 1
        J = (set(I) - {j} - {b for (_, b) in F}) | {a for (a, _) in F}
        ref = set(I) if self.opt["connector_set"] == "I" else J
        nin = sum(1 for p in F if connector[p] in ref) if self.opt["connector_sign"] else 0
        eps = self.kappa(I, j, F) * (-1) ** nin
        sigma = list(range(1, self.m + 1))
        for (a, b) in F:
            sigma[a - 1], sigma[b - 1] = b, a
        return {
            "src": sorted(I),
            "tgt": sorted(J),
            "sigma": sigma,
            "params": sorted([k, v] for k, v in mono.items()),
            "coeff": eps,
            "mover": j,
            "pairs": sorted(F),
            "parent": {str(p): str(q) for p, q in parent.items()},
            "connector": {str(p): x for p, x in connector.items()},
        }

    def terms_from(self, I):
        I = frozenset(I)
        out = []
        for j in sorted(I):
            for F in self.pair_sets(I, j):
                t = self.term(I, j, F)
                if t is not None:
                    out.append(t)
        return out

    def all_terms(self):
        out = []
        for mask in range(1 << self.m):
            I = {s + 1 for s in range(self.m) if mask >> s & 1}
            out.extend(self.terms_from(I))
        return out


# ---------------------------------------------------------------------------
# Comparison with the JSON data
# ---------------------------------------------------------------------------
def key(t):
    return (tuple(t["src"]), tuple(t["tgt"]), tuple(t["sigma"]))


def mono(t):
    return tuple(sorted((k, v) for k, v in t["params"]))


def compare(path, options=None, verbose=False):
    d = json.load(open(path))
    st = Stacks(d["cartan_type"], d["strand_colour"], d["strand_puncture"], options)
    gen = st.all_terms()

    data = defaultdict(list)
    for t in d["terms"]:
        if t.get("dots"):
            raise ValueError("terms with dots not supported")
        data[key(t)].append(t)
    mine = defaultdict(list)
    for t in gen:
        mine[key(t)].append(t)

    dup_data = sum(1 for v in data.values() if len(v) > 1)
    dup_mine = sum(1 for v in mine.values() if len(v) > 1)

    full_d = {(k, mono(t)): t for k, v in data.items() for t in v}
    full_m = {(k, mono(t)): t for k, v in mine.items() for t in v}
    missing = [k for k in full_d if k not in full_m]
    extra = [k for k in full_m if k not in full_d]
    signs = [k for k in full_d if k in full_m and full_d[k]["coeff"] != full_m[k]["coeff"]]
    mono_mis = [k for k in data if k in mine and
                {mono(t) for t in data[k]} != {mono(t) for t in mine[k]}]

    res = {
        "file": path,
        "n_file": len(d["terms"]),
        "n_gen": len(gen),
        "missing": len(missing),
        "extra": len(extra),
        "sign": len(signs),
        "mono_mismatch_same_diagram": len(mono_mis),
        "dup_keys_file": dup_data,
        "dup_keys_gen": dup_mine,
        "missing_list": missing,
        "extra_list": extra,
        "sign_list": signs,
        "mine": full_m,
        "data": full_d,
    }
    if verbose:
        for k in missing:
            t = full_d[k]
            print("  MISSING", t["src"], "->", t["tgt"], "sigma", t["sigma"], t["params"], t["coeff"])
        for k in extra:
            t = full_m[k]
            print("  EXTRA  ", t["src"], "->", t["tgt"], "sigma", t["sigma"], t["params"], t["coeff"],
                  "mover", t["mover"], "pairs", t["pairs"], "conn", t["connector"])
        for k in signs:
            t = full_m[k]
            print("  SIGN   ", t["src"], "->", t["tgt"], "sigma", t["sigma"], t["params"],
                  "file", full_d[k]["coeff"], "mine", t["coeff"],
                  "mover", t["mover"], "pairs", t["pairs"], "conn", t["connector"])
    return res


DEFAULT_DIRS = ["data", "data_fast", "data2"]


def main(argv):
    verbose = False
    if argv and argv[0] == "-v":
        verbose = True
        argv = argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(here)
    files = argv or [f for d in DEFAULT_DIRS for f in sorted(glob.glob(os.path.join(base, d, "*.json")))]
    hdr = "%-40s %7s %7s %7s %7s %6s %6s" % ("file", "n_file", "n_gen", "missing", "extra", "sign", "mono")
    print(hdr)
    print("-" * len(hdr))
    tot = Counter()
    for f in files:
        r = compare(f, verbose=verbose)
        name = os.path.relpath(f, base) if f.startswith(base) else f
        print("%-40s %7d %7d %7d %7d %6d %6d" % (name, r["n_file"], r["n_gen"], r["missing"],
                                                r["extra"], r["sign"], r["mono_mismatch_same_diagram"]))
        if r["dup_keys_file"] or r["dup_keys_gen"]:
            print("   (duplicate (src,tgt,sigma) keys: file %d, generated %d)"
                  % (r["dup_keys_file"], r["dup_keys_gen"]))
        for k in ("n_file", "n_gen", "missing", "extra", "sign"):
            tot[k] += r[k]
    print("-" * len(hdr))
    print("%-40s %7d %7d %7d %7d %6d" % ("TOTAL", tot["n_file"], tot["n_gen"], tot["missing"],
                                         tot["extra"], tot["sign"]))


if __name__ == "__main__":
    main(sys.argv[1:])
