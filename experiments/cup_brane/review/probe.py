#!/usr/bin/env python3
"""Probe which readings of NOTES.md section 2 the data actually distinguishes.

Runs reimpl.py under alternative interpretations and prints totals over all
files in data/, data_fast/, data2/, plus statistics of the generated terms
(|F|, depth of the tree, kind of connector, configurations with cycles).
"""

import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reimpl  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [f for d in reimpl.DEFAULT_DIRS for f in sorted(glob.glob(os.path.join(BASE, d, "*.json")))]

ALTERNATIVES = [
    ("literal reading", {}),
    ("x_p counted in J instead of I", {"connector_set": "J"}),
    ("no (-1)^{#x_p in I} factor", {"connector_sign": False}),
    ("kappa without b_p -> a_p replacement", {"kappa_replace": False}),
    ("t_{c(x_p),c(p)} instead of t_{c(p),c(x_p)}", {"t_order": "parent_first"}),
    ("drop admissibility 1", {"cond1": False}),
    ("drop admissibility 2", {"cond2": False}),
    ("drop 1 and 2", {"cond1": False, "cond2": False}),
    ("'tree' read as 'forest'", {"graph": "forest"}),
    ("'tree' read as 'connected' (cycles allowed)", {"graph": "connected"}),
    ("mover link without adjacency (any other colour)", {"mover_link_needs_adjacent": False}),
    ("pairs/links across stacks allowed", {"same_stack_only": False}),
]


def totals(opts):
    tot = Counter()
    per_file = {}
    for f in FILES:
        r = reimpl.compare(f, options=opts)
        for k in ("n_file", "n_gen", "missing", "extra", "sign"):
            tot[k] += r[k]
        per_file[os.path.relpath(f, BASE)] = (r["missing"], r["extra"], r["sign"])
    return tot, per_file


def stats():
    print("\nStatistics of generated terms (literal reading)")
    print("%-36s %5s %18s %22s %8s" % ("file", "terms", "|F| distribution", "connector j/a_q/b_q", "depth>=2"))
    for f in FILES:
        d = json.load(open(f))
        st = reimpl.Stacks(d["cartan_type"], d["strand_colour"], d["strand_puncture"])
        T = st.all_terms()
        nF = Counter(len(t["pairs"]) for t in T)
        conn = Counter()
        deep = 0
        for t in T:
            par = t["parent"]
            for p, x in t["connector"].items():
                if par[p] == "M":
                    conn["j"] += 1
                else:
                    q = eval(par[p])
                    conn["a" if x == q[0] else "b"] += 1
                    deep += 1
        print("%-36s %5d %18s %22s %8d" % (
            os.path.relpath(f, BASE), len(T),
            "/".join(str(nF[k]) for k in range(max(nF) + 1)),
            "%d/%d/%d" % (conn["j"], conn["a"], conn["b"]), deep))


def cyclic_configs():
    """Configurations satisfying 1, 2, connected, but with a cycle."""
    print("\nConfigurations (I, j, F) obeying 1 and 2 whose link graph is connected but has a cycle")
    for f in FILES:
        d = json.load(open(f))
        st = reimpl.Stacks(d["cartan_type"], d["strand_colour"], d["strand_puncture"])
        n = 0
        n_forest_only = 0
        for mask in range(1 << st.m):
            I = frozenset(s + 1 for s in range(st.m) if mask >> s & 1)
            for j in sorted(I):
                for F in st.pair_sets(I, j):
                    # count edges / connectivity directly
                    E = sum(1 for p in F if st.linked_to_mover(j, p))
                    E += sum(1 for x in range(len(F)) for y in range(x + 1, len(F)) if st.linked(F[x], F[y]))
                    if E > len(F):
                        n += 1
        print("  %-36s %d" % (os.path.relpath(f, BASE), n))


def triangles():
    """Count terms whose (minimal) diagram has three black strands pairwise
    crossing with outer colours equal and middle colour adjacent, i.e. where
    the reduced expression is not unique up to braid moves without
    correction in an (ordinary) KLR algebra."""
    print("\nTerms whose diagram contains a black triangle (i, i', i) with i ~ i' adjacent")
    for f in FILES:
        d = json.load(open(f))
        st = reimpl.Stacks(d["cartan_type"], d["strand_colour"], d["strand_puncture"])
        stacks = sorted(set(d["strand_puncture"]))
        n_tri = 0
        n_any = 0
        for t in d["terms"]:
            I, J, sig = set(t["src"]), set(t["tgt"]), t["sigma"]

            def positions(V):
                pos = {}
                k = 0
                for P in stacks:
                    slots = [s for s in range(1, st.m + 1) if st.stack[s - 1] == P]
                    left = sorted([s for s in slots if s not in V], reverse=True)
                    right = sorted([s for s in slots if s in V])
                    for s in left:
                        pos[("L", s)] = k
                        k += 1
                    pos[("W", P)] = k
                    k += 1
                    for s in right:
                        pos[("R", s)] = k
                        k += 1
                return pos
            ps, pt = positions(I), positions(J)
            strands = []
            for s in range(1, st.m + 1):
                side_s = "R" if s in I else "L"
                s2 = sig[s - 1]
                side_t = "R" if s2 in J else "L"
                strands.append((ps[(side_s, s)], pt[(side_t, s2)], st.c(s)))
            strands.sort()
            found = False
            found_any = False
            L = len(strands)
            for x in range(L):
                for y in range(x + 1, L):
                    if strands[x][1] < strands[y][1]:
                        continue
                    for z in range(y + 1, L):
                        if strands[y][1] > strands[z][1] and strands[x][1] > strands[z][1]:
                            found_any = True
                            if strands[x][2] == strands[z][2] and st.adj(strands[x][2], strands[y][2]):
                                found = True
            n_tri += found
            n_any += found_any
        print("  %-36s terms %5d   any triangle %5d   (i,i',i) triangle %5d"
              % (os.path.relpath(f, BASE), len(d["terms"]), n_any, n_tri))


def _sign_mismatches(cls, files):
    n = 0
    for f in files:
        d = json.load(open(f))
        st = cls(d["cartan_type"], d["strand_colour"], d["strand_puncture"])
        gen = {reimpl.key(t): t["coeff"] for t in st.all_terms()}
        n += sum(1 for t in d["terms"] if gen[reimpl.key(t)] != t["coeff"])
    return n


class SignAllPairs(reimpl.Stacks):
    """eps = kappa * (-1)^{|F|}  (every connector counted)."""
    def term(self, I, j, F):
        t = super().term(I, j, F)
        if t is not None:
            n_a = sum(1 for p, x in t["connector"].items()
                      if t["parent"][p] != "M" and x == eval(t["parent"][p])[0])
            t["coeff"] *= (-1) ** n_a
        return t


class PerStackKappa(reimpl.Stacks):
    """kappa computed inside the mover's stack only."""
    def kappa(self, I, j, F):
        P = self.stack[j - 1]
        return reimpl.Stacks.kappa(self, [s for s in I if self.stack[s - 1] == P], j, F)


class TextbookKoszul(reimpl.Stacks):
    """d(x (x) y) = dx (x) y + (-1)^{|x|} x (x) dy, x = earlier stack."""
    def kappa(self, I, j, F):
        P = self.stack[j - 1]
        before = sum(1 for s in I if self.stack[s - 1] < P)
        return (-1) ** before * reimpl.Stacks.kappa(self, [s for s in I if self.stack[s - 1] == P], j, F)


def extra_sign_checks():
    print("\nFurther sign readings (number of sign mismatches)")
    print("  eps = kappa*(-1)^|F|, all files:              ", _sign_mismatches(SignAllPairs, FILES))
    d2 = [f for f in FILES if "/data2/" in f]
    print("  kappa per stack, data2:                       ", _sign_mismatches(PerStackKappa, d2))
    print("  textbook Koszul sign for the product, data2:  ", _sign_mismatches(TextbookKoszul, d2))


def word_check():
    """The undocumented 'word' field vs the position permutation built from
    the idempotent of NOTES.md section 1 (stacks left to right in list order)."""
    res = Counter()
    for f in FILES:
        d = json.load(open(f))
        m, stk, npu = len(d["strand_colour"]), d["strand_puncture"], len(d["punctures"])

        def positions(V):
            pos, k = {}, 0
            for P in range(npu):
                slots = [s for s in range(1, m + 1) if stk[s - 1] == P]
                for s in sorted([s for s in slots if s not in V], reverse=True):
                    pos[("L", s)] = k
                    k += 1
                pos[("W", P)] = k
                k += 1
                for s in sorted([s for s in slots if s in V]):
                    pos[("R", s)] = k
                    k += 1
            return pos, k
        for t in d["terms"]:
            I, J, sig = set(t["src"]), set(t["tgt"]), t["sigma"]
            ps, n = positions(I)
            pt, _ = positions(J)
            perm = [None] * n
            for P in range(npu):
                perm[ps[("W", P)]] = pt[("W", P)]
            for s in range(1, m + 1):
                s2 = sig[s - 1]
                perm[ps[("R" if s in I else "L", s)]] = pt[("R" if s2 in J else "L", s2)]
            inv = sum(1 for x in range(n) for y in range(x + 1, n) if perm[x] > perm[y])
            cur = list(range(n))
            for i in t["word"]:
                cur[i - 1], cur[i] = cur[i], cur[i - 1]
            out = [None] * n
            for pos, s in enumerate(cur):
                out[s] = pos
            res[(len(t["word"]) == inv, out == perm)] += 1
    print("\n'word' field: (len(word) == #inversions, word reproduces permutation) ->", dict(res))


def main():
    print("%-50s %7s %7s %7s %7s %6s  files affected" % ("reading", "n_file", "n_gen", "missing", "extra", "sign"))
    for name, opts in ALTERNATIVES:
        tot, pf = totals(opts)
        bad = sum(1 for v in pf.values() if any(v))
        print("%-50s %7d %7d %7d %7d %6d  %d/%d" % (name, tot["n_file"], tot["n_gen"], tot["missing"],
                                                   tot["extra"], tot["sign"], bad, len(pf)))
    extra_sign_checks()
    stats()
    cyclic_configs()
    triangles()
    word_check()


if __name__ == "__main__":
    main()
