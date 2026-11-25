"""
Print the correction terms of a stored case in "slot" language.

    python3 analyze.py data/CASE.json [max_terms_per_class]
"""
import json
import sys
from collections import defaultdict


def cycles(sigma):
    seen, out = set(), []
    for s in range(1, len(sigma) + 1):
        if s in seen or sigma[s - 1] == s:
            continue
        cyc, t = [], s
        while t not in seen:
            seen.add(t)
            cyc.append(t)
            t = sigma[t - 1]
        out.append(tuple(cyc))
    return out


def mono(m):
    return "*".join(n if p == 1 else "%s^%d" % (n, p) for n, p in m) or "1"


def main(path, limit=6):
    data = json.load(open(path))
    col = data["strand_colour"]
    n = len(col)
    print(path)
    print("slots:", " ".join("%d:c%d" % (s + 1, c) for s, c in enumerate(col)))
    for o in data["orders"]:
        print("  order", o["monomial"], o["equations"], "x", o["unknowns"], "rank", o["rank"], "terms", o["solution_terms"])
    classes = defaultdict(list)
    for t in data["terms"]:
        if not t["params"]:
            assert all(t["sigma"][s] == s + 1 for s in range(n)), t
            continue
        src, tgt = set(t["src"]), set(t["tgt"])
        key = (mono(t["params"]), mono(t["dots"]), tuple(cycles(t["sigma"])),
               tuple(sorted(src - tgt)), tuple(sorted(tgt - src)))
        classes[key].append(t)
    print("correction classes: (monomial, dots, sigma cycles, slots leaving I, slots entering I) x count")
    for key, ts in sorted(classes.items()):
        involved = set(key[3]) | set(key[4])
        for c in key[2]:
            involved |= set(c)
        # which other slots are constrained?
        fixed = {}
        for s in range(1, n + 1):
            if s in involved:
                continue
            vals = set((s in t["src"]) for t in ts)
            if len(vals) == 1:
                fixed[s] = "R" if vals.pop() else "L"
        signs = sorted(set(t["coeff"] for t in ts))
        print("  %-28s dots %-6s sigma %-22s out %-12s in %-10s x%-3d fixed %s coeffs %s" % (
            key[0], key[1], key[2], key[3], key[4], len(ts), fixed, signs))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
