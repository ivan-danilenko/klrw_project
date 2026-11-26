"""
Catalogue of the true terms beyond the tree formula in a local data file, in "physical" form:
which strand crosses the red strand (R slot -> L slot), which right strands move inwards,
which left strands move outwards.      python3 catalogue.py pieces/FILE.json
"""
import sys, json, os
from collections import defaultdict
from formula import Stack

def physical(V, U, sigma):
    V, U = set(V), set(U)
    mover = []; rmoves = []; lmoves = []; other = []
    for s, t in enumerate(sigma, 1):
        if s in V and t not in U:
            mover.append((s, t))
        elif s in V and t in U:
            if s != t: rmoves.append((s, t))
        elif s not in V and t not in U:
            if s != t: lmoves.append((s, t))
        else:
            other.append((s, t))
    return mover, sorted(rmoves), sorted(lmoves), other

def main(path, only_new=True):
    data = json.load(open(path))
    st = Stack(data["cartan_type"], data["word"])
    col = st.col
    truth = defaultdict(dict)
    for src, tgt, sigma, mono, dots, c in data["terms"]:
        truth[(frozenset(src), frozenset(tgt))][(tuple(sigma), tuple(tuple(x) for x in mono), tuple(tuple(x) for x in dots))] = c
    print("=====", os.path.basename(path), " slots:", " ".join("%d:c%d" % (s + 1, c) for s, c in enumerate(col)))
    seen = set()
    for (V, U), terms in sorted(truth.items(), key=lambda kv: (sorted(kv[0][0]), sorted(kv[0][1]))):
        extra = dict(terms)
        for J, sigma, mono, sign in st.terms_from(V):
            if J == U:
                k = (sigma, mono, ())
                extra[k] = extra.get(k, 0) - sign
        extra = {k: v for k, v in extra.items() if v}
        if not extra:
            continue
        # show each family once (ignore unaffected slots): key by out/in sets
        fam = (tuple(sorted(V - U)), tuple(sorted(U - V)))
        if fam in seen:
            continue
        seen.add(fam)
        print("V=%s  out %s  in %s" % (sorted(V), sorted(V - U), sorted(U - V)))
        for (sigma, mono, dots), c in sorted(extra.items(), key=lambda kv: (len(kv[0][2]) == 0, str(kv[0][1]))):
            mover, rm, lm, other = physical(V, U, sigma)
            f = lambda lst: " ".join("%d>%d(c%d)" % (a, b, col[a - 1]) for a, b in lst) or "-"
            print("   %+d %-40s dots %-8s mover %s | R: %s | L: %s%s" % (
                c, "*".join(n if p == 1 else "%s^%d" % (n, p) for n, p in mono),
                ",".join(d for d, _ in dots) or "-", f(mover), f(rm), f(lm), " | OTHER %s" % other if other else ""))

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
