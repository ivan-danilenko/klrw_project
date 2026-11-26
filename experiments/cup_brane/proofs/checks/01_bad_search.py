"""
01_bad_search.py  (pure Python)

    python3 proofs/checks/01_bad_search.py exhaustive TYPE RANK MAXLEN
    python3 proofs/checks/01_bad_search.py random TYPE RANK N MAXLEN [SEED]
    python3 proofs/checks/01_bad_search.py crosscheck        # compare with 01_triples.py

Fast search for BAD triples in the diagrams beta(I; j, F) of admissible configurations,
using the interval form of Lemma B.4 of 01_foundations.md (a bad triple exists iff one of
(B1)-(B4) holds; this does not depend on I).  For every word: is it cyclic (find_cycle,
exhaustive: max_pairs = m // 2), and does some admissible configuration have a bad triple?
Also tests the explicit "shortening" construction of Proposition B.6 (a bad triple gives a
cyclic configuration) on every bad configuration found.
"""
import itertools
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from formula import Stack          # noqa: E402
from cyclefree import find_cycle, links   # noqa: E402


def configurations(st):
    """All admissible (j, F) of the word (independent of the vertex I): F a set of disjoint
    same-colour pairs avoiding j, no blocks, link graph a tree.  Grown leaf by leaf."""
    out = []
    for j in range(1, st.m + 1):
        avail = [p for p in st.pairs if j not in p]
        found = {()}
        frontier = [()]
        while frontier:
            new = []
            for F in frontier:
                used = {x for p in F for x in p}
                for p in avail:
                    if p[0] in used or p[1] in used:
                        continue
                    G = tuple(sorted(F + (p,)))
                    if G in found:
                        continue
                    if st.analyse(j, G) is None:
                        continue
                    found.add(G)
                    new.append(G)
            frontier = new
        out += [(j, F) for F in found]
    return out


def adj(st, c, d):
    return frozenset((c, d)) in st.E


def bad_triples(st, j, F):
    """List of the bad triples of (j, F) in interval form (B1)-(B4)."""
    col = lambda s: st.col[s - 1]
    cj = col(j)
    used = {j} | {x for p in F for x in p}
    unaff = [y for y in range(1, st.m + 1) if y not in used]
    res = []
    for p in F:
        a, b = p
        if adj(st, col(a), cj) and b < j:
            for y in unaff:
                if a < y < b and col(y) == cj:
                    res.append(("B1", y, p))
    for p, q in itertools.permutations(F, 2):
        if p[0] < q[0] < q[1] < p[1]:                      # q nested in p
            if p[1] < j and col(q[0]) == cj and adj(st, col(p[0]), cj):
                res.append(("B2", p, q))
            if adj(st, col(p[0]), col(q[0])):
                for y in unaff:
                    if q[0] < y < q[1] and col(y) == col(p[0]):
                        res.append(("B3", p, q, y))
                for r in F:
                    if q[0] < r[0] < r[1] < q[1] and col(r[0]) == col(p[0]):
                        res.append(("B4", p, q, r))
    return res


def is_cyclic_config(st, j, F):
    r = links(st, j, F)
    return r is not None and r[1] and r[0] > len(F)


def has_cyclic_subconfig(st, j, F):
    """Does some subset of F, with mover j, form a cyclic configuration?"""
    F = list(F)
    for k in range(1, len(F) + 1):
        for G in itertools.combinations(F, k):
            if is_cyclic_config(st, j, tuple(sorted(G))):
                return tuple(sorted(G))
    return None


def shortening(st, j, F, bad):
    """Proposition B.6: from a bad triple build a configuration that contains a cyclic one.
    B1: new mover y.   B2: new mover a_q (if a_q is not ... ) -- we simply try all natural
    candidates and report the first cyclic sub-configuration found."""
    kind = bad[0]
    cands = []
    if kind == "B1":
        _, y, p = bad
        cands.append((y, tuple(F)))
    elif kind == "B2":
        _, p, q = bad
        rest = tuple(x for x in F if x != q)
        cands += [(q[0], rest), (q[1], rest)]
    elif kind == "B3":
        _, p, q, y = bad
        rest = [x for x in F if x != p]
        cands += [(j, tuple(sorted(rest + [(p[0], y)]))), (j, tuple(sorted(rest + [(y, p[1])])))]
    elif kind == "B4":
        _, p, q, r = bad
        rest = [x for x in F if x not in (p, r)]
        cands += [(j, tuple(sorted(rest + [(p[0], r[0]), ]))), (j, tuple(sorted(rest + [(r[1], p[1])])))]
        cands += [(j, tuple(sorted(rest + [(p[0], r[0]), (r[1], p[1])])))]
    for (jj, G) in cands:
        if jj in {x for g in G for x in g}:
            continue
        sub = has_cyclic_subconfig(st, jj, G)
        if sub is not None:
            return (jj, sub)
    return None


def scan(ct, word, test_shortening=True):
    st = Stack(ct, word)
    n_bad = Counter()
    ex = None
    short_fail = []
    for j, F in configurations(st):
        bt = bad_triples(st, j, F)
        for b in bt:
            n_bad[b[0]] += 1
            if ex is None:
                ex = (j, F, b)
            if test_shortening and shortening(st, j, F, b) is None:
                short_fail.append((j, F, b))
    return st, n_bad, ex, short_fail


def report(ct, words, label):
    stats = Counter()
    types = {"cyclic": Counter(), "cycle-free": Counter()}
    first = {}
    sfail = []
    cf_bad_words = []
    for word in words:
        st, n_bad, ex, short_fail = scan(ct, word)
        cyc = find_cycle(st, max_pairs=max(1, len(word) // 2))
        key = ("cyclic" if cyc else "cycle-free", "bad" if n_bad else "no bad")
        stats[key] += 1
        types[key[0]].update(n_bad)
        if n_bad and key not in first:
            first[key] = ("".join(map(str, word)), ex)
        if n_bad and not cyc:
            cf_bad_words.append("".join(map(str, word)))
        sfail += [("".join(map(str, word)),) + f for f in short_fail]
    print(label, ct, dict(stats), flush=True)
    print("    bad triples by type (counted per configuration):", {k: dict(v) for k, v in types.items()})
    if cf_bad_words:
        print("    cycle-free words with bad triples:", " ".join(cf_bad_words[:40]))
    for k, v in first.items():
        print("    first", k, v)
    print("    bad configurations for which the shortening construction found no cycle:", len(sfail))
    for f in sfail[:5]:
        print("      ", f)
    return stats, sfail


def crosscheck():
    """Interval form (B1)-(B4) agrees with the strand-level enumeration of 01_triples.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("t01", os.path.join(HERE, "01_triples.py"))
    t01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t01)
    n = 0
    for ct, L in ((["A", 2], 8), (["A", 3], 6), (["D", 4], 5)):
        for word in itertools.product(range(1, ct[1] + 1), repeat=L):
            st = Stack(ct, word)
            for j, F in configurations(st):
                used = {j} | {x for p in F for x in p}
                # put every unaffected slot on both sides in turn: badness must not depend on I
                for side in (0, 1):
                    I = {j} | {b for a, b in F} | {y for y in range(1, st.m + 1) if y not in used and (y + side) % 2}
                    strand_bad = sorted(t for t, names, bad in t01.classify(st, frozenset(I), j, F) if bad)
                    interval_bad = bad_triples(st, j, F)
                    # every interval-form bad triple appears exactly once as a strand triple
                    exp = sorted({"B1": {True: "T2", False: "T1"}, "B2": None, "B3": None, "B4": None}[b[0]][b[1] in I]
                                 if b[0] == "B1" else {"B2": "T3", "B3": None, "B4": "T7"}[b[0]] if b[0] != "B3"
                                 else ("T6" if b[3] in I else "T5") for b in interval_bad)
                    exp = sorted(exp + ["T4" for b in interval_bad if b[0] == "B2"] + ["T8" for b in interval_bad if b[0] == "B4"])
                    assert strand_bad == exp, (word, j, F, I, strand_bad, exp)
                    n += 1
    print("crosscheck interval form vs strand enumeration: %d (configuration, side) cases agree" % n)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "crosscheck":
        crosscheck()
        sys.exit()
    ct = [sys.argv[2], int(sys.argv[3])]
    if mode == "exhaustive":
        maxL = int(sys.argv[4])
        for L in range(3, maxL + 1):
            report(ct, itertools.product(range(1, ct[1] + 1), repeat=L), "length %d" % L)
    elif mode == "random":
        N, maxL = int(sys.argv[4]), int(sys.argv[5])
        seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
        rng = random.Random(seed)
        words = set()
        while len(words) < N:
            L = rng.randint(6, maxL)
            words.add(tuple(rng.randint(1, ct[1]) for _ in range(L)))
        report(ct, sorted(words), "random words (seed %d, length 6..%d)" % (seed, maxL))


# ---------------------------------------------------------------------------------------
# shields (Definition B.7): for a bad triple (X, Y, Z) a strand S of the outer colour that
# crosses X and Z but not Y.
# ---------------------------------------------------------------------------------------
def _t01():
    import importlib.util
    spec = importlib.util.spec_from_file_location("t01", os.path.join(HERE, "01_triples.py"))
    t01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t01)
    return t01


def shields(st, I, j, F, t01):
    """For every bad strand-triple of beta(I; j, F): (triple names, list of shield names)."""
    S = t01.strands_of(st, I, j, F)
    by_name = {s[0]: s for s in S}
    out = []
    for typ, names, bad in t01.classify(st, frozenset(I), j, F):
        if not bad:
            continue
        X, Y, Z = (by_name[n] for n in names)
        sh = [s[0] for s in S if s[1] == X[1] and s[0] not in names
              and t01.cross(s, X) and t01.cross(s, Z) and not t01.cross(s, Y)]
        out.append((typ, names, sh))
    return out


def shield_scan(ct, words):
    t01 = _t01()
    stats = Counter()
    ex = {}
    for word in words:
        st = Stack(ct, word)
        cyc = find_cycle(st, max_pairs=max(1, len(word) // 2))
        tag = "cyclic" if cyc else "cycle-free"
        for j, F in configurations(st):
            if not bad_triples(st, j, F):
                continue
            used = {j} | {x for p in F for x in p}
            free = [y for y in range(1, st.m + 1) if y not in used]
            base = {j} | {b for a, b in F}
            for k in range(len(free) + 1):
                for Y in itertools.combinations(free, k):
                    I = base | set(Y)
                    for typ, names, sh in shields(st, I, j, F, t01):
                        kinds = tuple(sorted(set(n if isinstance(n, str) else n[0] for n in sh)))
                        key = (tag, typ, "shielded" if sh else "UNSHIELDED")
                        stats[key] += 1
                        stats[(tag, typ, "shield kinds", kinds)] += 1
                        if key not in ex:
                            ex[key] = ("".join(map(str, word)), sorted(I), j, F, names, sh)
    return stats, ex
