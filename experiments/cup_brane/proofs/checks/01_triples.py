"""
01_triples.py  (pure Python:  python3 proofs/checks/01_triples.py [MODE])

Combinatorics of the diagram beta(I; j, F) of a tree term (01_foundations.md, section B).

For every admissible configuration (formula.Stack.admissible) of the words searched we
  * list the strands (mover M, left strands L_p, right strands R_p, unaffected y, red W)
    with their signed source / target positions (pos_I(s) = +s / -s, red = 0),
  * check the crossing rules of Lemma B.2 (which pairs cross) against the definition
    "two strands cross iff their order is reversed",
  * enumerate all triples of pairwise crossing strands, check that every one is of one of
    the types T1..T8 of Lemma B.3 (with the predicted middle strand), and record which of
    them are BAD (outer strands black of one colour c, middle strand black of a colour
    adjacent to c; or outer strands of the red colour with the red strand in the middle).

MODE = all (default) | arbitrary | minuscule
"""
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from formula import Stack          # noqa: E402
from cyclefree import find_cycle   # noqa: E402


def strands_of(st, I, j, F):
    """List of (name, colour, src, tgt); colour 'W' for the red strand."""
    I = set(I)
    out = [("W", "W", 0, 0), ("M", st.col[j - 1], +j, -j)]
    in_pair = {j}
    for (a, b) in F:
        out.append((("L", (a, b)), st.col[a - 1], -a, -b))
        out.append((("R", (a, b)), st.col[b - 1], +b, +a))
        in_pair |= {a, b}
    for y in range(1, st.m + 1):
        if y not in in_pair:
            p = y if y in I else -y
            out.append((("y", y), st.col[y - 1], p, p))
    return out


def cross(s, t):
    return (s[2] - t[2]) * (s[3] - t[3]) < 0


def predicted_cross(st, I, j, s, t):
    """Lemma B.2: which pairs of strands cross."""
    def kind(x):
        return x[0] if isinstance(x[0], str) else x[0][0]
    ks, kt = kind(s), kind(t)
    order = "WMLRy"
    if order.index(ks) > order.index(kt):
        s, t, ks, kt = t, s, kt, ks
    if ks == "W":
        return kt == "M"
    if ks == "M":
        if kt == "y":
            return s[0] == "M" and t[0][1] < j
        a, b = t[0][1]
        return b < j
    if ks in "LR" and kt in "LR":
        if ks != kt:
            return False
        (a, b), (a2, b2) = s[0][1], t[0][1]
        return (a < a2 < b2 < b) or (a2 < a < b < b2)          # nested
    if ks in "LR" and kt == "y":
        a, b = s[0][1]
        y = t[0][1]
        return a < y < b and ((ks == "L") == (y not in I))
    if ks == "y" and kt == "y":
        return False
    raise AssertionError((s, t))


def scan_word(ct, word, red=None, examples=None, max_ex=3):
    st = Stack(ct, word)
    cnt = Counter()
    n_conf = 0
    m = len(word)
    for h in range(1, m + 1):
        for I in itertools.combinations(range(1, m + 1), h):
            I = frozenset(I)
            for j in sorted(I):
                for F, parent in st.admissible(I, j).items():
                    n_conf += 1
                    for typ, names, bad in classify(st, I, j, F, red):
                        cnt[typ, bad] += 1
                        if bad and examples is not None:
                            key = (ct[0], ct[1], tuple(word), typ)
                            if len(examples[typ]) < max_ex:
                                examples[typ].append(dict(ct=ct, word=list(word), I=sorted(I), j=j, F=list(F), triple=names))
    return n_conf, cnt


def classify(st, I, j, F, red_colour=None):
    S = strands_of(st, I, j, F)
    for s, t in itertools.combinations(S, 2):
        assert cross(s, t) == predicted_cross(st, I, j, s, t), (I, j, F, s, t)
    res = []
    for tr in itertools.combinations(S, 3):
        if not all(cross(a, b) for a, b in itertools.combinations(tr, 2)):
            continue
        a, mid, b = sorted(tr, key=lambda s: s[2])
        assert [x[3] for x in sorted(tr, key=lambda s: s[2])] == sorted([x[3] for x in tr], reverse=True)
        if "W" in (a[1], mid[1], b[1]):
            raise AssertionError("red strand in a triple: " + repr(tr))
        kinds = tuple(sorted((x[0] if isinstance(x[0], str) else x[0][0]) for x in tr))
        typ = {("L", "M", "y"): "T1", ("M", "R", "y"): "T2", ("L", "L", "M"): "T3",
               ("M", "R", "R"): "T4", ("L", "L", "y"): "T5", ("R", "R", "y"): "T6",
               ("L", "L", "L"): "T7", ("R", "R", "R"): "T8"}[kinds]
        mk = mid[0] if isinstance(mid[0], str) else mid[0][0]
        pred_mid = {"T1": "L", "T2": "R", "T3": "L", "T4": "R", "T5": "L", "T6": "R", "T7": "L", "T8": "R"}[typ]
        assert mk == pred_mid, (typ, tr)
        # which L / R is the middle one (Lemma B.3): L of the inner pair, R of the outer pair
        Ls = sorted([x for x in tr if not isinstance(x[0], str) and x[0][0] == "L"], key=lambda x: x[0][1][0])
        Rs = sorted([x for x in tr if not isinstance(x[0], str) and x[0][0] == "R"], key=lambda x: x[0][1][0])
        if typ == "T3":
            assert mid is Ls[0], (typ, tr)          # outer pair has the smaller a
        if typ == "T5":
            assert mid is Ls[1], (typ, tr)          # inner pair has the larger a
        if typ == "T7":
            assert mid is Ls[1], (typ, tr)
        if typ == "T4":
            assert mid is Rs[0], (typ, tr)          # outer pair has the smaller a
        if typ == "T6":
            assert mid is Rs[1], (typ, tr)          # inner pair
        if typ == "T8":
            assert mid is Rs[1], (typ, tr)
        bad = a[1] == b[1] and frozenset((a[1], mid[1])) in st.E
        res.append((typ, tuple(x[0] for x in (a, mid, b)), bad))
    return res


def all_words(rank, L):
    return itertools.product(range(1, rank + 1), repeat=L)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}
    if mode in ("all", "arbitrary"):
        print("== (i)/(ii) arbitrary words ==")
        for ct, maxL in [tuple(x) for x in json.loads(os.environ.get("TRIPLE_CASES", "[[[\"A\", 2], 8], [[\"A\", 3], 7], [[\"D\", 4], 6]]"))]:
            tot = Counter()
            tot_cf = Counter()
            n_words = n_cf = 0
            ex = defaultdict(list)
            ex_cf = defaultdict(list)
            for L in range(3, maxL + 1):
                for word in all_words(ct[1], L):
                    st = Stack(ct, word)
                    cyc = find_cycle(st, max_pairs=L // 2)
                    n_words += 1
                    e = ex if cyc is not None else ex_cf
                    n, cnt = scan_word(ct, word, examples=e)
                    tot.update(cnt)
                    if cyc is None:
                        n_cf += 1
                        tot_cf.update(cnt)
            print(ct, "words of length 3..%d: %d words, %d without cyclic configurations" % (maxL, n_words, n_cf))
            print("   all words        bad triples by type:", dict(sorted((k[0], v) for k, v in tot.items() if k[1])),
                  " good:", dict(sorted((k[0], v) for k, v in tot.items() if not k[1])))
            print("   cycle-free words bad triples by type:", dict(sorted((k[0], v) for k, v in tot_cf.items() if k[1])))
            for typ in sorted(ex_cf):
                for e in ex_cf[typ][:2]:
                    print("      cycle-free example", typ, e)
            for typ in sorted(ex):
                for e in ex[typ][:1]:
                    print("      cyclic-word example", typ, e)
            out["%s%d" % tuple(ct)] = dict(cf={typ: v for typ, v in ex_cf.items()}, cyc={typ: v for typ, v in ex.items()})
    if mode in ("all", "minuscule"):
        print("== (iii) minuscule words (words.json, and their prefixes) ==")
        words = json.load(open(os.path.join(HERE, "..", "..", "words.json")))
        for name, (ct, red, word) in sorted(words.items()):
            if len(word) > 12:
                continue
            n, cnt = scan_word(ct, word, red)
            bad = {k[0]: v for k, v in cnt.items() if k[1]}
            print("  %-8s %-26s %6d configurations, bad triples: %s" % (name, "".join(map(str, word)), n, bad or "none"))
    json.dump(out, open(os.path.join(HERE, "01_triples_examples.json"), "w"), indent=1, default=str)
