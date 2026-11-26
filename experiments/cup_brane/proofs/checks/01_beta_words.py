"""
01_beta_words.py   (sage -python proofs/checks/01_beta_words.py TYPE RANK RED WORD [N_SAMPLES] [SEED])

Does the element beta(I; j, F) depend on the reduced word?  For every admissible term of the
tree formula whose diagram contains a BAD triple (01_bad_search.bad_triples), sample reduced
words of its bijection uniformly-ish (random descents), record for every bad triple its
orientation in the word (the order in which the three pairs of strands cross), and compare
the elements.  Prints, per term: the bad triples, the orientations seen, and whether the
element is constant on reduced words.  With a second check: two reduced words that differ by
ONE braid move at a bad triple differ by exactly  +- r_c t_{cc'} * (smoothed word).
"""
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from cupbrane import *          # noqa
from fast import braid_for       # noqa
from formula import Stack        # noqa
import importlib.util
_spec = importlib.util.spec_from_file_location("bs01", os.path.join(HERE, "01_bad_search.py"))
bs01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs01)


def position_perm(cube, I, J, sigma):
    """perm[p] = target position of the strand at source position p (0-based), labels."""
    n = cube.n
    eps_I = tuple(1 if s + 1 in I else 0 for s in range(n))
    eps_J = tuple(1 if s + 1 in J else 0 for s in range(n))
    left = list(cube.labels(eps_I))
    inv = {sigma[s] - 1: s for s in range(n)}
    right = [x if isinstance(x, tuple) else inv[x] for x in cube.labels(eps_J)]
    perm = [right.index(x) for x in left]
    return eps_I, eps_J, left, perm


def random_reduced_word(perm, rng):
    cur = list(perm)
    word = []
    while True:
        desc = [p for p in range(len(cur) - 1) if cur[p] > cur[p + 1]]
        if not desc:
            return tuple(word)
        p = rng.choice(desc)
        cur[p], cur[p + 1] = cur[p + 1], cur[p]
        word.append(p + 1)


def crossing_order(left_labels, word):
    """For each unordered pair of strand labels that cross: index of the letter where they cross."""
    lab = list(left_labels)
    order = {}
    for t, i in enumerate(word):
        a, b = lab[i - 1], lab[i]
        order[frozenset((a, b))] = t
        lab[i - 1], lab[i] = b, a
    return order


def strand_label(cube, eps_I, name, j):
    """label (as in cube.labels) of the strand called name in 01_triples conventions."""
    if name == "M":
        return j - 1
    kind, obj = name
    if kind == "y":
        return obj - 1
    a, b = obj
    return (a - 1) if kind == "L" else (b - 1)


def run(ct, red, word, n_samples=200, seed=0, max_terms=None):
    P = [(red, tuple(word))]
    KLRW = make_algebra(ct, P)
    cube = Cube(KLRW, P, build=False)
    st = Stack(ct, word)
    rng = random.Random(seed)
    t01 = bs01.importlib if False else None
    spec = importlib.util.spec_from_file_location("t01", os.path.join(HERE, "01_triples.py"))
    t01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t01)
    n_bad_terms = n_dep = 0
    results = []
    for (I, J, sigma, mono), sign in sorted(st.all_terms().items(), key=lambda kv: (sorted(kv[0][0]), sorted(kv[0][1]))):
        I = frozenset(I)
        # recover (j, F) from (I, J, sigma)
        F = tuple(sorted((s, sigma[s - 1]) for s in range(1, st.m + 1) if sigma[s - 1] > s))
        (j,) = [s for s in I if s not in J and sigma[s - 1] == s]
        triples = [t for t in t01.classify(st, I, j, F) if t[2]]
        if not triples:
            continue
        sh = bs01.shields(st, I, j, F, t01)
        all_shielded = all(len(s[2]) > 0 for s in sh)
        n_bad_terms += 1
        eps_I, eps_J, left, perm = position_perm(cube, I, J, sigma)
        _, _, braid = braid_for(cube, I, J, sigma)
        base = KLRW.monomial(braid)
        orients = defaultdict(set)
        values = {}
        for k in range(n_samples):
            w = random_reduced_word(perm, rng)
            assert len(w) == len(braid.word())
            order = crossing_order(left, w)
            key = []
            for typ, names, bad in triples:
                A, B, C = [strand_label(cube, eps_I, nm, j) for nm in names]
                o = (order[frozenset((A, B))], order[frozenset((A, C))], order[frozenset((B, C))])
                key.append("lex" if o[0] < o[1] < o[2] else ("antilex" if o[2] < o[1] < o[0] else "?"))
            key = tuple(key)
            assert "?" not in key
            if key not in values:
                values[key] = (w, KLRW.element_from_braid(w, left_state=cube.state(eps_I)))
            else:
                el = KLRW.element_from_braid(w, left_state=cube.state(eps_I))
                assert el == values[key][1], "same triple orientations, different element?!"
        bw = crossing_order(left, braid.word())
        bkey = []
        for typ, names, bad in triples:
            A, B, C = [strand_label(cube, eps_I, nm, j) for nm in names]
            o = (bw[frozenset((A, B))], bw[frozenset((A, C))], bw[frozenset((B, C))])
            bkey.append("lex" if o[0] < o[1] < o[2] else "antilex")
        dep = len(set(str(v[1]) for v in values.values())) > 1
        n_dep += dep
        results.append((sorted(I), j, F, mono, triples, tuple(bkey), {k: str(v[1] - base)[:160] for k, v in values.items()}, all_shielded, dep))
        if max_terms and n_bad_terms >= max_terms:
            break
    return n_bad_terms, n_dep, results


def tally(cases, n_samples=40):
    """cases: list of (ct, red, word).  Count terms by (all bad triples shielded?, depends on word?)."""
    from collections import Counter
    c = Counter()
    for ct, red, word in cases:
        n_bad, n_dep, results = run(ct, red, word, n_samples)
        for r in results:
            c[("shielded" if r[7] else "UNSHIELDED", "depends" if r[8] else "independent")] += 1
        print("  ", "".join(map(str, ct)), red, "".join(map(str, word)), "terms with bad triples:", n_bad,
              "dependent:", n_dep, flush=True)
    return c


if __name__ == "__main__" and sys.argv[1] == "tally":
    # sage -python 01_beta_words.py tally TYPE RANK WORD1,WORD2,... [N_SAMPLES]
    ct = [sys.argv[2], int(sys.argv[3])]
    words = [tuple(int(ch) for ch in w) for w in sys.argv[4].split(",")]
    ns = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    cases = [(ct, red, w) for w in words for red in sorted(set(w))[:1]]
    print("TALLY", dict(tally(cases, ns)))
    sys.exit()

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3])
    word = tuple(int(c) for c in sys.argv[4])
    n_samples = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    n_bad, n_dep, results = run(ct, red, word, n_samples, seed)
    print("%s red %d word %s: %d terms with a bad triple, %d of them depend on the reduced word"
          % ("".join(map(str, ct)), red, "".join(map(str, word)), n_bad, n_dep))
    for r in results[:6]:
        I, j, F, mono, triples, bkey, vals, shielded, dep = r
        print("  I=%s j=%d F=%s mono=%s" % (I, j, F, mono))
        print("     bad triples:", [(t[0], t[1]) for t in triples], " basis braid orientation:", bkey, " all shielded:", shielded, " depends:", dep)
        for k, v in vals.items():
            print("     orientation", k, ": element - basis braid =", v)
