"""
Theorem 9.1(d) of proofs/04_first_order.md: the completion of the tree formula KEEPING ITS
r^2-PART (not "the true differential": a gauge-equivalent dot-free completion also exists,
Theorem 9.1(e),(f)) for the smallest cyclic word found so far:  A_2, red strand of colour 2,
word 1221122 (7 strands, 128 vertices).

    sage -python proofs/checks/04_cyclic7_solve.py [WORD] [RED] [start]      start = tree | delta0

Starting from the tree formula (start=tree, default) or from delta_0, the order-by-order
solver of cupbrane.correct_differential is run with the COMPLETE ansatz (all basis elements
of the right multidegree, dots allowed; ``interval.interval_corrections`` on the full cube).
Printed: per order the number of equations, unknowns, rank and kernel; the terms of the
result that are not tree-formula terms.  All terms are written to
proofs/checks/04_data/cyclic7_<word>_W<red>_<start>.json in the format read by
../../cycle3.py (``sage -python cycle3.py FILE -v`` compares with the single-cycle rule).
"""
import sys, os, json, time
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from cupbrane import make_algebra, Cube, correct_differential, differential_terms, term_record  # noqa: E402
from interval import interval_corrections, show  # noqa: E402
from verify import formula_differential  # noqa: E402
from formula import Stack  # noqa: E402
from degree_model import Model  # noqa: E402


def run(word, red, start="tree", verbose=True):
    ct = ["A", 2]
    P = [(red, tuple(word))]
    KLRW = make_algebra(ct, P)
    cube = Cube(KLRW, P)
    stack = Stack(ct, word)
    model = Model(ct, P)
    by_h = {h: [frozenset(s + 1 for s in range(cube.n) if eps[s]) for eps in lst] for h, lst in cube.vertices.items()}
    interval_corrections.degrees = {V: cube.projectives[h][k].equivariant_degree for h, lst in by_h.items() for k, V in enumerate(lst)}
    # start = "tree": start from delta_0 and weight the l1 objective so that tree-formula terms
    # cost 1 and all other unknowns cost 1000; at orders where the system has a kernel the
    # solver then returns the tree-formula solution whenever it is a solution.  (Starting the
    # library solver directly from the tree-formula differential hits a state mismatch inside
    # klrw.perfect_complex_corrections; the weighted run is equivalent and is checked below:
    # every tree term must come out with its coefficient.)
    from cupbrane import split_exponent
    tree = set(stack.all_terms())
    def weight(h, row, col, braid, exp):
        if start != "tree":
            return 1.0
        src, tgt = by_h[h][col], by_h[h - 1][row]
        eps = tuple(1 if s + 1 in src else 0 for s in range(cube.n))
        arrive = cube.strand_map(eps, braid)
        labels = cube.labels(tuple(1 if s + 1 in tgt else 0 for s in range(cube.n)))
        sig = {a + 1: t + 1 for a, t in zip(arrive, labels) if not isinstance(a, tuple)}
        params, dots = split_exponent(KLRW, exp)
        key = (src, tgt, tuple(sig[s] for s in range(1, cube.n + 1)), tuple(sorted(params)))
        return 1.0 if (not dots and key in tree) else 1000.0
    d, log = correct_differential(KLRW, cube.projectives, cube.d0, verbose=verbose, record_rank=True,
                                  weight_function=weight,
                                  corrections_function=lambda exp: interval_corrections(cube, model, by_h, exp))
    if d is None:
        return cube, stack, None, log
    actual = {}
    for t in differential_terms(cube, d):
        rec = term_record(cube, t)
        key = (frozenset(rec["src"]), frozenset(rec["tgt"]), tuple(rec["sigma"]),
               tuple(sorted(tuple(x) for x in rec["params"])), tuple(sorted(tuple(x) for x in rec["dots"])))
        actual[key] = actual.get(key, 0) + rec["coeff"]
    actual = {k: c for k, c in actual.items() if c}
    return cube, stack, actual, log


if __name__ == "__main__":
    word = tuple(int(c) for c in (sys.argv[1] if len(sys.argv) > 1 else "1221122"))
    red = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    start = sys.argv[3] if len(sys.argv) > 3 else "tree"
    t0 = time.time()
    cube, stack, actual, log = run(word, red, start)
    if actual is None:
        print("NO SOLUTION")
        sys.exit(1)
    predicted = {}
    for (I, J, sigma, mono), sign in stack.all_terms().items():
        predicted[(I, J, sigma, mono, ())] = sign
    A, P = set(actual), set(predicted)
    wrong = [k for k in A & P if actual[k] != predicted[k]]
    print("terms: actual {}, tree formula {}, actual-not-tree {}, tree-not-actual {}, different coefficient {}  [{:.0f}s]".format(
        len(A), len(P), len(A - P), len(P - A), len(wrong), time.time() - t0))
    for title, keys in (("ACTUAL but not in tree formula", A - P), ("TREE but not actual", P - A), ("DIFFERENT coefficient", wrong)):
        if keys:
            print(title)
            for k in sorted(keys, key=lambda k: (sorted(k[0]), sorted(k[1]), k[2], k[3], k[4])):
                print("   ", show(k, actual.get(k, predicted.get(k))))
    os.makedirs(os.path.join(HERE, "04_data"), exist_ok=True)
    path = os.path.join(HERE, "04_data", "cyclic7_%s_W%d_%s.json" % ("".join(map(str, word)), red, start))
    with open(path, "w") as f:
        json.dump(dict(cartan_type=["A", 2], red=red, word=list(word), I=list(range(1, len(word) + 1)), K=[],
                       vertices=[sorted(s + 1 for s in range(cube.n) if eps[s]) for lst in cube.vertices.values() for eps in lst],
                       orders=[dict(monomial=r["monomial"], shape=list(map(int, r["shape"])), rank=int(r.get("rank", -1)))
                               for r in log],
                       terms=[[sorted(k[0]), sorted(k[1]), list(k[2]), [list(x) for x in k[3]], [list(x) for x in k[4]], c]
                              for k, c in actual.items()]), f)
    print("saved", path)
