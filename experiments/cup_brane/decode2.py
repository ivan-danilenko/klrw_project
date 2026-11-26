"""
Decode the cyclic (dotted) part of the true differential in "chamber" form.

For every source/target pair of a local data file with terms beyond the tree formula:
write   truth - tree  =  sum c * M * [reduced exchange diagram with one dot on strand s,
placed after the first k same-colour crossings of s]   and print the solution.

    sage -python decode2.py pieces/FILE.json
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cupbrane import *
from fast import exp_for
from formula import Stack
from events import simulate, precedences, linear_extension
from decode import term_keys
from cycle import link_list, configurations


def trace(cube, V, word):
    """For each letter: (strand ids crossing), and the state after each prefix (as source-slot labels)."""
    n = cube.n
    eps = tuple(1 if s + 1 in V else 0 for s in range(n))
    cur = [x if isinstance(x, tuple) else x + 1 for x in cube.labels(eps)]
    states = [list(cur)]
    crossings = []
    for letter in word:
        a, b = cur[letter - 1], cur[letter]
        crossings.append((a, b))
        cur[letter - 1], cur[letter] = b, a
        states.append(list(cur))
    return crossings, states


def dotted_diagram(cube, V, word, strand, k):
    """Element of the diagram ``word`` (from vertex V) with a dot on ``strand`` (source slot)
    inserted right after its k-th crossing with a strand of its own colour (k = 0: at the source)."""
    KLRW = cube.KLRW
    n = cube.n
    col = cube.strand_colour
    crossings, states = trace(cube, V, word)
    idx = 0
    count = 0
    if k > 0:
        for i, (a, b) in enumerate(crossings):
            if strand in (a, b):
                other = b if a == strand else a
                if not isinstance(other, tuple) and col[other - 1] == col[strand - 1]:
                    count += 1
                    if count == k:
                        idx = i + 1
                        break
        else:
            return None
    eps = tuple(1 if s + 1 in V else 0 for s in range(n))
    left_state = cube.state(eps)
    prefix = KLRW.element_from_braid(word[:idx], left_state=left_state) if idx else KLRW.idempotent(left_state)
    mid_state = left_state
    for letter in word[:idx]:
        mid_state = mid_state.act_by_s(letter)
    suffix = KLRW.element_from_braid(word[idx:], left_state=mid_state) if idx < len(word) else KLRW.idempotent(mid_state)
    # dot variable of the strand at the intermediate state
    pos = states[idx].index(strand)
    rank = sum(1 for x in states[idx][: pos + 1] if not isinstance(x, tuple) and col[x - 1] == col[strand - 1])
    x = KLRW.base()("x_%d_%d" % (col[strand - 1], rank))
    return prefix * (x * suffix)


def n_same_colour_crossings(cube, V, word, strand):
    col = cube.strand_colour
    crossings, _ = trace(cube, V, word)
    return sum(1 for a, b in crossings if strand in (a, b) and not isinstance(a, tuple) and not isinstance(b, tuple)
               and col[a - 1] == col[b - 1])


def decode_file(path, verbose=True):
    data = json.load(open(path))
    ct, red, word = data["cartan_type"], data["red"], tuple(data["word"])
    P = [(red, word)]
    KLRW = make_algebra(ct, P); cube = Cube(KLRW, P, build=False); stack = Stack(ct, word)
    D = KLRW.base()
    truth_all = {}
    for src, tgt, sigma, mono, dots, c in data["terms"]:
        truth_all.setdefault((frozenset(src), frozenset(tgt)), {})[
            (tuple(sigma), tuple(tuple(x) for x in mono), tuple(tuple(x) for x in dots))] = c
    results = []
    for (V, U), truth in sorted(truth_all.items(), key=lambda kv: (sorted(kv[0][0]), sorted(kv[0][1]))):
        truth = dict(truth)
        for J, sigma, mono, sign in stack.terms_from(V):
            if J == U:
                truth[(sigma, mono, ())] = truth.get((sigma, mono, ()), 0) - sign
        truth = {k: v for k, v in truth.items() if v}
        if not truth:
            continue
        # find the cyclic configuration(s) from V to U
        confs = []
        for j in sorted(V - U):
            for F, links in configurations(stack, V, j, max_b1=3).items():
                if len(links) > len(F) and (V - {j} - {b for a, b in F}) | {a for a, b in F} == U:
                    confs.append((j, F, links))
        if len(confs) != 1:
            print("V=%s U=%s: %d cyclic configurations, skipping" % (sorted(V), sorted(U), len(confs)))
            continue
        j, F, links = confs[0]
        rel = [(x, y) for x, y, _, _ in links]
        ro = linear_extension(("m",) + F, rel)
        lo = linear_extension(("m",) + F, [(y, x) for x, y in rel])
        w = simulate(cube, V, j, F, ro, lo)
        monos = sorted(set(k[1] for k in truth if k[2]))
        cands = {}
        involved = [j] + [x for p in F for x in p]
        for (sg, mono, dots) in truth:
            if not dots:
                continue
            (dname, power), = dots
            colour = int(dname.split("_")[1])
            M = D.monomial(*exp_for(KLRW, mono))
            for s in involved:
                if cube.strand_colour[s - 1] != colour:
                    continue
                for k in range(n_same_colour_crossings(cube, V, w, s) + 1):
                    if (mono, s, k) in cands:
                        continue
                    el = dotted_diagram(cube, V, w, s, k)
                    cands[(mono, s, k)] = term_keys(cube, V, U, M * el)
        names = sorted(cands)
        rows = sorted(set(k for c in cands.values() for k in c) | set(truth), key=str)
        ridx = {k: i for i, k in enumerate(rows)}
        A = sp.lil_matrix((len(rows), len(names)), dtype=np.int64)
        for jx, nm in enumerate(names):
            for k, v in cands[nm].items():
                A[ridx[k], jx] = v
        b = sp.lil_matrix((len(rows), 1), dtype=np.int64)
        for k, v in truth.items():
            b[ridx[k], 0] = v
        x = solve_integer(A.tocsr(), b.tocsc())
        print("V=%s  mover %d (c%d)  pairs %s  links %s" % (
            sorted(V), j, cube.strand_colour[j - 1], ["(%d,%d)c%d" % (a, b_, cube.strand_colour[a - 1]) for a, b_ in F],
            [("m" if x_ == "m" else "(%d,%d)" % x_) + "-" + "(%d,%d)" % y_ for x_, y_, _, _ in links]))
        if x is None:
            print("    NO decomposition into single-dot diagrams; extra terms:", len(truth))
            results.append((V, U, j, F, None))
            continue
        res = {nm: int(c) for nm, c in zip(names, x) if c}
        for (mono, s, k), c in sorted(res.items(), key=lambda kv: (kv[0][1], kv[0][2])):
            role = "mover" if s == j else [("L" if s == a else "R") + "(%d,%d)" % (a, b_) for a, b_ in F if s in (a, b_)][0]
            tot = n_same_colour_crossings(cube, V, w, s)
            print("    %+d  dot on %-10s after %d of its %d same-colour crossings   %s" % (
                c, role, k, tot, "*".join(n_ if p == 1 else "%s^%d" % (n_, p) for n_, p in mono)))
        results.append((V, U, j, F, res))
    return results


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("=====", os.path.basename(p))
        decode_file(p)
