"""
Solve for the true differential on a small sub-quotient of a (possibly huge) stack.

The cube is filtered by the colourwise dominance order: J <= I if for every colour the
slots of J can be matched injectively to larger-or-equal slots of I.  All known terms of
the differential are decreasing for this order, so the vertices of an interval [K, I]
span a sub-quotient complex.  We run the complete ansatz (dots allowed) on that interval.

    sage -python interval.py TYPE RANK RED [min|max|WORD] LENGTH "I" "K"
"""
import sys, os, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cupbrane import *
from formula import Stack


def dominated_sets(slots, upper, lower):
    """Subsets S of ``slots`` (sorted) with lower <= S <= upper in dominance order."""
    upper, lower = sorted(upper), sorted(lower)
    out = []
    for size in range(len(lower), len(upper) + 1):
        for S in itertools.combinations(slots, size):
            # S <= upper: i-th largest of S <= i-th largest of upper
            if any(s > u for s, u in zip(reversed(S), reversed(upper))):
                continue
            if any(l > s for l, s in zip(reversed(lower), reversed(S))):
                continue
            out.append(S)
    return out


def interval_vertices(stack, I, K):
    per_colour = []
    for (p, c), slots in stack.same.items():
        per_colour.append(dominated_sets(slots, [s for s in slots if s in I], [s for s in slots if s in K]))
    return [frozenset(x for part in combo for x in part) for combo in itertools.product(*per_colour)]


def build_interval(cube, vertices):
    """Projectives and delta_0 restricted to the given vertices."""
    KLRW = cube.KLRW
    Braid = KLRW.KLRWBraid
    n = cube.n
    eps_of = lambda S: tuple(1 if s + 1 in S else 0 for s in range(n))
    by_h = defaultdict(list)
    for V in sorted(vertices, key=lambda S: (len(S), sorted(S))):
        by_h[len(V)].append(V)
    index = {V: (h, k) for h, lst in by_h.items() for k, V in enumerate(lst)}

    deg_cache = {}

    def move_entry(V, s):
        """delta_0 component V -> V - {s} (s 1-based) with sign."""
        eps = eps_of(V)
        new = eps[: s - 1] + (0,) + eps[s:]
        sign = -1 if sum(eps[s:]) % 2 else 1
        word = braid_from_marked_state(cube.marked_state(eps), cube.marked_state(new))
        braid = Braid._element_constructor_(cube.state(new), word)
        return KLRW.base()(sign) * KLRW.monomial(braid)

    def eq_degree(V):
        if V not in deg_cache:
            if not V:
                deg_cache[V] = KLRW.grading_group.zero()
            else:
                s = max(V)
                deg_cache[V] = eq_degree(V - {s}) - move_entry(V, s).degree(check_if_homogeneous=True)
        return deg_cache[V]

    projectives = {h: [KLRWIrreducibleProjectiveModule(cube.state(eps_of(V)), eq_degree(V)) for V in lst]
                   for h, lst in by_h.items()}
    d0 = {h: {} for h in by_h if h - 1 in by_h}
    for V in vertices:
        h, col = index[V]
        for s in V:
            W = V - {s}
            if W in index:
                d0[h][index[W][1], col] = move_entry(V, s)
    return by_h, index, projectives, d0


def matching_bijections(model, posV, posU, target):
    """
    Colour preserving bijections sigma of the slots whose reduced diagram from V to U has
    crossing content exactly ``target`` (a Counter of 's', 'e', 'w' types).  Colour by
    colour, so that the same-colour and red-strand counts prune early.
    """
    import itertools
    classes = {}
    for s in range(1, model.m + 1):
        classes.setdefault(model.col[s - 1], []).append(s)
    BIG = model.BIG
    per_colour = []
    for c, sl in classes.items():
        want_s = target.get(("s", c), 0)
        want_w = target.get(("w", c), 0)
        reds = [p for p, r in enumerate(model.reds) if r == c]
        good = []
        for perm in itertools.permutations(sl):
            cnt = 0
            ok = True
            for a in range(len(sl)):
                for b in range(a + 1, len(sl)):
                    if (posV[sl[a]] - posV[sl[b]]) * (posU[perm[a]] - posU[perm[b]]) < 0:
                        cnt += 1
                        if cnt > want_s:
                            ok = False
                            break
                if not ok:
                    break
            if not ok or cnt != want_s:
                continue
            w = 0
            for a in range(len(sl)):
                for p in reds:
                    if (posV[sl[a]] - p * BIG) * (posU[perm[a]] - p * BIG) < 0:
                        w += 1
            if w != want_w:
                continue
            good.append(dict(zip(sl, perm)))
        if not good:
            return
        per_colour.append((c, sl, good))
    # adjacent-colour pairs
    colours = [c for c, _, _ in per_colour]
    adj = {}
    for i, (c, sl, _) in enumerate(per_colour):
        for k, (c2, sl2, _) in enumerate(per_colour[:i]):
            if frozenset((c, c2)) in model.E:
                adj[i, k] = ("e", min(c, c2), max(c, c2))
    for ty, v in target.items():
        if ty[0] == "e" and v and not any(t == ty for t in adj.values()):
            return

    def rec(i, sigma, remaining):
        if i == len(per_colour):
            if not any(remaining.values()):
                yield dict(sigma)
            return
        c, sl, good = per_colour[i]
        for g in good:
            rem = dict(remaining)
            ok = True
            for k in range(i):
                ty = adj.get((i, k))
                if ty is None:
                    continue
                sl2 = per_colour[k][1]
                cnt = 0
                for a in sl:
                    for b in sl2:
                        if (posV[a] - posV[b]) * (posU[g[a]] - posU[sigma[b]]) < 0:
                            cnt += 1
                rem[ty] = rem.get(ty, 0) - cnt
                if rem[ty] < 0:
                    ok = False
                    break
            if not ok:
                continue
            sigma.update(g)
            yield from rec(i + 1, sigma, rem)

    remaining = {ty: v for ty, v in target.items() if ty[0] == "e"}
    yield from rec(0, {}, remaining)


def interval_corrections(cube, model, by_h, exp):
    """
    Complete ansatz at the parameter order ``exp`` on the interval: for every pair of
    vertices in adjacent degrees, every braid (colour preserving bijection of slots) and
    every placement of dots of the correct multidegree.  Replaces the library's
    ``possible_corrections``, which enumerates all braids out of a state.
    """
    from collections import Counter
    from fast import corrections_matrix, braid_for
    KLRW = cube.KLRW
    D = KLRW.base()
    names = D.variable_names()
    n = cube.n
    # crossing content of the parameter monomial
    pcont = Counter()
    for pos, power in exp.sparse_iter():
        nm = names[pos].split("_")
        if nm[0] == "r":
            pcont[("s", int(nm[1]))] += power
        elif nm[0] == "t":
            a, b = int(nm[1]), int(nm[2])
            pcont[("e", min(a, b), max(a, b))] += 2 * power
        elif nm[0] == "u":
            pcont[("w", int(nm[1]))] += 2 * power
    if not hasattr(model, "_bij"):
        model._bij = list(model.bijections())
    slots = range(1, n + 1)
    Dc = {}
    out = {}
    for h in by_h:
        if h - 1 not in by_h:
            continue
        entries = defaultdict(list)
        for col, V in enumerate(by_h[h]):
            if V not in Dc:
                Dc[V] = model.D(V)
            posV = {s: model.pos(s, V) for s in slots}
            for row, U in enumerate(by_h[h - 1]):
                if U not in Dc:
                    Dc[U] = model.D(U)
                target = Counter(Dc[V])
                target.subtract(Dc[U])
                target.subtract(pcont)
                if any(v < 0 for v in target.values()):
                    continue
                posU = {s: model.pos(s, U) for s in slots}
                src_deg = interval_corrections.degrees[V]
                tgt_deg = interval_corrections.degrees[U]
                for sigma in matching_bijections(model, posV, posU, target):
                    _, _, braid = braid_for(cube, V, U, tuple(sigma[s] for s in slots))
                    for e in D.exps_by_degree_and_parameter_part(tgt_deg - src_deg - KLRW.braid_degree(braid), exp):
                        entries[row, col].append((braid, e))
        out[h] = corrections_matrix(len(by_h[h - 1]), len(by_h[h]), entries)
    return out


def interval_terms(cube, by_h, d_csc):
    """Terms of a differential on the interval, in the (I, J, sigma, mono, dots) -> coeff format."""
    KLRW = cube.KLRW
    n = cube.n
    eps_of = lambda S: tuple(1 if s + 1 in S else 0 for s in range(n))
    out = {}
    for h in sorted(d_csc.keys()):
        for (row, col), entry in d_csc[h].dict().items():
            src, tgt = by_h[h][col], by_h[h - 1][row]
            for braid, params, dots, c in entry_terms(KLRW, entry):
                arrive = cube.strand_map(eps_of(src), braid)
                labels = cube.labels(eps_of(tgt))
                sigma = {a + 1: t + 1 for a, t in zip(arrive, labels) if not isinstance(a, tuple)}
                key = (src, tgt, tuple(sigma[s] for s in range(1, n + 1)), tuple(sorted(params)), tuple(sorted(dots)))
                out[key] = out.get(key, 0) + c
    return out


def solve_interval(ct, red, word, I, K, verbose=True):
    P = [(red, tuple(word))]
    KLRW = make_algebra(ct, P)
    cube = Cube(KLRW, P, build=False)
    stack = Stack(ct, word)
    vertices = interval_vertices(stack, frozenset(I), frozenset(K))
    by_h, index, projectives, d0 = build_interval(cube, vertices)
    if verbose:
        print("interval: {} vertices, by degree {}".format(len(vertices), {h: len(v) for h, v in sorted(by_h.items())}), flush=True)
    from degree_model import Model
    model = Model(ct, P)
    interval_corrections.degrees = {V: projectives[h][k].equivariant_degree for V, (h, k) in index.items()}
    d, log = correct_differential(
        KLRW, projectives, d0, verbose=verbose, record_rank=True,
        corrections_function=lambda exp: interval_corrections(cube, model, by_h, exp),
    )
    if d is None:
        return cube, stack, vertices, None, None
    actual = interval_terms(cube, by_h, d)
    predicted = {}
    for V in vertices:
        for J, sigma, mono, sign in stack.terms_from(V):
            if J in index:
                predicted[(V, J, sigma, mono, ())] = sign
    return cube, stack, vertices, actual, predicted


def show(key, coeff):
    I, J, sigma, mono, dots = key
    mv = {s + 1: t for s, t in enumerate(sigma) if s + 1 != t}
    return "%+d * %-34s dots %-18s out %-18s in %-12s sigma %s" % (
        coeff, "*".join(n if p == 1 else "%s^%d" % (n, p) for n, p in mono) or "1",
        "*".join(n if p == 1 else "%s^%d" % (n, p) for n, p in dots) or "-",
        sorted(I - J), sorted(J - I), mv)


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]; red = int(sys.argv[3]); spec = sys.argv[4]
    length = int(sys.argv[5]) if sys.argv[5] != "full" else None
    word = minuscule_word(ct, red, {"min": min, "max": max}[spec], length=length) if spec in ("min", "max") else tuple(int(c) for c in spec)
    I = eval(sys.argv[6]); K = eval(sys.argv[7])
    print("word", "".join(map(str, word)), " slots:", " ".join("%d:c%d" % (s + 1, c) for s, c in enumerate(word)))
    cube, stack, vertices, actual, predicted = solve_interval(ct, red, word, I, K)
    if actual is None:
        print("NO SOLUTION on the interval")
        sys.exit()
    import json
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pieces"), exist_ok=True)
    name = "{}{}_W{}_{}__I_{}__K_{}.json".format(ct[0], ct[1], red, "".join(map(str, word)),
                                                "-".join(map(str, sorted(I))), "-".join(map(str, sorted(K))))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pieces", name), "w") as f:
        json.dump(dict(cartan_type=ct, red=red, word=list(word), I=sorted(I), K=sorted(K),
                       vertices=[sorted(V) for V in vertices],
                       terms=[[sorted(k[0]), sorted(k[1]), list(k[2]), [list(x) for x in k[3]], [list(x) for x in k[4]], c]
                              for k, c in actual.items()]), f)
    print("saved pieces/" + name)
    A, Pd = set(actual), set(predicted)
    wrong = [k for k in A & Pd if actual[k] != predicted[k]]
    print("terms: actual {}, formula {}, actual-not-formula {}, formula-not-actual {}, different coefficient {}".format(
        len(A), len(Pd), len(A - Pd), len(Pd - A), len(wrong)))
    for title, keys in (("ACTUAL but not in formula", A - Pd), ("FORMULA but not actual", Pd - A), ("DIFFERENT coefficient", wrong)):
        if keys:
            print(title)
            for k in sorted(keys, key=lambda k: (len(k[3]), sorted(k[0]), sorted(k[1])))[:60]:
                print("   ", show(k, actual.get(k, predicted.get(k))), "" if k not in wrong else " (formula %+d)" % predicted[k])
