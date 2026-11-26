"""
Pure Python check of the crossing bookkeeping of 06_main_intertwiner.md, section 6
(Lemmas 6.4 - 6.8): for random pair structures (every colour at most twice), random vertices I
and random configurations, the set of pairs of strands that cross in BOTH factors of each
product is the one stated in the text, and the two sides have the same bijection.

    python3 proofs/checks/06_check_bigons.py [N_SAMPLES] [SEED]

A diagram is a map  label -> (start position, end position);  W has position 0, slot s has
position +s on the right (s in the vertex) and -s on the left.
"""
import sys, random, itertools


def pos(V, s):
    return s if s in V else -s


def inv(start, end):
    """pairs of labels inverted by the diagram (start, end: dict label -> position)."""
    labs = list(start)
    return {frozenset((x, y)) for x, y in itertools.combinations(labs, 2)
            if (start[x] - start[y]) * (end[x] - end[y]) < 0}


def diagram(m, V, Vend, sigma):
    """strand in slot s of V goes to slot sigma[s] of Vend."""
    start = {s: pos(V, s) for s in range(1, m + 1)}
    end = {s: pos(Vend, sigma.get(s, s)) for s in range(1, m + 1)}
    start["W"] = end["W"] = 0
    return start, end


def compose(d1, d2_by_slot, sig1, m):
    """second diagram is given in slot labels of the middle vertex; relabel by the source slots."""
    start2, end2 = d2_by_slot
    s2 = {s: start2[sig1.get(s, s)] for s in range(1, m + 1)}
    e2 = {s: end2[sig1.get(s, s)] for s in range(1, m + 1)}
    s2["W"] = e2["W"] = 0
    assert all(d1[1][x] == s2[x] for x in s2)
    return inv(*d1) & inv(s2, e2), e2


def exch(F):
    sig = {}
    for a, b in F:
        sig[a], sig[b] = b, a
    return sig


def target(V, F, remove=()):
    return (set(V) - {b for a, b in F} - set(remove)) | {a for a, b in F}


def interlaced(p, q):
    (a, b), (a2, b2) = sorted((p, q))
    return a < a2 < b < b2


def inter_part(F1, F2):
    out = set()
    for p in F1:
        for q in F2:
            if interlaced(p, q):
                out.add(frozenset((p[0], q[0])) if p < q else frozenset((p[1], q[1])))   # L_p L_q  or  R_p R_q
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    count = {"T0": 0, "T1": 0, "T2": 0}
    for _ in range(n):
        m = rng.randint(3, 11)
        slots = list(range(1, m + 1))
        rng.shuffle(slots)
        npairs = rng.randint(1, m // 2)
        pairs = [tuple(sorted(slots[2 * i:2 * i + 2])) for i in range(npairs)]
        I = {s for s in range(1, m + 1) if rng.random() < 0.5}
        avail = [p for p in pairs if p[0] not in I and p[1] in I]
        kind = rng.choice(["T0", "T1", "T2"])
        if kind == "T0":
            free = [j for j in I if all(j not in p for p in avail)] + [j for j in I if all(j != p[1] for p in avail)]
            movers = [j for j in I]
            if not movers:
                continue
            j = rng.choice(movers)
            FF = [p for p in avail if j not in p and rng.random() < 0.7]
            F1 = [p for p in FF if rng.random() < 0.5]; F2 = [p for p in FF if p not in F1]
            # left side: Phi_j then exchange FF
            dA = diagram(m, I, I - {j}, {})
            dbl, endL = compose(dA, diagram(m, I - {j}, target(I - {j}, FF), exch(FF)), {}, m)
            assert dbl == {frozenset((j, a)) for a, b in FF if a < j < b}
            # right side: exchange F1, then mover j with F2
            I1 = target(I, F1)
            dB = diagram(m, I, I1, exch(F1))
            dbl, endR = compose(dB, diagram(m, I1, target(I1, F2, [j]), exch(F2)), exch(F1), m)
            pred = {frozenset((b, j)) for a, b in F1 if a < j < b} | inter_part(F1, F2)
            assert dbl == pred, (I, j, F1, F2, dbl, pred)
            assert endL == endR
        elif kind == "T1":
            cand = [p for p in pairs if p[0] in I and p[1] in I]
            if not cand:
                continue
            a, b = p = rng.choice(cand)
            FF = [q for q in avail if rng.random() < 0.7]
            F1 = [q for q in FF if rng.random() < 0.5]; F2 = [q for q in FF if q not in F1]
            # left: Phi_a then exchange FF + p at I - a
            dA = diagram(m, I, I - {a}, {})
            dbl, endL = compose(dA, diagram(m, I - {a}, target(I - {a}, FF + [p]), exch(FF + [p])), {}, m)
            assert dbl == {frozenset((a, q[0])) for q in FF if q[0] < a and b < q[1]}
            # right: exchange F1, then the resolved mover b with F2 :  b -> slot a (right), a -> slot b (left)
            I1 = target(I, F1)
            sig = exch(F2); sig[a], sig[b] = b, a
            K = target(I1, F2, [b])
            dbl, endR = compose(diagram(m, I, I1, exch(F1)), diagram(m, I1, K, sig), exch(F1), m)
            pred = inter_part(F1, F2) | {frozenset((q[1], b)) for q in F1 if a < q[0] < b < q[1]} | {frozenset((q[1], a)) for q in F1 if q[0] < a < q[1]}
            assert dbl == pred, (I, p, F1, F2, dbl, pred)
            assert endL == endR
        else:
            if not avail:
                continue
            a, b = p = rng.choice(avail)
            FF = [q for q in avail if q != p and rng.random() < 0.7]
            F1 = [q for q in FF if rng.random() < 0.5]; F2 = [q for q in FF if q not in F1]
            # (i): exchange F1 + p, then mover a with F2
            I1 = target(I, F1 + [p])
            K = target(I1, F2, [a])
            dbl, end_i = compose(diagram(m, I, I1, exch(F1 + [p])), diagram(m, I1, K, exch(F2)), exch(F1 + [p]), m)
            pred = inter_part(F1, F2) | {frozenset((a, q[0])) for q in F2 if a < q[0] < b < q[1]} | {frozenset((b, q[1])) for q in F1 if q[0] < a and b < q[1]}
            assert dbl == pred, ("i", I, p, F1, F2, dbl, pred)
            # (ii): exchange F1, then resolved mover b with F2 (a not in the vertex): b -> slot a (left), a -> slot b (left)
            F1b = [q for q in FF if rng.random() < 0.5]; F2b = [q for q in FF if q not in F1b]
            I1 = target(I, F1b)
            sig = exch(F2b); sig[a], sig[b] = b, a
            K2 = target(I1, F2b, [b])
            assert K2 == K
            dbl, end_ii = compose(diagram(m, I, I1, exch(F1b)), diagram(m, I1, K2, sig), exch(F1b), m)
            pred = inter_part(F1b, F2b) | {frozenset((q[0], a)) for q in F1b if q[0] < a < q[1] < b} | {frozenset((q[1], b)) for q in F1b if q[0] < b < q[1]}
            assert dbl == pred, ("ii", I, p, F1b, F2b, dbl, pred)
            assert end_i == end_ii
        count[kind] += 1
    print("crossing bookkeeping verified:", count)
