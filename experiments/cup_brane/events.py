"""
Honest ("time-ordered") diagrams for a configuration (V; mover j, exchange pairs F).

Events on the right of the red strand:  R_p (the right strand of pair p moves inwards
from slot b_p to slot a_p) and the mover leaving.  Events on the left: L_p (the left strand
of p moves outwards from a_p to b_p) and the mover arriving.  Two linked events drawn in
the "wrong" order produce a bigon; drawn in the right order they do not cross.

``diagram(cube, V, j, F, right_order, left_order)`` returns the KLRW element of the diagram
in which the right-hand events happen in ``right_order`` and the left-hand events in
``left_order`` (lists containing the pairs and the string "m").
"""
import itertools


def simulate(cube, V, j, F, right_order, left_order, tags=None, spans=None):
    """Word (1-based letters, read from the source) of the time-ordered diagram.
    If ``tags`` is a list, it receives for every letter the pair (side, event)."""
    n = cube.n
    eps = tuple(1 if s + 1 in V else 0 for s in range(n))
    labels = list(cube.labels(eps))             # position -> slot-1 or ('W', 0)
    wpos = labels.index(("W", 0))
    left = [x + 1 for x in labels[:wpos]]       # decreasing slot labels
    right = [x + 1 for x in labels[wpos + 1:]]  # increasing slot labels
    # current "label" of each strand (strand identified by its source slot)
    word = []
    Fd = {p: p for p in F}

    current = [None]

    def swap(pos):                               # swap positions pos, pos+1 (0-based) of the full state
        word.append(pos + 1)
        if tags is not None:
            tags.append(current[0])

    state = left + ["W"] + right                 # entries: current labels (unique)

    def move_right_side(strand_label, new_label):
        # strand on the right moves inwards past right strands with labels in (new_label, strand_label)
        pos = state.index(strand_label)
        w = state.index("W")
        while pos - 1 > w and state[pos - 1] > new_label:
            state[pos - 1], state[pos] = state[pos], state[pos - 1]
            swap(pos - 1)
            pos -= 1
        state[pos] = new_label

    def move_left_side(strand_label, new_label):
        # strand on the left moves outwards past left strands with labels in (strand_label, new_label)
        pos = state.index(strand_label)
        while pos - 1 >= 0 and state[pos - 1] < new_label:
            state[pos - 1], state[pos] = state[pos], state[pos - 1]
            swap(pos - 1)
            pos -= 1
        state[pos] = new_label

    # labels must stay unique: temporarily mark moving strands with fractional labels
    def do_right(ev):
        current[0] = ("R", ev)
        if ev == "m":
            move_right_side(j, 0.5)             # mover next to W on the right
        else:
            a, b = ev
            move_right_side(b, a + 0.25)        # right strand of the pair: slot b -> slot a
    def do_left(ev):
        current[0] = ("L", ev)
        if ev == "m":
            pos = state.index(0.5)
            # cross the red strand
            w = state.index("W")
            assert pos == w + 1
            state[w], state[pos] = state[pos], state[w]
            swap(w)
            state[w] = 0.6
            move_left_side(0.6, j)
        else:
            a, b = ev
            move_left_side(a, b - 0.25)         # left strand of the pair: slot a -> slot b

    def run(side, ev):
        start = len(word)
        (do_right if side == "R" else do_left)(ev)
        if spans is not None:
            spans[side, ev] = (start, len(word))

    mr = right_order.index("m")
    ml = left_order.index("m")
    for ev in right_order[: mr + 1]:
        run("R", ev)
    for ev in left_order[:ml]:
        run("L", ev)
    run("L", "m")
    for ev in right_order[mr + 1:]:
        run("R", ev)
    for ev in left_order[ml + 1:]:
        run("L", ev)
    return tuple(word)


def diagram(cube, V, j, F, right_order, left_order):
    word = simulate(cube, V, j, F, right_order, left_order)
    n = cube.n
    eps = tuple(1 if s + 1 in V else 0 for s in range(n))
    return cube.KLRW.element_from_braid(word, left_state=cube.state(eps))


def precedences(stack, j, F):
    """
    Right-hand precedences (x before y) that give no bigons: mover before every linked pair,
    and for interlaced linked pairs the one starting earlier first.  On the left all are reversed.
    Returns the list of links as ordered pairs (first, second) for the right-hand side.
    """
    rel = []
    cj = stack.col[j - 1]
    for p in F:
        a, b = p
        if a < j < b and stack.adjacent(a, j):
            rel.append(("m", p))
    for p, q in itertools.combinations(sorted(F), 2):
        (a, b), (a2, b2) = p, q
        if a < a2 < b < b2 and stack.adjacent(a, a2):
            rel.append((p, q))
    return rel


def linear_extension(nodes, rel):
    """Some linear order of ``nodes`` with x before y for (x, y) in rel, or None if cyclic."""
    nodes = list(nodes)
    order = []
    remaining = set(nodes)
    while remaining:
        free = [x for x in nodes if x in remaining and not any(a in remaining and b == x for a, b in rel)]
        if not free:
            return None
        order.append(free[0])
        remaining.discard(free[0])
    return order


def canonical_orders(j, F):
    """Right-hand events from the inside out (increasing b, the mover counting as b = j),
    left-hand events from the outside in."""
    key = lambda ev: j if ev == "m" else ev[1]
    ro = sorted(["m"] + list(F), key=key)
    return ro, list(reversed(ro))
