"""
R01_rank.py  (referee of 01_foundations.md; own code, no klrw, no 01_*.py)

    sage -python proofs/checks/R01_rank.py

Theorem A.5 through the referee's own polynomial representation (R01_lib.PolyRepR):

 (1) independence: for pairs of states (s, s') the operators of x^alpha psi_w,
     w in S(s,s'), |alpha| <= D, are linearly independent (rank over GF(p) at a random
     specialisation of the parameters = number of elements), for TWO choices of reduced words:
     the Lehmer words of Def. B.10 and random reduced words;
 (2) the right basis psi_w x^beta (dots at the target) is independent as well;
 (3) spanning, degree by degree: random (non-reduced, dotted) words of q-degree d act inside
     the span of the claimed basis elements of q-degree d;
 (4) what happens at degenerate parameters (r = 0, t = 0, u = 0): the representation stops
     being faithful (rank drops) -- so (F1) at r = 0 cannot come from V, only from base change.
"""
import itertools
import random
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from R01_lib import (is_red, seq_str, parse_seq, PolyRepR, random_params, lehmer_word,  # noqa
                     random_reduced_word, colour_perms, states_of, dynkin_edges)
from sage.all import GF, matrix  # noqa

PR = 1000003
F = GF(PR)


def monos(rep, positions, deg):
    out = []
    for d in range(deg + 1):
        for combo in itertools.combinations_with_replacement(positions, d):
            m = rep.R(1)
            for q in combo:
                m *= rep.y[q]
            out.append((m, d))
    return out


def qdeg_crossing(ct, a, b):
    E = dynkin_edges(ct)
    if is_red(a) and is_red(b):
        raise ValueError
    if is_red(a) or is_red(b):
        c, w = (b, a) if is_red(a) else (a, b)
        return 1 if c == w[1] else 0
    if a == b:
        return -2
    return 1 if frozenset((a, b)) in E else 0


def word_qdeg(ct, s, w):
    d = 0
    s = list(s)
    for i in w:
        d += qdeg_crossing(ct, s[i], s[i + 1])
        s[i], s[i + 1] = s[i + 1], s[i]
    return d


def operator_rows(rep, s, elems, tests):
    """elems: list of (left poly, word, right poly at target); returns sparse rows."""
    rows = []
    for (g, w, h) in elems:
        vec = {}
        for ti, f in enumerate(tests):
            t, out = rep.word(s, w, g * f)
            out = out * h
            for mono, c in out.dict().items():
                vec[(ti, t, mono)] = c
        rows.append(vec)
    return rows


def rank_of(rows):
    cols = sorted(set(k for r in rows for k in r), key=str)
    idx = {c: i for i, c in enumerate(cols)}
    M = matrix(F, len(rows), max(1, len(cols)), {(i, idx[c]): v for i, r in enumerate(rows) for c, v in r.items()},
               sparse=True)
    return M.rank()


def check_pair(ct, rep, s, t, D, rng, which="lehmer", side="left"):
    perms = colour_perms(s, t)
    blacks_s = [q for q in range(len(s)) if not is_red(s[q])]
    blacks_t = [q for q in range(len(t)) if not is_red(t[q])]
    elems = []
    for perm in perms:
        w = lehmer_word(perm) if which == "lehmer" else random_reduced_word(perm, rng)
        if side == "left":
            for m, _ in monos(rep, blacks_s, D):
                elems.append((m, w, rep.R(1)))
        else:
            for m, _ in monos(rep, blacks_t, D):
                elems.append((rep.R(1), w, m))
    maxlen = max(len(lehmer_word(p)) for p in perms)
    # generic test vectors: |S| + 3 random polynomials of degree D + maxlen + 1 (a full-rank
    # result on any set of test vectors proves independence; a deficient one would be re-run)
    tests = []
    pool = [m for m, _ in monos(rep, blacks_s, min(D + maxlen + 1, D + 3))]
    for _ in range(len(perms) + 3):
        f = rep.R(0)
        for m in pool:
            f += rep.B(rng.randint(0, PR - 1)) * m
        tests.append(f)
    rows = operator_rows(rep, s, elems, tests)
    return len(elems), rank_of(rows), len(perms)


def part1(rng):
    print("(1)/(2) independence of x^a psi_w (left) and psi_w x^b (right), rank over GF(p):")
    ok = True
    cases = [(["A", 2], "1 1 2 W1", 2, 14), (["A", 2], "1 W1 1 1", 2, 10),
             (["A", 2], "1 2 1 W1 1", 1, 5), (["A", 3], "1 2 W2 2 3", 1, 5),
             (["D", 4], "2 1 W2 2 3", 1, 4), (["A", 2], "1 1 1 W1", 1, 4)]
    for ct, seq, D, npairs in cases:
        s0 = parse_seq(seq)
        nodes = list(range(1, ct[1] + 1))
        params = random_params(ct, rng, F, nodes)
        rep = PolyRepR(ct, len(s0), params, F)
        st = states_of(s0)
        pairs = [(a, b) for a in st for b in st]
        if npairs is not None:
            pairs = rng.sample(pairs, npairs)
        for s, t in pairs:
            for which in ("lehmer", "random"):
                for side in ("left", "right"):
                    if side == "right" and which == "random":
                        continue
                    ne, rk, npm = check_pair(ct, rep, s, t, D, rng, which, side)
                    good = ne == rk
                    ok &= good
                    print("   %-12s -> %-12s %-6s %-5s |S|=%2d  elements %4d  rank %4d %s"
                          % (seq_str(s), seq_str(t), which, side, npm, ne, rk, "ok" if good else "DEPENDENT"))
    return ok


def part3(rng):
    print("(3) spanning degree by degree: random dotted words lie in the span of the basis of their degree")
    ok = True
    for ct, seq, nwords in [(["A", 2], "1 1 2 W1", 40), (["A", 2], "1 2 1 W1", 40), (["A", 2], "1 W1 1 2", 40),
                            (["A", 3], "2 1 2 W2 3", 30)]:
        s0 = parse_seq(seq)
        nodes = list(range(1, ct[1] + 1))
        params = random_params(ct, rng, F, nodes)
        rep = PolyRepR(ct, len(s0), params, F)
        st = states_of(s0)
        n_in = 0
        for _ in range(nwords):
            s = rng.choice(st)
            L = rng.randint(2, 9)
            w = []
            cur = s
            dots = []
            for _ in range(L):
                i = rng.randrange(len(s) - 1)
                if is_red(cur[i]) and is_red(cur[i + 1]):
                    continue
                w.append(i)
                cur = list(cur)
                cur[i], cur[i + 1] = cur[i + 1], cur[i]
                cur = tuple(cur)
            t = cur
            blacks = [q for q in range(len(s)) if not is_red(s[q])]
            g = rep.R(1)
            nd = rng.randint(0, 2)
            for _ in range(nd):
                g *= rep.y[rng.choice(blacks)]
            d = 2 * nd + word_qdeg(ct, s, w)
            # claimed basis elements of degree d
            elems = []
            for perm in colour_perms(s, t):
                lw = lehmer_word(perm)
                dd = d - word_qdeg(ct, s, lw)
                if dd < 0 or dd % 2:
                    continue
                for m, k in monos(rep, blacks, dd // 2):
                    if k == dd // 2:
                        elems.append((m, lw, rep.R(1)))
            tests = [m for m, _ in monos(rep, blacks, 3)]
            rows_b = operator_rows(rep, s, elems, tests)
            rows_w = operator_rows(rep, s, [(g, w, rep.R(1))], tests)
            r1 = rank_of(rows_b) if rows_b else 0
            r2 = rank_of(rows_b + rows_w)
            if r2 == r1:
                n_in += 1
            else:
                ok = False
                print("   NOT IN SPAN", seq_str(s), w, d)
        print("   %-10s %d/%d random words in the span of the basis of their q-degree" % (seq, n_in, nwords))
    return ok


def part4(rng):
    print("(4) degenerate specialisations: rank of the claimed basis through V (A_2, 1 1 2 W1, D = 1)")
    ct, seq = ["A", 2], "1 1 2 W1"
    s0 = parse_seq(seq)
    for label, zero in [("generic", ()), ("r_1 = 0", (("r", 1),)), ("t_12 = t_21 = 0", (("t", 1, 2), ("t", 2, 1))),
                        ("u_1 = 0", (("u", 1),))]:
        params = random_params(ct, rng, F, [1, 2], zero=zero)
        rep = PolyRepR(ct, len(s0), params, F)
        tot_e = tot_r = 0
        prs = [(a, b) for a in states_of(s0) for b in states_of(s0)]
        for s, t in random.Random(99).sample(prs, 16):
            if True:
                ne, rk, _ = check_pair(ct, rep, s, t, 1, rng, "lehmer", "left")
                tot_e += ne
                tot_r += rk
        print("   %-18s elements %4d   total rank %4d" % (label, tot_e, tot_r))


if __name__ == "__main__":
    rng = random.Random(11)
    ok = part1(rng)
    ok &= part3(rng)
    part4(rng)
    print("ALL OK" if ok else "FAILURES")
