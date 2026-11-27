"""
Referee (second pass): Lemma 4.3 of 05_main_sliding.md from its statement.

    sage -python R05b_lemma43.py NTRIALS SEED

For a random word w = w'.c, a slot a of w', a vertex K'' of w' containing a, mu = iota_L(m_a(K''))
(e_0 = c.e_{K''}) and a random element z of e' A e_0 (random parameter/dot coefficients, including
powers of r, on random reduced diagrams), let v(.) be the r-adic valuation (min total r-degree of the
coefficients in the basis; infinity for 0).  Lemma 4.3.1 says: if a is the first occurrence of its
colour in w', then  z mu in m^{N+1} A  implies  z in m^{N+1} A,  i.e.  v(z mu) = v(z)  (v(z mu) >= v(z)
is trivial).  Part 2 likewise for mu' = iota_R(m_a(K'')) and  v(mu' z) = v(z), z in e_1 A e'.
For slots that are not first occurrences we search for z with v(z mu) > v(z) (control: the lemma's
hypothesis is needed).
"""
import sys, random
from collections import defaultdict
from R05_lib import *


def rvaluation(S, x, ridx):
    best = None
    for braid, coeff in x:
        for exp, cc in coeff.iterator_exp_coeff():
            d = sum(exp[i] for i in ridx)
            best = d if best is None else min(best, d)
    return best


def random_coeff(S, arr, rng, colours):
    """Random polynomial in parameters and dots of arr (times idempotent), with some r-powers."""
    c = 0 * S.idem(arr)
    for _ in range(rng.randint(1, 3)):
        mono = S.idem(arr) * rng.choice([1, -1, 2, 3])
        for _ in range(rng.randint(0, 2)):
            mono = mono * S.r(rng.choice(colours))
        for _ in range(rng.randint(0, 1)):
            x, y = rng.sample(colours, 2) if len(colours) > 1 else (colours[0], colours[0])
            if S.adj(x, y):
                mono = mono * S.t(x, y)
        for _ in range(rng.randint(0, 2)):
            lab = rng.choice([l for l in arr if l != "W"])
            mono = mono * S.dot(arr, lab)
        c = c + mono
    return c


def random_element_to(S, src, tgt_labels_arr, rng, colours):
    """Random element from arrangement src to (any colour-compatible relabelling of) tgt_labels_arr."""
    by = defaultdict(list)
    for lab in src:
        by[S.col(lab)].append(lab)
    for v in by.values():
        rng.shuffle(v)
    tgt = [by[S.col(lab)].pop() for lab in tgt_labels_arr]
    return S.el(src, rand_reduced(src, tgt, rng)), tgt


if __name__ == "__main__":
    ntrials, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    st = defaultdict(int)
    TYPES = [["A", 2], ["A", 3], ["D", 4]]
    for _ in range(ntrials):
        ct = rng.choice(TYPES); rank = ct[1]
        n = rng.randint(2, 5)
        word = tuple(rng.randint(1, rank) for _ in range(n)) + (rng.randint(1, rank),)
        red = rng.randint(1, rank)
        S = Setup(ct, red, word)
        m = n + 1
        colours = sorted(set(word))
        ridx = [i for i, e in enumerate(S.r(c).exponents()[0]) if e] if False else None
        ridx = sorted({i for c in range(1, rank + 1) for i, e in enumerate(S.r(c).exponents()[0]) if e})
        a = rng.randint(1, n)
        first = word.index(word[a - 1]) == a - 1
        Kpp = set(rng.sample(range(1, n + 1), rng.randint(1, n))) | {a}
        for side in ("L", "R"):
            V = Kpp if side == "L" else Kpp | {m}          # e_0 = vertex V of w (m left / right)
            e0 = S.arr(V); e1 = S.arr(V - {a})
            p = e0.index(a) + 1
            l = sum(1 for lab in e0 if lab != a and (lab == "W" or lab < a))
            mu = S.el(e0, tuple(range(p - 1, p - l - 1, -1)))
            # a random other vertex for e'
            other = set(rng.sample(range(1, m + 1), rng.randint(0, m)))
            ep = S.arr(other)
            if side == "L":
                # z in e' A e_0:  build z from e' to e0
                z = 0 * S.idem(ep)
                for _ in range(rng.randint(1, 3)):
                    d, tgt = random_element_to(S, ep, e0, rng, colours)
                    # d ends at an arrangement with e0's colours; it is an element of e' A e0 as idempotents
                    z = z + random_coeff(S, ep, rng, colours) * d
                prod = z * mu
            else:
                z = 0 * S.idem(e1)
                for _ in range(rng.randint(1, 3)):
                    d, tgt = random_element_to(S, e1, ep, rng, colours)
                    z = z + random_coeff(S, e1, rng, colours) * d
                prod = mu * z
            if z.is_zero():
                continue
            vz, vp = rvaluation(S, z, ridx), rvaluation(S, prod, ridx)
            key = "first" if first else "NOT first"
            st["%s, side %s: cases" % (key, side)] += 1
            if vp is None or vp > vz:
                st["%s, side %s: v(product) > v(z)" % (key, side)] += 1
                if first:
                    print("FAIL 4.3", ct, red, word, a, sorted(Kpp), side, vz, vp, flush=True)
    for k_ in sorted(st):
        print("   %-50s %d" % (k_, st[k_]))
