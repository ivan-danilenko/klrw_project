"""
R06: Proposition 4.2 (naturality of Phi) and Proposition 4.4 (zeta_m commutes) on random
elements a of the algebra of w' (random products of crossings and dots, sums of several),
including the red strand and states where equal colours are adjacent.  Negative control:
psi in place of phi.

    sage -python R06_natural.py
"""
import random
from R06_lib import Setup

rng = random.Random(42)


def rand_element(S, arr0, shift, rng, nterms=3, length=5):
    """Random element on positions (1+shift .. len(arr0)+shift) of the big state; returns
    (element of the big algebra starting at big state, final small arr)."""
    total = None
    end = None
    # all terms must end at the same state: fix the word of crossings, vary dots/coefficients
    word = [rng.randint(1, len(arr0) - 1) for _ in range(length)]
    for k in range(nterms):
        cur = list(arr0)
        big = lambda a: (["M"] + a) if shift else (a + ["M"])
        el = S.idem(S._big(big(cur)))
        for i in word:
            if rng.random() < 0.5:
                # random dot on a black strand of the block
                ps = [p for p, lab in enumerate(cur, 1) if lab != "W"]
                p = rng.choice(ps)
                el = el * S.dotpos(S._big(big(cur)), p + shift)
            el = el * S.psi(S._big(big(cur)), i + shift)
            cur[i - 1], cur[i] = cur[i], cur[i - 1]
        el = rng.randint(-3, 3) * el
        total = el if total is None else total + el
        end = cur
    return total, end


class Nat(Setup):
    def _big(self, a):
        return [self.m if lab == "M" else lab for lab in a]


cases = [
    (["A", 2], 1, (1, 2, 1)), (["A", 2], 1, (1, 1, 1)), (["A", 2], 2, (1, 2, 1)), (["A", 2], 1, (2, 1, 1)),
    (["A", 2], 1, (1, 2, 2, 1)), (["A", 3], 2, (2, 1, 3, 2)), (["A", 3], 1, (1, 2, 1, 1)),
    (["D", 4], 2, (2, 1, 3, 4, 2)), (["D", 4], 1, (1, 2, 1, 3, 1)), (["A", 2], 2, (2, 2, 2)),
]
tot = bad = badneg = badz = 0
for ct, red, w in cases:
    S = Nat(ct, red, w)
    m = S.m
    labels = list(range(1, m)) + ["W"]
    nb = nn = nz = 0
    for trial in range(15):
        s = labels[:]
        rng.shuffle(s)
        seed = rng.random()
        aL, endL = rand_element(S, s, 0, random.Random(seed))
        aR, endR = rand_element(S, s, 1, random.Random(seed))
        assert endL == endR
        Phi_s = S.D(s + [m], [m] + s)
        Phi_e = S.D(endL + [m], [m] + endL)
        lhs = aL * Phi_e
        rhs = Phi_s * aR
        nb += not (lhs - rhs).is_zero()
        # negative control: psi-only move
        def psiword(src, tgt):
            from R06_lib import reduced_word
            el = S.idem(src)
            cur = list(src)
            for i in reduced_word(src, tgt):
                el = el * S.psi(cur, i)
                cur[i - 1], cur[i] = cur[i], cur[i - 1]
            return el
        lhs2 = aL * psiword(endL + [m], [m] + endL)
        rhs2 = psiword(s + [m], [m] + s) * aR
        nn += not (lhs2 - rhs2).is_zero()
        # zeta_m = prod_{a<m, c(a)=c(m)} (x_a - x_m) commutes with a (x) 1
        c = S.col(m)
        def zeta(arr):
            z = S.idem(arr)
            for lab in arr:
                if lab != "W" and lab != m and S.col(lab) == c:
                    z = z * (S.dot(arr, lab) - S.dot(arr, m))
            return z
        nz += not (zeta(s + [m]) * aL - aL * zeta(endL + [m])).is_zero()
        nz += not (zeta([m] + s) * aR - aR * zeta([m] + endL)).is_zero()
    print("%s red %d word %s: naturality failures %d/15, zeta failures %d/30, negative control (psi only) failures %d/15"
          % ("".join(map(str, ct)), red, "".join(map(str, w)), nb, nz, nn), flush=True)
    tot += 15
    bad += nb
    badz += nz
    badneg += nn
print("RESULT naturality: %d/%d failures; zeta: %d failures; negative control: %d/%d differ" % (bad, tot, badz, badneg, tot))
