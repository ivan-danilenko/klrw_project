"""
R06 basic checks (own code): conventions, Lemma 1.1, Lemma 2.1, Lemma 2.3 (all 3-strand
labelled states), Prop. 3.2/3.3/3.5 on random labelled states, Lemma 6.5 on tree terms.

    sage -python R06_basic.py
"""
import sys, random, itertools
from R06_lib import Setup, reduced_word

rng = random.Random(6)
fails = 0


def report(name, ok):
    global fails
    fails += (not ok)
    print(("ok   " if ok else "FAIL ") + name, flush=True)


# --- conventions ------------------------------------------------------------------------------
S = Setup(["A", 2], 1, (1, 2))
arr = [1, 2, "W"]          # labels; slot 1 colour 1, slot 2 colour 2
# psi^2 on (1,2) at positions 1,2
p2 = S.psi(arr, 1) * S.psi([2, 1, "W"], 1)
Qv = S.t(1, 2) * S.dotpos(arr, 1) + S.t(2, 1) * S.dotpos(arr, 2)
report("psi^2 on (1,2) = t12 x_1 + t21 x_2", (p2 - Qv).is_zero())
report("D('t_1_2') == edge_variable(1,2)", S.Dt("t_1_2") == S.t(1, 2))
report("D('r_1') == vertex_variable(1)", S.Dt("r_1") == S.r(1))

# --- same colour: (S), Lemma 1.1, Lemma 2.1 ---------------------------------------------------
S = Setup(["A", 2], 1, (1, 1, 1))
a = [1, 2, "W", 3]
x1, x2 = S.dotpos(a, 1), S.dotpos(a, 2)
ps = S.psi(a, 1)
r = S.r(1) * S.idem(a)
report("(S) x1 psi - psi x2 = r", (x1 * ps - ps * x2 - r).is_zero())
report("(S) psi x1 - x2 psi = r", (ps * x1 - x2 * ps - r).is_zero())
phi = S.g(a, 1)
report("phi = -psi(x1-x2) + r", (phi + ps * (x1 - x2) - r).is_zero())
report("x1 phi = phi x2", (x1 * phi - phi * x2).is_zero())
report("x2 phi = phi x1", (x2 * phi - phi * x1).is_zero())
report("phi^2 = r^2", (phi * phi - S.r(1) ** 2 * S.idem(a)).is_zero())
# Lemma 1.1 for f = x1^2 x2 + 3 x1 x2^3
f = x1 * x1 * x2 + 3 * x1 * x2 * x2 * x2
sf = x2 * x2 * x1 + 3 * x2 * x1 * x1 * x1
# divided difference (f - sf)/(x1 - x2) computed by hand: x1 x2 (x1 - x2)/(x1-x2) ... :
# f - sf = x1 x2 (x1 - x2) + 3 x1 x2 (x2^2 - x1^2) = (x1 - x2) x1 x2 (1 - 3(x1 + x2))
df = x1 * x2 - 3 * x1 * x2 * (x1 + x2)
report("Lemma 1.1 psi f = (s f) psi + r df", (ps * f - sf * ps - S.r(1) * df).is_zero())

# --- Lemma 2.3 on all 3-strand states --------------------------------------------------------
for ct, words in ((["A", 3], [(1, 2, 1), (1, 1, 2), (2, 1, 1), (1, 1, 1), (1, 3, 1), (1, 2, 3), (2, 2, 3)]),
                  (["D", 4], [(2, 1, 2), (1, 3, 1), (2, 2, 2)])):
    for w in words:
        for red in sorted(set(w)) + [c for c in (1, 2, 3) if c not in w][:1]:
            S = Setup(ct, red, w)
            labels = [1, 2, 3, "W"]
            nbad = ntot = 0
            for perm in itertools.permutations(labels):
                perm = list(perm)
                for i in (1, 2):
                    # positions i, i+1, i+2
                    def gw(word):
                        el = S.idem(perm)
                        cur = list(perm)
                        for l in word:
                            el = el * S.g(cur, l)
                            cur[l - 1], cur[l] = cur[l], cur[l - 1]
                        return el
                    lhs = gw((i, i + 1, i))
                    rhs = gw((i + 1, i, i + 1))
                    ntot += 1
                    nbad += not (lhs - rhs).is_zero()
                    # second statement on (c, c, c) and naturality-type identity psi_i g g = g g psi_{i+1}
                    A_, B_ = perm[i - 1], perm[i]
                    if "W" not in (A_, B_) and S.col(A_) == S.col(B_):
                        el1 = S.psi(perm, i)
                        cur = list(perm); cur[i - 1], cur[i] = cur[i], cur[i - 1]
                        el1 = el1 * S.g(cur, i + 1)
                        cur[i], cur[i + 1] = cur[i + 1], cur[i]
                        el1 = el1 * S.g(cur, i)
                        cur[i - 1], cur[i] = cur[i], cur[i - 1]
                        cur2 = list(perm)
                        el2 = S.g(cur2, i + 1)
                        cur2[i], cur2[i + 1] = cur2[i + 1], cur2[i]
                        el2 = el2 * S.g(cur2, i)
                        cur2[i - 1], cur2[i] = cur2[i], cur2[i - 1]
                        el2 = el2 * S.psi(cur2, i + 1)
                        ntot += 1
                        nbad += not (el1 - el2).is_zero()
            report("Lemma 2.3 %s word %s red %s: %d identities, %d failures" % (ct, w, red, ntot, nbad), nbad == 0)

# negative control: braid relation with psi in place of phi fails on (c, c', c)
S = Setup(["A", 2], 1, (1, 2, 1))
perm = [1, 2, 3, "W"]
l = S.psi(perm, 1) * S.psi([2, 1, 3, "W"], 2) * S.psi([2, 3, 1, "W"], 1)
rr = S.psi(perm, 2) * S.psi([1, 3, 2, "W"], 1) * S.psi([3, 1, 2, "W"], 2)
report("negative control: psi braid on (1,2,1) is NOT exact", not (l - rr).is_zero())

# --- Prop. 3.2, 3.3, 3.5 on random labelled states ---------------------------------------------
for ct, red, w in ((["A", 3], 2, (2, 1, 3, 2, 1)), (["D", 4], 1, (2, 1, 3, 4, 2, 1)), (["A", 2], 1, (1, 2, 1, 2, 1)),
                   (["A", 2], 2, (2, 2, 1, 2))):
    S = Setup(ct, red, w)
    labels = list(range(1, len(w) + 1)) + ["W"]
    n32 = n33 = n35 = 0
    for trial in range(12):
        s0 = labels[:]; rng.shuffle(s0)
        s1 = labels[:]; rng.shuffle(s1)
        s2 = labels[:]; rng.shuffle(s2)
        d1 = S.D(s0, s1, rng=rng)
        d1b = S.D(s0, s1, rng=rng)
        n32 += not (d1 - d1b).is_zero()
        lab = rng.choice([x for x in labels if x != "W"])
        n33 += not (S.dot(s0, lab) * d1 - d1 * S.dot(s1, lab)).is_zero()
        d2 = S.D(s1, s2, rng=rng)
        d02 = S.D(s0, s2, rng=rng)
        pos = lambda s: {x: i for i, x in enumerate(s)}
        P0, P1, P2 = pos(s0), pos(s1), pos(s2)
        coef = S.idem(s0)
        for a_, b_ in itertools.combinations(labels, 2):
            inv01 = (P0[a_] - P0[b_]) * (P1[a_] - P1[b_]) < 0
            inv12 = (P1[a_] - P1[b_]) * (P2[a_] - P2[b_]) < 0
            if inv01 and inv12:
                if "W" in (a_, b_):
                    x = a_ if b_ == "W" else b_
                    q = S.Dt("u_%d" % red) * S.dot(s0, x) if S.col(x) == red else S.idem(s0)
                elif S.col(a_) == S.col(b_):
                    q = S.r(S.col(a_)) ** 2 * S.idem(s0)
                elif S.adj(S.col(a_), S.col(b_)):
                    q = S.t(S.col(a_), S.col(b_)) * S.dot(s0, a_) + S.t(S.col(b_), S.col(a_)) * S.dot(s0, b_)
                else:
                    q = S.idem(s0)
                coef = coef * q
        n35 += not (d1 * d2 - coef * d02).is_zero()
    report("Prop 3.2/3.3/3.5 %s red %d word %s: failures %d %d %d (12 trials)" % (ct, red, w, n32, n33, n35),
           n32 == n33 == n35 == 0)

print("TOTAL FAILURES", fails)
