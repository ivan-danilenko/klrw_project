"""
Check of the explicit braid-move path ("sweep") used in the proof of Theorem B of
proofs/04_first_order.md.

    sage -python proofs/checks/04_sweep.py TYPE RANK MAXLEN [MINLEN]

For every word, red colour, vertex I and i < j in I it builds the words
    D_s = [i: z_1..z_s] [j: U_R, z_1..z_s] [j: i] [j: z_{s+1}..z_N] [i: z_{s+1}..z_N] [j: U_L]
(s = 0..N, z_1..z_N = Z_R decreasing, W, Z_L increasing; "[u: v_1..v_k]" = strand u
crosses v_1, ..., v_k, moving to the left) and checks

  (S1) every crossing of the words is between neighbours (the words are well defined);
  (S2) D_N is the concatenation of the words of m_i(I) and m_j(I - i);
       D_0 = m_j(I) m_i(I - j) in the algebra (they differ by commutations only);
  (S3) D_s = D'_s and D_{s-1} = D''_s in the algebra, where D'_s, D''_s are obtained by
       moving [i: z_s] (resp. the crossings of the triangle) together, and the local state
       of the triangle {z_s, i, j} is (z_s, i, j) left to right; the triangle is bad iff
       z_s is black with c(z_s) = c(j) ~ c(i);
  (S4) the smoothed word X_s is reduced and realises tau = (exchange of the endpoints of
       z_s and j); X_s = beta(tau) modulo r; we also record whether X_s = beta(tau) exactly;
  (S5) m_i(I) m_j(I-i) - m_j(I) m_i(I-j) = sum over bad s of r_{c(j)} t_{c(j)c(i)} X_s, exactly.
"""
import sys, os, itertools, importlib, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L = importlib.import_module("04_lib")


class Sim:
    def __init__(self, labels):
        self.state = list(labels)
        self.word = []

    def cross(self, u, v):
        p = self.state.index(u)
        assert p > 0 and self.state[p - 1] == v, ("not neighbours", u, v, self.state)
        self.state[p - 1], self.state[p] = u, v
        self.word.append(p)          # 1-based position of the left one = p
        return p

    def run(self, moves):
        for u, v in moves:
            self.cross(u, v)


def labels_of(C, I):
    return [("W" if isinstance(x, tuple) else x + 1) for x in C.cube.labels(C.eps(I))]


def data(I, i, j, m):
    ZR = sorted([z for z in I if z < i], reverse=True)
    ZL = sorted([z for z in range(1, i) if z not in I])
    zs = ZR + ["W"] + ZL
    UR = sorted([z for z in I if i < z < j], reverse=True)
    UL = sorted([z for z in range(i + 1, j) if z not in I])
    return zs, UR, UL


def D_moves(s, zs, UR, UL, i, j):
    N = len(zs)
    mv = [(i, z) for z in zs[:s]]
    mv += [(j, z) for z in UR + zs[:s]]
    mv += [(j, i)]
    mv += [(j, z) for z in zs[s:]]
    mv += [(i, z) for z in zs[s:]]
    mv += [(j, z) for z in UL]
    return mv


def check_word(ct, red, word, stats):
    C = L.Ctx(ct, red, word)
    ok = True
    for I in L.subsets(C.m):
        for i, j in itertools.combinations(sorted(I), 2):
            zs, UR, UL = data(I, i, j, C.m)
            N = len(zs)
            lab = labels_of(C, I)
            Ds = []
            for s in range(N + 1):
                sim = Sim(lab)
                sim.run(D_moves(s, zs, UR, UL, i, j))
                Ds.append(tuple(sim.word))
                assert sim.state == labels_of(C, I - {i, j}), (sim.state, labels_of(C, I - {i, j}))
            P1w = tuple(C.move(I, i).leading_support().word()) + tuple(C.move(I - {i}, j).leading_support().word())
            if Ds[N] != P1w:
                print("S2 FAIL D_N", word, sorted(I), i, j); ok = False
            P1 = C.move(I, i) * C.move(I - {i}, j)
            P2 = C.move(I, j) * C.move(I - {j}, i)
            if C.word_element(I, Ds[0]) != P2:
                print("S2 FAIL D_0", word, sorted(I), i, j); ok = False
            total = C.K.zero()
            for s in range(1, N + 1):
                y = zs[s - 1]
                pre = [(i, z) for z in zs[:s - 1]] + [(j, z) for z in UR + zs[:s - 1]]
                sim = Sim(lab); sim.run(pre)
                p = sim.state.index(y) + 1
                local = tuple(sim.state[p - 1:p + 2])
                if local != (y, i, j):
                    print("S3 FAIL local state", word, sorted(I), i, j, y, local); ok = False
                prew = list(sim.word)
                suf = [(j, z) for z in zs[s:]] + [(i, z) for z in zs[s:]] + [(j, z) for z in UL]
                s1 = Sim(sim.state); s1.run([(i, y), (j, y), (j, i)]); s1.run(suf)
                s2 = Sim(sim.state); s2.run([(j, i), (j, y), (i, y)]); s2.run(suf)
                Dp = tuple(prew + s1.word); Dpp = tuple(prew + s2.word)
                if C.word_element(I, Dp) != C.word_element(I, Ds[s]) or C.word_element(I, Dpp) != C.word_element(I, Ds[s - 1]):
                    print("S3 FAIL commutation", word, sorted(I), i, j, s); ok = False
                bad = (y != "W" and C.col(y) == C.col(j) and C.adj(C.col(i), C.col(j)))
                diff = C.word_element(I, Dp) - C.word_element(I, Dpp)
                # smoothed word: y plays the role of j in the suffix, j does not move
                sx = Sim(sim.state)
                sx.run([(y, z) for z in zs[s:]] + [(i, z) for z in zs[s:]] + [(y, z) for z in UL])
                Xw = tuple(prew + sx.word)
                X = C.word_element(I, Xw)
                if bad:
                    stats["bad triangles"] += 1
                    if diff != (C.r(C.col(j)) * C.t(C.col(j), C.col(i))) * X:
                        print("S3 FAIL bad move", word, sorted(I), i, j, y); ok = False
                    cnt, final = L.crossing_counts(C, I, Xw)
                    if max(cnt.values()) > 1:
                        print("S4 FAIL X not reduced", word, sorted(I), i, j, y); ok = False
                    tau = list(C.ident); tau[y - 1], tau[j - 1] = j, y
                    B = C.basis(I, I - {i, j}, tuple(tau))
                    if C.r_part(X - B, 0) != 0:
                        print("S4 FAIL X != beta mod r", word, sorted(I), i, j, y); ok = False
                    if X == B:
                        stats["X_s = beta(tau) exactly"] += 1
                    else:
                        stats["X_s != beta(tau) (differ by O(r))"] += 1
                    total += (C.r(C.col(j)) * C.t(C.col(j), C.col(i))) * X
                else:
                    stats["good triangles"] += 1
                    if not diff.is_zero():
                        print("S3 FAIL good move not exact", word, sorted(I), i, j, y); ok = False
            if P1 - P2 != total:
                print("S5 FAIL", word, sorted(I), i, j); ok = False
            stats["pairs (I,i,j)"] += 1
    return ok


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    maxlen = int(sys.argv[3]); minlen = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    stats = Counter(); nf = 0; t0 = time.time(); nw = 0
    for w in L.all_words(ct[1], maxlen, minlen):
        for red in range(1, ct[1] + 1):
            nf += not check_word(ct, red, w, stats)
            nw += 1
    print("type", ct, "lengths", minlen, "..", maxlen, "word x red", nw, "failures", nf, "[%.0fs]" % (time.time() - t0))
    for k, v in sorted(stats.items()):
        print("   %-40s %d" % (k, v))
