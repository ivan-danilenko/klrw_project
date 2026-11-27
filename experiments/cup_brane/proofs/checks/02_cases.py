"""
Case scheme of the proof of Theorem 5.2 (proofs/02_omega1_ibrane.md, section 5), checked on ALL
composable pairs of terms of the R-model, pure Python.

    python3 proofs/checks/02_cases.py N_MAX

Every composable pair (t1: v -> u, t2: u -> w) of terms of the R-model (universal q's) is one
product term of  delta^2 = sum_{j <= k} Phi_{j,k}.  The script assigns to it the label of the
proof (exactly one label; the labelling uses only the two term tags), and checks:
  * each non-W label has the partner predicted in the proof, inside the same component (v, w),
    and the two contributions cancel;
  * labels predicted to be impossible ("zero" products: AC, CB, ...) never occur;
  * the W-labelled paths of a component are exactly n+1, one for each edge x = 0..n, with
    coefficient  (-1)^(j+1) (-1)^x q_x * (sign of N_{j,k} on the source);
  * no component contains anything else.

Labels (block (j, k], D = k - j, relative colour c' = c - j; see Theorem 5.2 and Lemmas 5.3, 5.4):
  KOS                      both internal (j = k)
  dO                       an internal move on an outer colour and a gluing term k -> j whose
                           extra colour (if any) is a different colour        partner: other order
  Wsat                     the same, extra colour = the internal colour       (W-path, edge e-1 / e)
  T1a,T1b,T2a,T2b,AA,...   Lemma 5.3 (inner part)
  P1,P2,NA,NB,NC,AN,BN,CN  Lemma 5.4 (O-linear part)
  OO                       two gluing steps both with outer extra colours     partner: extras swapped
"""
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
R = importlib.import_module("02_rmodel")


def outer(e, j, k):
    return e <= j or e > k


def label(n, k, t1, l, t2, j):
    """t1 = (target envelope, R, coeff, tag) out of envelope k; t2 out of envelope l; final envelope j."""
    tag1, tag2 = t1[3], t2[3]
    if j == k:
        return ("KOS", tag1[1], tag2[1])
    D = k - j
    if tag1[0] == "int" and tag2[0] != "int":          # internal at k, then gluing k -> j
        c = tag1[1]
        X = tag2[0]
        if outer(c, j, k):
            if X == "iii" and tag2[2] == c:
                return ("Wsat", c)
            return ("dO", "int-first", c, tag2)
        cc = c - j
        if X == "iii":
            return ("P2", cc)
        if X == "i":
            return ("T2a", cc)
        return ("T2b", cc)
    if tag1[0] != "int" and tag2[0] == "int":          # gluing k -> j, then internal at j
        c = tag2[1]
        X = tag1[0]
        if outer(c, j, k):
            if X == "iii" and tag1[2] == c:
                return ("Wsat", c)
            return ("dO", "glue-first", c, tag1)
        cc = c - j
        if X == "iii":
            return ("P1", cc)
        if X == "i":
            return ("T1a", cc)
        return ("T1b", cc)
    # two gluing steps k -> l -> j
    X1, X2 = tag1[0], tag2[0]
    o1 = X1 == "iii" and outer(tag1[2], j, k)
    o2 = X2 == "iii" and outer(tag2[2], j, k)
    T = {"i": "A", "ii": "B", "iii": "C"}
    if o1 and o2:
        return ("OO", l, tag1[2], tag2[2])
    if o2:
        return ("N" + T[X1], l)                  # N_{j,l} Z^in_{l,k}
    if o1:
        return (T[X2] + "N", l)                  # Z^in_{j,l} N_{l,k}
    return (T[X2] + T[X1], l)                    # Z^in_{j,l} Z^in_{l,k}: first letter = second step


def partner(lab, j, k):
    """Partner label predicted by the proof, or 'W' or 'zero'."""
    D = k - j
    kind = lab[0]
    if kind == "KOS":
        return ("KOS", lab[2], lab[1])
    if kind == "dO":
        _, order, c, tag = lab
        return ("dO", "glue-first" if order == "int-first" else "int-first", c, tag)
    if kind == "Wsat":
        return "W"
    if kind == "OO":
        _, l, e1, e2 = lab
        return ("OO", l, e2, e1)
    if kind == "T1a":
        return ("AA", j + lab[1]) if 1 <= lab[1] <= D - 1 else "zero"
    if kind == "AA":
        return ("T1a", lab[1] - j)
    if kind == "T1b":
        cc = lab[1]
        if cc == D:
            return "W"
        return ("CA", j + cc)
    if kind == "CA":
        return ("T1b", lab[1] - j)
    if kind == "BA":
        return "W"
    if kind == "T2a":
        cc = lab[1]
        if cc == 1:
            return "W"
        return ("BC", j + cc - 1)
    if kind == "BC":
        return ("T2a", lab[1] - j + 1)
    if kind == "T2b":
        return ("BB", j + lab[1] - 1) if lab[1] >= 2 else "zero"
    if kind == "BB":
        return ("T2b", lab[1] - j + 1)
    if kind == "AB":
        return ("CC", lab[1])
    if kind == "CC":
        return ("AB", lab[1])
    if kind in ("AC", "CB"):
        return "zero"
    if kind == "P1":
        return ("NA", j + lab[1]) if 1 <= lab[1] <= D - 1 else "zero"
    if kind == "NA":
        return ("P1", lab[1] - j)
    if kind == "P2":
        return ("BN", j + lab[1] - 1) if lab[1] >= 2 else "zero"
    if kind == "BN":
        return ("P2", lab[1] - j + 1)
    if kind == "NB":
        return ("CN", lab[1])
    if kind == "CN":
        return ("NB", lab[1])
    if kind == "NC":
        return ("AN", lab[1])
    if kind == "AN":
        return ("NC", lab[1])
    raise ValueError(lab)


def W_edge(lab, j, k, tag1, tag2):
    """Edge x of a W-path."""
    kind = lab[0]
    if kind == "Wsat":
        e = lab[1]
        return e - 1 if e <= j else e
    if kind == "T1b":
        return k
    if kind == "BA":
        return lab[1]
    if kind == "T2a":
        return j
    raise ValueError(lab)


def main(N):
    for n in range(1, N + 1):
        verts = list(R.rmodel_vertices(n))
        terms = {v: R.rmodel_terms(n, *v) for v in verts}
        comps = defaultdict(list)
        for v in verts:
            for t1 in terms[v]:
                u = (t1[0], t1[1])
                for t2 in terms[u]:
                    w = (t2[0], t2[1])
                    lab = label(n, v[0], t1, u[0], t2, w[0])
                    comps[(v, w)].append((lab, R.qmul(t1[2], t2[2]), t1, t2))
        counts = defaultdict(int)
        bad = 0
        npairs = 0
        for (v, w), paths in comps.items():
            k, j = v[0], w[0]
            labs = defaultdict(list)
            for lab, coef, t1, t2 in paths:
                npairs += 1
                counts[lab[0]] += 1
                labs[lab].append(coef)
            for lab, coefs in labs.items():
                if len(coefs) != 1:
                    bad += 1          # a label must identify a path uniquely inside a component
            Wsum = {}
            Wedges = []
            for lab, coef, t1, t2 in paths:
                p = partner(lab, j, k)
                if p == "zero":
                    bad += 1
                    continue
                if p == "W":
                    Wsum = R.qadd(Wsum, coef)
                    Wedges.append(W_edge(lab, j, k, t1[3], t2[3]))
                    continue
                if p not in labs or R.qadd(coef, labs[p][0]):
                    bad += 1
            if Wedges:
                r = R.N_op(n, j, k, v[1])
                if r is None or r[0] != w[1] or sorted(Wedges) != list(range(n + 1)):
                    bad += 1
                elif Wsum != R.qscale(R.W(n), (-1) ** (j + 1) * r[1]):
                    bad += 1
        print("n=%d: %d composable pairs in %d components; violations of the case scheme: %d" % (
            n, npairs, len(comps), bad))
        print("      label counts:", dict(sorted(counts.items())), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
