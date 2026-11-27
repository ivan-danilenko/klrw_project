"""
Term-by-term check of the coefficient formulas used in the proofs of 07_conjecture_H.md:

  "diag":  Proposition 3.2 (coefficients of Phi_m, zeta X_m, h_chain X_{c_1}, X_j h_chain on the
           brackets tau_b, and nothing else);
  "adj0":  Proposition 5.4 (the five kinds of terms (i)-(v) of the group of an adjacent mover j at
           K = (I - {j, m}) + {a}, a < j, on the brackets (a, Y, m, Z)), the chain sums with a
           fixed first element taken together (compact form, Prop. 2.6).

Every contribution of the expansion (07_lab) is compared with the predicted closed formula; a
bracket that is not of the predicted shape counts as a failure.

    sage -python 07_formulas.py TYPE RANK RED WORD [RED WORD ...]
"""
import sys, importlib, itertools
from collections import defaultdict
L = importlib.import_module("07_lab")
G = importlib.import_module("07_groups")
G.ADJ_BY_J = True


def chain_of(tau, start, stop):
    """follow tau from start until stop is reached; returns the list of labels strictly between."""
    out, x = [], tau[start - 1]
    while x != stop:
        out.append(x)
        x = tau[x - 1]
        if len(out) > len(tau):
            return None
    return out


def check_diag(S, I, lst):
    m = S.m
    R = sorted(x for x in I if x != m)
    same = [s for s in range(1, m) if S.col(s) == S.col(m)]
    time = [s for s in reversed(R) if s in same] + [s for s in range(1, m) if s in same and s not in I]
    rank = {s: i for i, s in enumerate(time)}
    xm = S.x(m)
    r = S.r(S.col(m))
    bad = 0
    n = 0
    for label, prods, sgn in lst:
        el = S.expand(I, prods)
        for t, f in el.items():
            n += 1
            tau = S.tau_of(I, I - {m}, t)
            b = chain_of(tau, m, m)
            moved = {s for s in range(1, m + 1) if tau[s - 1] != s}
            if b is None or moved != (set(b) | {m} if b else set()) or [rank.get(x, -1) for x in b] != sorted(rank.get(x, -1) for x in b) or any(x not in same for x in b):
                bad += 1
                continue
            bset = set(b)
            mu = {}
            last = m
            for s in time:
                mu[s] = last
                if s in bset:
                    last = s
            A = {s: S.x(s) - xm for s in same}
            B = {s: S.x(s) - S.x(mu[s]) for s in same}
            N = r ** len(b)
            for s in same:
                N /= B[s]
            if label == "Phi":
                pred = 1 if not b else None
            elif label == "zF":
                pred = -N
                for s in same:
                    pred *= A[s]
            else:
                H = prods[1] if label == "dR.h" else prods[0]
                ch = list(H.info[0])
                if not set(ch) <= bset:
                    pred = None
                else:
                    pred = -(-1) ** len(ch) * N
                    for s in same:
                        if s in ch:
                            pred *= B[s]
                        elif label == "dR.h" and s in bset and s < prods[0].info[0]:
                            pred *= A[s] - B[s]
                        else:
                            pred *= A[s]
            if pred is None or sgn * f != pred:
                bad += 1
    return n, bad


def check_adj0(S, I, K, j, a, lst):
    """the group of the adjacent mover j at K = (I - {j, m}) + {a}, a < j."""
    m = S.m
    c = S.col(m)
    e = S.col(j)
    same = [s for s in range(1, m) if S.col(s) == c]
    Lin = [s for s in same if a < s < m and s not in I]
    Rin = [s for s in same if a < s < m and s in I]
    xm = S.x(m)
    r, t, t2 = S.r(c), S.t(c, e), S.t(e, c)
    Q = lambda x: t * x + t2 * S.x(j)
    sigma = S.stack.koszul(I, j, ((a, m),))
    # accumulate the five kinds, chain sums taken together
    acc = defaultdict(lambda: defaultdict(lambda: S.F.zero()))
    for label, prods, sgn in lst:
        if label == "zF":
            k = ("iii",)
        elif label == "dR.h":
            T, H = prods
            jj, F = T.info
            k = ("i",) if not F else ("iv", F[0][1])
        else:
            H, T = prods
            jj, F = T.info
            k = ("ii",) if not F else ("v", H.info[0][0])
        for tt, f in S.expand(I, prods).items():
            acc[k][S.tau_of(I, K, tt)] += sgn * f
    n = bad = 0
    for k, row in acc.items():
        for tau, f in row.items():
            if f == 0:
                continue
            n += 1
            if tau[j - 1] != j:
                bad += 1; continue
            cyc = chain_of(tau, a, a)
            if cyc is None or m not in cyc:
                bad += 1; continue
            Y = cyc[:cyc.index(m)]
            Z = cyc[cyc.index(m) + 1:]
            moved = {s for s in range(1, m + 1) if tau[s - 1] != s}
            if moved != set(cyc) | {a} or Y != sorted(Y) or Z != sorted(Z, reverse=True) or not set(Y) <= set(Lin) or not set(Z) <= set(Rin):
                bad += 1; continue
            mu, last = {}, a
            for s in Lin:
                mu[s] = last
                if s in Y:
                    last = s
            nu, last = {}, m
            for s in reversed(Rin):
                nu[s] = last
                if s in Z:
                    last = s
            A = lambda s: S.x(s) - xm
            B = {s: S.x(s) - S.x(mu[s]) for s in Lin}
            Bp = {s: S.x(s) - S.x(nu[s]) for s in Rin}
            base = sigma * r * t * r ** (len(Y) + len(Z))
            for s in Lin:
                base /= B[s]
            for s in Rin:
                base /= Bp[s]
                base *= A(s)
            for s in same:
                if s < a:
                    base *= A(s)
            PY = S.F.one()
            for s in Lin:
                PY *= (A(s) - B[s]) if s in Y else A(s)
            lam = max([a] + [y for y in Y if y < j])
            nuj = min([m] + [z for z in Z if z > j])
            if k == ("i",):
                pred = base * (-Q(S.x(lam)) / t) * PY
            elif k == ("ii",):
                pred = base * (Q(S.x(nuj)) / t) * PY
            elif k == ("iii",):
                pred = base * A(a)
                for s in Lin:
                    pred *= A(s)
            elif k[0] == "iv":
                bb = k[1]
                pred = base * (-Bp[bb]) * PY if bb in Z else 0
            else:
                cc = k[1]
                if cc not in Y:
                    pred = 0
                else:
                    pred = base * (-B[cc]) * A(a)
                    for s in Lin:
                        if s < cc:
                            pred *= A(s)
                        elif s > cc:
                            pred *= (A(s) - B[s]) if s in Y else A(s)
            if f != pred:
                bad += 1
                if VERBOSE:
                    print("   ", L.fmt(I), "->", L.fmt(K), k, L.cycles(tau), "actual", f.factor(), " predicted", pred.factor() if pred != 0 else 0)
    return n, bad


VERBOSE = "v" in sys.argv


def run(ct, red, word):
    S = L.Lab(ct, red, word)
    m = S.m
    c = S.col(m)
    stat = defaultdict(lambda: [0, 0])
    for h in range(1, m + 1):
        for rest in itertools.combinations(range(1, m), h - 1):
            I = frozenset(rest) | {m}
            for (K, key), lst in G.groups(S, I).items():
                if key == "diag":
                    n, bad = check_diag(S, I, lst)
                    stat["diag"][0] += n; stat["diag"][1] += bad
                elif isinstance(key, tuple) and key[0] == "J":
                    j = key[1]
                    new = sorted(K - I)
                    gone = sorted((I - {m}) - K)
                    if len(new) == 1 and gone == [j] and new[0] < j:
                        n, bad = check_adj0(S, I, K, j, new[0], lst)
                        stat["adj0"][0] += n; stat["adj0"][1] += bad
    print("{} red {} word {}: ".format("".join(map(str, ct)), red, "".join(map(str, word))) +
          "; ".join("{}: {} coefficients compared, {} mismatches".format(k, v[0], v[1]) for k, v in sorted(stat.items())), flush=True)
    return all(v[1] == 0 for v in stat.values())


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    args = [x for x in sys.argv[3:] if x != "v"]
    for red, word in zip(args[0::2], args[1::2]):
        run(ct, int(red), tuple(int(x) for x in word))
