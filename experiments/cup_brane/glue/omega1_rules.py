"""
Tests of combinatorial rules for the gluing maps of A_n, omega_1 (all colours
distinct).  A state is [A decreasing] W1 [B increasing] Wn [C decreasing];
S_k has A in {1..k}, C in {k+1..n}.

Rule 1 (support, k - l = 1): the components S_k -> S_(k-1) are exactly the
single moves "one strand crosses one red strand from left to right"
(A -> B or B -> C) whose result is a vertex of S_(k-1).

Rule 2 (sign, k - l = 1): searched among parities of simple counts.

    python3 glue/omega1_rules.py glue/data/A2_w1.json glue/data/A3_w1.json ...
"""
import sys
import json
import itertools
from collections import defaultdict


def regions(state):
    out, r = {}, 0
    for x in state:
        if x[0] == "W":
            r += 1
        else:
            out[int(x[1:])] = "ABC"[r]
    return out


def load(path):
    D = json.load(open(path))
    n = D["cartan_type"][1]
    comps = {}
    for t in D["terms"]:
        k, l = t["src_envelope"], t["tgt_envelope"]
        if k == l:
            continue
        rs, rt = regions(t["left_state"]), regions(t["right_state"])
        key = (k, l, tuple(sorted(rs.items())), tuple(sorted(rt.items())))
        assert key not in comps
        comps[key] = t
    return D, n, comps


def vertices(n, k):
    """all (region map) of S_k"""
    for regs in itertools.product("ABC", repeat=n):
        reg = dict(zip(range(1, n + 1), regs))
        if all(c <= k for c in reg if reg[c] == "A") and all(c > k for c in reg if reg[c] == "C"):
            yield reg


def rule1(D, n, comps):
    pred = set()
    for k in range(1, n + 1):
        for reg in vertices(n, k):
            for c in reg:
                for a, b in (("A", "B"), ("B", "C")):
                    if reg[c] != a:
                        continue
                    new = dict(reg)
                    new[c] = b
                    if all(x <= k - 1 for x in new if new[x] == "A") and all(x > k - 1 for x in new if new[x] == "C"):
                        pred.add((k, k - 1, tuple(sorted(reg.items())), tuple(sorted(new.items()))))
    actual = set(key for key in comps if key[0] - key[1] == 1)
    return len(actual), len(pred), len(actual - pred), len(pred - actual)


def features(n, k, reg, c, move):
    A = [x for x in reg if reg[x] == "A"]
    B = [x for x in reg if reg[x] == "B"]
    C = [x for x in reg if reg[x] == "C"]
    return {
        "B<c": sum(1 for x in B if x < c),
        "B>c": sum(1 for x in B if x > c),
        "C<c": sum(1 for x in C if x < c),
        "C>c": sum(1 for x in C if x > c),
        "A<c": sum(1 for x in A if x < c),
        "A>c": sum(1 for x in A if x > c),
        "|B|": len(B),
        "|C|": len(C),
        "|A|": len(A),
        "k": k,
        "c": c,
        "n": n,
        "1": 1,
    }


def rule2(datasets):
    rows = defaultdict(list)
    for D, n, comps in datasets:
        for (k, l, rs, rt), t in comps.items():
            if k - l != 1:
                continue
            rs, rt = dict(rs), dict(rt)
            (c,) = [x for x in rs if rs[x] != rt[x]]
            move = rs[c] + rt[c]
            sgn = 1 if t["coeff"] == "1" else -1 if t["coeff"] == "-1" else None
            rows[move].append((features(n, k, rs, c, move), sgn))
    names = list(next(iter(rows.values()))[0][0].keys())
    found = {}
    for move, data in rows.items():
        sols = []
        for r in range(0, 5):
            for subset in itertools.combinations(names, r):
                if all((-1) ** sum(f[x] for x in subset) == s for f, s in data):
                    sols.append(subset)
            if sols:
                break
        found[move] = (len(data), sols)
    return found


if __name__ == "__main__":
    datasets = [load(p) for p in sys.argv[1:]]
    for (D, n, comps), p in zip(datasets, sys.argv[1:]):
        a, b, miss, extra = rule1(D, n, comps)
        print("{}: rule 1 (k-l=1 components = single rightward moves): actual {}, predicted {}, "
              "actual-not-predicted {}, predicted-not-actual {}".format(D["case"], a, b, miss, extra))
    for move, (count, sols) in rule2(datasets).items():
        print("sign of {} moves ({} components): minimal parity rules (-1)^sum(...) that fit all data: {}".format(
            move, count, sols[:12]))
