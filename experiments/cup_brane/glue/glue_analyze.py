"""
Catalogue of the gluing-map components of glued complexes dumped by glue.py
(pure Python).

    python3 glue/glue_analyze.py glue/data/*.json
"""
import sys
import json
import itertools
from collections import Counter, defaultdict


def mono(params):
    return "*".join(n if p == 1 else "{}^{}".format(n, p) for n, p in sorted(params.items())) or "1"


def crossings(t):
    """list of (left strand name, right strand name) for every crossing of the diagram"""
    st = list(t["left_state"])
    out = []
    for letter in t["word"]:
        a, b = st[letter - 1], st[letter]
        out.append((a, b))
        st[letter - 1], st[letter] = b, a
    assert st == t["right_state"]
    return out


def crossing_profile(t):
    c = Counter()
    for a, b in crossings(t):
        if a[0] == "W" or b[0] == "W":
            c["V-W same colour" if a[1:] == b[1:] else "V-W other colour"] += 1
        elif a == b:
            c["V-V same colour"] += 1
        else:
            c["V-V different colour"] += 1
    return c


def bellman_ford(n, edges):
    """edges (u, v, w): constraint a_v - a_u <= w.  Feasible?"""
    dist = [0] * n
    for _ in range(n):
        changed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            return True
    return False


def rescaling_analysis(D, terms, variables):
    """
    Rescaling all projectives of envelope k by a unit monomial m_k multiplies a
    component from envelope k to envelope l by m_l / m_k (and leaves the
    differentials inside the envelopes unchanged).  For every variable x, with
    a_k = exponent of x in m_k, the new exponent is e + a_l - a_k.
      zero:    can all x-exponents be made 0 ?
      nonneg:  can all x-exponents be made >= 0 ?
      nonpos:  can all x-exponents be made <= 0 ?
    """
    n = len(D["envelopes"])
    out = {}
    for x in variables:
        if x.startswith("x_"):
            continue
        es = [(t["src_envelope"], t["tgt_envelope"], t["params"].get(x, 0)) for t in terms]
        if not any(e for _, _, e in es):
            continue
        # e + a_l - a_k >= 0  <=>  a_k - a_l <= e   (edge l -> k, weight e)
        nonneg = bellman_ford(n, [(l, k, e) for k, l, e in es])
        # e + a_l - a_k <= 0  <=>  a_l - a_k <= -e  (edge k -> l, weight -e)
        nonpos = bellman_ford(n, [(k, l, -e) for k, l, e in es])
        zero = bellman_ford(n, [(l, k, e) for k, l, e in es] + [(k, l, -e) for k, l, e in es])
        spread = defaultdict(set)
        for k, l, e in es:
            spread[k, l].add(e)
        out[x] = dict(zero=zero, nonneg=nonneg, nonpos=nonpos,
                      max_spread_within_a_pair=max(max(s) - min(s) for s in spread.values()))
    return out


def bellman_ford_solution(n, edges):
    dist = [0] * n
    for _ in range(n + 1):
        changed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            return dist
    return None


def nonneg_rescaling(D, terms, variables):
    """
    Per-envelope rescaling (a_0 = 0, minimal total exponent) making every exponent of every
    variable non-negative (per variable independent difference constraints).
    Returns (rescaling per envelope, rescaled terms) or None if infeasible.
    """
    n = len(D["envelopes"])
    resc = [dict() for _ in range(n)]
    for x in variables:
        if x.startswith("x_"):
            continue
        es = [(t["src_envelope"], t["tgt_envelope"], t["params"].get(x, 0)) for t in terms]
        if not any(e for _, _, e in es):
            continue
        if bellman_ford_solution(n, [(l, k, e) for k, l, e in es]) is None:
            return None
        # among all feasible rescalings take one minimising the total exponent
        # (difference constraints: the LP is totally unimodular, so integral)
        import numpy as np
        from scipy.optimize import linprog
        c = np.zeros(n)
        A, b = [], []
        for k, l, e in es:
            c[l] += 1
            c[k] -= 1
            row = np.zeros(n)
            row[k] += 1
            row[l] -= 1          # a_k - a_l <= e
            A.append(row)
            b.append(e)
        bounds = [(0, 0)] + [(None, None)] * (n - 1)
        res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
        assert res.status == 0, res.message
        a = [int(round(v)) for v in res.x]
        assert all(e + a[l] - a[k] >= 0 for k, l, e in es)
        for k in range(n):
            if a[k]:
                resc[k][x] = a[k]
    new = []
    for t in terms:
        k, l = t["src_envelope"], t["tgt_envelope"]
        pr = dict(t["params"])
        for x in set(resc[k]) | set(resc[l]):
            pr[x] = pr.get(x, 0) + resc[l].get(x, 0) - resc[k].get(x, 0)
        pr = {x: e for x, e in pr.items() if e}
        assert all(e >= 0 for e in pr.values())
        new.append(dict(t, params=pr))
    return resc, new


def analyze(path):
    D = json.load(open(path))
    P = D["projectives"]
    env = D["envelopes"]
    T = D["terms"]
    glue = [t for t in T if t["src_envelope"] != t["tgt_envelope"]]
    print("=" * 100)
    print("case {}  ({} envelopes, {} moving strands, {} projectives, {} terms of d, {} of them gluing; {:.1f}s)".format(
        D["case"], len(env), env[0]["n_strands"], len(P), len(T), len(glue), D["timings"]["total"]))
    for e in env:
        print("   S_{}: {:<34} left {:<14} right {:<14} {:>4} terms in d".format(
            e["index"], e["weight"], str(e["left_word"]), str(e["right_word"]), e["n_differential_terms"]))
    print(" steps (Hom^0 of S_k[shift] -> E_(k-1)):")
    for s in D["steps"]:
        rem = {k: v for k, v in s.get("removable_by_homotopy", {}).items() if v is not None}
        print("   k={}: graded maps {:>4}, cycles {:>3}, boundaries {:>3}, Ext {}; map has {:>3} terms, {:>3} rigid; "
              "removable by homotopy: {}".format(
                  s["step"], s["hom_dims"]["graded_maps"], s["hom_dims"]["cycles"], s["hom_dims"]["boundaries"],
                  s["hom_dims"]["ext"], s["n_map_terms"], s.get("n_rigid_terms", -1),
                  "none" if not any(rem.values()) else [k for k, v in rem.items() if v]))
    # pairs
    pairs = Counter((t["src_envelope"], t["tgt_envelope"]) for t in glue)
    print(" gluing terms by (k -> l):", dict(sorted(pairs.items())))
    by_diff = defaultdict(set)
    for (k, l), c in pairs.items():
        by_diff[k - l].add(c)
    print(" ... by k - l:", {d: sorted(v) for d, v in sorted(by_diff.items())})
    entries = Counter((t["source"], t["target"]) for t in glue)
    print(" matrix entries of the gluing part: {}, with more than one term: {}".format(
        len(entries), sum(1 for c in entries.values() if c > 1)))
    print(" coefficients:", dict(Counter(t["coeff"] for t in glue)))
    print(" empty braid word:", sum(1 for t in glue if not t["word"]),
          "  (identity: empty word, no parameters, no dots:",
          sum(1 for t in glue if not t["word"] and not t["params"] and not t["dots"]), ")")
    print(" dots:", dict(Counter(mono(t["dots"]) for t in glue)))
    # variables
    var_hist = defaultdict(Counter)
    for t in glue:
        for x in D["variables"]:
            if not x.startswith("x_"):
                var_hist[x][t["params"].get(x, 0)] += 1
    print(" exponent histograms over the gluing terms (variable: {exponent: count}):")
    for x, h in var_hist.items():
        if set(h) != {0}:
            print("   {:<6} {}".format(x, dict(sorted(h.items()))))
    print(" variables never occurring in gluing terms:",
          [x for x, h in var_hist.items() if set(h) == {0}])
    print(" terms with: u {}, u^+ {}, u^- {}, r^- {}, r^+ {}, t^- {}, any negative exponent {}, no parameter {}".format(
        sum(1 for t in glue if any(k.startswith("u_") for k in t["params"])),
        sum(1 for t in glue if any(k.startswith("u_") and v > 0 for k, v in t["params"].items())),
        sum(1 for t in glue if any(k.startswith("u_") and v < 0 for k, v in t["params"].items())),
        sum(1 for t in glue if any(k.startswith("r_") and v < 0 for k, v in t["params"].items())),
        sum(1 for t in glue if any(k.startswith("r_") and v > 0 for k, v in t["params"].items())),
        sum(1 for t in glue if any(k.startswith("t_") and v < 0 for k, v in t["params"].items())),
        sum(1 for t in glue if any(v < 0 for v in t["params"].values())),
        sum(1 for t in glue if not t["params"])))
    print(" distinct parameter monomials: {}".format(len(set(mono(t["params"]) for t in glue))))
    for m, c in Counter(mono(t["params"]) for t in glue).most_common():
        print("   {:>4} x {}".format(c, m))
    prof = Counter()
    for t in glue:
        for k, v in crossing_profile(t).items():
            prof[k, v] += 1
    print(" crossing profile of the gluing diagrams ((type, number): count):", dict(sorted(prof.items())))
    print(" word lengths:", dict(sorted(Counter(len(t["word"]) for t in glue).items())))
    # rigid terms (homotopy invariant given E_(k-1))
    rigid = [t for s in D["steps"] for t in s.get("map_terms", []) if t.get("rigid")]
    nonrigid = [t for s in D["steps"] for t in s.get("map_terms", []) if not t.get("rigid")]
    if nonrigid:
        print(" rigid terms: {} / {}; their monomials: {}".format(
            len(rigid), len(rigid) + len(nonrigid), dict(Counter(mono(t["params"]) for t in rigid))))
        print("   rigid terms with u: {}, with r^-1: {}, with dots: {}".format(
            sum(1 for t in rigid if any(k.startswith("u_") for k in t["params"])),
            sum(1 for t in rigid if any(k.startswith("r_") and v < 0 for k, v in t["params"].items())),
            sum(1 for t in rigid if t["dots"])))
    res = nonneg_rescaling(D, glue, D["variables"])
    if res is None:
        print(" no per-envelope rescaling makes all exponents non-negative")
    else:
        resc, new = res
        print(" after a per-envelope rescaling making all exponents >= 0 with minimal total degree "
              "(S_k multiplied by m_k, component k->l by m_l/m_k):")
        print("   m_k:", [mono(r) for r in resc])
        for m, c in Counter(mono(t["params"]) for t in new).most_common():
            print("   {:>4} x {}".format(c, m))
    print(" per-envelope rescaling by unit monomials (can a variable be removed / made non-negative?):")
    for x, r in rescaling_analysis(D, glue, D["variables"]).items():
        print("   {:<6} zero {:<5} nonneg {:<5} nonpos {:<5} max spread within one (k,l) {}".format(
            x, str(r["zero"]), str(r["nonneg"]), str(r["nonpos"]), r["max_spread_within_a_pair"]))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
