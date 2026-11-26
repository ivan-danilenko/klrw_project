"""Tabulate the terms of the glued complex for A_n omega_1 as moves of colours between the three regions
A (left of W_1), B (between the red strands), C (right of W_n).   python3 glue/omega1_table.py N [dmin]"""
import json, sys
from collections import defaultdict

def load(n):
    d = json.load(open("glue/data/A%d_w1.json" % n))
    P = d["projectives"]
    def region(p):
        st = p["state"]; i1 = st.index("W1"); i2 = st.index("W%d" % n)
        A = frozenset(int(x[1:]) for x in st[:i1]); C = frozenset(int(x[1:]) for x in st[i2 + 1:])
        return (p["envelope"], A, C)
    V = [region(p) for p in P]
    terms = []
    for t in d["terms"]:
        terms.append((V[t["source"]], V[t["target"]], dict(t["params"]), int(t["coeff"]), len(t["word"])))
    return d, P, V, terms

def fmt(v, n):
    k, A, C = v
    return "S%d[%s|%s|%s]" % (k, "".join(map(str, sorted(A))) or "-",
                              "".join(str(c) for c in range(1, n + 1) if c not in A and c not in C) or "-",
                              "".join(map(str, sorted(C))) or "-")

if __name__ == "__main__":
    n = int(sys.argv[1]); dmin = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    d, P, V, terms = load(n)
    for (v, w, params, coeff, length) in sorted(terms, key=lambda t: (t[0][0] - t[1][0], t[0][0], sorted(t[0][1]), sorted(t[0][2]))):
        dk = v[0] - w[0]
        if dk < dmin:
            continue
        mono = " ".join("%s^%d" % kv for kv in sorted(params.items())) or "1"
        print("d=%d  %-16s -> %-16s  %+d  len %d  %s" % (dk, fmt(v, n), fmt(w, n), coeff, length, mono))
