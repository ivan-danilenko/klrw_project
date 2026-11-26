"""
Homology of the omega_1 I-brane (proofs/02_omega1_ibrane.md, Theorem 8.1), numerical check, pure Python.

    python3 proofs/checks/02_homology.py N_MAX [STATES_PER_N] [SEED]

For a state s (any ordering of the labels 1..n, W_1, W_n with W_1 left of W_n), e_s E is the
complex of free S-modules (S = R[x_1..x_n]) with basis the vertices and entries
c_{vw} Q_{D(s,v,w)} (Lemma 2.3 and F1).  Theorem 8.1 says: H(e_s E) = 0 unless s = s_B
(= W_1 1 2 ... n W_n), and H(e_{s_B} E) = S/(x_1..x_n) in degree 0.  We test two consequences
after specialising the parameters t, u to random non-zero residues mod p:
  (a) at a random point x in F_p^n the specialised complex is exact (for every s);
  (b) at x = 0 the specialised complex has  dim H_i = binom(n, i)  for s = s_B  (Tor of the
      residue field), and is exact for every other s.
"""
import itertools
import os
import random
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
M = importlib.import_module("02_model")

P = 1000003


def rank_mod(rows, ncols):
    """rows: list of dict col -> value mod P."""
    basis = {}
    r = 0
    for row in rows:
        row = {c: v % P for c, v in row.items() if v % P}
        while row:
            piv = min(row)
            if piv in basis:
                b = basis[piv]
                f = row[piv]
                for c, v in b.items():
                    row[c] = (row.get(c, 0) - f * v) % P
                    if not row[c]:
                        del row[c]
            else:
                inv = pow(row[piv], P - 2, P)
                basis[piv] = {c: v * inv % P for c, v in row.items()}
                r += 1
                break
    return r


def main(N, per_n, seed):
    rng = random.Random(seed)
    for n in range(1, N + 1):
        labels = list(range(1, n + 1)) + ["Wa", "Wb"]
        states = [p for p in itertools.permutations(labels) if p.index("Wa") < p.index("Wb")]
        sB = tuple(["Wa"] + list(range(1, n + 1)) + ["Wb"])
        if per_n and len(states) > per_n:
            chosen = rng.sample(states, per_n)
            if sB not in chosen:
                chosen[0] = sB
        else:
            chosen = states
        verts = list(M.vertices(n))
        idx = {}
        byh = {}
        for v in verts:
            h = M.hdeg(n, v)
            byh.setdefault(h, []).append(v)
        for h, lst in byh.items():
            for i, v in enumerate(lst):
                idx[v] = i
        terms = M.all_terms(n)
        params = {}
        def pval(name):
            if name not in params:
                params[name] = rng.randrange(1, P)
            return params[name]
        def mono_val(m):
            val = 1
            for name, e in m:
                x = pval(name)
                val = val * pow(x, e % (P - 1), P) % P
            return val
        bad_a = bad_b = 0
        for s in chosen:
            pos = {lab: i for i, lab in enumerate(s)}
            def obit(x):
                a = "Wa" if x == 0 else x
                b = "Wb" if x == n else x + 1
                return pos[a] < pos[b]
            bs = [obit(x) for x in range(n + 1)]
            for point in ("random", "zero"):
                xs = [rng.randrange(1, P) for _ in range(n)] if point == "random" else [0] * n
                def Qval(x):
                    val = 0
                    if x >= 1:
                        val += pval(M.tname(n, x, x + 1)) * xs[x - 1]
                    if x + 1 <= n:
                        val += pval(M.tname(n, x + 1, x)) * xs[x]
                    return val % P
                mats = {}
                for t in terms:
                    v, w = t.src, t.tgt
                    ov, ow = M.order_bits(n, v), M.order_bits(n, w)
                    val = t.sign * mono_val(t.coeff) % P
                    for x in range(n + 1):
                        if ov[x] != bs[x] and ow[x] == bs[x]:
                            val = val * Qval(x) % P
                    h = M.hdeg(n, v)
                    mats.setdefault(h, {}).setdefault(idx[v], {})[idx[w]] = val
                ranks = {h: rank_mod(list(mats.get(h, {}).values()), len(byh.get(h - 1, []))) for h in byh}
                homology = {h: len(byh[h]) - ranks[h] - ranks.get(h + 1, 0) for h in byh}
                if point == "random":
                    if any(homology.values()):
                        bad_a += 1
                else:
                    expect = {h: (comb(n, h) if s == sB and h <= n else 0) for h in byh}
                    if homology != expect:
                        bad_b += 1
        print("n=%d: %d states tested; (a) not exact at a random point: %d; (b) wrong homology at x = 0: %d"
              % (n, len(chosen), bad_a, bad_b), flush=True)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(N, per, seed)
