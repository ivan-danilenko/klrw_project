"""
Referee check R02: compare a glued complex produced by the real gluing procedure (glue/glue.py, i.e.
standard_ebranes with klrw's Hom/Ext machinery; dump written by R02_glue_run.py or the repository
dumps glue/data/A<n>_w1.json) with the closed formula as implemented in R02_formula.py.

    python3 proofs/checks/R02_compare_glue.py DUMP.json

Compared: the set of (source, target) pairs; signs, with the gauge (-1)^{k(k-1)/2 + l(l-1)/2} of GLUING.md;
the coefficient monomials; absence of dots; reduced length of every word (= number of inverted pairs);
the vertex grading shifts of standard_ebranes against Def. 6.2 (Delta) up to a global constant (and sign).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
F = importlib.import_module('R02_formula')


def main(path):
    D = json.load(open(path))
    n = D['cartan_type'][1]
    P = D['projectives']

    def region(p):
        st = p['state']
        i1 = st.index('W1'); i2 = len(st) - 1 - st[::-1].index('W%d' % n)
        k = p['envelope']
        A = set(int(x[1:]) for x in st[:i1]); C = set(int(x[1:]) for x in st[i2 + 1:])
        s = []
        for c in range(1, n + 1):
            s.append('A' if c in A else 'C' if c in C else ('L' if c <= k else 'R'))
        v = (k, tuple(s))
        assert F.seq(n, v) == tuple(0 if x == 'W1' and j == i1 else (n + 1) if j == i2 else
                                    (int(x[1:]) if x[0] == 'V' else None) for j, x in enumerate(st)), (st, v)
        return v
    V = [region(p) for p in P]
    assert len(set(V)) == len(V) == (n + 1) * 2 ** n
    pname = lambda nm: ('u_1' if nm == 'u1' else 'u_%d' % n if nm == 'un' else 't_' + nm[1:].replace(',', '_'))
    actual = {}
    dots = 0; nonreduced = 0
    for t in D['terms']:
        v, w = V[t['source']], V[t['target']]
        assert (v, w) not in actual
        actual[(v, w)] = (int(t['coeff']), {a: b for a, b in t['params'].items() if b})
        dots += bool(t['dots'])
        sv, sw = F.seq(n, v), F.seq(n, w)
        pv = {x: i for i, x in enumerate(sv)}; pw = {x: i for i, x in enumerate(sw)}
        inv = sum(1 for a in sv for b in sv if pv[a] < pv[b] and pw[a] > pw[b])
        nonreduced += (len(t['word']) != inv)
    TT = F.all_terms(n)
    pred = {}
    for v, lst in TT.items():
        for w, tag, sg, e in lst:
            pred[(v, w)] = (sg, {pname(a): b for a, b in e.items()}, tag)
    missing = set(pred) - set(actual); extra = set(actual) - set(pred)
    bad_sign = bad_coef = 0
    for key in set(pred) & set(actual):
        (v, w) = key
        k, l = v[0], w[0]
        gauge = (-1) ** (k * (k - 1) // 2 + l * (l - 1) // 2)
        if actual[key][0] != gauge * pred[key][0]:
            bad_sign += 1
        merged = {}
        for a, b in pred[key][1].items():
            merged[a] = merged.get(a, 0) + b
        merged = {a: b for a, b in merged.items() if b}
        if actual[key][1] != merged:
            bad_coef += 1
    # degrees: Delta(v) in the dump's symbols
    def edge_key(x):
        if x == 0:
            return 'V1--W1'
        if x == n:
            return [kk for kk in ('W%d--V%d' % (n, n), 'V%d--W%d' % (n, n))]
        return [kk for kk in ('V%d--V%d' % (x, x + 1), 'V%d--V%d' % (x + 1, x))]
    keys = set(kk for p in P for kk in p['equivariant_degree_dict'])
    def ek(x):
        c = edge_key(x)
        if isinstance(c, str):
            return c
        hit = [kk for kk in c if kk in keys]
        return hit[0] if hit else c[0]
    def Delta(v):
        k, s = v
        o = F.obits(n, v)
        out = {}
        for x in range(n + 1):
            m = o[x] + 2 * (x <= k - 1) + 2 * (1 <= x <= n and s[x - 1] == 'C')
            for kk in ('0', ek(x)) if n > 1 else ('0', 'V1--W1'):
                out[kk] = out.get(kk, 0) + m
        return out
    plus, minus = set(), set()
    for v, p in zip(V, P):
        dv = Delta(v); ev = p['equivariant_degree_dict']
        ks = set(dv) | set(ev)
        plus.add(tuple(sorted((kk, ev.get(kk, 0) + dv.get(kk, 0)) for kk in ks)))
        minus.add(tuple(sorted((kk, ev.get(kk, 0) - dv.get(kk, 0)) for kk in ks)))
    ext = [s['hom_dims'] for s in D['steps']]
    print('n=%d: dump %d terms, formula %d; missing %d, extra %d, wrong sign %d, wrong coefficient %d, '
          'terms with dots %d, non-reduced words %d; equivariant shift = -Delta + const: %s, = +Delta + const: %s'
          % (n, len(actual), len(pred), len(missing), len(extra), bad_sign, bad_coef, dots, nonreduced,
             len(plus) == 1, len(minus) == 1))
    print('   Hom^0 per step (graded maps / cycles / boundaries / Ext):',
          [(e['graded_maps'], e['cycles'], e['boundaries'], e['ext']) for e in ext])
    # predicted number of graded maps at step k = number of formula terms out of S_k
    cnt = [sum(1 for v, lst in TT.items() if v[0] == k for w, *_ in lst if w[0] < k) for k in range(1, n + 1)]
    print('   formula terms out of S_k into lower envelopes:', cnt)


if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)
