import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code')
from deodhar_minuscule import cartan, reflect, minuscule_words, addpoly
from collections import defaultdict

def product(A, word):
    n = len(A); rho = tuple([1]*n)
    elt = {rho: {0: 1}}
    for s in word:
        new = defaultdict(dict)
        for mu, p in elt.items():
            nu = reflect(mu, s, A)
            addpoly(new[nu], p)
            addpoly(new[mu], p, shift=(1 if mu[s] > 0 else -1))
        elt = {m: q for m, q in new.items() if q}
    return elt

def peel(A, mu):
    """mu = y^{-1} rho -> reduced word of y (left-to-right)"""
    word = []
    mu = list(mu)
    while True:
        s = next((i for i in range(len(mu)) if mu[i] < 0), None)
        if s is None: break
        word.append(s)          # y = (y s) s ; s is rightmost remaining
        mu = list(reflect(tuple(mu), s, A))
    return word[::-1]

def parabolic(A, k, word, chi):
    """project product to M = H (x)_{H_J} chi ; chi='triv' (H_s->v^-1) or 'sign' (H_s-> -v)"""
    n = len(A)
    lam = tuple(1 if j == k else 0 for j in range(n))
    elt = product(A, word)
    out = defaultdict(dict)
    for mu, p in elt.items():
        yw = peel(A, mu)
        ly = len(yw)
        nu = lam
        for s in reversed(yw): nu = reflect(nu, s, A)
        # length of minimal coset rep = height of lam - nu
        out_key = nu
        out[out_key] = out.get(out_key, {})
        out[(nu, ly)] = None
        # store per (nu) : need l(x) = number of steps; compute via words dict later
        out.setdefault(('raw', nu), []).append((ly, p))
    return out

def check(typ, n, k):
    A = cartan(typ, n)
    words = minuscule_words(A, k)
    lx = {nu: len(w) for nu, w in words.items()}
    print(f"{typ}{n} omega_{k+1}")
    for mu_w, wd in sorted(words.items(), key=lambda t: len(t[1])):
        elt = product(A, wd)
        res = {}
        for chi in ('triv', 'sign'):
            coeff = defaultdict(dict)
            for mu, p in elt.items():
                yw = peel(A, mu)
                nu = tuple(1 if j == k else 0 for j in range(n))
                for s in reversed(yw): nu = reflect(nu, s, A)
                lu = len(yw) - lx[nu]
                if chi == 'triv':
                    addpoly(coeff[nu], p, shift=-lu)
                else:
                    addpoly(coeff[nu], p, shift=lu, sgn=(-1)**lu)
            coeff = {a: b for a, b in coeff.items() if b}
            assert coeff[mu_w] == {0: 1}, coeff[mu_w]
            ok = all(min(b) >= 1 for a, b in coeff.items() if a != mu_w)
            res[chi] = ok
        if not (res['triv'] and res['sign']) or len(wd) >= max(lx.values()) - 4:
            print(f"  |heap|={len(wd)}: parabolic-triv tight={res['triv']}, parabolic-sign tight={res['sign']}")

for t, n, k in [('A',7,3), ('D',6,5), ('D',5,4), ('E',6,0), ('A',8,2)]:
    check(t, n, k)
