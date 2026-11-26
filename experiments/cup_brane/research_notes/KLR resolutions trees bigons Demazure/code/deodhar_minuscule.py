"""
For each Lambda-minuscule element w (one per weight mu = w Lambda of a minuscule rep),
test whether the Kazhdan-Lusztig basis element equals the product of the C'_s along a
reduced word (Deodhar / 'tight' property; Billey-Warrington, Billey-Jones).
Soergel normalisation: H_s^2 = 1 + (v^-1 - v) H_s, KL generator Hs + v.
Product is KL basis element  <=>  every coefficient of H_y (y != w) lies in v Z[v].
Group elements are encoded by mu = x^{-1} rho in fundamental-weight coordinates.
"""
import sys
from collections import defaultdict

def cartan(typ, n):
    A = [[0]*n for _ in range(n)]
    for i in range(n): A[i][i] = 2
    def edge(i, j):
        A[i][j] = A[j][i] = -1
    if typ == 'A':
        for i in range(n-1): edge(i, i+1)
    elif typ == 'D':   # Bourbaki: 1-2-...-(n-2), (n-2)-(n-1), (n-2)-n
        for i in range(n-2): edge(i, i+1)
        edge(n-3, n-1)
    elif typ == 'E':   # Bourbaki: 1-3-4-5-6(-7-8), 2-4
        edge(0, 2); edge(1, 3)
        for i in range(2, n-1): edge(i, i+1)
    return A

def reflect(mu, s, A):
    c = mu[s]
    if c == 0: return mu
    return tuple(mu[j] - c*A[s][j] for j in range(len(mu)))

def minuscule_words(A, k):
    n = len(A)
    lam = tuple(1 if j == k else 0 for j in range(n))
    words = {lam: ()}
    frontier = [lam]
    while frontier:
        new = []
        for mu in frontier:
            for i in range(n):
                if mu[i] == 1:
                    nu = reflect(mu, i, A)
                    if nu not in words:
                        words[nu] = (i,) + words[mu]   # w' = s_i w
                        new.append(nu)
        frontier = new
    return words  # weight -> reduced word (left to right = product order)

def addpoly(p, q, shift=0, sgn=1):
    for e, c in q.items():
        p[e+shift] = p.get(e+shift, 0) + sgn*c
        if p[e+shift] == 0: del p[e+shift]

def is_tight(A, word):
    n = len(A)
    rho = tuple([1]*n)
    elt = {rho: {0: 1}}          # H_e
    for s in word:
        new = defaultdict(dict)
        for mu, p in elt.items():
            nu = reflect(mu, s, A)
            if mu[s] > 0:        # xs > x : H_x H_s = H_xs ; plus v H_x
                addpoly(new[nu], p)
                addpoly(new[mu], p, shift=1)
            else:                # xs < x : H_x(H_s+v) = H_xs + v^-1 H_x
                addpoly(new[nu], p)
                addpoly(new[mu], p, shift=-1)
        elt = {m: q for m, q in new.items() if q}
    # identify w: the key reached by the full word (top term has coefficient 1 = v^0)
    w = rho
    for s in word: w = reflect(w, s, A)
    bad = []
    for mu, p in elt.items():
        if mu == w:
            assert p == {0: 1}, p
            continue
        if min(p) <= 0:
            bad.append((mu, p))
    return len(bad) == 0, len(elt), bad

def run(typ, n, k, verbose=True):
    A = cartan(typ, n)
    words = minuscule_words(A, k)
    res = {}
    for mu, wd in sorted(words.items(), key=lambda t: len(t[1])):
        ok, size, bad = is_tight(A, wd)
        res[mu] = (len(wd), ok)
    nonD = [mu for mu in res if not res[mu][1]]
    lens = sorted(res[mu][0] for mu in nonD)
    maxlen = max(r[0] for r in res.values())
    print(f"{typ}{n}, omega_{k+1}: dim={len(words)}, |P|={maxlen}, non-Deodhar weights={len(nonD)}, their heap sizes={lens}")
    return res, words, A

if __name__ == '__main__':
    for (t, n, k) in [('A',7,3),('A',8,2),('A',8,3),('A',9,3),('A',9,4),('D',4,0),('D',4,3),('D',5,4),('D',6,0),('D',7,0),('D',6,5),('E',6,0)]:
        run(t, n, k)
