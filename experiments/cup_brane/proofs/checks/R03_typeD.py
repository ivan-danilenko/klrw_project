"""
Referee test of Prop. 4.1, Lemma 4.2, Lemma 4.3, Theorem D, Cor. 4.4, Prop. 2.11(c) (D_6), type D.

  python3 R03_typeD.py [NMAX]     (default 9: all strict partitions with parts <= 8)

Half-spin model derived by the referee from the sign-vector picture: weights of V(omega_n) are
1/2(s_1,...,s_n), s_i = +-1, even number of '-'; <mu, alpha_i^vee> = (s_i - s_{i+1})/2 (i<n),
(s_{n-1}+s_n)/2 (i=n).  Row u of the full chain moves the u-th minus sign into position u:
for odd u it is created (with a partner at n) by s_n at positions n-1, n, for even u the partner
is moved from n to n-1 by s_{n-1}; then s_{n-2}, ..., s_u move it left.  Box (u,v), 1<=u<v<=n,
is the step that brings that sign to position n-(v-u) (v-u>=2), resp. the creation/partner step
(v=u+1).  So colour(u,v) = n-(v-u) if v-u>=2, = n if v=u+1 and u odd, = n-1 if v=u+1 and u even.
(This is the labelling of the draft; it is checked below against the numbers game and words.json.)
"""
import sys, os, json, time, itertools
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R03_core import *

HERE = os.path.dirname(os.path.abspath(__file__))


def colour(n, u, v):
    r = v - u
    if r >= 2:
        return n - r
    return n if u % 2 == 1 else n - 1


def strict_partitions(maxpart):
    for r in range(maxpart + 1):
        for S in itertools.combinations(range(maxpart, 0, -1), r):
            yield tuple(S)


def ideal_word(n, ell):
    w = []
    for u, l in enumerate(ell, 1):
        for v in range(u + 1, u + l + 1):
            w.append(colour(n, u, v))
    return w


def fund_to_eps(n, mu):
    """fundamental-weight coordinates (tuple of n) -> epsilon coordinates (Fractions)."""
    eps = [Fraction(0)] * n
    for i in range(1, n + 1):
        c = mu[i - 1]
        if i <= n - 2:
            for t in range(i):
                eps[t] += c
        elif i == n - 1:
            for t in range(n - 1):
                eps[t] += Fraction(c, 2)
            eps[n - 1] -= Fraction(c, 2)
        else:
            for t in range(n):
                eps[t] += Fraction(c, 2)
    return eps


def main(NMAX):
    words = json.load(open(os.path.join(HERE, "..", "..", "words.json")))
    for n in range(4, NMAX + 1):
        t0 = time.time()
        E = edges("D", n); adj = make_adj(E)
        full = ideal_word(n, tuple(range(n - 1, 0, -1)))
        key = "D%d_w%d" % (n, n)
        wj = words.get(key)
        same = (wj[2] == full) if wj else None
        heaps, covers, order = all_weights_with_heaps(E, n, n)
        single = all(len(h) == 1 for h in heaps.values())
        st = dict(ideals=0, cyclic=0, disagree=0, pat_disagree=0, not_chain=0, mult_fail=0, heap_mismatch=0)
        wt = {}
        cyc_set = []
        for ell in strict_partitions(n - 1):
            w = ideal_word(n, ell)
            ok, mu = is_chain_word(w, E, n, n)
            if not ok:
                st["not_chain"] += 1
                continue
            wt[ell] = mu
            if heaps.get(mu) != {heap_key(tuple(w), E)}:
                st["heap_mismatch"] += 1
            st["ideals"] += 1
            T = Tables(w, adj)
            res, _ = conn_search_fast(T)
            cyc = res is not None
            l3 = ell[2] if len(ell) > 2 else 0
            l4 = ell[3] if len(ell) > 3 else 0
            crit = l3 >= 3 and l4 >= 1
            if cyc:
                st["cyclic"] += 1
                cyc_set.append(ell)
            if cyc != crit:
                st["disagree"] += 1
                print("THEOREM D DISAGREE", n, ell, res)
            if has_pattern(w, E) != cyc:
                st["pat_disagree"] += 1
                print("PATTERN DISAGREE", n, ell)
            if not cyc and w and max(w.count(c) for c in set(w)) > 3:
                st["mult_fail"] += 1
            if not cyc and n <= 8 and w:
                # brute force over all configurations as well
                if T.brute() is not None:
                    print("BRUTE FINDS CYCLE", n, ell)
        bij = len(set(wt.values())) == len(wt) == len(heaps)
        # weight order vs inclusion of ideals
        def contains(a, b):
            return all((a[i] if i < len(a) else 0) >= b[i] for i in range(len(b)))
        cont = {ell: [ideal_word(n, ell).count(c) for c in range(1, n + 1)] for ell in wt}
        badord = sum(1 for a in wt for b in wt
                     if contains(a, b) != all(x >= y for x, y in zip(cont[a], cont[b])))
        # unique minimal cyclic ideal
        mins = [a for a in cyc_set if not any(b != a and contains(a, b) for b in cyc_set)]
        line = "D_%d half-spin: full word = words.json %s: %s; chain-word heaps unique: %s; model hits all %d weights bijectively: %s; %s; order mismatches %d" % (
            n, key, same, single, len(heaps), bij, st, badord)
        if mins:
            (mc,) = mins if len(mins) == 1 else (mins,)
            mu = wt[mc]
            eps = fund_to_eps(n, mu)
            sig = "".join("+" if e > 0 else "-" for e in eps)
            line += "; minimal cyclic ideals %s (|I|=%d), mu_c fund=%s eps=1/2(%s), content %s" % (
                mins, sum(mc), mu, sig, cont[mc])
        print(line, "(%.1fs)" % (time.time() - t0), flush=True)
        # vector representation
        heaps1, _, _ = all_weights_with_heaps(E, n, 1)
        nv = 0; mm = 0; cy = 0
        for mu, hs in heaps1.items():
            (h,) = hs
            nv += 1
            if h:
                mm = max(mm, max(h.count(c) for c in set(h)))
            if conn_search_fast(Tables(list(h), adj))[0] is not None:
                cy += 1
        print("   D_%d vector: %d weights, max colour multiplicity %d, cyclic %d" % (n, nv, mm, cy), flush=True)
        if n == 6:
            # Prop 2.11(c): cyclic ignoring blocks
            ib = []
            for ell in wt:
                w = ideal_word(n, ell)
                if conn_search_fast(Tables(w, adj), ignore_blocks=True)[0] is not None and ell not in cyc_set:
                    ib.append((sum(ell), ell))
            print("   D_6: ideals cyclic ignoring blocks but not cyclic:", sorted(ib))
        if n == 4:
            for k in (3, 4):
                hk, _, _ = all_weights_with_heaps(E, 4, k)
                print("   D_4 omega_%d: max multiplicity %d" % (k, max(max(h.count(c) for c in set(h)) for (h,) in hk.values() if h)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 9)
