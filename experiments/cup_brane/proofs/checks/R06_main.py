"""
R06: own checks of Prop. 4.3 (d_Phi^2 = 0), Theorem 6.2 (d_Phi G = G d_tree, G from the text of
Definition 6.2), Corollary 6.3 (d_tree^2 = 0), with perturbations.

    sage -python R06_main.py word TYPE RANK RED WORD [variant ...]
    sage -python R06_main.py all TYPE RANK MAXLEN [MINLEN] [part/nparts]   all words with multiplicity <= 2, every red colour
    sage -python R06_main.py random TYPE RANK N MINLEN MAXLEN SEED [variant]
variants: text noQ LL RR plusr nosign
"""
import sys, time, random, itertools
from collections import Counter
from R06_lib import Setup


def run_word(ct, red, w, variants=("text",), check_d2=True, vertices=None, quiet=False):
    t0 = time.time()
    S = Setup(ct, red, w)
    res = {}
    V = list(S.vertices()) if vertices is None else vertices
    # d_Phi^2 = 0
    if check_d2:
        bad2 = 0
        for I in V:
            tot = {}
            for J, x in S.dPhi(I).items():
                for K, y in S.dPhi(J).items():
                    tot[K] = tot.get(K, S.K.zero()) + x * y
            bad2 += any(not v.is_zero() for v in tot.values())
        res["dPhi2"] = bad2
        # d_tree^2 = 0 (Corollary 6.3, direct)
        badt = 0
        for I in V:
            tot = {}
            for J, x in S.dtree(I).items():
                for K, y in S.dtree(J).items():
                    tot[K] = tot.get(K, S.K.zero()) + x * y
            badt += any(not v.is_zero() for v in tot.values())
        res["dtree2"] = badt
    for var in variants:
        nb = 0
        ncomp = 0
        for I in V:
            bad, n = S.check_vertex(I, var)
            nb += len(bad)
            ncomp += n
        res[var] = (nb, ncomp)
    if not quiet:
        print("%s red %d word %s: %s  [%.1fs]" % ("".join(map(str, ct)), red, "".join(map(str, w)), res,
                                               time.time() - t0), flush=True)
    return res


def words_mult2(colours, maxlen, minlen=1, maxmult=2):
    out = []
    for L in range(minlen, maxlen + 1):
        for w in itertools.product(colours, repeat=L):
            if max(Counter(w).values()) <= maxmult:
                out.append(w)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "word":
        ct = [sys.argv[2], int(sys.argv[3])]
        red = int(sys.argv[4])
        w = tuple(int(c) for c in sys.argv[5])
        variants = sys.argv[6:] or ["text"]
        run_word(ct, red, w, variants)
    elif mode == "all":
        ct = [sys.argv[2], int(sys.argv[3])]
        maxlen = int(sys.argv[4])
        minlen = int(sys.argv[5]) if len(sys.argv) > 5 else 1
        part, nparts = (map(int, sys.argv[6].split("/")) if len(sys.argv) > 6 else (0, 1))
        cols = list(range(1, ct[1] + 1))
        W = words_mult2(cols, maxlen, minlen)
        W = W[part::nparts]
        t0 = time.time()
        nfail = 0
        ncases = 0
        for w in W:
            for red in cols:
                res = run_word(ct, red, w, ("text",), check_d2=True, quiet=True)
                ncases += 1
                bad = res["dPhi2"] or res["dtree2"] or res["text"][0]
                if bad:
                    nfail += 1
                    print("FAIL", red, w, res, flush=True)
            if len(w) >= maxlen - 1 and ncases % 50 == 0:
                print("... %d cases, %d failures [%.0fs]" % (ncases, nfail, time.time() - t0), flush=True)
        print("RESULT all %s len %d..%d part %d/%d: %d (word, red) cases, %d failures [%.0fs]" % (
            "".join(map(str, ct)), minlen, maxlen, part, nparts, ncases, nfail, time.time() - t0), flush=True)
    elif mode == "random":
        ct = [sys.argv[2], int(sys.argv[3])]
        N, minlen, maxlen, seed = map(int, sys.argv[4:8])
        variants = sys.argv[8:] or ["text"]
        rng = random.Random(seed)
        cols = list(range(1, ct[1] + 1))
        t0 = time.time()
        nfail = 0
        for k in range(N):
            L = rng.randint(minlen, maxlen)
            while True:
                w = tuple(rng.choice(cols) for _ in range(L))
                if max(Counter(w).values()) <= 2:
                    break
            red = rng.choice(cols)
            res = run_word(ct, red, w, variants, check_d2=True)
            nfail += bool(res["dPhi2"] or res["dtree2"] or res[variants[0]][0])
        print("RESULT random %s N=%d: %d failures [%.0fs]" % ("".join(map(str, ct)), N, nfail, time.time() - t0))
