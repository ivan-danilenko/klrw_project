"""
Referee check 1: d^2 = 0 for the tree complex, with MY implementation of the formula (R05_lib.my_terms)
and RANDOM reduced words for every term (a fresh one per use of the term, unless mode 'basis').

    sage -python R05_d2.py all   TYPE RANK MINLEN MAXLEN MAXMULT [NPROC] [reds: all|one]
    sage -python R05_d2.py word  TYPE RANK RED WORD
    sage -python R05_d2.py sample TYPE RANK NWORDS LEN NVERT SEED [MAXMULT] [pattern]

'all': every word with colour multiplicities <= MAXMULT and MINLEN <= length <= MAXLEN, every red colour.
'sample': random words of given length, d^2 e_I = 0 at NVERT random vertices I (cube never built).
pattern: 'nb' = force two equal neighbouring letters; 'red' = both occurrences of some colour adjacent
to the red colour placed around an occurrence of the red colour (c k c), and red colour doubled.
"""
import sys, itertools, random, time
from multiprocessing import Pool
from R05_lib import *


def d2_vertex(S, I, rng, cache=None):
    """Returns (number of products, list of targets K with (d^2)_{I,K} != 0)."""
    arrI = S.arr(I)
    total = {}
    nprod = 0
    for t1 in S.my_terms(I):
        x = S.coef(t1) * S.el(arrI, S.beta_word(I, t1.j, t1.F, rng))
        arrJ = S.arr(t1.J)
        for t2 in S.my_terms(t1.J):
            y = S.coef(t2) * S.el(arrJ, S.beta_word(t1.J, t2.j, t2.F, rng))
            p = x * y
            nprod += 1
            total[t2.J] = total[t2.J] + p if t2.J in total else p
    return nprod, [K for K, v in total.items() if not v.is_zero()]


def run_word(args):
    ct, red, word, seed = args
    rng = random.Random(seed)
    S = Setup(ct, red, word)
    m = len(word)
    nprod = 0
    bad = []
    for size in range(2, m + 1):
        for I in itertools.combinations(range(1, m + 1), size):
            n, b = d2_vertex(S, frozenset(I), rng)
            nprod += n
            if b:
                bad.append((I, b))
    return ct, red, word, nprod, bad


def words_with_mult(rank, minlen, maxlen, maxmult):
    for L in range(minlen, maxlen + 1):
        for w in itertools.product(range(1, rank + 1), repeat=L):
            if all(w.count(c) <= maxmult for c in set(w)):
                yield w


if __name__ == "__main__":
    mode = sys.argv[1]
    ct = [sys.argv[2], int(sys.argv[3])]
    rank = ct[1]
    t0 = time.time()
    if mode == "word":
        red = int(sys.argv[4]); word = tuple(int(c) for c in sys.argv[5])
        print(run_word((ct, red, word, 0)))
    elif mode == "all":
        minlen, maxlen, maxmult = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
        nproc = int(sys.argv[7]) if len(sys.argv) > 7 else 10
        reds = sys.argv[8] if len(sys.argv) > 8 else "all"
        jobs = []
        for w in words_with_mult(rank, minlen, maxlen, maxmult):
            for red in (range(1, rank + 1) if reds == "all" else [1 + (hash(w) % rank)]):
                jobs.append((ct, red, w, len(jobs)))
        nw = nprod = nfail = 0
        with Pool(nproc) as pool:
            for ct_, red, word, n, bad in pool.imap_unordered(run_word, jobs, chunksize=4):
                nw += 1; nprod += n
                if bad:
                    nfail += 1
                    print("FAIL", ct_, "red", red, "word", "".join(map(str, word)), bad[:3], flush=True)
                if nw % 500 == 0:
                    print("  ... %d / %d (word, red) cases, %d products, %d failures [%.0fs]" % (nw, len(jobs), nprod, nfail, time.time() - t0), flush=True)
        print("RESULT all %s%d len %d..%d mult<=%d reds=%s: %d (word, red) cases, %d products, %d FAILURES [%.0fs]" % (
            ct[0], rank, minlen, maxlen, maxmult, reds, nw, nprod, nfail, time.time() - t0), flush=True)
    elif mode == "sample":
        nwords, L, nvert, seed = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
        maxmult = int(sys.argv[8]) if len(sys.argv) > 8 else 2
        pattern = sys.argv[9] if len(sys.argv) > 9 else ""
        rng = random.Random(seed)
        E = dynkin_edges(ct)
        nfail = nprod = nv = 0
        done = 0
        while done < nwords:
            red = rng.randint(1, rank)
            # random word with multiplicities <= maxmult
            pool_ = [c for c in range(1, rank + 1) for _ in range(maxmult)]
            if L > len(pool_):
                raise SystemExit("length too large")
            word = rng.sample(pool_, L)
            if pattern == "nb":
                doubles = [c for c in set(word) if word.count(c) == 2]
                if not doubles:
                    continue
                c = rng.choice(doubles)
                rest = [x for x in word if x != c]
                p = rng.randint(0, len(rest))
                word = rest[:p] + [c, c] + rest[p:]
            elif pattern == "red":
                nbrs = [c for c in range(1, rank + 1) if frozenset((c, red)) in E]
                c = rng.choice(nbrs)
                rest = [x for x in word if x not in (c, red)]
                p = rng.randint(0, len(rest))
                q = rng.randint(0, len(rest))
                rest = rest[:p] + [c, red, c] + rest[p:]
                q = rng.randint(0, len(rest))
                rest = rest[:q] + [red] + rest[q:]
                word = rest[:L] if len(rest) >= L else rest
                if any(word.count(x) > maxmult for x in set(word)):
                    continue
            word = tuple(word)
            m = len(word)
            S = Setup(ct, red, word)
            bad_here = []
            t1 = time.time()
            for _ in range(nvert):
                size = rng.randint(2, m)
                I = frozenset(rng.sample(range(1, m + 1), size))
                n, b = d2_vertex(S, I, rng)
                nprod += n; nv += 1
                if b:
                    bad_here.append((sorted(I), b))
            done += 1
            nfail += bool(bad_here)
            print("  %s%d red %d word %s: %d vertices, ok=%s [%.0fs]" % (ct[0], rank, red, "".join(map(str, word)), nvert, not bad_here, time.time() - t1), flush=True)
            if bad_here:
                print("FAIL", bad_here[:2], flush=True)
        print("RESULT sample %s%d len %d mult<=%d pattern '%s': %d words, %d vertices, %d products, %d FAILURES [%.0fs]" % (
            ct[0], rank, L, maxmult, pattern, done, nv, nprod, nfail, time.time() - t0), flush=True)
