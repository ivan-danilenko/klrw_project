"""
Referee (second pass) check 1b: d^2 = 0 for the tree formula AS DEFINED IN THE PROJECT, i.e.
formula.Stack(...).terms_from(I) with the klrw basis braid (fast.braid_for, via sample_verify.check_vertex).
Complements R05_d2.py, which uses an independent brute-force implementation of NOTES.md section 2 and
random reduced words.

    sage -python R05b_d2.py all    TYPE RANK MINLEN MAXLEN NPROC [lastonly]
    sage -python R05b_d2.py sample TYPE RANK NWORDS LEN NVERT SEED NPROC [pattern]

'all': every word with colour multiplicities <= 2 and MINLEN <= length <= MAXLEN, every red colour,
       every vertex (full complex).  'lastonly': only words whose last colour occurs twice (the case of
       section 8 at the top step; the other words are covered by Theorem A from shorter words).
'sample': random words with multiplicities <= 2; NVERT random vertices each.  Patterns:
   ''      : uniform;
   'nb'    : some colour's two occurrences are neighbours;
   'red'   : c k c with c adjacent to the red colour k, and k occurring twice;
   'twinred': the two occurrences of the red colour k surround an occurrence of a colour adjacent to k;
   'branch': the last letter is a colour with >= 3 neighbours (families with k = 2, 3 in Lemma 8.4),
             occurring twice, its neighbours occurring twice.
"""
import os, sys, itertools, random, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from cupbrane import make_algebra, Cube          # noqa
from formula import Stack                        # noqa
from sample_verify import check_vertex           # noqa
from degree_model import edges as dynkin_edges   # noqa


def run_word(args):
    ct, red, word, verts = args
    P = [(red, tuple(word))]
    K = make_algebra(ct, P)
    cube = Cube(K, P, build=False)
    st = Stack(ct, word)
    cache = {}
    m = len(word)
    if verts is None:
        verts = [frozenset(I) for k in range(2, m + 1) for I in itertools.combinations(range(1, m + 1), k)]
    nprod = 0
    bad = []
    for I in verts:
        n, _, b, _ = check_vertex(cube, st, frozenset(I), cache)
        nprod += n
        if b:
            bad.append((sorted(I), [sorted(x) for x in b][:3]))
    return ct, red, tuple(word), len(verts), nprod, bad


def words_T1(rank, minlen, maxlen, lastonly):
    for L in range(minlen, maxlen + 1):
        for w in itertools.product(range(1, rank + 1), repeat=L):
            if all(w.count(c) <= 2 for c in set(w)) and (not lastonly or w.count(w[-1]) == 2):
                yield w


def random_word(rng, ct, L, pattern):
    rank = ct[1]
    E = dynkin_edges(ct)
    nb = lambda c: [d for d in range(1, rank + 1) if frozenset((c, d)) in E]
    pool = [c for c in range(1, rank + 1) for _ in range(2)]
    for _ in range(1000):
        red = rng.randint(1, rank)
        w = rng.sample(pool, L)
        if pattern == "":
            return red, w
        if pattern == "nb":
            dbl = [c for c in set(w) if w.count(c) == 2]
            if not dbl:
                continue
            c = rng.choice(dbl)
            rest = [x for x in w if x != c]
            p = rng.randint(0, len(rest))
            return red, rest[:p] + [c, c] + rest[p:]
        if pattern in ("red", "twinred"):
            c = rng.choice(nb(red))
            outer, inner = (c, red) if pattern == "red" else (red, c)
            rest = [x for x in w if x not in (c, red)]
            p = rng.randint(0, len(rest))
            rest = rest[:p] + [outer, inner, outer] + rest[p:]
            q = rng.randint(0, len(rest))
            rest = rest[:q] + [inner] + rest[q:]
            w2 = rest[-L:] if len(rest) > L else rest
            if all(w2.count(x) <= 2 for x in set(w2)):
                return red, w2
            continue
        if pattern == "k2":
            # d1, d2 (neighbours of c) once before and once after a; d3 (third neighbour) after a;
            # the remaining letters random, inserted anywhere before m.
            br = [c for c in range(1, rank + 1) if len(nb(c)) >= 3]
            c = rng.choice(br)
            d1, d2, d3 = rng.sample(nb(c), 3)
            before = [d1, d2]; rng.shuffle(before)
            after = [d1, d2, d3]; rng.shuffle(after)
            used = [c, c, d1, d1, d2, d2, d3]
            rest_pool = [x for x in pool if x not in (c, d1, d2, d3)] + [d3]
            if L < 7 or L > 7 + len(rest_pool):
                raise SystemExit("length incompatible with pattern 'k2'")
            body = before + [c] + after
            for x in rng.sample(rest_pool, L - 7):
                p = rng.randint(0, len(body))
                body = body[:p] + [x] + body[p:]
            return red, body + [c]
        if pattern == "branch":
            br = [c for c in range(1, rank + 1) if len(nb(c)) >= 3]
            c = rng.choice(br)
            nbs = [d for d in nb(c) for _ in range(2)]
            if L < 2 + len(nbs):            # one neighbour may occur only once (it can be the mover N)
                nbs.remove(rng.choice(nb(c)))
            if L < 2 + len(nbs) or L > 2 * rank:
                raise SystemExit("length incompatible with pattern 'branch'")
            others = [x for x in pool if x != c and x not in nb(c)]
            rest = rng.sample(others, L - 2 - len(nbs))
            body = [c] + rest + nbs
            rng.shuffle(body)
            return red, body + [c]
    raise SystemExit("no word found")


if __name__ == "__main__":
    mode = sys.argv[1]
    ct = [sys.argv[2], int(sys.argv[3])]
    rank = ct[1]
    t0 = time.time()
    jobs = []
    if mode == "all":
        minlen, maxlen, nproc = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
        lastonly = len(sys.argv) > 7 and sys.argv[7] == "lastonly"
        for w in words_T1(rank, minlen, maxlen, lastonly):
            for red in range(1, rank + 1):
                jobs.append((ct, red, w, None))
    else:
        nwords, L, nvert, seed, nproc = (int(x) for x in sys.argv[4:9])
        pattern = sys.argv[9] if len(sys.argv) > 9 else ""
        rng = random.Random(seed)
        for _ in range(nwords):
            red, w = random_word(rng, ct, L, pattern)
            m = len(w)
            verts = [frozenset(rng.sample(range(1, m + 1), rng.randint(2, m))) for _ in range(nvert)]
            jobs.append((ct, red, tuple(w), verts))
    nw = nv = nprod = nfail = 0
    with Pool(nproc) as pool:
        for ct_, red, word, n_v, n_p, bad in pool.imap_unordered(run_word, jobs, chunksize=1 if mode == "sample" else 8):
            nw += 1; nv += n_v; nprod += n_p
            if bad:
                nfail += 1
                print("FAIL", ct_, "red", red, "word", "".join(map(str, word)), bad[:2], flush=True)
            if mode == "sample":
                print("  %s%d red %d word %s: %d vertices, %d products, ok=%s" % (
                    ct[0], rank, red, "".join(map(str, word)), n_v, n_p, not bad), flush=True)
            elif nw % 1000 == 0:
                print("  ... %d / %d [%.0fs]" % (nw, len(jobs), time.time() - t0), flush=True)
    print("RESULT %s %s%d %s: %d (word, red) cases, %d vertices, %d products, %d FAILURES [%.0fs]" % (
        mode, ct[0], rank, " ".join(sys.argv[4:]), nw, nv, nprod, nfail, time.time() - t0), flush=True)
