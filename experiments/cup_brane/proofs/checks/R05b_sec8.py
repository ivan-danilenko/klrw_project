"""
Referee (second pass): run every assertion of R05_sec8.check_word (Lemma 3.2, 8.1, 8.3-8.7, E = 0 in all
components, from the statements of section 8) on words chosen to stress section 8:

    sage -python R05b_sec8.py TYPE RANK NWORDS LEN SEED NPROC PATTERN

PATTERN as in R05b_d2.random_word ('branch' = the last colour is a node with >= 3 neighbours, so that
data with k = 2 components occur; 'red', 'twinred', 'nb', '').  Only words whose last colour occurs
twice are kept.
"""
import sys, random, time
from multiprocessing import Pool
from collections import defaultdict
from R05_sec8 import check_word
from R05b_d2 import random_word

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    nwords, L, seed, nproc = (int(x) for x in sys.argv[3:7])
    pattern = sys.argv[7] if len(sys.argv) > 7 else ""
    rng = random.Random(seed)
    jobs = []
    while len(jobs) < nwords:
        red, w = random_word(rng, ct, L, pattern)
        if w.count(w[-1]) != 2:
            continue
        jobs.append((ct, red, tuple(w), seed * 100000 + len(jobs)))
    t0 = time.time()
    tot = defaultdict(int)
    with Pool(nproc) as pool:
        for ct_, red, word, st in pool.imap_unordered(check_word, jobs, chunksize=1):
            for k_, v in st.items():
                tot[k_] += v
            print("  %s%d red %d word %s: k=2 data %d, E FAIL %d" % (ct[0], ct[1], red, "".join(map(str, word)),
                  st.get("8.4 data with k=2", 0), st.get("E FAIL", 0)), flush=True)
    print("RESULT sec8 %s%d len %d pattern '%s': %d words, no assertion failed [%.0fs]" % (
        ct[0], ct[1], L, pattern, len(jobs), time.time() - t0))
    for k_ in sorted(tot):
        print("   %-45s %d" % (k_, tot[k_]))
