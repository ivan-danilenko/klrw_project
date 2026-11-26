"""
Statistics of the cancellation patterns in d^2 over families of words.

    sage -python 05_explore_stats.py TYPE RANK MINLEN MAXLEN [T1|all] [maxwords] [seed]
"""
import sys, importlib, itertools, random, time
from collections import defaultdict
L = importlib.import_module("05_lib")
E = importlib.import_module("05_explore")
from cyclefree import find_cycle
from formula import Stack

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    lo, hi = int(sys.argv[3]), int(sys.argv[4])
    mode = sys.argv[5] if len(sys.argv) > 5 else "T1"
    maxwords = int(sys.argv[6]) if len(sys.argv) > 6 else 10**9
    rng = random.Random(int(sys.argv[7]) if len(sys.argv) > 7 else 0)
    total = defaultdict(int)
    t0 = time.time()
    for n in range(lo, hi + 1):
        if mode == "T1":
            words = list(L.words_T1(ct[1], n))
        else:
            words = [w for w in itertools.product(range(1, ct[1] + 1), repeat=n)
                     if find_cycle(Stack(ct, w), max_pairs=4) is None]
        if len(words) > maxwords:
            words = rng.sample(words, maxwords)
        for w in words:
            stats = E.explore(ct, w, verbose=False)
            for k, v in stats.items():
                total[k] += v
            if stats.get("(I,K) FAIL"):
                print("FAIL", w, flush=True)
        print("length", n, "words", len(words), "[%.0fs]" % (time.time() - t0), flush=True)
    for k, v in sorted(total.items()):
        print("%-50s %d" % (k, v))
