"""
Does the tree formula give d^2 = 0 for ARBITRARY words (not only minuscule chains)?

    sage -python proofs/checks/general_words.py TYPE RANK N_WORDS MAXLEN [SEED]

For each random word: is there a cyclic configuration / a blocked connected configuration,
and does the tree formula satisfy d^2 = 0 (full complex)?
"""
import sys, os, random, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from cupbrane import *
from formula import Stack
from cyclefree import find_cycle
from verify import verify

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    n_words = int(sys.argv[3]); maxlen = int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    rng = random.Random(seed)
    rank = ct[1]
    seen = set()
    stats = {}
    while len(seen) < n_words:
        L = rng.randint(3, maxlen)
        word = tuple(rng.randint(1, rank) for _ in range(L))
        red = rng.randint(1, rank)
        if (red, word) in seen:
            continue
        seen.add((red, word))
        st = Stack(ct, word)
        cyc = find_cycle(st, max_pairs=4)
        try:
            ok = verify(ct, red, word, verbose=False)
        except AssertionError as e:
            ok = "inhomogeneous"
        key = (cyc is not None, ok)
        stats[key] = stats.get(key, 0) + 1
        if ok is not True:
            print("FAIL red", red, "word", "".join(map(str, word)), "cyclic:", cyc, "result:", ok, flush=True)
    print("summary (cyclic, d2=0): count", stats)
