"""
01_shield_scan.py   (python3 proofs/checks/01_shield_scan.py TYPE RANK MAXLEN [N SEED])

Section B.6 of 01_foundations.md: for every bad triple of every admissible configuration at
every vertex, is it shielded?  Split by cyclic / cycle-free words.  Exhaustive over words of
length <= MAXLEN, or N random words of length 6..MAXLEN.
"""
import sys, os, itertools, importlib.util, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", ".."))
spec = importlib.util.spec_from_file_location("bs", os.path.join(HERE, "01_bad_search.py"))
bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)
ct = [sys.argv[1], int(sys.argv[2])]; L = int(sys.argv[3])
if len(sys.argv) > 4:
    rng = random.Random(int(sys.argv[5])); N = int(sys.argv[4])
    words = sorted({tuple(rng.randint(1, ct[1]) for _ in range(rng.randint(6, L))) for _ in range(N)})
else:
    words = [w for LL in range(3, L + 1) for w in itertools.product(range(1, ct[1] + 1), repeat=LL)]
stats, ex = bs.shield_scan(ct, words)
for k in sorted(stats, key=str): print(k, stats[k])
for k, v in ex.items(): print("example", k, v)
