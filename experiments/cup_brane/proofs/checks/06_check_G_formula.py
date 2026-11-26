"""
Check the closed formula for G (every colour at most twice):  d_Phi G = G d_tree  exactly.

    sage -python proofs/checks/06_check_G_formula.py TYPE RANK RED WORD [RED WORD ...]
    sage -python proofs/checks/06_check_G_formula.py TYPE RANK random N MAXLEN [SEED]
"""
import sys, importlib, random, time
L = importlib.import_module("06_lib")


def check(ct, red, word):
    S = L.Setup(ct, red, word)
    cache = {}
    def G(I):
        if I not in cache:
            cache[I] = L.G_formula_row(S, I)
        return cache[I]
    t0 = time.time()
    bad = L.is_chain_map(S, S.dPhi, G, S.d)
    n = sum(len(G(I)) for I in L.subsets(S.m))
    print("{} red {} word {}: {} components of G, d_Phi G = G d_tree: {}   [{:.0f}s]".format(
        "".join(map(str, ct)), red, "".join(map(str, word)), n, "OK" if not bad else "FAILS at {}".format(bad[:3]), time.time() - t0), flush=True)
    return not bad


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    if sys.argv[3] == "random":
        n, maxlen = int(sys.argv[4]), int(sys.argv[5])
        rng = random.Random(int(sys.argv[6]) if len(sys.argv) > 6 else 0)
        seen = set(); ok = 0
        while len(seen) < n:
            Lw = rng.randint(3, maxlen)
            word = tuple(rng.randint(1, ct[1]) for _ in range(Lw))
            red = rng.randint(1, ct[1])
            if max(word.count(c) for c in set(word)) > 2 or (red, word) in seen:
                continue
            if not any(word.count(c) == 2 for c in set(word)):
                continue
            seen.add((red, word))
            ok += check(ct, red, word)
        print("RESULT: {} of {} words OK".format(ok, len(seen)))
    else:
        args = sys.argv[3:]
        for red, word in zip(args[0::2], args[1::2]):
            check(ct, int(red), tuple(int(c) for c in word))
