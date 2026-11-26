"""
Check the conjectural closed formula for the one-step maps Gamma^(k) : C^(k-1) -> C^(k)
(Conjecture H of 06_main_intertwiner.md): is it a chain map, for every k?

    sage -python proofs/checks/06_check_step_formula.py TYPE RANK RED WORD [RED WORD ...]
    sage -python proofs/checks/06_check_step_formula.py TYPE RANK random N MAXLEN MAXMULT [SEED]
"""
import sys, importlib, random, time
L = importlib.import_module("06_lib")
from cyclefree import find_cycle


def check(ct, red, word):
    t0 = time.time()
    S = L.Setup(ct, red, word)
    res = []
    for k in range(1, S.m + 1):
        cache = {}
        def G(I, k=k, cache=cache):
            if I not in cache:
                cache[I] = L.step_formula_row(S, I, k)
            return cache[I]
        bad = L.is_chain_map(S, L.d_mixed(S, k - 1), G, L.d_mixed(S, k))
        res.append(not bad)
    cyc = find_cycle(S.stack, max_pairs=4) is not None
    print("{} red {} word {}: cyclic {}, steps OK: {}   [{:.0f}s]".format(
        "".join(map(str, ct)), red, "".join(map(str, word)), cyc, "".join("Y" if r else "N" for r in res), time.time() - t0), flush=True)
    return all(res)


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    if sys.argv[3] == "random":
        n, maxlen, maxmult = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
        rng = random.Random(int(sys.argv[7]) if len(sys.argv) > 7 else 0)
        seen = set(); ok = 0
        while len(seen) < n:
            Lw = rng.randint(3, maxlen)
            word = tuple(rng.randint(1, ct[1]) for _ in range(Lw))
            red = rng.randint(1, ct[1])
            mult = max(word.count(c) for c in set(word))
            if mult > maxmult or mult < 2 or (red, word) in seen:
                continue
            seen.add((red, word))
            ok += check(ct, red, word)
        print("RESULT: {} of {} words OK".format(ok, len(seen)))
    else:
        args = sys.argv[3:]
        for red, word in zip(args[0::2], args[1::2]):
            check(ct, int(red), tuple(int(c) for c in word))
