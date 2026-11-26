"""
Validation of the exhaustive search of 03_lib against a brute force over ALL configurations
(every mover, every set of disjoint same-colour pairs), on random arbitrary words.

    python3 proofs/checks/03_search_vs_brute.py [N_WORDS] [MAXLEN] [SEED]

Checks, for every word, that the following agree between search and brute force:
  (a) cyclic (no blocks)            (b) mover-on-a-cycle variant      (c) blocks ignored
"""
import importlib, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L = importlib.import_module("03_lib")

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    maxlen = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    types = [("A", 2), ("A", 3), ("A", 4), ("D", 4), ("A", 5)]
    stats = {}
    for it in range(N):
        typ, n = rng.choice(types)
        m = rng.randint(4, maxlen)
        col = [rng.randint(1, n) for _ in range(m)]
        W = L.Word(col, L.adjacency(typ, n))
        for mode in [dict(), dict(mover_on_cycle=True), dict(allow_blocks=True)]:
            s = W.search(**mode)
            b = W.brute(**mode)
            assert (s is None) == (b is None), (typ, n, col, mode, s, b)
            if s is not None:
                j, F = s
                assert W.is_cyclic_config(j, F, mode.get("allow_blocks", False)), (col, s)
                if mode.get("mover_on_cycle"):
                    assert L.mover_on_some_cycle(j, F, W)
            key = tuple(sorted(mode)) or ("plain",)
            stats.setdefault(key, [0, 0])[s is not None] += 1
    print("all %d words agree; (non-cyclic, cyclic) per mode:" % N, stats)
