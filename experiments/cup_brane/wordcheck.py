"""
Is the diagram of each formula term independent of the reduced word chosen for it?
Compare the lexmin basis braid with products of crossings along other reduced words.
    sage -python wordcheck.py TYPE RANK RED [min|max|WORD] [LENGTH]
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cupbrane import *
from fast import braid_for
from formula import Stack

def reduced_words(perm, how):
    """perm: list, perm[p] = target position of the strand at source position p. Returns a reduced word (1-based letters)."""
    cur = list(perm)  # cur[position] = target of strand currently there
    word = []
    n = len(cur)
    rng = random.Random(how)
    while True:
        desc = [p for p in range(n - 1) if cur[p] > cur[p + 1]]
        if not desc:
            return tuple(word)
        if how == "right":
            p = desc[-1]
        elif how == "left":
            p = desc[0]
        else:
            p = rng.choice(desc)
        cur[p], cur[p + 1] = cur[p + 1], cur[p]
        word.append(p + 1)

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]; red = int(sys.argv[3])
    spec = sys.argv[4] if len(sys.argv) > 4 else "min"
    length = int(sys.argv[5]) if len(sys.argv) > 5 else None
    word = minuscule_word(ct, red, {"min": min, "max": max}[spec], length=length) if spec in ("min", "max") else tuple(int(c) for c in spec)
    P = [(red, word)]
    KLRW = make_algebra(ct, P)
    cube = Cube(KLRW, P)
    st = Stack(ct, word)
    n_terms = n_dep = 0
    examples = []
    for (I, J, sigma, mono), sign in st.all_terms().items():
        eps_I, eps_J, braid = braid_for(cube, I, J, sigma)
        base = KLRW.monomial(braid)
        # position permutation
        left = list(cube.labels(eps_I)); inv = {sigma[s] - 1: s for s in range(cube.n)}
        right = [x if isinstance(x, tuple) else inv[x] for x in cube.labels(eps_J)]
        perm = [right.index(x) for x in left]
        dep = False
        for how in ("left", "right", 1, 2, 3):
            w = reduced_words(perm, how)
            assert len(w) == len(braid.word())
            el = KLRW.element_from_braid(w, left_state=cube.state(eps_I))
            if el != base:
                dep = True
                if len(examples) < 5:
                    examples.append((sorted(I), sorted(J), mono, w, tuple(braid.word()), str(el - base)[:200]))
                break
        n_terms += 1; n_dep += dep
    print(ct, red, word, ": terms", n_terms, " whose diagram depends on the reduced word:", n_dep)
    for e in examples: print("   ", e)
