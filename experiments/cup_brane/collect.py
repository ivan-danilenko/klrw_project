"""
Collect corrected differentials of stacks of U-branes.

    sage -python experiments/cup_brane/collect.py TYPE RANK RED [min|max|WORD] [LENGTH]

e.g. ``collect.py A 4 2`` (full stack, lexicographically smallest word),
``collect.py D 5 1 max`` or ``collect.py A 3 2 2132``.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cupbrane import minuscule_word, run_case

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3])
    spec = sys.argv[4] if len(sys.argv) > 4 else "min"
    length = int(sys.argv[5]) if len(sys.argv) > 5 else None
    if spec in ("min", "max"):
        word = minuscule_word(ct, red, {"min": min, "max": max}[spec], length=length)
    else:
        word = tuple(int(c) for c in spec)
    print("type", ct, "red", red, "word", word, flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    run_case(ct, [(red, word)], out_dir=out)
