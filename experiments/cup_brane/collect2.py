"""
Two-puncture stable envelopes S_mu:

    sage -python collect2.py TYPE RANK RED [min|max|WORD] SPLIT

The full minuscule word is cut after SPLIT letters: the first part is nested
around the left puncture (red colour RED), the rest around the right puncture
(red colour = dual of RED), innermost strand = last letter.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cupbrane import minuscule_word, run_case
from sage.combinat.root_system.cartan_type import CartanType

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3])
    spec = sys.argv[4]
    split = int(sys.argv[5])
    if spec in ("min", "max"):
        word = minuscule_word(ct, red, {"min": min, "max": max}[spec])
    else:
        word = tuple(int(c) for c in spec)
    dual = CartanType(ct).opposition_automorphism()[red]
    punctures = [(red, word[:split]), (dual, tuple(reversed(word[split:])))]
    print("type", ct, "punctures", punctures, flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data2")
    run_case(ct, punctures, out_dir=out)
