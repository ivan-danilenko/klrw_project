"""
One-step formula (Conjecture H) and, when every colour occurs at most twice, the closed
formula for G (Theorem 6.2), on minuscule words.

    sage -python proofs/checks/06_check_minuscule.py TYPE RANK RED [min|max] [LENGTH]
"""
import sys, importlib, time
L = importlib.import_module("06_lib")
from cupbrane import minuscule_word
step = importlib.import_module("06_check_step_formula")
glob = importlib.import_module("06_check_G_formula")

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3])
    choice = {"min": min, "max": max}[sys.argv[4]] if len(sys.argv) > 4 else min
    length = int(sys.argv[5]) if len(sys.argv) > 5 else None
    word = minuscule_word(ct, red, choice, length=length)
    print("minuscule word", "".join(map(str, word)), "multiplicities", sorted(word.count(c) for c in set(word)), flush=True)
    if max(word.count(c) for c in set(word)) <= 2:
        glob.check(ct, red, word)
    step.check(ct, red, word)
