"""
Test "rule C": a degree-allowed candidate (I, J, sigma, mono) is used iff
  * exactly one slot j of I is fixed by sigma and leaves (the mover), and
  * sigma is a product of disjoint transpositions (a b), a < b, with a not in I, b in I
    (so a enters and b leaves), and nothing else changes side.
"""
import sys, json
from collections import Counter
from used_classes import load

def rule_c(I, J, sigma):
    m = len(sigma)
    movers = 0
    for s in range(1, m + 1):
        t = sigma[s - 1]
        if t == s:
            if (s in I) != (s in J):
                if s in I and s not in J:
                    movers += 1
                else:
                    return False
            continue
        if sigma[t - 1] != s:
            return False  # not an involution
        a, b = min(s, t), max(s, t)
        if a in I or b not in I or a not in J or b in J:
            return False
    return movers == 1

if __name__ == "__main__":
    for path in sys.argv[1:]:
        data, model, cands, actual = load(path)
        C = set(c for c in cands if rule_c(c[0], c[1], c[2]))
        A = set(actual)
        print("%-44s used %5d  cand %5d  rule-C %5d  used-not-C %4d  C-not-used %4d" % (
            path.split("/")[-1], len(A), len(cands), len(C), len(A - C), len(C - A)))
