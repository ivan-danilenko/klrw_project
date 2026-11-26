"""
Decode the gluing components of A_n, omega_1 (all colours distinct).
A state is [A decreasing] W1 [B increasing] Wn [C decreasing]; S_k has
A in {1..k}, C in {k+1..n}.  A component S_k -> S_l is printed as
(A|B|C) -> (A'|B'|C') with the moves of the strands.

    python3 glue/decode_omega1.py glue/data/A3_w1.json
"""
import sys
import json
from collections import Counter, defaultdict


def regions(state):
    out, r = {}, 0
    for x in state:
        if x[0] == "W":
            r += 1
        else:
            out[int(x[1:])] = "ABC"[r]
    return out


def fmt(reg, n):
    return "|".join("".join(str(c) for c in range(1, n + 1) if reg[c] == R) for R in "ABC")


def main(path):
    D = json.load(open(path))
    n = D["cartan_type"][1]
    P = D["projectives"]
    rows = []
    for t in D["terms"]:
        k, l = t["src_envelope"], t["tgt_envelope"]
        if k == l:
            continue
        rs, rt = regions(t["left_state"]), regions(t["right_state"])
        moves = tuple(sorted((c, rs[c] + rt[c]) for c in rs if rs[c] != rt[c]))
        rows.append((k, l, fmt(rs, n), fmt(rt, n), moves, t["coeff"], t["params"], len(t["word"])))
    by_type = Counter()
    for k, l, s, g, moves, c, pr, L in sorted(rows):
        print("S_{} {:<10} -> S_{} {:<10} moves {:<40} {:>3} {}".format(
            k, s, l, g, " ".join("{}:{}".format(c_, m) for c_, m in moves), c,
            "*".join("{}^{}".format(a, b) for a, b in sorted(pr.items())) or "1"))
        by_type[k - l, tuple(sorted(m for _, m in moves))] += 1
    print("\n(k - l, moves): count")
    for key, c in sorted(by_type.items()):
        print("  ", key, c)


if __name__ == "__main__":
    main(sys.argv[1])
