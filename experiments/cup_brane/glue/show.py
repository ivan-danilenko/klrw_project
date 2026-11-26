"""
Pretty-print a glued complex dumped by glue.py.

    python3 glue/show.py glue/data/A2_w1.json [--all]
"""
import sys
import json
from collections import Counter, defaultdict
from fractions import Fraction


def mono(params):
    return "*".join(n if p == 1 else "{}^{}".format(n, p) for n, p in sorted(params.items())) or "1"


def eps_str(eps, nl):
    return "".join(map(str, eps[:nl])) + "|" + "".join(map(str, eps[nl:]))


def main(path, show_all=False):
    D = json.load(open(path))
    env = D["envelopes"]
    P = D["projectives"]
    print("case", D["case"], " gluing order:", D["weights_in_gluing_order"])
    for e in env:
        print("  S_{}: weight {:<28} left {} right {}".format(e["index"], e["weight"], e["left_word"], e["right_word"]))
    for s in D["steps"]:
        print("  step {}: hom dims {}  ext {:.1f}s".format(s["step"], s["hom_dims"], s["seconds_ext"]))
    print("projectives:")
    for p in P:
        nl = len(env[p["envelope"]]["left_word"])
        print("  #{:<3} h={:<3} S_{} eps={:<8} {:<40} {}".format(
            p["index"], p["hom_deg"], p["envelope"], eps_str(p["eps"], nl), " ".join(p["state"]), p["equivariant_degree"]))
    print("terms" + ("" if show_all else " (gluing only)") + ":")
    for t in sorted(D["terms"], key=lambda t: (t["src_envelope"], t["tgt_envelope"], t["source"], t["target"])):
        if not show_all and t["src_envelope"] == t["tgt_envelope"]:
            continue
        s, g = P[t["source"]], P[t["target"]]
        nls = len(env[s["envelope"]]["left_word"])
        nlt = len(env[g["envelope"]]["left_word"])
        print("  S_{} {:<7} -> S_{} {:<7}  #{:<3}->#{:<3} {:>4} * {:<30} dots {:<10} word {:<16} {} => {}".format(
            s["envelope"], eps_str(s["eps"], nls), g["envelope"], eps_str(g["eps"], nlt),
            t["source"], t["target"], t["coeff"], mono(t["params"]), mono(t["dots"]),
            "".join(map(str, t["word"])) or "-", " ".join(t["left_state"]), " ".join(t["right_state"])))


if __name__ == "__main__":
    main(sys.argv[1], "--all" in sys.argv)
