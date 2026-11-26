"""
Terms of a glued complex in "poset element" language.

Every moving strand slot is an element of the minuscule poset, named by its position in the
chain lambda -> lowest weight (p1, p2, ...; the chain is left word + right word of any
envelope, elements identified by colour and order of occurrence).  In the envelope of an
ideal J an element is in state  A (in J, left of W), BL (in J, right of W),
BR (not in J, left of W'), C (not in J, right of W').

    python3 glue/poset_table.py CASE [src_env tgt_env]
"""
import json, sys
from collections import defaultdict, Counter

def load(case, poly=False):
    d = json.load(open("glue/data/%s.json" % case))
    if poly:
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import glue_analyze as analyze
        glue_terms = [t for t in d["terms"] if t["src_envelope"] != t["tgt_envelope"]]
        resc, new_glue = analyze.nonneg_rescaling(d, glue_terms, d["variables"])
        d["terms"] = [t for t in d["terms"] if t["src_envelope"] == t["tgt_envelope"]] + new_glue
        d["rescaling"] = resc
        d["steps"] = []
    envs = d["envelopes"]
    full = envs[-1]["left_word"] + envs[-1]["right_word"]
    # canonical naming: k-th occurrence of colour c
    def names(word):
        cnt = Counter(); out = []
        for c in word:
            cnt[c] += 1; out.append((c, cnt[c]))
        return out
    canon = names(full)
    elem_name = {x: "p%d" % (i + 1) for i, x in enumerate(canon)}
    info = {}
    for e in envs:
        chain = names(e["left_word"] + e["right_word"])
        info[e["index"]] = ([elem_name[x] for x in chain], len(e["left_word"]))
    def layout(env, eps):
        """position -> element name (or W), and element -> state"""
        chain, nl = info[env]
        left = list(range(nl)); right = list(range(nl, len(chain)))
        pos = [chain[s] for s in reversed(left) if eps[s] == 0] + ["W"] + [chain[s] for s in left if eps[s] == 1] \
            + [chain[s] for s in right if eps[s] == 0] + ["W'"] + [chain[s] for s in reversed(right) if eps[s] == 1]
        state = {}
        for s in left: state[chain[s]] = "A" if eps[s] == 0 else "BL"
        for s in right: state[chain[s]] = "BR" if eps[s] == 0 else "C"
        return pos, state
    P = d["projectives"]
    colour = {elem_name[x]: x[0] for x in canon}
    rows = []
    rigid = {}
    for st in d["steps"]:
        for t in st["map_terms"]:
            key = (t["src_envelope"], tuple(t["src_eps"]), t["tgt_envelope"], tuple(t["tgt_eps"]), tuple(t["word"]),
                   tuple(sorted(t["params"].items())), tuple(sorted(t["dots"].items())))
            rigid[key] = t.get("rigid")
    for t in d["terms"]:
        s, g = P[t["source"]], P[t["target"]]
        pos_s, st_s = layout(s["envelope"], s["eps"]); pos_t, st_t = layout(g["envelope"], g["eps"])
        lab = list(pos_s)
        for letter in t["word"]:
            lab[letter - 1], lab[letter] = lab[letter], lab[letter - 1]
        sigma = {a: b for a, b in zip(lab, pos_t) if a not in ("W", "W'")}
        assert all(colour[a] == colour[b] for a, b in sigma.items())
        key = (s["envelope"], tuple(s["eps"]), g["envelope"], tuple(g["eps"]), tuple(t["word"]),
               tuple(sorted(t["params"].items())), tuple(sorted(t["dots"].items())))
        rows.append(dict(src_env=s["envelope"], tgt_env=g["envelope"], st_s=st_s, st_t=st_t, sigma=sigma,
                         params=t["params"], dots=t["dots"], coeff=t["coeff"], length=len(t["word"]),
                         rigid=rigid.get(key)))
    return d, sorted(colour, key=lambda x: int(x[1:])), colour, rows

def show(row, elems):
    moves = []
    for p in elems:
        a, b = row["st_s"][p], row["st_t"][p]
        q = row["sigma"][p]
        moves.append("%s:%s>%s%s" % (p, a, row["st_t"][q], "" if q == p else "(%s)" % q))
    changed = [m for m, p in zip(moves, elems) if row["st_s"][p] != row["st_t"][row["sigma"][p]] or row["sigma"][p] != p
               or row["st_s"][p] != row["st_t"][p]]
    mono = " ".join("%s^%d" % kv for kv in sorted(row["params"].items())) or "1"
    dots = " ".join(row["dots"]) or "-"
    return "%s%+d  [%s]  dots %-6s %-34s | state: %s" % (
        "R " if row["rigid"] else ("  " if row["rigid"] is not None else "i "), int(row["coeff"]), " ".join(changed), dots, mono,
        " ".join("%s=%s" % (p, row["st_s"][p]) for p in elems))

if __name__ == "__main__":
    poly = "--poly" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--poly"]
    case = sys.argv[1]
    d, elems, colour, rows = load(case, poly)
    if poly:
        print("rescaling of the envelopes:", [{x: e for x, e in v.items() if e} for v in d["rescaling"]])
    print("elements:", " ".join("%s:c%d" % (p, colour[p]) for p in elems))
    if len(sys.argv) > 3:
        a, b = int(sys.argv[2]), int(sys.argv[3])
        for r in rows:
            if r["src_env"] == a and r["tgt_env"] == b:
                print(show(r, elems))
    else:
        c = Counter((r["src_env"], r["tgt_env"]) for r in rows)
        print(sorted(c.items()))
