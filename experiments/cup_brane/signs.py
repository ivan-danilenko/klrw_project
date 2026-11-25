"""Residual signs: coefficient divided by the Koszul sign, per class of terms."""
import json, sys
from collections import defaultdict

def perm_sign(seq):
    seq = list(seq); s = 1
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                s = -s
    return s

def koszul(src, tgt, sigma):
    """Wedge sign: write I ascending, apply sigma in place, contract the mover, sort."""
    I = sorted(src); J = set(tgt)
    img = [sigma[s - 1] for s in I]
    moving = [k for k, s in enumerate(img) if s not in J]
    assert len(moving) == 1
    k = moving[0]
    sign = (-1) ** (len(I) - 1 - k)
    rest = img[:k] + img[k + 1:]
    assert set(rest) == J
    return sign * perm_sign(rest), I[k]

def cycles(sigma):
    seen, out = set(), []
    for s in range(1, len(sigma) + 1):
        if s in seen or sigma[s - 1] == s:
            continue
        cyc, t = [], s
        while t not in seen:
            seen.add(t); cyc.append(t); t = sigma[t - 1]
        out.append(tuple(cyc))
    return tuple(out)

def residuals(path):
    data = json.load(open(path))
    res = defaultdict(lambda: defaultdict(int))
    for t in data["terms"]:
        K, mv = koszul(t["src"], t["tgt"], t["sigma"])
        mono = "*".join(n for n, p in t["params"] for _ in range(p)) or "1"
        src, tgt = set(t["src"]), set(t["tgt"])
        key = (mono, cycles(t["sigma"]), mv, tuple(sorted(src - tgt)), tuple(sorted(tgt - src)))
        res[key][t["coeff"] * K] += 1
    return data, res

if __name__ == "__main__":
    for path in sys.argv[1:]:
        data, res = residuals(path)
        print(path, " slots:", " ".join("%d:c%d" % (s + 1, c) for s, c in enumerate(data["strand_colour"])))
        bad = 0
        for key, v in sorted(res.items(), key=lambda kv: (len(kv[0][0]), kv[0])):
            if key[0] == "1":
                assert dict(v).keys() == {1}
                continue
            if len(v) > 1: bad += 1
            print("   %-40s sigma %-26s mover %d out %-14s in %-10s residual %s" % (key[0], key[1], key[2], key[3], key[4], dict(v)))
        print("   classes with non-constant residual:", bad)
