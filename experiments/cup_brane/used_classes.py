"""Classes of used / unused candidates: python3 used_classes.py data_fast/CASE.json"""
import json, sys
from collections import defaultdict
from degree_model import Model
from signs import cycles, koszul

def load(path, processes=12):
    data = json.load(open(path))
    model = Model(data["cartan_type"], data["punctures"])
    cands = set(model.candidates(processes))
    actual = {}
    for t in data["terms"]:
        actual[(frozenset(t["src"]), frozenset(t["tgt"]), tuple(t["sigma"]),
                tuple(sorted((n, p) for n, p in t["params"])))] = t["coeff"]
    return data, model, cands, actual

def class_table(model, items):
    m = model.m
    classes = defaultdict(list)
    for I, J, sigma, mono in items:
        key = ("*".join(n for n, p in mono for _ in range(p)) or "1", cycles(sigma),
               tuple(sorted(I - J)), tuple(sorted(J - I)))
        classes[key].append((I, J))
    rows = []
    for key, lst in sorted(classes.items(), key=lambda kv: (len(kv[0][0]), kv[0])):
        involved = set(key[2]) | set(key[3])
        fixed = {}
        for s in range(1, m + 1):
            if s in involved: continue
            vals = set((s in I) for I, J in lst)
            if len(vals) == 1:
                fixed[s] = "R" if vals.pop() else "L"
        free = m - len(involved) - len(fixed)
        rows.append((key, len(lst), fixed, free))
    return rows

if __name__ == "__main__":
    data, model, cands, actual = load(sys.argv[1])
    only = sys.argv[2] if len(sys.argv) > 2 else None
    print("slots:", " ".join("%d:c%d" % (s + 1, c) for s, c in enumerate(model.col)))
    for title, items in (("USED", [k for k in actual if k[3]]), ("UNUSED", [k for k in cands - set(actual)])):
        print(title)
        for key, n, fixed, free in class_table(model, items):
            if only and only not in key[0]: continue
            flag = "" if n == 2 ** free else " [%d of 2^%d]" % (n, free)
            print("  %-36s %-26s out %-13s in %-10s fixed %s%s" % (key[0], key[1], key[2], key[3], fixed, flag))
