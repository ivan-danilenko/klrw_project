"""Classify the degree-allowed candidates of a stack: python3 classes.py TYPE RANK RED WORD"""
import sys
from collections import defaultdict
from degree_model import Model

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

def classify(ct, red, word, show=True, processes=12):
    model = Model(ct, [(red, word)])
    m = model.m
    classes = defaultdict(list)
    for I, J, sigma, mono in model.candidates(processes):
        if not mono:
            continue
        key = (mono, cycles(sigma), tuple(sorted(I - J)), tuple(sorted(J - I)))
        classes[key].append((I, J))
    if show:
        print(ct, red, word, "slots:", " ".join("%d:c%d" % (s + 1, c) for s, c in enumerate(word)))
        for key, lst in sorted(classes.items(), key=lambda kv: (sum(p for _, p in kv[0][0]), kv[0])):
            mono, cyc, out, inn = key
            involved = set(out) | set(inn)
            fixed = {}
            for s in range(1, m + 1):
                if s in involved: continue
                vals = set((s in I) for I, J in lst)
                if len(vals) == 1:
                    fixed[s] = "R" if vals.pop() else "L"
            free = m - len(involved) - len(fixed)
            ok = "" if len(lst) == 2 ** free else "  (!! count %d != 2^%d)" % (len(lst), free)
            print("  %-44s sigma %-28s out %-14s in %-10s fixed %s%s" % (
                "*".join(n for n, p in mono for _ in range(p)), cyc, out, inn, fixed, ok))
    return classes

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    classify(ct, int(sys.argv[3]), [int(c) for c in sys.argv[4]])
