"""
08 -- convert an 08_data differential (terms [src, tgt, sigma, {name: power}, coeff]) into the
pieces/ format read by ../../cycle3.py (terms [src, tgt, sigma, params, dots, coeff], vertices).

    python3 08_topieces.py 08_data/FILE.json OUT.json
"""
import json, sys, itertools
d = json.load(open(sys.argv[1]))
m = len(d["word"])
terms = []
for src, tgt, sg, mono, c in d["terms"]:
    params = sorted([n, p] for n, p in mono.items() if not n.startswith("x_"))
    dots = sorted([n, p] for n, p in mono.items() if n.startswith("x_"))
    terms.append([src, tgt, sg, params, dots, c])
verts = [list(c) for k in range(m + 1) for c in itertools.combinations(range(1, m + 1), k)]
json.dump(dict(cartan_type=d["cartan_type"], red=d["red"], word=d["word"], I=[], K=[], vertices=verts, terms=terms),
          open(sys.argv[2], "w"))
