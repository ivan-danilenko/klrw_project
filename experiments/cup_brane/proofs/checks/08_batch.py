"""
08 -- batch of dot-free completion problems (proofs/08_cyclic_normal_form.md, §3).

    sage -python 08_batch.py LIST [--monotone] [--procs N]

LIST:  len7      all cyclic words of length 7 up to symmetry (A_3 classes, every red colour 1..3,
                 plus red 4 in D_4 for the words using colours 1, 2, 3)
       a2len8    all cyclic A_2 words of length 8 up to the symmetry 1 <-> 2, both reds
       a3len8    the cyclic A_3 words of length 8 up to symmetry (every red), as many as listed
       FILE      a text file with lines  TYPE RANK RED WORD

Appends one JSON line per case to 08_data/batch_<LIST>[_mon].jsonl and saves every solved
differential (08_solve.save).  Summary table printed at the end.
"""
import os
import sys
import json
import time
import itertools
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
S8 = importlib.import_module("08_solve")
D8 = importlib.import_module("08_dotted")
C8 = importlib.import_module("08_comb")
L3 = importlib.import_module("03_lib")


def heapkey(w, adj):
    w = list(w)
    out = []
    while w:
        best = None
        for i, c in enumerate(w):
            if all(not (adj(c, d) or c == d) for d in w[:i]):
                if best is None or c < w[best]:
                    best = i
        out.append(w.pop(best))
    return tuple(out)


def classes(typ, n, m, autos):
    adj = L3.adjacency(typ, n)
    cl = {}
    for w in itertools.product(range(1, n + 1), repeat=m):
        W = L3.Word(w, adj)
        if W.search() is None:
            continue
        key = min(heapkey(tuple(a[c] for c in w), adj) for a in autos)
        cl.setdefault(key, w)
    return sorted(cl)


def case_list(name):
    if name == "len7":
        out = []
        for w in classes("A", 3, 7, [{1: 1, 2: 2, 3: 3}, {1: 3, 2: 2, 3: 1}]):
            for red in (1, 2, 3):
                out.append((["A", 3], red, w))
            if 3 in w:
                out.append((["D", 4], 4, w))
        return out
    if name == "a2len8":
        return [(["A", 2], red, w) for w in classes("A", 2, 8, [{1: 1, 2: 2}, {1: 2, 2: 1}]) for red in (1, 2)]
    if name == "a3len8":
        return [(["A", 3], red, w) for w in classes("A", 3, 8, [{1: 1, 2: 2, 3: 3}, {1: 3, 2: 2, 3: 1}]) if 3 in w and 1 in w
                for red in (1, 2, 3)]
    out = []
    for line in open(name):
        line = line.split("#")[0].split()
        if len(line) >= 4:
            out.append(([line[0], int(line[1])], int(line[2]), tuple(int(c) for c in line[3])))
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    monotone = "--monotone" in sys.argv
    procs = 12
    if "--procs" in sys.argv:
        procs = int(sys.argv[sys.argv.index("--procs") + 1])
    start = 0
    if "--start" in sys.argv:
        start = int(sys.argv[sys.argv.index("--start") + 1])
    basis_mode = "klrw"
    if "--basis" in sys.argv:
        basis_mode = sys.argv[sys.argv.index("--basis") + 1]
    dotted = "--dotted" in sys.argv
    cases = case_list(name)
    print("%d cases" % len(cases), flush=True)
    out_path = os.path.join(HERE, "08_data", "batch_%s%s%s%s.jsonl" % (os.path.basename(name), "_mon" if monotone else "", "" if basis_mode == "klrw" else "_basis-" + basis_mode, "_dottedR3" if dotted else ""))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for k, (ct, red, w) in enumerate(cases):
        if k < start:
            continue
        t0 = time.time()
        try:
            if dotted:
                out, res, C, cands = D8.run(ct, red, list(w), R=3, save_it=True, procs=procs, verbose=True, basis_mode=basis_mode)
                out["patterns"] = sorted(C.W.patterns())
            else:
                out, res, C, cands = S8.run(ct, red, list(w), monotone=monotone, save_it=True, procs=procs, verbose=True, basis_mode=basis_mode)
        except Exception as err:  # keep the batch running
            out = dict(status="ERROR %r" % (err,))
        rec = dict(type=ct, red=red, word="".join(map(str, w)), **{k2: v for k2, v in out.items() if k2 not in ("kernel",)})
        rec["kernel_preview"] = out.get("kernel", [])[:3]
        rows.append(rec)
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("CASE %d/%d %s%d red %d %s: %s, kernel %s, extra %s, tree missing %s wrong %s [%.0fs]" % (
            k + 1, len(cases), ct[0], ct[1], red, "".join(map(str, w)), out.get("status"), out.get("kernel_dim"),
            out.get("n_extra"), out.get("tree_missing"), out.get("tree_wrong"), time.time() - t0), flush=True)
    print("\nSUMMARY")
    for r in rows:
        print("  %s%d red %d %-10s %-8s %-40s kernel %-3s extra %-4s dotted %-4s tree missing %-3s wrong %-3s" % (
            r["type"][0], r["type"][1], r["red"], r["word"], ",".join(r.get("patterns", [])), r.get("status"),
            r.get("kernel_dim"), r.get("n_extra"), r.get("n_dotted"), r.get("tree_missing"), r.get("tree_wrong")))
