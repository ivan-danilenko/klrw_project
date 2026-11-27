"""
08 -- completions with dots allowed (proofs/08_cyclic_normal_form.md, §4).

    sage -python 08_dotted.py TYPE RANK RED WORD [--from R] [--dotfree-below S] [--save] [--basis klrw|left|right]

Finds ALL differentials  d = (tree terms with < R pairs) + eps,  eps = sum_c lambda_c c,  where c runs over
ALL homogeneous basis elements  x^alpha mu beta_sigma  (dots allowed) of homological degree -1 and
r-degree >= R (default R = 3, i.e. the tree formula is kept up to r^2).  With --dotfree-below S the
candidates of r-degree < S are restricted to dot-free ones.  Same deferred-kernel solver as 08_solve.py.
"""
import os
import sys
import time
import json
import argparse
import itertools
import importlib
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
A8 = importlib.import_module("08_alg")
C8 = importlib.import_module("08_comb")
S8 = importlib.import_module("08_solve")
from sage.all import QQ  # noqa: E402


def dot_monomials(C, I, a):
    """all dot monomials with a[c] dots on the strands of colour c at the source e_I (klrw names x_c_k)."""
    labs = [l for l in C.W.labels(I) if l != "W"]
    per = []
    for c, k in sorted(a.items()):
        nstr = sum(1 for l in labs if C.W.c(l) == c)
        vars_ = ["x_%d_%d" % (c, i) for i in range(1, nstr + 1)]
        per.append(list(itertools.combinations_with_replacement(vars_, k)))
    out = []
    for combo in itertools.product(*per):
        cnt = Counter(v for grp in combo for v in grp)
        out.append(tuple(sorted(cnt.items())))
    return out


def dotted_candidates(C, rmin, rmax=None, dotfree_below=None):
    W = C.W
    out = []
    for (I, J, s, mono, a) in W.candidates(-1, rmin=rmin, rmax=rmax, dots=True):
        rd = C8.rdeg(mono)
        if dotfree_below is not None and rd < dotfree_below and a:
            continue
        for dm in dot_monomials(C, I, a):
            out.append((I, J, s, mono, dm))
    return out


def element(C, cand):
    I, J, s, mono, dm = cand
    e = C.D.one()
    for nm, p in dm:
        e *= C.D(nm) ** p
    return (e * C.mono(mono)) * C.basis(I, J, s)


def fmt(C, cand, v):
    I, J, s, mono, dm = cand
    return "%+s %-30s dots %-14s {%s} -> {%s}  sigma %-24s %s" % (
        v, C8.mono_str(mono), C8.mono_str(dm) if dm else "-", ",".join(map(str, sorted(I))), ",".join(map(str, sorted(J))),
        C.W.cycle_str(s), C.W.describe(I, J, s))


def run(ct, red, word, R=3, dotfree_below=None, save_it=False, procs=12, basis_mode="klrw", verbose=True, base_maxpairs=None):
    t0 = time.time()
    C = A8.Ctx(ct, red, word, basis_mode=basis_mode)
    cands = dotted_candidates(C, R, dotfree_below=dotfree_below)
    base = C.tree_terms(maxpairs=(R - 1) if base_maxpairs is None else base_maxpairs)
    if verbose:
        print("%s%d red %d word %s: candidates (dots allowed) r>=%d: %s; with dots %d [%.1fs]" % (
            ct[0], ct[1], red, "".join(map(str, word)), R, dict(sorted(Counter(C8.rdeg(c[3]) for c in cands).items())),
            sum(1 for c in cands if c[4]), time.time() - t0), flush=True)
    els = [(c[0], c[1], element(C, c)) for c in cands]
    rdegs = [C8.rdeg(c[3]) for c in cands]
    res = S8.complete(C, None, base, verbose=verbose, procs=procs, els=els, rdegs=rdegs, start=R)
    out = dict(status=res["status"], kernel_dim=len(res["K"]), steps=res["steps"])
    if verbose:
        print("  status %s, kernel %d [%.0fs]" % (res["status"], len(res["K"]), time.time() - t0))
    if res["status"] == "solved":
        tree = S8.tree_index(C)
        sol = {cands[i]: v for i, v in res["p"].items()}
        extra = []
        tree_ok = True
        for cnd, v in sorted(sol.items(), key=lambda kv: (C8.rdeg(kv[0][3]), sorted(kv[0][0]), sorted(kv[0][1]))):
            I, J, s, mono, dm = cnd
            tk = (I, J, s, mono)
            if not dm and tk in tree and tree[tk] == v:
                continue
            extra.append((cnd, v))
        # tree terms of r-degree >= R that are absent
        miss = [k for k, v in tree.items() if C8.rdeg(k[3]) >= R and sol.get(k + ((),)) != v]
        out.update(n_extra=len(extra), tree_missing=len(miss), n_dotted=sum(1 for c, v in extra if c[4]))
        if verbose:
            print("  extra terms (beyond the tree formula): %d (%d dotted); tree terms of r>=%d missing/changed: %d" % (
                len(extra), out["n_dotted"], R, len(miss)))
            for cnd, v in extra:
                print("    " + fmt(C, cnd, v))
        if res["K"] and verbose:
            for kv in res["K"][:8]:
                print("    kernel vector:", [("%s" % v, sorted(cands[i][0]), sorted(cands[i][1]), C.W.cycle_str(cands[i][2]), C8.mono_str(cands[i][4])) for i, v in kv.items()][:10])
        if save_it:
            terms = []
            full = defaultdict(lambda: QQ(0))
            for (I, J, s, mono, sign) in base:
                full[(I, J, s, mono, ())] += sign
            for cnd, v in sol.items():
                full[cnd] += v
            for (I, J, s, mono, dm), v in sorted(full.items(), key=lambda kv: (len(kv[0][0]), sorted(kv[0][0]), sorted(kv[0][1]))):
                if v:
                    md = dict(mono)
                    md.update(dict(dm))
                    terms.append([sorted(I), sorted(J), list(s), md, int(v) if QQ(v).denominator() == 1 else str(v)])
            data = dict(cartan_type=C.ct, red=C.red, word=list(C.word), basis="%s basis braid; dots at the left end, klrw names x_c_k" % basis_mode,
                        what="tree terms with < %d pairs + complete dotted ansatz at r >= %d" % (R, R), n_terms=len(terms),
                        terms=terms, **{k: v for k, v in out.items() if k != "steps"})
            path = os.path.join(A8.DATA, S8.case_name(ct, red, word, False, basis_mode) + "_dotted_R%d.json" % R)
            json.dump(data, open(path, "w"), indent=0)
            print("  saved", path)
    out["seconds"] = time.time() - t0
    return out, res, C, cands


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("type")
    ap.add_argument("rank", type=int)
    ap.add_argument("red", type=int)
    ap.add_argument("word")
    ap.add_argument("--from", dest="R", type=int, default=3)
    ap.add_argument("--dotfree-below", type=int, default=None)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--basis", default="klrw")
    a = ap.parse_args()
    run([a.type, a.rank], a.red, [int(c) for c in a.word], R=a.R, dotfree_below=a.dotfree_below, save_it=a.save,
        procs=a.procs, basis_mode=a.basis)
