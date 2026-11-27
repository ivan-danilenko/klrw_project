"""
08 -- dot-free completions of delta_0 + delta_1 for a word (proofs/08_cyclic_normal_form.md, §3).

    sage -python 08_solve.py TYPE RANK RED WORD [--monotone] [--save] [--procs N]

Finds ALL differentials of the form

    d = delta_0 + delta_1 + eps,     eps = sum_c lambda_c c,   lambda_c in QQ,

where c runs over the dot-free homogeneous basis elements of homological degree -1 and
r-degree >= 2 (08_comb.WordData.candidates; with --monotone only the monotone ones), i.e.
d^2 = 0 exactly.  delta_0 + delta_1 = the tree terms with at most one pair (formula.Stack).

Method: d^2 = 0 is graded by the r-degree n of the coefficients (the exponent of every basis
coefficient of a product of homogeneous elements is fixed by the degree).  Order by order the
solution set is kept as an affine space  p + span(K)  over the candidates of r-degree < n;
at order n the unknowns are the kernel coordinates lambda and the new candidates x_n, and the
equation  (d_p^2)_n + sum lambda_i [d_p, K_i]_n + [delta_0, x_n]_n + sum lambda_i lambda_j (K_i K_j + K_j K_i)_n = 0
is linear as long as the quadratic terms vanish (checked; if not, the script stops and says so).
No minimisation and no choice: the output is the complete solution set.

Output (with --save): 08_data/<TYPE><RANK>_W<RED>_<WORD>[_mon].json with the particular solution
(the full differential as terms [src, tgt, sigma, {name: power}, coeff]), the kernel basis
(if the solution is not unique) and the log.
"""
import os
import sys
import time
import json
import argparse
import importlib
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
A8 = importlib.import_module("08_alg")
C8 = importlib.import_module("08_comb")
from sage.all import QQ, ZZ, lcm, matrix, vector  # noqa: E402


def _cand_col(G, i):
    C, d0, els, n = G["C"], G["d0"], G["els"], G["n"]
    I, J, el = els[i]
    return A8.anticomm_vec(C, d0, I, J, el, keep=lambda e: C.rdeg(e) == n)


def complete(C, cands, base_terms, verbose=True, procs=12, log=None, els=None, rdegs=None, start=2):
    """returns dict(status, p (dict cand index -> QQ), K (list of dicts), steps).
    cands: list of (I, J, sigma, mono) (dot-free), or pass els = [(I, J, element)] and rdegs directly."""
    t0 = time.time()
    base = C.map_from_terms(base_terms)
    d0 = C.map_from_terms([t for t in base_terms if C8.rdeg(t[3]) == 0])
    if els is None:
        els = [(I, J, C.term(I, J, s, mono, 1)) for (I, J, s, mono) in cands]
        rdegs = [C8.rdeg(c[3]) for c in cands]
    by_r = defaultdict(list)
    for i, rd in enumerate(rdegs):
        by_r[rd].append(i)
    maxr_c = max(by_r) if by_r else 0
    p = {}
    K = []
    steps = []

    def as_map(vec):
        mp_ = defaultdict(lambda: defaultdict(lambda: C.K.zero()))
        for i, v in vec.items():
            if v:
                I, J, el = els[i]
                mp_[I][J] += C.D(v) * el
        return {I: dict(r) for I, r in mp_.items()}

    n = start
    status = None
    while True:
        dp = A8.add_maps(C, base, as_map(p))
        sq = A8.vec_map(C, A8.compose(C, dp, dp))
        maxr_sq = max((C.rdeg(k[3]) for k in sq), default=-1)
        Kmaps = [as_map(k) for k in K]
        kcols_full = []
        for km in Kmaps:
            ac = A8.add_maps(C, A8.compose(C, dp, km), A8.compose(C, km, dp))
            kcols_full.append(A8.vec_map(C, ac))
        maxr_k = max((C.rdeg(k[3]) for col in kcols_full for k in col), default=-1)
        if n > max(maxr_sq, maxr_k, maxr_c):
            status = "solved" if not sq else "FAILED (residue)"
            break
        # quadratic terms in the kernel coordinates
        quad = 0
        for a in range(len(Kmaps)):
            for b in range(a, len(Kmaps)):
                pr = A8.add_maps(C, A8.compose(C, Kmaps[a], Kmaps[b]), A8.compose(C, Kmaps[b], Kmaps[a]))
                v = A8.vec_map(C, pr, keep=lambda e: C.rdeg(e) == n)
                if v:
                    quad += 1
        if quad:
            status = "NONLINEAR at r^%d (%d quadratic kernel products)" % (n, quad)
            break
        rhs = {k: -v for k, v in sq.items() if C.rdeg(k[3]) == n}
        kcols = [{k: v for k, v in col.items() if C.rdeg(k[3]) == n} for col in kcols_full]
        new = by_r.get(n, [])
        A8._G.clear()
        A8._G.update(C=C, d0=d0, els=els, n=n)
        ncols = A8.par_map(_cand_col, new, procs=procs)
        cols = kcols + ncols
        res = A8.solve_affine(cols, rhs)
        st = dict(order=n, kernel_in=len(K), new_unknowns=len(new), equations=len(set(k for c in cols for k in c) | set(rhs)),
                  rhs_nonzero=len(rhs))
        if res is None:
            st["result"] = "inconsistent"
            steps.append(st)
            status = "NO SOLUTION (inconsistent at r^%d)" % n
            if verbose:
                print("  r^%d: %s" % (n, st), flush=True)
            break
        y0, kb = res
        st["kernel_out"] = len(kb)
        steps.append(st)
        if verbose:
            print("  r^%d: kernel in %d, new unknowns %d, equations %d, rhs %d -> kernel out %d  [%.0fs]" % (
                n, len(K), len(new), st["equations"], len(rhs), len(kb), time.time() - t0), flush=True)
        # update p and K
        nk = len(K)

        def combine(y):
            out = defaultdict(lambda: QQ(0))
            for i in range(nk):
                if y[i]:
                    for c, v in K[i].items():
                        out[c] += y[i] * v
            for j, c in enumerate(new):
                if y[nk + j]:
                    out[c] += y[nk + j]
            return {c: v for c, v in out.items() if v}

        dy = combine(y0)
        for c, v in dy.items():
            p[c] = p.get(c, QQ(0)) + v
        p = {c: v for c, v in p.items() if v}
        K = [combine(y) for y in kb]
        n += 1
    return dict(status=status, p=p, K=K, steps=steps, seconds=time.time() - t0, els=els)


def tree_index(C):
    return {(I, J, s, mono): sign for (I, J, s, mono, sign) in C.tree_terms()}


def analyse(C, cands, p, verbose=True):
    """compare the solution with the tree formula; list the extra terms."""
    tree = tree_index(C)
    sol = {}
    for i, v in p.items():
        I, J, s, mono = cands[i]
        sol[(I, J, s, mono)] = v
    # the full differential: delta_0 + delta_1 + p
    full = {k: QQ(v) for k, v in tree.items() if C8.rdeg(k[3]) <= 1}
    for k, v in sol.items():
        full[k] = full.get(k, QQ(0)) + v
    full = {k: v for k, v in full.items() if v}
    missing = [k for k in tree if k not in full]
    wrong = [k for k in tree if k in full and full[k] != tree[k]]
    extra = sorted([k for k in full if k not in tree], key=lambda k: (C8.rdeg(k[3]), sorted(k[0]), sorted(k[1])))
    return full, missing, wrong, extra


def fmt_term(C, k, v):
    I, J, s, mono = k
    LR = [x for x in C.W.slots if x not in I and s[x - 1] in J]
    return "%+d %-28s {%s} -> {%s}  sigma %-22s %s%s" % (
        int(v), C8.mono_str(mono), ",".join(map(str, sorted(I))), ",".join(map(str, sorted(J))),
        C.W.cycle_str(s), C.W.describe(I, J, s), "  NON-MONOTONE" if LR else "")


def save(C, path, full, info):
    terms = [[sorted(I), sorted(J), list(s), dict(mono), int(v)] for (I, J, s, mono), v in sorted(
        full.items(), key=lambda kv: (len(kv[0][0]), sorted(kv[0][0]), sorted(kv[0][1]), kv[0][2]))]
    data = dict(cartan_type=C.ct, red=C.red, word=list(C.word),
                basis="klrw basis braid of sigma (fast.braid_for), dots at the left end; terms [src, tgt, sigma, {name: power}, coeff]",
                **{k: v for k, v in info.items() if k not in ("n_terms", "terms")}, n_terms=len(terms), terms=terms)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(data, open(path, "w"), indent=0)


def case_name(ct, red, word, monotone=False, basis_mode="klrw"):
    return "%s%d_W%d_%s%s%s" % (ct[0], ct[1], red, "".join(map(str, word)), "_mon" if monotone else "",
                                "" if basis_mode == "klrw" else "_basis-" + basis_mode)


def run(ct, red, word, monotone=False, save_it=False, procs=12, verbose=True, basis_mode="klrw"):
    t0 = time.time()
    C = A8.Ctx(ct, red, word, basis_mode=basis_mode)
    W = C.W
    cands = W.candidates(-1, rmin=2, monotone=monotone)
    base = C.tree_terms(maxpairs=1)
    cyc = W.minimal_cyclic()
    if verbose:
        print("%s%d red %d word %s: cyclic %s %s; dot-free candidates r>=2: %s%s [%.1fs]" % (
            ct[0], ct[1], red, "".join(map(str, word)), bool(cyc), sorted(W.patterns()),
            dict(sorted(Counter(C8.rdeg(c[3]) for c in cands).items())), " (monotone only)" if monotone else "",
            time.time() - t0), flush=True)
    res = complete(C, cands, base, verbose=verbose, procs=procs)
    out = dict(status=res["status"], kernel_dim=len(res["K"]), steps=res["steps"], cyclic_configs=[[j, [list(p) for p in F]] for j, F in cyc],
               patterns=sorted(W.patterns()), monotone_ansatz=monotone, basis_mode=basis_mode,
               candidates_by_rdeg={str(k): v for k, v in sorted(Counter(C8.rdeg(c[3]) for c in cands).items())})
    if res["status"] == "solved":
        integral = all(QQ(v).denominator() == 1 for v in res["p"].values())
        full, missing, wrong, extra = analyse(C, cands, res["p"])
        out.update(integral=integral, n_terms=len(full), tree_missing=len(missing), tree_wrong=len(wrong), n_extra=len(extra))
        if verbose:
            print("  status %s; kernel dim %d; integral %s; terms %d; tree terms missing %d, wrong coefficient %d; extra %d" % (
                res["status"], len(res["K"]), integral, len(full), len(missing), len(wrong), len(extra)), flush=True)
            for k in extra:
                print("    " + fmt_term(C, k, full[k]))
            for k in missing[:20]:
                print("    MISSING tree term " + fmt_term(C, k, 1))
            for k in wrong[:20]:
                print("    WRONG coefficient " + fmt_term(C, k, full[k]))
        if res["K"]:
            kern = []
            for kv in res["K"]:
                kern.append([[sorted(cands[i][0]), sorted(cands[i][1]), list(cands[i][2]), dict(cands[i][3]), str(v)] for i, v in kv.items()])
            out["kernel"] = kern
            if verbose:
                for kv in res["K"][:6]:
                    print("    kernel vector:", ["%s %s->%s %s" % (v, sorted(cands[i][0]), sorted(cands[i][1]), W.cycle_str(cands[i][2])) for i, v in kv.items()][:12])
        if save_it and integral:
            save(C, os.path.join(A8.DATA, case_name(ct, red, word, monotone, basis_mode) + ".json"), full, out)
    else:
        if verbose:
            print("  status", res["status"], flush=True)
    out["seconds"] = time.time() - t0
    return out, res, C, cands


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("type")
    ap.add_argument("rank", type=int)
    ap.add_argument("red", type=int)
    ap.add_argument("word")
    ap.add_argument("--monotone", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--basis", default="klrw")
    a = ap.parse_args()
    run([a.type, a.rank], a.red, [int(c) for c in a.word], monotone=a.monotone, save_it=a.save, procs=a.procs, basis_mode=a.basis)
