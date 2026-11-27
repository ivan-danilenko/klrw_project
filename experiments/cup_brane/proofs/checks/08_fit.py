"""
08 -- where do the dots of the cyclic term sit?  (proofs/08_cyclic_normal_form.md, §5)

    sage -python 08_fit.py 08_data/FILE_dotted_R3.json [...]

For the cyclic component I0 -> J0 of a stored N_T differential (4-cycle configurations only):
  delta_s = beta_sigma * x_{sigma(s)}  -  x_s * beta_sigma        (dot at the target end minus at the source end)
for every strand s of the configuration (klrw expansion, r-terms only), and a least-squares-free exact
fit of the non-dotted part R of the component (bijections != sigma) by  sum_s mu_s delta_s,  mu_s in
QQ[t] (one unknown per strand and monomial).
"""
import os
import sys
import json
import importlib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
A8 = importlib.import_module("08_alg")
C8 = importlib.import_module("08_comb")
AN = importlib.import_module("08_analyse")
from sage.all import QQ, matrix, vector  # noqa: E402


def dot_at(C, J, slot):
    """the dot on the strand sitting in slot `slot` of the state e_J, times e_J."""
    st = C.cube.state(C.eps(J))
    pos = C.W.labels(J).index(slot)
    idx = st.index_among_same_color(pos)
    return C.D.dot_variable(st[pos], idx).monomial * C.K.idempotent(st)


def main(path):
    W, data, terms = AN.load(path)
    C = A8.Ctx(data["cartan_type"], data["red"], data["word"])
    rep = AN.analyse(path, verbose=False)
    for I, J, sp, dotted, plain in rep:
        (j, F) = None, None
        # recover the configuration from the first spanning tree
        sigma = None
        for (sg, m, c) in terms[I, J]:
            if any(n.startswith("x_") for n in m):
                sigma = sg
        if sigma is None:
            continue
        beta = C.basis(I, J, sigma)
        strands = [s for s in W.slots if sigma[s - 1] != s or (s in I and s not in J)]
        # observed: the whole component as a klrw element
        obs = C.K.zero()
        for (sg, m, c) in terms[I, J]:
            if sg == sigma:
                continue
            obs += C.term(I, J, sg, tuple(sorted(m.items())), c)
        obsv = A8.vec_map(C, {I: {J: obs}})
        # candidate columns: delta_s times each monomial occurring in the dotted part
        monos = sorted({tuple(sorted((n, p) for n, p in m.items() if not n.startswith("x_"))) for m, c in dotted})
        cols, labels = [], []
        for s in strands:
            el = beta * dot_at(C, J, sigma[s - 1]) - dot_at(C, I, s) * beta
            for mo in monos:
                v = A8.vec_map(C, {I: {J: C.mono(mo) * el}})
                if v:
                    cols.append(v)
                    labels.append((s, mo))
        res = A8.solve_affine(cols, obsv)
        print("%s red %d %s: component %s -> %s, sigma %s" % (W.ct, W.red, "".join(map(str, W.col)), sorted(I), sorted(J), W.cycle_str(sigma)))
        print("   observed non-dotted part: %d coefficients" % len(obsv))
        for s in strands:
            el = beta * dot_at(C, J, sigma[s - 1]) - dot_at(C, I, s) * beta
            v = A8.vec_map(C, {I: {J: el}})
            print("   delta_%d = %s" % (s, " ".join("%+d %s %s" % (c, C8.mono_str(tuple((C.names[p], e) for p, e in enumerate(k[3]) if e)),
                                                               W.cycle_str(W.sigma_from_word(I, J, k[2]))) for k, c in v.items()) or "0"))
        for k, c in obsv.items():
            print("   observed %+d %s %s" % (c, C8.mono_str(tuple((C.names[p], e) for p, e in enumerate(k[3]) if e)), W.cycle_str(W.sigma_from_word(I, J, k[2]))))
        if res is None:
            print("   NOT in the span of the delta_s")
        else:
            y0, kb = res
            print("   fit: " + ", ".join("mu[%d, %s] = %s" % (s, C8.mono_str(mo), v) for (s, mo), v in zip(labels, y0) if v), "; freedom", len(kb))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
