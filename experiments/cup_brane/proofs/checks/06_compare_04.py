"""
Compare the differential obtained by iterated division with H+ (06_divide_all.py) with the
corrected differential of the cyclic word 1221122, red 2, stored by the first-order front in
proofs/checks/04_data/cyclic7_1221122_W2_tree.json (terms [src, tgt, sigma, params, dots, coeff];
the file in which the correction starts from the tree formula; default) or another file of 04_data.

    sage -python proofs/checks/06_compare_04.py [FILE]
"""
import sys, os, json, importlib
from collections import defaultdict
L = importlib.import_module("06_lib")
da = importlib.import_module("06_divide_all")

if __name__ == "__main__":
    L.SHARED = True
    data = json.load(open(os.path.join(L.HERE, "04_data", sys.argv[1] if len(sys.argv) > 1 else "cyclic7_1221122_W2_tree.json")))
    word = tuple(data["word"]); red = data["red"]; ct = data["cartan_type"]
    S = L.Setup(ct, red, word)
    stored = defaultdict(lambda: defaultdict(lambda: S.KLRW.zero()))
    for src, tgt, sigma, params, dots, coeff in data["terms"]:
        I, J = frozenset(src), frozenset(tgt)
        c = S.D(coeff)
        for name, power in list(params) + list(dots):
            c *= S.D(name) ** power
        stored[I][J] += c * S.term_element(I, J, tuple(sigma))
    st = S.full(lambda I: dict(stored.get(I, {})))
    print("stored differential of 04: d^2 = 0:", all(not r for r in S.compose(st, st).values()))
    d = S.dPhi
    for k in range(1, S.m + 1):
        d, failures = da.divide_step(S, d, k)
        assert not failures
    diff = [(I, K) for I in L.subsets(S.m) for K in set(d(I)) | set(st[I])
            if not (d(I).get(K, S.KLRW.zero()) - st[I].get(K, S.KLRW.zero())).is_zero()]
    print("components in which the division differential differs from the stored one:", len(diff))
    for I, K in diff[:6]:
        print("  {} -> {}".format(L.fmt_set(I), L.fmt_set(K)))
        for sigma, w, coeff in L.terms_of(S, I, K, d(I).get(K, S.KLRW.zero())):
            print("      division: sigma {:<22} {}".format(L.cycles_str(sigma), coeff.factor()))
        for sigma, w, coeff in L.terms_of(S, I, K, st[I].get(K, S.KLRW.zero())):
            print("      stored:   sigma {:<22} {}".format(L.cycles_str(sigma), coeff.factor()))
