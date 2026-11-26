"""
Exhaustive determination of the cyclic order ideals of a minuscule poset (Theorems A, D, E of
proofs/03_cyclic_combinatorics.md).  Pure Python.

    python3 proofs/checks/03_minuscule.py TYPE RANK NODE [--blocks] [--all]

What it does, exactly:
 1. builds a chain omega_k -> lowest weight (03_lib.minuscule_full_word; every step
    subtracts alpha_i with <mu, alpha_i^vee> = 1), and the heap P of this word;
 2. checks that the number of order ideals of P equals the number of weights reachable by
    chains (the W-orbit of omega_k), and that the subword of every ideal is again a chain
    (so ideals <-> weights, the ideal of mu being the slots of any chain omega_k -> mu);
 3. for every ideal I (by increasing size): I is cyclic if some lower cover I - {x}
    (x maximal) is cyclic (PROOFS.md Prop. 7: I - {x} is a prefix of a linear extension of
    I); otherwise the EXHAUSTIVE search 03_lib.Word.search decides (no bound on the number
    of pairs; see Lemma 2 of the write-up).  With --all the search is run on every ideal
    and the inheritance shortcut is only used as a cross-check;
 4. prints the minimal cyclic ideals, as weights, root contents and slot lists, and a
    witness for each; with --blocks also the variant in which blocks are ignored.
"""
import importlib, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L = importlib.import_module("03_lib")


def setup(typ, n, k):
    word = L.minuscule_full_word(typ, n, k)
    adj = L.adjacency(typ, n)
    less = L.heap_order(word, adj)
    ideals = L.order_ideals(len(word), less)
    orbit = L.weight_orbit(typ, n, k)
    assert len(ideals) == len(orbit), (len(ideals), len(orbit))
    wt = {}
    for I in ideals:
        sub = [word[s] for s in I]
        w = L.chain_weights(typ, n, k, sub)[-1]       # raises unless a chain
        wt[I] = w
    assert len(set(wt.values())) == len(ideals)
    # every chain word omega_k -> mu is a linear extension of the ideal of mu: compare the
    # number of chains (paths in the weight graph) with the number of linear extensions
    A = L.cartan(typ, n)
    lam = tuple(1 if j == k else 0 for j in range(n + 1))
    paths = {lam: 1}
    for mu in sorted(orbit, key=lambda mu: -sum(mu[1:])):   # not a topological order in general
        pass
    order = sorted(ideals, key=len)
    by_wt = {wt[I]: I for I in ideals}
    for I in order[1:]:
        mu = wt[I]
        tot = 0
        for i in range(1, n + 1):
            nu = tuple(mu[j] + A[i][j] for j in range(n + 1))          # mu = nu - alpha_i
            if nu in paths and nu[i] == 1:
                tot += paths[nu]
        paths[mu] = tot
    ext = {(): 1}
    for I in order[1:]:
        Is = set(I)
        ext[I] = sum(ext[tuple(y for y in I if y != x)] for x in I if not (less[x] & Is))
    for I in ideals:
        assert paths[wt[I]] == ext[I], (typ, n, k, I)
    return word, adj, less, ideals, wt


def weight_order_check(typ, n, k):
    """mu <= nu (nu - mu a sum of simple roots)  <=>  ideal(mu) contains ideal(nu); returns #ideals."""
    word, adj, less, ideals, wt = setup(typ, n, k)
    cont = {}
    for I in ideals:
        c = [0] * (n + 1)
        for s in I:
            c[word[s]] += 1
        cont[I] = c
    for I in ideals:
        for J in ideals:
            assert all(a <= b for a, b in zip(cont[I], cont[J])) == set(I).issubset(J), (typ, n, k)
    return len(ideals)


def classify(typ, n, k, allow_blocks=False, run_all=False, verbose=True):
    word, adj, less, ideals, wt = setup(typ, n, k)
    maximal = lambda I: [x for x in I if not (less[x] & set(I))]
    cyc = {}
    wit = {}
    t0 = time.time()
    ntrees = 0
    for I in sorted(ideals, key=len):
        inherited = any(cyc[tuple(y for y in I if y != x)] for x in maximal(I))
        if inherited and not run_all:
            cyc[I] = True
            continue
        W = L.Word([word[s] for s in I], adj)
        r = W.search(allow_blocks=allow_blocks)
        ntrees += W.ntrees
        cyc[I] = r is not None
        if r is not None:
            wit[I] = r
        if run_all:
            assert not (inherited and r is None), I        # Prop. 7 cross-check
    minimal = [I for I in ideals if cyc[I] and not any(cyc[tuple(y for y in I if y != x)] for x in maximal(I))]
    if verbose:
        print("%s%d omega_%d: |P| = %d, %d ideals, %d cyclic, %d minimal cyclic%s  [%d tree configurations, %.1fs]" % (
            typ, n, k, len(word), len(ideals), sum(cyc.values()), len(minimal),
            " (blocks ignored)" if allow_blocks else "", ntrees, time.time() - t0))
        for I in minimal:
            cont = {}
            for s in I:
                cont[word[s]] = cont.get(word[s], 0) + 1
            j, F = wit[I]
            sub = [word[s] for s in I]
            print("   minimal cyclic ideal: %d slots, weight %s, root content %s" % (
                len(I), list(wt[I][1:]), [cont.get(c, 0) for c in range(1, n + 1)]))
            print("      word %s" % "".join(map(str, sub)) if n < 10 else "      word %s" % sub)
            print("      witness: mover slot %d (colour %d), pairs %s" % (
                j + 1, sub[j], ["(%d,%d)c%d" % (a + 1, b + 1, sub[a]) for a, b in F]))
    return dict(word=word, ideals=ideals, cyc=cyc, minimal=minimal, wt=wt, less=less)


if __name__ == "__main__":
    typ, n, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    classify(typ, n, k, allow_blocks="--blocks" in sys.argv, run_all="--all" in sys.argv)
