"""
Pure-Python checks for section 7 of proofs/04_first_order.md.

    python3 proofs/checks/04_degree_defect.py defect TYPE RANK MAXLEN SAMPLES
    python3 proofs/checks/04_degree_defect.py mult2  TYPE RANK MAXLEN

defect:  Proposition 7.1 (degree defect of an arbitrary bijection) against the direct count of
         degree_model.Model, for SAMPLES random (word, red, I, J, sigma) with |J| arbitrary.
mult2:   Proposition 7.4: for every word in which every colour occurs at most twice, the
         monotone (no strand crosses W from left to right) homogeneous basis elements of
         r-degree 1 are, in homological degree -1, exactly r_c(p) t_c(p)c(j) beta(I; j, p) for
         the linked one-pair configurations (no dots), and in homological degree 0 there are none.
"""
import sys, os, random, itertools, importlib
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from degree_model import Model  # noqa: E402
Dg = importlib.import_module("04_degree")


def bracket(model, c, c2):
    if c == c2:
        return ("s", c)
    if frozenset((c, c2)) in model.E:
        return ("e", min(c, c2), max(c, c2))
    return None


def defect_formula(model, I, J, sigma):
    """Proposition 7.1."""
    out = Counter()
    m = model.m
    for sA, sB in itertools.combinations(range(1, m + 1), 2):     # sA < sB (source slots)
        br = bracket(model, model.col[sA - 1], model.col[sB - 1])
        if br is None:
            continue
        tA, tB = sigma[sA], sigma[sB]
        if tA < tB:
            f = -2 if (sB not in I and tB in J) else 0
        else:
            f = 1 if (sB in I and tA not in J) else -1
        out[br] += f
    red = model.reds[0]
    for s in range(1, m + 1):
        if model.col[s - 1] == red and s not in I and sigma[s] in J:
            out[("w", red)] -= 2
    return out


def direct(model, I, J, sigma):
    t = Counter(model.D(I))
    t.subtract(model.D(J))
    t.subtract(model.content(I, J, sigma))
    return t


def norm(c):
    return {k: v for k, v in c.items() if v}


def run_defect(ct, maxlen, samples, seed=1):
    rng = random.Random(seed)
    bad = 0
    for _ in range(samples):
        L = rng.randint(1, maxlen)
        w = tuple(rng.randint(1, ct[1]) for _ in range(L))
        red = rng.randint(1, ct[1])
        model = Model(ct, [(red, w)])
        I = frozenset(s for s in range(1, L + 1) if rng.random() < 0.5)
        J = frozenset(s for s in range(1, L + 1) if rng.random() < 0.5)
        bij = list(model.bijections())
        sigma = rng.choice(bij)
        if norm(defect_formula(model, I, J, sigma)) != norm(direct(model, I, J, sigma)):
            bad += 1
            print("FAIL", w, red, sorted(I), sorted(J), sigma)
    print("defect formula:", samples, "samples,", bad, "failures")


def run_mult2(ct, maxlen):
    from formula import Stack
    bad = 0; n = Counter()
    for L in range(1, maxlen + 1):
        for w in itertools.product(range(1, ct[1] + 1), repeat=L):
            if max(w.count(c) for c in set(w)) > 2:
                continue
            st = Stack(ct, w)
            for red in range(1, ct[1] + 1):
                model = Model(ct, [(red, w)])
                for hdeg in (-1, 0):
                    found = set()
                    for (I, J, sigma, nn, mm, ll, aa) in Dg.all_elements(model, hdeg):
                        if sum(nn.values()) != 1:
                            continue
                        if any(s not in I and sigma[s - 1] in J for s in range(1, L + 1)):
                            continue
                        mono = tuple(sorted([("r_%d" % c, v) for c, v in nn.items() if v] +
                                            [("t_%d_%d" % k, v) for k, v in mm.items() if v]))
                        found.add((I, J, sigma, mono, tuple(sorted((c, v) for c, v in aa.items() if v))))
                    if hdeg == 0:
                        n["hdeg 0 monotone first-order elements"] += len(found)
                        bad += len(found)
                        continue
                    pred = set()
                    for h in range(L + 1):
                        for I in itertools.combinations(range(1, L + 1), h):
                            for J, sigma, mono, sign in st.terms_from(frozenset(I)):
                                if sum(p for nm, p in mono if nm.startswith("r_")) == 1:
                                    pred.add((frozenset(I), J, sigma, mono, ()))
                    n["hdeg -1 monotone first-order elements"] += len(found)
                    n["one-pair terms"] += len(pred)
                    if found != pred:
                        bad += 1
                        print("FAIL", w, red, len(found), len(pred))
    print("mult<=2 classification:", dict(n), "failures", bad)


if __name__ == "__main__":
    mode = sys.argv[1]
    ct = [sys.argv[2], int(sys.argv[3])]
    if mode == "defect":
        run_defect(ct, int(sys.argv[4]), int(sys.argv[5]))
    else:
        run_mult2(ct, int(sys.argv[4]))
