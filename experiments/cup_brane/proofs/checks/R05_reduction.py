"""
Referee check 3: the Reduction Lemma (Prop. 4.5) and the cocycle identity (Lemma 4.1) under
deliberate perturbations of F.

    sage -python R05_reduction.py NWORDS MAXLEN SEED [T1|any]

For a random word w = w'.c whose prefix satisfies d'^2 = 0 (checked numerically), let F be the family
of section 3 and E(F) = iota_R(d') F - F iota_L(d').  E is linear in F.  For perturbations G:

  drop / flip / swap-t  : remove one (gamma)-term, change its sign, replace its t_{x,y} by t_{y,x}
  random-1              : one random element in one random component
  random-a              : several random elements, all in components (J', K') with a in K'
                          (a = a first-occurrence slot of w') -- the adversarial case
  coboundary            : G = iota_R(d') H + H iota_L(d')  (then E(G) must be 0)

we compute E(F + G) and test
  (i)   Lemma 4.1 for E(F+G)  (must hold for every F),
  (ii)  Prop. 4.5: if E(F+G) != 0 then for EVERY first-occurrence slot a of w' there is a non-zero
        component E_{I',K'} with a not in K'.
  (iii) control: for slots that are NOT first occurrences, how often all non-zero components have
        the slot in K' (the lemma does not claim anything there).
"""
import sys, random, itertools, time
from R05_lib import *


def random_element(S, src, tgt_vertex_arr, rng):
    """Random basis-like element from arrangement src to the idempotent of tgt (colour-preserving bijection)."""
    # tgt_vertex_arr is a list of slot labels; choose for each colour class a random bijection
    by = defaultdict(list)
    for lab in src:
        by[S.col(lab)].append(lab)
    for v in by.values():
        rng.shuffle(v)
    tgt = [by[S.col(lab)].pop() for lab in tgt_vertex_arr]
    w = rand_reduced(src, tgt, rng)
    x = S.el(src, w)
    if rng.random() < 0.4:
        lab = rng.choice([l for l in src if l != "W"])
        x = S.dot(src, lab) * x
    return rng.choice([1, -1, 2]) * x


def run(ct, red, word, rng, st):
    word = tuple(word)
    m = len(word); n = m - 1
    c = word[-1]
    S = Setup(ct, red, word)
    Sp = Setup(ct, red, word[:-1])
    subsets = [frozenset(x) for k in range(n + 1) for x in itertools.combinations(range(1, n + 1), k)]
    Xword = tuple(range(m, 0, -1))
    arrR = lambda Ip: S.arr(set(Ip) | {m})
    arrL = lambda Ip: S.arr(Ip)
    dR, dL, F = {}, {}, {}

    def add(d, k_, x):
        d[k_] = d[k_] + x if k_ in d else x
    gam_terms = []
    for Ip in subsets:
        for t in Sp.my_terms(Ip):
            w_ = rand_reduced(Sp.arr(Ip), Sp.target_arr(Ip, t.j, t.F), rng)
            cf = S.coef(t)
            add(dR, (Ip, t.J), cf * S.el(arrR(Ip), w_))
            add(dL, (Ip, t.J), cf * S.el(arrL(Ip), tuple(i + 1 for i in w_)))
        add(F, (Ip, Ip), S.el(arrR(Ip), Xword))
        for t in S.my_terms(Ip | {m}):
            if t.j != m and any(m in p for p in t.F):
                el = S.el(arrR(Ip), S.beta_word(Ip | {m}, t.j, t.F, rng))
                add(F, (Ip, t.J), S.coef(t) * el)
                gam_terms.append((Ip, t, el))
    succ = defaultdict(list); pred = defaultdict(list)
    for (Ip, Jp) in dR:
        succ[Ip].append(Jp); pred[Jp].append(Ip)

    # d'^2 = 0 ?
    for Ip in subsets:
        tot = {}
        for Jp in succ[Ip]:
            for Kp in succ[Jp]:
                add(tot, Kp, dR[Ip, Jp] * dR[Jp, Kp])
        if any(not v.is_zero() for v in tot.values()):
            st["words skipped (d'^2 != 0)"] += 1
            return

    def Eof(G):
        E = {}
        for (Jp, Kp), g in G.items():
            for Ip in pred[Jp]:
                add(E, (Ip, Kp), dR[Ip, Jp] * g)
            for Lp in succ[Kp]:
                add(E, (Jp, Lp), -(g * dL[Kp, Lp]))
        return {k_: v for k_, v in E.items() if not v.is_zero()}

    def cocycle_ok(E):
        tot = {}
        for (Jp, Kp), e in E.items():
            for Ip in pred[Jp]:
                add(tot, (Ip, Kp), dR[Ip, Jp] * e)
            for Lp in succ[Kp]:
                add(tot, (Jp, Lp), e * dL[Kp, Lp])
        return all(v.is_zero() for v in tot.values())

    E0 = Eof(F)
    st["words"] += 1
    if E0:
        st["words with E(F) != 0 (tree formula fails for w)"] += 1
    firsts = [s for s in range(1, n + 1) if word.index(word[s - 1]) == s - 1]
    others = [s for s in range(1, n + 1) if s not in firsts]

    def test(name, G):
        E = Eof(G)
        # E(F + G) = E(F) + E(G)
        for k_, v in E0.items():
            add(E, k_, v)
        E = {k_: v for k_, v in E.items() if not v.is_zero()}
        st[name + ": perturbations"] += 1
        if not cocycle_ok(E):
            st["FAIL Lemma 4.1"] += 1
            print("FAIL 4.1", ct, red, word, name, flush=True)
        if not E:
            st[name + ": E = 0"] += 1
            return
        st[name + ": E != 0"] += 1
        for a in firsts:
            if not any(a not in Kp for (Ip, Kp) in E):
                st["FAIL Prop 4.5"] += 1
                print("FAIL 4.5", ct, red, word, name, "slot", a, [(sorted(i), sorted(k)) for i, k in E][:5], flush=True)
            else:
                st[name + ": (E != 0, first-occurrence slot) violation seen with a notin K'"] += 1
        for a in others:
            st["control: (E != 0, non-first slot)"] += 1
            if not any(a not in Kp for (Ip, Kp) in E):
                st["control: all non-zero components have the non-first slot in K'"] += 1

    test("unperturbed", {})
    for Ip, t, el in rng.sample(gam_terms, min(len(gam_terms), 6)):
        cf = S.coef(t)
        test("drop", {(Ip, t.J): -cf * el})
        test("flip", {(Ip, t.J): -2 * cf * el})
        cf2 = S.D(t.sign)
        for cp, cx in t.mono:
            cf2 = cf2 * S.r(cp) * S.t(cx, cp)
        test("swap-t", {(Ip, t.J): (cf2 - cf) * el})
    if gam_terms:
        test("drop all of F_1", {(Ip, t.J): 0 * el for Ip, t, el in gam_terms} and
             {k_: -v for k_, v in _sum_terms(S, gam_terms).items()})
    for _ in range(6):
        Jp, Kp = rng.choice(subsets), rng.choice(subsets)
        test("random-1", {(Jp, Kp): random_element(S, arrR(Jp), arrL(Kp), rng)})
    for a in firsts:
        for _ in range(3):
            G = {}
            for _ in range(rng.randint(2, 5)):
                Jp = rng.choice(subsets)
                Kp = rng.choice([K for K in subsets if a in K])
                add(G, (Jp, Kp), random_element(S, arrR(Jp), arrL(Kp), rng))
            test("random-a", G)
    for _ in range(3):
        H = {}
        for _ in range(rng.randint(1, 3)):
            Jp, Kp = rng.choice(subsets), rng.choice(subsets)
            add(H, (Jp, Kp), random_element(S, arrR(Jp), arrL(Kp), rng))
        G = {}
        for (Jp, Kp), h in H.items():
            for Ip in pred[Jp]:
                add(G, (Ip, Kp), dR[Ip, Jp] * h)
            for Lp in succ[Kp]:
                add(G, (Jp, Lp), h * dL[Kp, Lp])
        E = Eof(G)
        st["coboundary: perturbations"] += 1
        if E:
            st["FAIL coboundary (E(G) != 0)"] += 1
            print("FAIL coboundary", ct, red, word, flush=True)


def _sum_terms(S, gam_terms):
    out = {}
    for Ip, t, el in gam_terms:
        k_ = (Ip, t.J)
        x = S.coef(t) * el
        out[k_] = out[k_] + x if k_ in out else x
    return out


if __name__ == "__main__":
    nwords, maxlen, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    kind = sys.argv[4] if len(sys.argv) > 4 else "T1"
    rng = random.Random(seed)
    st = defaultdict(int)
    t0 = time.time()
    done = 0
    while done < nwords:
        ct = rng.choice([["A", 2], ["A", 3], ["A", 4], ["D", 4]])
        rank = ct[1]
        L = rng.randint(3, maxlen)
        if kind == "T1":
            if L > 2 * rank:
                continue
            word = rng.sample([x for x in range(1, rank + 1) for _ in range(2)], L)
            if word.count(word[-1]) != 2:
                continue
        else:
            word = [rng.randint(1, rank) for _ in range(L)]
            if word.count(word[-1]) < 2:
                continue
        red = rng.randint(1, rank)
        before = st["words"]
        run(ct, red, word, rng, st)
        done += 1
    print("RESULT reduction (%s words, maxlen %d, seed %d) [%.0fs]" % (kind, maxlen, seed, time.time() - t0))
    for k_ in sorted(st):
        print("   %-80s %d" % (k_, st[k_]))
