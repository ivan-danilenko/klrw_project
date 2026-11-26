"""
Checks for proofs/06_main_intertwiner.md, sections 2-4:

  (a) local identities of the intertwiner phi = (x_1 - x_2) psi - r
  (b) naturality:  (a (x) 1) * Phi = Phi * (1 (x) a)  for every generator a (dot, crossing) of the
      algebra of w' at EVERY state (every ordering of the strands of w' and the red strand)
  (c) z = prod (x_m - x_y) commutes with (a (x) 1) and (1 (x) a)
  (d) d_Phi^2 = 0 on the cube
  (e) Phi_j(I) = sum over subsets (expansion of the intertwiners)   [only printed for inspection]

    sage -python proofs/checks/06_check_basic.py TYPE RANK RED WORD [RED WORD ...]
"""
import sys, itertools, importlib
L = importlib.import_module("06_lib")


def all_states(S, slots):
    for perm in itertools.permutations(list(slots) + ["W"]):
        yield tuple(perm)


def Phi_state(S, lab, m):
    """strand m at the far right of lab + (m,) moves to the far left, intertwiners for equal colours."""
    full = tuple(lab) + (m,)
    n = len(full)
    el, new = S.word_el(full, tuple(range(n - 1, 0, -1)), intertwine=True)
    assert new == (m,) + tuple(lab)
    return el


def check_naturality(S):
    m = S.m
    slots = list(range(1, m))
    n_checked = 0
    for lab in all_states(S, slots):
        n = len(lab)
        PhiL = Phi_state(S, lab, m)
        # dots
        for pos in range(1, n + 1):
            if lab[pos - 1] == "W":
                continue
            lhs = S.dot(lab + (m,), pos) * PhiL
            rhs = PhiL * S.dot((m,) + lab, pos + 1)
            assert (lhs - rhs).is_zero(), ("dot", lab, pos)
            n_checked += 1
        # the dot of strand m itself
        lhs = S.dot(lab + (m,), n + 1) * PhiL
        rhs = PhiL * S.dot((m,) + lab, 1)
        assert (lhs - rhs).is_zero(), ("dot m", lab)
        # crossings
        for i in range(1, n):
            a_right, new = S.psi(lab + (m,), i)            # a (x) 1, read from lab
            a_left, new2 = S.psi((m,) + lab, i + 1)        # 1 (x) a
            lab2 = new[:-1]
            lhs = a_right * Phi_state(S, lab2, m)
            rhs = PhiL * a_left
            assert (lhs - rhs).is_zero(), ("crossing", lab, i)
            n_checked += 1
    return n_checked


def check_z(S):
    m = S.m
    slots = list(range(1, m))
    for lab in all_states(S, slots):
        n = len(lab)
        for full, off in ((lab + (m,), 0), ((m,) + lab, 1)):
            z = S.z(None, m, lab=full)
            for pos in range(1, n + 1):
                if lab[pos - 1] != "W":
                    d = S.dot(full, pos + off)
                    assert (z * d - d * z).is_zero()
            for i in range(1, n):
                a, new = S.psi(full, i + off)
                z2 = S.z(None, m, lab=new)
                assert (z * a - a * z2).is_zero(), ("z", full, i)
    return True


def check_dPhi2(S):
    dP = S.full(S.dPhi)
    sq = S.compose(dP, dP)
    return all(not row for row in sq.values())


def check_tree(S):
    d = S.full(S.d)
    sq = S.compose(d, d)
    return all(not row for row in sq.values())


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    args = sys.argv[3:]
    for red, word in zip(args[0::2], args[1::2]):
        S = L.Setup(ct, int(red), tuple(int(c) for c in word))
        n = check_naturality(S)
        print("{} red {} word {}: naturality of Phi_m on all states: OK ({} identities); z central: {}; d_Phi^2 = 0: {}; tree d^2 = 0: {}".format(
            "".join(map(str, ct)), red, word, n, check_z(S), check_dPhi2(S), check_tree(S)), flush=True)
