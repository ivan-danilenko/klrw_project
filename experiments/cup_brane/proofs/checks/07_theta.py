"""
Local nil-Hecke identities used in 07_conjecture_H.md (one colour; labelled calculus of 07_lab).

  omega_y = (x_y - x_m) psi - r      (x_y = dot of the left (stationary) strand before the crossing,
                                      x_m = dot of a strand m not involved)

Theta(r):  J passes a, y_1, ..., y_r  (psi), then the position of a passes y_1..y_r and J with omega's.
           Claim: Theta = 0.

    sage -python 07_theta.py RMAX
"""
import sys, importlib
L = importlib.import_module("07_lab")


def omega_letter(S, src, el, i, mlabel):
    """right multiplication by omega at positions i, i+1: (x_left - x_m) psi_i - r."""
    out = {}
    # (x_left - x_m) is read at the state before the crossing: positional dot of position i
    tmp = {}
    for t, f in el.items():
        tmp[t] = f * (S.x(t[i - 1]) - S.x(mlabel))
    res = S.mul_letter(src, tmp, i, False)
    for t, f in el.items():
        res[t] = res.get(t, S.F.zero()) - S.r(S.col(t[i - 1])) * f
    return {k: v for k, v in res.items() if v != 0}


def theta(rr):
    # labels: m = 0 is not used as a slot; use slots 1..rr+3 all of colour 1; m = rr+3 sits at the far right
    n = rr + 3
    S = L.Lab(["A", 1], 1, tuple([1] * n))
    # positions: y_rr .. y_1, a, J, m      (labels: y_k = k+2 for k = 1..rr, a = 2, J = 1, m = n)
    ys = list(range(3, rr + 3))
    src = tuple(reversed(ys)) + (2, 1, n)
    el = {src: S.F.one()}
    # J (position rr+2) passes a, y_1..y_rr : letters rr+1, rr, ..., 1
    for i in range(rr + 1, 0, -1):
        el = S.mul_letter(src, el, i, False)
    # now (J, y_rr, ..., y_1, a, m): the position of a (rr+2) passes y_1..y_rr and J with omegas
    for i in range(rr + 1, 0, -1):
        el = omega_letter(S, src, el, i, n)
    return el


if __name__ == "__main__":
    for rr in range(0, int(sys.argv[1]) + 1):
        el = theta(rr)
        print("r = {}: Theta = 0: {}   ({} nonzero brackets)".format(rr, not el, len(el)), flush=True)
