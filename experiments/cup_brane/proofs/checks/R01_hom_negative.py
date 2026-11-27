"""
R01_hom_negative.py -- negative control for R01_klrw.py hom: a polynomial representation with a
wrong sign (same-colour crossing acting by -r*d instead of r*d, or t_{ab} and t_{ba} exchanged)
must be detected by the homomorphism test.   sage -python proofs/checks/R01_hom_negative.py
"""
import random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import R01_lib
import R01_klrw

orig_psi = R01_lib.PolyRepR.psi
orig_Q = R01_lib.PolyRepR.Q


def bad_psi(self, s, i, f):
    t, g = orig_psi(self, s, i, f)
    a, b = s[i], s[i + 1]
    if a == b and not R01_lib.is_red(a):
        g = -g
    return t, g


def bad_Q(self, a, b, yi, yj):
    if not R01_lib.is_red(a) and not R01_lib.is_red(b) and a != b and self.adj(a, b):
        return self.p[("t", b, a)] * yi + self.p[("t", a, b)] * yj
    return orig_Q(self, a, b, yi, yj)


for name, patch in [("psi_same -> -psi_same", ("psi", bad_psi)), ("t_ab <-> t_ba in Q", ("Q", bad_Q))]:
    setattr(R01_lib.PolyRepR, patch[0], patch[1])
    print("negative control:", name)
    R01_klrw.CASES = R01_klrw.CASES[:3]
    R01_klrw.test_hom(random.Random(5), n_pairs=15)
    R01_lib.PolyRepR.psi = orig_psi
    R01_lib.PolyRepR.Q = orig_Q
