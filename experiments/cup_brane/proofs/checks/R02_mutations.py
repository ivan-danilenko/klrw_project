"""
Referee check R02: sensitivity of the own checks (R02_formula.py).  Each mutation of the formula, of the
basis change of Prop. 4.2, or of the R-model must be DETECTED; otherwise the corresponding check is void.

    python3 proofs/checks/R02_mutations.py
"""
import os
import sys
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
F = importlib.import_module('R02_formula')

orig_gsign, orig_coef, orig_terms, orig_sigma, orig_rm = F.gsign, F.coefficient, F.terms, F.sigma, F.rmodel_apply


def reset():
    F.gsign, F.coefficient, F.terms, F.sigma, F.rmodel_apply = orig_gsign, orig_coef, orig_terms, orig_sigma, orig_rm


def mut_sign_i(n, v, tag, w):
    s = orig_gsign(n, v, tag, w)
    return -s if tag[0] == 'i' and tag[1] == 2 else s


def mut_sign_iii(n, v, tag, w):
    s = orig_gsign(n, v, tag, w)
    return -s if tag[0] == 'iii' and tag[2] > v[0] else s


def mut_coef(n, v, tag):
    e = orig_coef(n, v, tag)
    if tag[0] == 'iii' and tag[2] <= v[0] - tag[1]:
        return {k: -x for k, x in e.items()}
    return e


def mut_terms(n, v):
    return [(w, tag) for (w, tag) in orig_terms(n, v) if not (tag[0] == 'iii' and tag[2] == v[0] + 1)]


def mut_sigma(n, v):
    k, s = v
    A = [c for c in range(1, n + 1) if s[c - 1] == 'A']
    C = [c for c in range(1, n + 1) if s[c - 1] == 'C']
    a, g = len(A), len(C)
    return (-1) ** (a * (a - 1) // 2 + g * (g - 1) // 2 + sum(A))


def mut_rm(n, k, R):
    out = orig_rm(n, k, R)
    return [(l, R2, -sg if (l < k and qi is None and k - l == 2) else sg, qi) for (l, R2, sg, qi) in out]


if __name__ == '__main__':
    n = 4
    print('unmutated: d2 (components, non-zero) =', F.check_d2(n, 'mod', 1)[2:], ' rmodel (terms, bad, extra) =',
          F.check_rmodel(n), ' WH =', F.check_WH(n))
    for name, attr, fn in [('sign of (i), d = 2', 'gsign', mut_sign_i),
                           ('sign of (iii), c > k', 'gsign', mut_sign_iii),
                           ('coefficient of (iii), c <= l inverted', 'coefficient', mut_coef),
                           ('drop (iii) with c = k+1', 'terms', mut_terms)]:
        setattr(F, attr, fn)
        print('mutation %-40s d2 non-zero components: %d' % (name, F.check_d2(n, 'mod', 1)[3]))
        reset()
    F.sigma = mut_sigma
    print('mutation %-40s rmodel differing entries: %d' % ('sigma without |B_L||C|', F.check_rmodel(n)[1]))
    reset()
    F.rmodel_apply = mut_rm
    print('mutation %-40s WH differing components: %d' % ('R-model, sign of d = 2 terms (ii)/(iii)', F.check_WH(n)[1]))
    reset()
