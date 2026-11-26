"""Deodhar masks on the heap read bottom-up (word of w^{-1}); list proper masks with
#zero-defects >= #plain-zeros (non-Deodhar witnesses), minimal by number of zeros."""
import sys, itertools
sys.path.insert(0, '/private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code')
from deodhar_minuscule import cartan, reflect, minuscule_words

def witnesses(A, seq, maxzeros):
    n = len(A); L = len(seq); rho = tuple([1]*n)
    lab = []; cnt = {}
    for c in seq:
        cnt[c] = cnt.get(c, 0) + 1; lab.append(f"{c+1}.{cnt[c]}")   # colour.occurrence (bottom-up)
    out = []
    for z in range(1, maxzeros+1):
        for zeros in itertools.combinations(range(L), z):
            zs = set(zeros); mu = rho; zd = pz = 0; defects = []
            for j, s in enumerate(seq):
                d = mu[s] < 0            # s is a right descent of current partial product
                if d: defects.append(j)
                if j in zs:
                    if d: zd += 1
                    else: pz += 1
                else:
                    mu = reflect(mu, s, A)
            if zd >= pz:
                out.append(([lab[j] for j in zeros], [lab[j] for j in defects if j in zs]))
        if out: return z, out
    return None, []

for t, n, k, size in [('A',7,3,15), ('D',6,5,13), ('E',6,0,13), ('E',6,0,14)]:
    A = cartan(t, n); words = minuscule_words(A, k)
    for mu, wd in words.items():
        if len(wd) == size:
            seq = list(reversed(wd))
            z, out = witnesses(A, seq, 8)
            print(f"{t}{n} w{k+1} heap {size}: bottom-up word {[c+1 for c in seq]}")
            print(f"   minimal #zeros in a violating mask: {z}; number of such masks: {len(out)}")
            for zeros, zdef in out[:4]:
                print(f"     zeros at {zeros}; of which zero-defects {zdef}")
