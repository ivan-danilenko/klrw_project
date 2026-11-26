import sys, itertools
sys.path.insert(0, '/private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code')
from deodhar_minuscule import cartan, reflect, minuscule_words, is_tight

def heap(A, word):
    # bottom-up order: first applied to Lambda = rightmost letter = bottom
    seq = list(reversed(word))
    L = len(seq)
    less = [[False]*L for _ in range(L)]
    for i in range(L):
        for j in range(i+1, L):
            if A[seq[i]][seq[j]] != 0:
                less[i][j] = True
    for k in range(L):          # transitive closure
        for i in range(L):
            if less[i][k]:
                for j in range(L):
                    if less[k][j]: less[i][j] = True
    return seq, less

def has_pattern(A, word):
    seq, less = heap(A, word)
    n = len(A); L = len(seq)
    for c in range(n):
        xs = [p for p in range(L) if seq[p] == c]   # increasing = chain bottom-up
        nbrs = [d for d in range(n) if A[c][d] == -1]
        for t in range(len(xs)-2):
            x1, x2, x3 = xs[t], xs[t+1], xs[t+2]
            good = []
            for d in nbrs:
                ds = [p for p in range(L) if seq[p] == d]
                low = any(less[x1][p] and less[p][x2] for p in ds)
                high = any(less[x3][p] for p in ds)
                if low and high: good.append(d)
            if len(good) >= 2:
                return True, (c, good)
    return False, None

def run(typ, n, k, deodhar=True):
    A = cartan(typ, n)
    words = minuscule_words(A, k)
    maxlen = max(len(w) for w in words.values())
    rows = []
    for mu, wd in sorted(words.items(), key=lambda t: len(t[1])):
        pat, info = has_pattern(A, wd)
        tight = is_tight(A, wd)[0] if deodhar else None
        rows.append((len(wd), pat, tight, mu, info))
    pat_sizes = sorted(r[0] for r in rows if r[1])
    nd_sizes = sorted(r[0] for r in rows if r[2] is False)
    print(f"{typ}{n} omega_{k+1} (dim {len(words)}, |P|={maxlen}): pattern at sizes {pat_sizes}; non-Deodhar at sizes {nd_sizes}")
    disagree = [(r[0], r[1], r[2], r[3]) for r in rows if r[2] is not None and r[1] != (not r[2])]
    if disagree: print("   disagreements (size, pattern, tight, weight):", disagree)
    return rows

for t, n, k in [('A',7,3), ('A',8,3), ('A',9,4), ('A',8,2), ('D',6,5), ('D',5,4), ('D',7,6), ('D',7,0), ('E',6,0)]:
    run(t, n, k)
