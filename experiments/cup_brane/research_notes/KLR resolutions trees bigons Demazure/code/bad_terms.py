import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code')
from deodhar_minuscule import cartan, minuscule_words, is_tight
from parabolic_check import peel
def show(t, n, k, size):
    A = cartan(t, n); words = minuscule_words(A, k)
    for mu, wd in words.items():
        if len(wd) != size: continue
        ok, sz, bad = is_tight(A, wd)
        print(f"{t}{n} w{k+1} heap {size} weight {mu}: Deodhar={ok}; #y with coefficient outside vZ[v]: {len(bad)}")
        for y, p in sorted(bad, key=lambda b: -len(peel(A, b[0])))[:6]:
            yw = peel(A, y)
            print("    y of length", len(yw), "word", [i+1 for i in yw], "coeff of H_y:", dict(sorted(p.items())))
show('A', 7, 3, 15); show('D', 6, 5, 13); show('E', 6, 0, 13); show('E', 6, 0, 14)
