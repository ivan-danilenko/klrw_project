import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code')
from deodhar_minuscule import cartan, minuscule_words
from pattern_vs_deodhar import heap
for t, n, k, size, c in [('E',6,0,13,3), ('E',6,0,14,3), ('D',6,5,13,3), ('A',7,3,15,3)]:
    A = cartan(t, n); words = minuscule_words(A, k)
    wd = [w for w in words.values() if len(w) == size][0]
    seq, less = heap(A, wd)
    lab = []; cnt = {}
    for x in seq:
        cnt[x] = cnt.get(x, 0) + 1; lab.append(f"{x+1}.{cnt[x]}")
    xs = [p for p in range(len(seq)) if seq[p] == c]
    # interleave: for each adjacent colour element, which gap of the colour-c chain it is in
    desc = []
    for p in range(len(seq)):
        if A[c][seq[p]] == -1:
            gap = sum(1 for x in xs if less[x][p])   # number of c-elements below p
            desc.append(f"{lab[p]}:gap{gap}")
    print(f"{t}{n} heap {size}: colour {c+1} chain {[lab[x] for x in xs]}; neighbours by gap (gap g = above g chain elements): {desc}")
