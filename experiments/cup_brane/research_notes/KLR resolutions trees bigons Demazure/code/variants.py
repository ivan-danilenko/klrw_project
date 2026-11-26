import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code')
from deodhar_minuscule import cartan, minuscule_words
from pattern_vs_deodhar import heap

def pattern_variant(A, word, variant):
    seq, less = heap(A, word)
    n = len(A); L = len(seq)
    for c in range(n):
        xs = [p for p in range(L) if seq[p] == c]
        for t in range(len(xs)-2):
            x1, x2, x3 = xs[t], xs[t+1], xs[t+2]
            pairs = []   # (colour, low element, high element)
            for p in range(L):
                if A[c][seq[p]] != -1: continue
                if less[x1][p] and less[p][x2]:
                    for q in range(L):
                        if seq[q] == seq[p] and less[x3][q]:
                            pairs.append((seq[p], p, q))
            if variant == 'distinct_colours':
                if len({pc for pc, _, _ in pairs}) >= 2: return True
            elif variant == 'distinct_lows':      # c' may equal c''
                if len({lo for _, lo, _ in pairs}) >= 2: return True
            elif variant == 'one_pair':
                if pairs: return True
    return False

for t, n, k in [('A',7,3), ('A',8,2), ('D',6,5), ('D',5,4), ('E',6,0), ('D',7,0)]:
    A = cartan(t, n); words = minuscule_words(A, k)
    out = []
    for v in ('distinct_colours', 'distinct_lows', 'one_pair'):
        sizes = sorted(len(wd) for wd in words.values() if pattern_variant(A, wd, v))
        out.append(f"{v}: {sizes}")
    print(f"{t}{n} w{k+1}: " + " | ".join(out))

# E6: print the heap of the 13- and 14-element ideals with colours by level
A = cartan('E', 6); words = minuscule_words(A, 0)
for mu, wd in sorted(words.items(), key=lambda t: len(t[1])):
    if len(wd) >= 12:
        print(len(wd), "weight", mu, "word(bottom-up, Bourbaki labels):", [i+1 for i in reversed(wd)])
