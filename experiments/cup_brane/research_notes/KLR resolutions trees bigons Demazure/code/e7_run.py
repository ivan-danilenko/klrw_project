import sys, time
sys.path.insert(0, '/private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code')
from deodhar_minuscule import cartan, minuscule_words, is_tight
from variants import pattern_variant
for (t, n, k) in [('E', 7, 6), ('D', 8, 7)]:
    A = cartan(t, n); words = minuscule_words(A, k)
    res = []
    for mu, wd in sorted(words.items(), key=lambda t: len(t[1])):
        t0 = time.time()
        ok, sz, _ = is_tight(A, wd)
        pv = pattern_variant(A, wd, 'distinct_lows'); pc = pattern_variant(A, wd, 'distinct_colours')
        res.append((len(wd), ok, pv, pc))
        print(t, n, "size", len(wd), "Deodhar", ok, "pattern(c'=c'' allowed)", pv, "pattern(distinct colours)", pc, "interval", sz, f"{time.time()-t0:.1f}s", flush=True)
    print("SUMMARY", t, n, "non-Deodhar sizes", sorted(r[0] for r in res if not r[1]), "pattern sizes", sorted(r[0] for r in res if r[2]), flush=True)
