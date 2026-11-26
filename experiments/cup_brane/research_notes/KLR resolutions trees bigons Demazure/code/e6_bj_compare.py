import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code')
from deodhar_minuscule import cartan, reflect, minuscule_words, is_tight
A = cartan('E', 6)
def elt(word):  # key x^{-1} rho for x = product of word left-to-right
    mu = tuple([1]*6)
    for s in word: mu = reflect(mu, s, A)
    return mu
# BJ labels: chain 0-1-2-3-4, 5 attached to 2  ->  Bourbaki (0-based): 0->0(node1),1->2(node3),2->3(node4),3->4(node5),4->5(node6),5->1(node2)
bj2b = {0:0, 1:2, 2:3, 3:4, 4:5, 5:1}
flip = {0:5, 2:4, 3:3, 4:2, 5:0, 1:1}   # E6 diagram automorphism (Bourbaki 0-based)
BJ = ["5 0 1 2 5 3 4 2 3 1 2 5 0 1", "5 1 2 3 0 1 2 5 4 3 2 1 0", "1 2 5 3 4 2 3 1 2 5 0 1 2", "2 5 1 2 3 0 1 2 5 4 3 2 1"]
pats = []
for b in BJ:
    w = [bj2b[int(t)] for t in b.split()]
    ok = is_tight(A, w)[0]
    pats.append(w)
    print("BJ E6 pattern", b, "len", len(w), "tight(Deodhar)?", ok)
words = minuscule_words(A, 0)
for mu, wd in words.items():
    if len(wd) in (13, 14):
        x = elt(wd)
        hits = []
        for i, p in enumerate(pats):
            for name, q in (("p", p), ("p^-1", p[::-1]), ("flip p", [flip[s] for s in p]), ("flip p^-1", [flip[s] for s in p[::-1]])):
                if len(q) == len(wd) and elt(q) == x: hits.append((i, name))
        print("minuscule heap size", len(wd), "weight", mu, "equals BJ pattern(s):", hits)
