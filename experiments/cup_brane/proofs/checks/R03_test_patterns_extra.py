"""
Extra coverage for Theorem 2.8: all words of length 8 over A_4 and D_4, all words of length 9 over
A_2 and A_3, all words of length <= 8 over a disconnected forest (A_2 + A_1), and words cut from
large Young-diagram stacks.   python3 R03_test_patterns_extra.py
"""
import itertools, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R03_core import *
import R03_test_patterns as TP

if __name__ == "__main__":
    random.seed(99)
    TP.run_family("A2 all, len=9", "A", 2, [9])
    TP.run_family("A3 all, len=9", "A", 3, [9], check_min=False)
    TP.run_family("A4 all, len=8", "A", 4, [8], check_min=False)
    TP.run_family("D4 all, len=8", "D", 4, [8], check_min=False)
    # disconnected forest A_2 + A_1: colours 1,2 joined; 3 isolated
    E = set([frozenset((1, 2))])
    import R03_core
    old = R03_core.edges
    TP.edges = lambda typ, n: E
    TP.nodes = lambda typ, n: [1, 2, 3]
    TP.run_family("forest A2+A1 all, len<=8", "X", 3, range(1, 9))
    TP.edges = old
    TP.nodes = R03_core.nodes
    # random subwords of the 6x6 box stack (type A_11), length 12-14
    k = 6
    full = [k + s - r for r in range(1, 7) for s in range(1, 7)]
    ws = []
    for _ in range(300):
        L = random.randint(12, 14)
        idx = sorted(random.sample(range(36), L))
        ws.append([full[i] for i in idx])
    TP.run_family("subwords of 6x6 stack x300", "A", 11, None, words=ws, check_min=True)
