"""
Minimal interface to the KLRW algebra for checking identities by hand.

    from alg import Alg
    A = Alg(["A", 2], "1 W1 2 1")       # colour sequence, left to right; W<k> = red strand of colour k
    e  = A.e("1 W1 2 1")                 # idempotent
    s  = A.x("1 W1 2 1", 3)              # crossing of positions 3,4 (1-based), read FROM the given state
    w  = A.word("1 W1 2 1", (3, 2, 3))   # product of crossings, first letter first
    y  = A.dot("1 W1 2 1", 4)            # dot on the strand at position 4 (times the idempotent)
    A.r(1), A.t(1, 2), A.u(1)            # parameters r_c, t_{cc'}, u_c  (elements of the base ring)

Products: ``a * b`` means "first a, then b" (a is drawn first, starting from its left state).
All colour sequences of one algebra must be permutations of each other.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from cupbrane import *   # noqa
from klrw.framed_dynkin import NodeInFramedQuiver, FramedDynkinDiagram_with_dimensions
from klrw.klrw_algebra import KLRWAlgebra
from sage.combinat.root_system.cartan_type import CartanType
from sage.rings.integer_ring import ZZ


def _node(tok):
    return NodeInFramedQuiver(int(tok[1:]), framing=True) if tok[0] == "W" else NodeInFramedQuiver(int(tok))


class Alg:
    def __init__(self, ct, seq, base=ZZ, invertible=False):
        toks = seq.split() if isinstance(seq, str) else list(seq)
        DD = FramedDynkinDiagram_with_dimensions(CartanType(ct))
        for t in toks:
            DD[_node(t)] += 1
        self.K = KLRWAlgebra(base, DD, dot_scaling=True, edge_scaling=True, vertex_scaling=True,
                             invertible_parameters=invertible, warnings=True)
        self.D = self.K.base()
        self.States = self.K.KLRWBraid.KLRWstate_set

    def state(self, seq):
        toks = seq.split() if isinstance(seq, str) else list(seq)
        return self.States._element_constructor_(tuple(_node(t) for t in toks))

    def e(self, seq):
        return self.K.idempotent(self.state(seq))

    def word(self, seq, word):
        """Product of crossings; letters are 1-based positions, the first letter acts first on ``seq``."""
        return self.K.element_from_braid(tuple(word), left_state=self.state(seq))

    def x(self, seq, i):
        return self.word(seq, (i,))

    def dot(self, seq, pos):
        st = self.state(seq)
        idx = st.index_among_same_color(pos - 1)
        return self.D.dot_variable(st[pos - 1], idx).monomial * self.K.idempotent(st)

    def r(self, c):
        return self.D.vertex_variable(NodeInFramedQuiver(c)).monomial

    def t(self, c, c2):
        return self.D.edge_variable(NodeInFramedQuiver(c), NodeInFramedQuiver(c2)).monomial

    def u(self, c):
        return self.D.edge_variable(NodeInFramedQuiver(c), NodeInFramedQuiver(c, framing=True)).monomial


if __name__ == "__main__":
    A = Alg(["A", 2], "1 2 1")
    s = "1 2 1"
    lhs = A.word(s, (1, 2, 1)); rhs = A.word(s, (2, 1, 2))
    print("bad-1, state (1,2,1):  psi1 psi2 psi1 - psi2 psi1 psi2 =", lhs - rhs)
    A = Alg(["A", 2], "2 1 1")
    for s in ("1 1 2", "2 1 1", "1 2 1"):
        print("  state", s, ": 121 - 212 =", A.word(s, (1, 2, 1)) - A.word(s, (2, 1, 2)))
    A = Alg(["A", 2], "1 W1 1")
    for s in ("1 W1 1", "W1 1 1", "1 1 W1"):
        print("bad-2? state", s, ": 121 - 212 =", A.word(s, (1, 2, 1)) - A.word(s, (2, 1, 2)))
    A = Alg(["A", 2], "1 2")
    print("bigon (1,2):", A.word("1 2", (1, 1)), "   bigon (2,1):", A.word("2 1", (1, 1)))
    A = Alg(["A", 2], "1 W1")
    print("red bigon (1,W1):", A.word("1 W1", (1, 1)), "   (W1,1):", A.word("W1 1", (1, 1)))
    A = Alg(["A", 2], "2 W1")
    print("red bigon (2,W1):", A.word("2 W1", (1, 1)))
    A = Alg(["A", 2], "1 1")
    s = "1 1"
    x1, x2, psi = A.dot(s, 1), A.dot(s, 2), A.x(s, 1)
    print("x1*psi - psi*x2 =", x1 * psi - psi * x2, "   psi*x1 - x2*psi =", psi * x1 - x2 * psi)
    phi = (x1 - x2) * psi - A.r(1) * A.e(s)
    print("phi = (x1-x2) psi - r:  phi^2 =", phi * phi, "  x1*phi - phi*x2 =", x1 * phi - phi * x2, "  phi + psi*(x1-x2) - r =", phi + psi * (x1 - x2) - A.r(1) * A.e(s))
    print("dot names:", x1, x2)
