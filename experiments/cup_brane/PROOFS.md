# What is proved, what is computed, what is conjectured

This file sorts every claim of `NOTES.md` and `GLUING.md` into three tiers:

* **Tier 1 — proved here.** A complete argument is written out below, using only the standing
  facts of section 1.
* **Tier 2 — proved by finite exact computation.** True for the listed cases because an
  exact computation in the KLRW algebra says so. These are proofs relative to the
  correctness of the `klrw` implementation; they are not human-checkable arguments and say
  nothing about other cases.
* **Tier 3 — conjecture.** Supported by Tier 2 evidence and by partial arguments only.

Short version: the grading and combinatorial statements are proved; the statement that the
tree formula squares to zero, and everything about uniqueness, the cyclic terms and the
gluing maps beyond finitely many cases, is not.

---

## 1. Setting and standing facts

### 1.1 The algebra

Fix a simply-laced Dynkin diagram with node set `Γ`. Write `c ∼ c'` if two nodes are joined
by an edge. Fix `k ∈ Γ` (the colour of the red strand) and a word `(i_1, …, i_m)` in `Γ`.
The algebra `A` is generated over

    R = ℤ[ r_c (c ∈ Γ),  t_{cc'} (c ∼ c', ordered),  u_c (c ∈ Γ) ]

by diagrams with one red strand of colour `k` and `m` black strands of colours
`i_1, …, i_m`, with idempotents `e(s)` for the orderings `s` of these `m+1` strands, dots
`x` on black strands and crossings `ψ` of neighbouring strands (red–red crossings do not
occur). Composition `ab` means "first `a`, then `b`". The relations are the local relations
of the weighted KLR algebra, in the form implemented in `klrw/klrw_algebra.py`:

* (R1) far-apart generators commute; dots commute with each other;
* (R2) a dot slides through a crossing of two strands of different colours, and through a
  red–black crossing; for two black strands of the same colour `c`,
  `x ψ − ψ x' = ± r_c` (the sign depending on which strand carries the dot);
* (R3) double crossings: `ψ² = 0` for equal colours; `ψ² = t_{cc'} x + t_{c'c} x'` for
  `c ∼ c'`; `ψ² = 1` for different, non-adjacent colours; for a black strand of colour `c`
  and the red strand: `ψ² = u_k x` if `c = k` and `ψ² = 1` otherwise;
* (R4) triple points: for three strands that cross pairwise, the two ways of drawing the
  triple point are equal, **except** in the following two *bad* cases, where they differ by
  a multiple of the diagram in which the three strands do not cross:
  * (bad-1) the two outer strands are black of the same colour `c` and the middle strand is
    black of colour `c' ∼ c`: the difference is `∓ r_c t_{cc'}`;
  * (bad-2) the two outer strands are black of colour `k` and the middle strand is the red
    strand: the difference is `∓ r_k u_k`.

  ("Outer" and "middle" refer to the positions of the three strands at one end of the
  triple point.)

### 1.2 Standing facts

* **(F1) Basis.** For every pair of idempotents fix, for every bijection `σ` of the strands
  compatible with colours and fixing the red strand, one diagram `β_σ` realising `σ` in
  which any two strands cross at most once. Then the elements
  `(monomial in dots) · β_σ` form an `R`-basis of `e(s) A e(s')`. This is the basis theorem
  for (weighted) KLR algebras [Khovanov–Lauda; Rouquier; Webster, *Weighted KLR algebras*,
  flatness of the deformation, Prop. 2.23]. **We use it for this particular deformed
  presentation without re-proving it.** It is used only in Proposition 4.
* **(F2) Grading.** Let `G` be the free abelian group on

      q;   σ_c (c ∈ Γ);   δ_c (c ∈ Γ);   ε_{cc'} (unordered edges c ∼ c');   ε_{cW} (c ∈ Γ).

  The relations (R1)–(R4) are homogeneous for

  | generator | degree |
  |---|---|
  | crossing of two black strands of the same colour `c` | `−2q + σ_c` |
  | crossing of black strands of colours `c ∼ c'` | `q + ε_{cc'}` |
  | crossing of a black strand of colour `k` with the red strand | `q + ε_{kW}` |
  | any other crossing | `0` |
  | dot on a strand of colour `c` | `2q + δ_c` |
  | `r_c` | `σ_c + δ_c` |
  | `t_{cc'}` | `2 ε_{cc'} − δ_c` |
  | `u_c` | `2 ε_{cW} − δ_c` |

  (Checked relation by relation: e.g. (R2) `2q + δ_c − 2q + σ_c = σ_c + δ_c`; (R3)
  `2(q + ε_{cc'}) = 2ε_{cc'} − δ_c + 2q + δ_c`; (bad-1)
  `(−2q + σ_c) + 2(q + ε_{cc'}) = (σ_c + δ_c) + (2ε_{cc'} − δ_c)`.) This is the grading of
  `klrw/gradings.py` with all three scalings on.
* **(F3) Reduced words.** Any two reduced words of a permutation are connected by
  commutation moves and braid moves (Matsumoto). A braid move applied to a diagram is an
  instance of (R4) for a triple of pairwise crossing strands.

Write `[c, c'] ∈ G` for the degree of a crossing of black strands of colours `c, c'`, and
`w_c` for the degree of a crossing of a black strand of colour `c` with the red strand
(`w_k = q + ε_{kW}`, `w_c = 0` otherwise).

### 1.3 The cube

*Slots* are `1, …, m`; slot `s` has colour `c(s) = i_s`. For `I ⊆ {1,…,m}` put

    pos_I(s) = +s  if s ∈ I,     pos_I(s) = −s  if s ∉ I,     pos(red) = 0,

and let `e_I` be the idempotent in which the strands are ordered by position. So the red
strand has the slots not in `I` on its left in decreasing order and the slots of `I` on its
right in increasing order.

For `x ∈ {1,…,m}` let

    d(x) = w_{c(x)} + Σ_{i < x} [c(i), c(x)]   ∈ G,        D(I) = Σ_{x ∈ I} d(x).

The term of the cube at `I` is the projective `P_I = A e_I` with grading shift `−D(I)`, in
homological degree `|I|`; a component `P_I → P_J` of a degree-preserving map is right
multiplication by an element of `e_I A e_J` of degree `D(I) − D(J)`.

Given `I`, `J` and a colour-preserving bijection `σ` of the slots, the *strand of slot `s`*
runs from position `pos_I(s)` to position `pos_J(σ(s))`. Two strands cross in a reduced
diagram of `σ` if and only if their order is reversed; a strand crosses the red strand if
and only if the sign of its position changes.

---

## 2. Tier 1: statements with complete proofs

### Lemma 1 (the elementary move)

Let `j ∈ I`. In the reduced diagram `m_j(I) : e_I → e_{I∖j}` of the identity bijection, the
strand of `j` crosses the red strand and every strand of a slot `i < j`, exactly once each,
and no other crossings occur. Its degree is `d(j)`.

*Proof.* Only the strand of `j` changes position, from `+j` to `−j`, so it crosses the red
strand. For `i ≠ j`: if `i ∈ I`, its position is `+i`, and `(j − i)(−j − i) < 0` iff
`i < j`. If `i ∉ I`, its position is `−i`, and `(j + i)(−j + i) < 0` iff `i < j`. ∎

### Proposition 2 (`δ₀` is homogeneous and well defined)

`δ₀ e_I = Σ_{j ∈ I} (−1)^{#{i ∈ I : i > j}} m_j(I)` is a degree-preserving map
`P_I → ⊕_j P_{I∖j}`.

*Proof.* By Lemma 1, `deg m_j(I) = d(j) = D(I) − D(I∖j)`. ∎

### Proposition 3 (degree count for a mover with exchange pairs)

Let `I` be a vertex, `j ∈ I`, and `F` a set of pairwise disjoint pairs `p = (a_p < b_p)`
with `c(a_p) = c(b_p) =: c(p)`, `a_p ∉ I`, `b_p ∈ I`, `b_p ≠ j`. Let
`J = (I ∖ ({j} ∪ {b_p})) ∪ {a_p}` and `σ = ∏_p (a_p b_p)`. Let `β` be a reduced diagram of
`σ` from `e_I` to `e_J`. Then

    D(I) − D(J) − deg β  =  Σ_{p ∈ F} [c(p), c(p)]
                           + Σ_{p : a_p < j < b_p} 2·[c(j), c(p)]
                           + Σ_{ {p,q} interlaced } 2·[c(p), c(q)],

where `p, q` are *interlaced* if `a_p < a_q < b_p < b_q` or the same with `p, q` exchanged.

*Proof.* *Left side, first part.* `D(I) − D(J) = d(j) + Σ_p (d(b_p) − d(a_p))`. Since
`c(a_p) = c(b_p)`, the terms with `i < a_p` cancel and

    d(b_p) − d(a_p) = [c(p), c(p)] + Σ_{a_p < i < b_p} [c(i), c(p)].            (∗)

*The crossings of `β`.* The strands are: the mover (from `+j` to `−j`); for each pair the
*left strand* `L_p` (from `−a_p` to `−b_p`) and the *right strand* `R_p` (from `+b_p` to
`+a_p`); and the unaffected strands, which do not move. Only the mover crosses the red
strand, contributing `w_{c(j)}`, which is the first term of `d(j)`. Unaffected strands do
not cross each other. For the remaining pairs of strands we compare positions before and
after:

1. *mover, unaffected `y`:* they cross iff `y < j` (Lemma 1). This accounts for the terms
   `[c(y), c(j)]`, `y < j` unaffected, of `d(j)`.
2. *mover, `R_p`:* `(j − b_p)(−j − a_p) < 0` iff `b_p < j`. *mover, `L_p`:*
   `(j + a_p)(−j + b_p) < 0` iff `b_p < j`. So the mover crosses both strands of `p` if
   `b_p < j` and neither otherwise. In `d(j)` the slots of `p` contribute
   `[c(p), c(j)]` once for each of `a_p, b_p` that is `< j`. Hence: if `b_p < j` both sides
   agree; if `j < a_p` both are zero; if `a_p < j < b_p`, `d(j)` contains `[c(p), c(j)]`
   once and `β` contains no crossing — a surplus of `[c(j), c(p)]` on the left side.
3. *`L_p` or `R_p`, unaffected `y`:* `L_p` crosses an unaffected `y ∉ I` iff
   `(−a_p + y)(−b_p + y) < 0`, i.e. iff `a_p < y < b_p`, and never crosses a `y ∈ I`.
   `R_p` crosses an unaffected `y ∈ I` iff `a_p < y < b_p`, and never a `y ∉ I`. So `p`
   crosses each unaffected `y` with `a_p < y < b_p` exactly once, matching the term `i = y`
   of (∗).
4. *the slot `j` in (∗):* if `a_p < j < b_p`, (∗) contains `[c(j), c(p)]`, and by 2. there is
   no crossing: a second surplus of `[c(j), c(p)]`.
5. *`L_p`, `R_p`:* `(−a_p − b_p)(−b_p − a_p) > 0`: they do not cross, while (∗) contains
   `[c(p), c(p)]`: surplus `[c(p), c(p)]`.
6. *two pairs `p ≠ q`:* `L_p, L_q` cross iff `(a_q − a_p)(b_q − b_p) < 0`, i.e. iff the
   pairs are nested; likewise `R_p, R_q`; `L_p, R_q` never cross. So `β` contains
   `2[c(p), c(q)]` if the pairs are nested and nothing otherwise. In (∗) for `p` the slots
   of `q` lying in `(a_p, b_p)` contribute, and symmetrically: `2[c(p),c(q)]` if nested
   (both slots of the inner pair), `2[c(p), c(q)]` if interlaced (one slot each), `0` if
   disjoint. Surplus: `2[c(p), c(q)]` exactly for interlaced pairs.

Adding the surpluses of 2., 4., 5., 6. gives the claim. ∎

(The identity was also checked numerically on 15 000 random configurations in five stacks.)

### Corollary 3.1 (which monomials are allowed; the tree monomial)

In the situation of Proposition 3 call `p` *linked to the mover* if `a_p < j < b_p` and
`c(j) ∼ c(p)`, *blocked by the mover* if `a_p < j < b_p` and `c(j) = c(p)`; call `p, q`
*linked* if interlaced with `c(p) ∼ c(q)`, *blocked* if interlaced with `c(p) = c(q)`. For
colours `a ∼ b` let `L_{ab}` be the number of links between a node of colour `a` and a node
of colour `b`, and for a colour `v` let

    S_v = #{p : c(p) = v} + 2·#{p of colour v blocked by the mover} + 2·#{blocked pairs {p,q} of colour v}.

Let `μ = ∏ r_v^{n_v} ∏ t_{ab}^{m_{ab}} ∏ u_v^{l_v}` be a monomial with non-negative
exponents. Then `μ·β` has degree `D(I) − D(J)` if and only if

    l_v = 0,    n_v = S_v,    m_{ab} + m_{ba} = L_{ab},    Σ_{b ∼ v} m_{vb} = S_v      for all v, a ∼ b.

Such a monomial is unique if it exists, and its existence forces
`Σ_v S_v = Σ L_{ab}`, i.e. (number of links) = `|F|` + 2·(number of blocks).

If there are no blocks and the graph on `{mover} ∪ F` with the links as edges is a tree,
the monomial exists and equals `∏_p r_{c(p)} t_{c(p), c(x_p)}`, where `x_p` is the slot of
the parent of `p` (in the tree rooted at the mover) lying in `(a_p, b_p)`.

*Proof.* By Proposition 3 the required degree of `μ` is
`Σ_v S_v(−2q + σ_v) + Σ_{a∼b} L_{ab}·2(q + ε_{ab})` — with the same-colour surpluses
collected in `S_v` and the adjacent-colour ones in `L_{ab}`; surpluses for unrelated colours
are zero. By (F2), `deg μ = Σ n_v(σ_v + δ_v) + Σ m_{ab}(2ε_{ab} − δ_a) + Σ l_v(2ε_{vW} − δ_v)`.
Comparing the coefficients of the basis elements of `G`: `σ_v` gives `n_v = S_v`;
`ε_{ab}` gives `m_{ab} + m_{ba} = L_{ab}`; `ε_{vW}` gives `l_v = 0`; `δ_v` gives
`n_v = Σ_b m_{vb} + l_v`; and the coefficient of `q`, `−2ΣS_v + 2ΣL_{ab} = 0`, follows by
summing the `δ_v`-equations. This is the stated system.

*Uniqueness.* The unknowns `m_{ab}` live on the oriented edges of the Dynkin diagram, which
is a forest. If two solutions differ by `(z_{ab})`, then `z_{ab} = −z_{ba}` and
`Σ_b z_{vb} = 0` for every `v`; at a leaf `v` with neighbour `b` this gives `z_{vb} = 0`,
and removing leaves inductively gives `z = 0`.

*Trees.* Put `m_{ab} = #{p : c(p) = a, c(x_p) = b}`. Every pair has exactly one parent, so
`Σ_b m_{ab} = #{p : c(p) = a} = S_a`; every link is the parent edge of exactly one pair, so
`m_{ab} + m_{ba} = L_{ab}`. ∎

*What this does and does not say.* It proves that every term of the tree formula is
homogeneous of the right degree, that its coefficient monomial is forced, and that a
connected, unblocked configuration with a cycle admits **no** dot-free homogeneous term
(links > pairs). It does not say that the tree terms occur in the differential, nor with
which sign.

### Proposition 4 (no `u`, and no red-coloured strand moves to the right)

Let `I, J` be vertices and let `z ∈ e_I A e_J` be homogeneous of degree `D(I) − D(J)`.
Write `z` in the basis (F1). Then no basis element occurring in `z` contains a variable
`u_c`, and in none of them does a strand of colour `k` pass from the left of the red strand
to its right. The same holds with several red strands (stable envelopes with two punctures),
for each red strand and its colour.

*Proof.* Take a basis element `μ·x^α·β_σ` of `z`, with `μ = ∏ r^n t^m u^l`, and compare
the coefficients of `ε_{kW}` in its degree and in `D(I) − D(J)`. Only crossings of
colour-`k` strands with the red strand and the variable `u_k` contribute, so

    N + 2 l_k = #{x ∈ I : c(x) = k} − #{x ∈ J : c(x) = k},

where `N` is the number of colour-`k` strands of `β_σ` that cross the red strand. Let `ρ`
(resp. `λ`) be the number of colour-`k` strands going from the right to the left (resp. from
the left to the right). The right-hand side is the decrease of the number of colour-`k`
strands on the right, `ρ − λ`, and `N = ρ + λ`. Hence `2 l_k = −2λ`, so `l_k = λ = 0`.
For `c ≠ k` no generator other than `u_c` has a component along `ε_{cW}`, and
`D(I) − D(J)` has none, so `l_c = 0`. With several red strands apply the argument to each
(for two red strands of the same colour, to the total). ∎

*Consequence.* Whatever the corrected differential of a stable envelope is, it does not
involve `u`. This uses polynomial (non-inverted) parameters.

### Theorem 5 (pairwise distinct colours: no correction)

If the colours `i_1, …, i_m` are pairwise distinct, then `δ₀² = 0` in `A`. In particular
the stable envelopes of `sl_n` with `ω_1` are the uncorrected cubes.

*Proof.* Let `i < j` be in `I`. The component of `δ₀²` from `e_I` to `e_{I∖{i,j}}` is

    ε_1 · m_i(I) m_j(I∖i)  +  ε_2 · m_j(I) m_i(I∖j),

with `ε_1 = (−1)^{#{x∈I: x>i}} (−1)^{#{x∈I∖i: x>j}}` and
`ε_2 = (−1)^{#{x∈I: x>j}} (−1)^{#{x∈I∖j: x>i}}`. Since `#{x ∈ I∖j : x > i} = #{x ∈ I : x > i} − 1`
and `#{x ∈ I∖i : x > j} = #{x ∈ I : x > j}`, we get `ε_2 = −ε_1`.

By Lemma 1, in `m_i(I) m_j(I∖i)` the strand of `i` crosses the red strand and all strands
`< i`, and then the strand of `j` crosses the red strand and all strands `< j` (including
that of `i`). The same pairs cross in `m_j(I) m_i(I∖j)`. In both products every pair of
strands crosses at most once, so both are reduced diagrams of the same bijection. By (F3)
they are connected by commutation moves, which are instances of (R1), and braid moves,
which are instances of (R4) for triples of pairwise crossing strands. A bad triple needs two
black strands of the same colour, which do not exist. So all these moves are equalities,
`m_i(I) m_j(I∖i) = m_j(I) m_i(I∖j)`, and the component vanishes. ∎

### Proposition 6 (two punctures: product of the two stacks)

Let `(C_1, d_1)` and `(C_2, d_2)` be complexes of the cube form for the two stacks of a
stable envelope, with `d_1² = 0`, `d_2² = 0`, all components degree-preserving, and such
that every component of `d_1` is a combination of diagrams in which the strands of the
first stack and the first red strand are the only ones that move (and likewise for `d_2`).
Then on the cube indexed by pairs `(I_1, I_2)`, with `e_{(I_1,I_2)}` the juxtaposition of
`e_{I_1}` and `e_{I_2}`,

    d(x ⊗ y) = (−1)^{|y|} d_1 x ⊗ y + x ⊗ d_2 y

is a degree-preserving differential, `d² = 0`.

*Proof.* Placing a diagram of the first stack to the left of an idempotent of the second
stack is an algebra homomorphism, because all relations are local; likewise on the other
side. By (R1), `(a ⊗ e)(e' ⊗ b) = (e'' ⊗ b)(a ⊗ e''')` for the appropriate idempotents
(far-apart generators commute). Hence
`d²(x ⊗ y) = d_1² x ⊗ y + x ⊗ d_2² y + ((−1)^{|y|} + (−1)^{|y|−1}) d_1 x ⊗ d_2 y = 0`.
Degrees add because `D` is additive over the two stacks (no strand of one stack is counted
in `d(·)` of the other). ∎

*Remark.* That this product, with the tree formula in each factor, equals the formula
"with global numbering" of `NOTES.md` is a sign comparison; it is confirmed on the 15
envelopes of `data2/` (Tier 2). That the differential of the stable envelope has no further
cross terms does not follow from Proposition 6; it is Tier 2/3.

### Proposition 7 (cyclicity depends only on the order ideal, and is inherited)

Whether a stack admits a cyclic configuration (a connected, unblocked configuration with
more links than pairs) depends only on the heap of the word, not on the chosen linear
extension; and if it holds for a word it holds for every word having it as a prefix.

*Proof.* The conditions defining pairs, links and blocks compare only slots whose colours
are equal or adjacent (`a_p < j < b_p` with `c(j) = c(p)` or `c(j) ∼ c(p)`; interlacing of
`p, q` with `c(p) = c(q)` or `c(p) ∼ c(q)`). Two slots with equal or adjacent colours are
comparable in the heap, and every word in the commutation class lists them in the heap
order. So the set of cyclic configurations is the same for all linear extensions. A
configuration of a prefix is a configuration of the longer word, with the same links. ∎

### Proposition 8 (type A: a diagram containing (4,4,4,3) is cyclic)

For `sl_n`, `Λ^k ℂ^n`, let the order ideal be a Young diagram (boxes `(row, column)`, the
box `(r, s)` of colour `k + s − r`). If it contains the shape `(4,4,4,3)`, the stack admits
a cyclic configuration.

*Proof.* Boxes of equal or adjacent colour are comparable, with `(r,s) ≤ (r',s')` iff
`r ≤ r'` and `s ≤ s'`. Take the mover `(3,3)` (colour `k`) and the pairs

    p = {(1,1), (2,2)}  (colour k),    q = {(2,1), (4,3)}  (colour k−1),    q' = {(1,2), (3,4)}  (colour k+1),

all inside `(4,4,4,3)`. Then `(2,1) < (3,3) < (4,3)` and `(1,2) < (3,3) < (3,4)`: `q` and
`q'` are linked to the mover. `(1,1) < (2,1) < (2,2) < (4,3)` and
`(1,1) < (1,2) < (2,2) < (3,4)`: `p` is linked to `q` and to `q'`. The pair `p` of the
mover's colour does not contain the mover (`(2,2) < (3,3)`), and no two pairs have the same
colour, so nothing is blocked. The graph is connected with 4 links and 3 pairs. Any vertex
`I` containing `(3,3), (2,2), (4,3), (3,4)` and not `(1,1), (2,1), (1,2)` realises it. ∎

---

## 3. Tier 2: proved by finite exact computation

Each line is an exact computation over ℤ in the KLRW algebra (commands in `NOTES.md` §5,
`GLUING.md` §2).

| statement | cases |
|---|---|
| The tree formula defines a degree-preserving differential, `d² = 0` (full complex) | every reduced word of `A_3 ω_2`, `A_4 ω_2`, `D_4 ω_1`, `A_5 ω_3`, `D_5 ω_5`; `A_6 ω_2`, `A_6 ω_3`, `D_5 ω_4`, `D_6 ω_1`; the 13-slot stack of `E_6` |
| `d² e_I = 0` for the tree formula at random vertices `I` | `A_7 ω_3` … `A_{10} ω_2`, `D_7 ω_1`, `D_8 ω_1`; 13-slot `E_6 ω_6`, 14-slot `E_7` |
| The tree formula equals the output of the order-by-order solver, term by term | all cases of `data/`, `data_fast/`, `data2/` (≤ 10 strands) |
| At every order that arises, the linear system for the correction has a unique solution (complete ansatz, dots allowed) | all one-stack cases with ≤ 6 strands |
| The tree formula fails `d² = 0` | vertices of `A_7 ω_4`, `D_6` spinor, `E_6` (16 slots) |
| The `ω_1` I-brane formula defines a differential, `d² = 0` | `sl_3 … sl_9` |
| The `ω_1` I-brane formula equals the iterated cone of the unique Ext classes | `sl_3 … sl_6` |
| A stack is cyclic iff `μ ≤ μ_c`, with a unique minimal `μ_c` | `A_n` (`n ≤ 10`), `D_n` (`n ≤ 8`), `E_6`, `E_7`, cycles with ≤ 4 pairs (≤ 6 near the threshold) |
| Type A: cyclic iff the diagram contains (4,4,4,3) | all `k`-subsets for `sl_8, sl_9, sl_{10}`, `k = 4, 5` |

The uniqueness line is weaker than it sounds: the solver only looks for corrections at the
parameter orders at which `d²` fails, so it does not exclude additional solutions that
differ at orders where there is no obstruction.

---

## 4. Tier 3: conjectures, with what a proof would need

**C1 (tree formula).** For every stack without cyclic configurations the tree formula
satisfies `d² = 0`.
*Status.* Not proved. Every term is homogeneous with the forced coefficient
(Corollary 3.1), and the case of distinct colours is Theorem 5. What is missing is the
cancellation in `d²`. A first step that looks within reach:

> *First-order cancellation (sketch, not written out).* Modulo terms of order two in the
> products `r_c t_{cc'}`: for `i < j` in `I`, the two reduced diagrams in the proof of
> Theorem 5 differ by one braid move for each triple `(k, i, j)`, `k < i`, with
> `c(k) = c(j) ∼ c(i)`. These are the only bad triples: the strands of `i` and `j` are the
> only ones that move, a third strand `y` crosses both only if `y < i`, and the red strand
> is never the middle strand, because both moving strands start on its right.
> Each contributes `∓ r t` times the diagram with that triple smoothed, which is the
> composite of the one-pair term for `(k, j)` with mover `i` and an elementary move of the
> strand in slot `k` — before it if `k ∈ I`, after it if `k ∉ I`. The signs have to be
> matched with `κ`.

Beyond first order one needs either an induction on the last slot (the stack of a word
`w·i` is the cone of a map between two copies of the stack of `w`, and the tree formula
predicts that map), or a solution of the Maurer–Cartan equation `κ + [δ₀, x] + x² = 0` by a
recursion over trees (report, section 4).

**C2 (uniqueness and completeness).** In the cycle-free case the tree formula is the only
degree-preserving differential on the cube of the form `δ₀ +` (terms divisible by some
parameter); equivalently the relevant `Ext¹` vanishes in the given multidegree.
*Status.* Not proved. Corollary 3.1 and Proposition 4 restrict the possible terms, but
other homogeneous elements exist (with dots, or with bijections that are not products of
transpositions), so a degree argument alone is not enough.

**C3 (threshold).** A stack is cyclic iff `μ ≤ μ_c`; in type A iff the diagram contains
(4,4,4,3).
*Status.* "If" in type A is Proposition 8; inheritance is Proposition 7. "Only if" and the
`D`, `E` statements are computed only. In types A and D the condition coincides, in all
computed cases, with the minuscule element not being Deodhar; this is false in type E
(13-slot `E_6`).

**C4 (cyclic terms).** The spanning-tree rule of `NOTES.md` §6.
*Status.* Partial even as a conjecture: it reproduces 95 of 101 computed components, and the
hand-over terms have no rule yet.

**C5 (`ω_1` I-brane).** The formula of `GLUING.md` §2 for all `n`, and its identification
with the iterated cone.
*Status.* Not proved. Here all colours are distinct, so by Theorem 5 the envelopes are
exact and the only relations entering `d² = 0` are (R1), (R3) and the exact form of (R4);
a proof by a finite case analysis of pairs of terms looks feasible but has not been done.

---

## 5. Remarks towards a formalisation

* **Purely combinatorial statements** (no algebra): Proposition 7, Proposition 8, the
  linear-algebra part of Corollary 3.1 (uniqueness on a forest, the tree solution). These
  need only finite sets, a graph and a free abelian group.
* **Statements about degrees**: Lemma 1, Propositions 2 and 3. They need the notion of the
  crossings of a reduced diagram of a bijection between two orderings ("order reversed")
  and the grading group; no relations of the algebra are used.
* **Statements that use the algebra**: Theorem 5 uses (R1), the exact case of (R4) and
  Matsumoto's theorem; Proposition 6 uses (R1) and locality of the relations;
  Proposition 4 uses the basis theorem (F1), which is the one deep input and would have to
  be either formalised for this deformed presentation or assumed.
* **Tier 2 statements** would have to be re-established inside the proof assistant (or by
  a verified computation); the present computations rely on unverified code.
* Definitions that must be fixed before formalising: the presentation of section 1.1 with
  explicit signs in (R2) and (R4) (the statements above do not depend on them, the tree
  formula's signs do); the choice of the diagrams `β_σ`; the sign `κ`.
