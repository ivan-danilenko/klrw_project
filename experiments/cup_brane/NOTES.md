# Closed form for the differential of a stable envelope (stack of U-branes)

Status of every statement is marked: **[verified]** = checked by exact computation in the
KLRW algebra (which cases: section 5), **[derived]** = pen-and-paper argument,
**[conjecture]** = extrapolation.

Conventions are those of `klrw/` with all three scalings switched on
(`dot_scaling`, `edge_scaling`, `vertex_scaling`), base ring ℤ. The parameters are all
independent: one `r_c` per colour (the ħ of that colour), one `t_{c c'}` for every *ordered*
pair of adjacent colours (`t_{c c'} ≠ t_{c' c}`), one `u_c` per colour.

## 1. Set-up

A *stack* is a red strand `W_k` together with a word `(i_1, …, i_m)` of colours: `m` nested
U-branes around the puncture, U-brane number 1 innermost. In the application the word is a
chain of weights `ω_k, ω_k − α_{i_1}, ω_k − α_{i_1} − α_{i_2}, …` of the minuscule
representation, so its *slots* `1, …, m` are the elements of an order ideal of the minuscule
poset, listed along a linear extension, and `c(s) = i_s` is the colour of slot `s`.

Vertices of the cube are subsets `I ⊆ {1,…,m}` (the slots whose strand is to the right of the
red strand); homological degree `|I|`. The idempotent of `I` is, read from left to right,

    [slots not in I, decreasing]   W_k   [slots in I, increasing].

The undeformed differential is `δ₀ e_I = Σ_{j∈I} ± (strand j moves to the left of W_k)`,
with the sign `(−1)^{#{i ∈ I : i > j}}` (the case `F = ∅` of the sign `κ` below).

## 2. The formula

**Scope.** The formula below is the complete differential exactly for the stacks without
cyclic configurations (section 4) **[verified, section 5]**; in general it gives a correct
part of the differential **[verified on local pieces, section 6]**. That it holds for all
`n` in the families of section 4 is a **[conjecture]**.

Fix a vertex `I`. Write `c(p) = c(a_p) = c(b_p)` for the colour of a pair. A **term** of
`d e_I` is given by

* a **mover** `j ∈ I`, and
* a set `F` of pairwise disjoint **exchange pairs** `p = (a_p < b_p)` of slots with
  `c(a_p) = c(b_p)`, `a_p ∉ I`, `b_p ∈ I`, `b_p ≠ j`,

subject to the admissibility condition below. Its target is

    J = ( I ∖ ({j} ∪ {b_p : p ∈ F}) ) ∪ {a_p : p ∈ F}.

**Diagram.** `β(I; j, F) : e_I → e_J` has no dots. The mover goes from slot `j` on the right
to slot `j` on the left. For every pair, the strand sitting in slot `a_p` to the left of the
red strand moves outwards to slot `b_p` (still on the left), and the strand sitting in slot
`b_p` to the right of the red strand moves inwards to slot `a_p` (still on the right). All
other strands keep their slots. So only the mover crosses the red strand, and the two
strands of a pair "exchange their labels". Strands cross minimally; in all tested cases the
element does not depend on the reduced word chosen (`wordcheck.py`), the stored data use the
basis braid of `klrw`.

**Links.** Call a pair `p` *linked to the mover* if `a_p < j < b_p` and `c(j)` is adjacent to
`c(p)` in the Dynkin diagram. Call two pairs `p, q` *linked* if they are interlaced
(`a_p < a_q < b_p < b_q` or `a_q < a_p < b_q < b_p`) and their colours are adjacent.

**Admissibility.**
1. no pair of the mover's colour has `a_p < j < b_p`;
2. no two pairs of the same colour are interlaced;
3. the graph with vertices `{mover} ∪ F` and the links as edges is a **tree**
   (connected and without cycles).

Condition 1 only matters for pairs attached through other pairs (a pair of the mover's
colour is never linked to the mover). Condition 2 and repeated factors such as `r_c²` need a
colour occurring four times; they do not occur in the cycle-free stacks and are tested only
on the local pieces of section 6.

Root the tree at the mover. Every pair `p` then has a parent, and a **connector** `x_p`: the
slot of the parent that lies strictly inside the interval `(a_p, b_p)`. Explicitly `x_p = j`
if the parent is the mover; if the parent is a pair `q` then `x_p = a_q` when
`a_p < a_q < b_p < b_q` and `x_p = b_q` when `a_q < a_p < b_q < b_p`.

**Coefficient.**

    coefficient(I; j, F)  =  ε · ∏_{p ∈ F}  r_{c(p)} · t_{c(p), c(x_p)}

    ε = κ(I; j, F) · (−1)^{ #{ p ∈ F : x_p ∈ I } }

(`x_p ∈ I` means: the connector is the mover, or the right-hand slot `b_q` of the parent pair.)

where `κ` is the Koszul sign: write `I` as an increasing sequence, replace every `b_p` by
`a_p` in place, delete `j` with the sign `(−1)^{number of entries to its right}`, and
multiply by the sign of the permutation that sorts what is left.

**The differential** is the sum of all admissible terms:

    d e_I = Σ_{(j, F) admissible}  coefficient(I; j, F) · β(I; j, F).

`F = ∅` gives `δ₀`. A single pair gives the first-order corrections `∓ r_c t_{c c'}`; chains
and trees of interlaced pairs give the higher orders `∏ r t`.

Consequences:

* no dots, no `u`, and all coefficients are `± (monomial in r_c t_{cc'})`;
* a stack with pairwise distinct colours (`sl_n`, `ω_1`) has `d = δ₀`;
* the slots not involved in a term are completely free (may be on either side).

**Two punctures.** For a stable envelope `S_μ` (one stack around each puncture) number the
slots globally, stack after stack from left to right; the idempotent of a vertex is the
concatenation of the idempotents of the two stacks, each as in section 1. The differential
is given by the same formula, where pairs and links only exist inside one stack and `κ` is
computed with the global numbering. Equivalently `S_μ` is the product of its two stacks with
no cross terms, with the sign convention `d(x ⊗ y) = (−1)^{|y|} dx ⊗ y + x ⊗ dy`
(`x` in the first stack). The red colours do not enter the formula. **[verified on the 15
envelopes of `data2/`]**

Implementation: `formula.py` (`Stack.terms_from(I)`, `Stack.all_terms()`), pure Python.

## 3. Where it comes from

**Uniqueness [verified].** In the fully graded algebra the linear system for the correction
at every order has a *unique* solution in the complete ansatz (all basis elements of the
right multidegree, dots allowed): kernel 0 in every case with ≤ 6 strands, and within the
restricted ansatz of section 3 for all cases up to 10 strands. So there is no gauge freedom
and the formula is not a choice of normal form.

**Degree principle [verified].** Every term of `d` is a dot-free basis diagram times a
monomial in `r, t`, with coefficient ±1. If every colour occurs at most twice in the stack,
`d` is exactly the sum of *all* such elements of the correct multidegree. With a colour
occurring three times this fails (extra degree-allowed diagrams exist in which
equal-coloured strands cross, or a second strand crosses the red strand); the terms that
occur are exactly the degree-allowed ones of the shape "mover + exchange pairs" above.

**Degree count [derived].** Write the multidegree additively in "crossing content":
`s_c` for a crossing of two strands of colour `c`, `e_{cc'}` for adjacent colours, `w` for a
strand of the red colour crossing `W`. Then `r_c t_{cc'} ↔ s_c + 2 e_{cc'}` and
`r_k u_k ↔ s_k + 2w`. For a configuration (mover, `F`) one finds

    deg(target) − deg(source) − content(β)
        =  Σ_p s_{c(p)}  +  Σ_{p : a_p < j < b_p} 2·[c(j), c(p)]  +  Σ_{p, q interlaced} 2·[c(p), c(q)]

(`[c, c']` = `s_c` if equal, `e_{cc'}` if adjacent, 0 otherwise): unaffected slots and nested
or disjoint pairs contribute nothing. This must be the content of a monomial, which forces
conditions 1 and 2 and exactly `|F|` links; for a tree the monomial is unique and is the one
in the formula. `u` can never occur: the number of red-coloured strands on the right of `W`
drops by the net flux across `W`, so the `w`-content left over is never positive.

**Mechanism [derived].** Sliding a strand of colour `c` over a crossing between another
`c`-strand and an adjacent-coloured strand `c'` fails by `−r_c t_{cc'}` times the diagram in
which nothing crosses, i.e. in which the two `c`-strands have exchanged their roles. This
is the label exchange. The triple `(V_k, W_k, V_k)` (which would produce `r_k u_k`) never
arises because only one strand crosses `W`.

## 4. Range of validity: the tree formula is complete exactly when no cycles are possible

Call a configuration (mover, `F`) *cyclic* if it has no blocked pairs, its link graph is
connected, and it has more links than pairs. `cyclefree.py` searches for such configurations.

| stack | cyclic configurations | status of the tree formula |
|---|---|---|
| `sl_n`, `ω_1, ω_2, ω_3` (and duals), checked `n ≤ 11` | none | complete **[verified]** |
| `so_{2n}`, vector `ω_1`, checked `n ≤ 8` | none | complete **[verified]** |
| `D_4` (all three), `D_5` spinors | none | complete **[verified]** |
| `sl_n`, `ω_k` with `4 ≤ k ≤ n−3` (first: `sl_8, ω_4`, 15 of its 16 slots) | exist | incomplete |
| `D_n` spinors, `n ≥ 6` (first 14 of 15 slots of `D_6`) | exist | incomplete |
| `E_6` (first 14 of 16 slots), `E_7` | exist | incomplete |

In the incomplete cases every term of the tree formula is still a term of the true
differential with the right coefficient **[verified on 32 local pieces, see section 6]**;
what is missing are additional terms attached to the cyclic configurations.

`u`-corrections do not occur inside a stable envelope in any case. They can only enter
through the gluing maps between different `S_μ` (not done).

## 5. Evidence for the tree formula

* **Independent re-implementation**: `review/reimpl.py` was written from the text of
  sections 1–2 alone (by a separate agent that did not read `formula.py`) and reproduces all
  31 files of `data/`, `data_fast/`, `data2/` — 40 084 terms — exactly; `review/probe.py`
  shows which alternative readings of the text the data exclude.
* **Exact agreement with the solver** (term by term, signs included): all one-stack cases
  in `data/` and `data_fast/` (`A_3 … A_6`, `D_4, D_5, D_6`, up to 10 strands, up to 7512
  terms) and all 15 two-puncture stable envelopes in `data2/`. Run `python3 formula.py FILE`.
* **Uniqueness**: the complete ansatz (dots allowed) has a unique solution at every order
  for all cases with ≤ 6 strands; the restricted ansatz for all cases up to 10 strands.
* **`d² = 0` from the formula alone, full complex** (`verify.py`): every reduced word of
  `A_3 ω_2`, `A_4 ω_2`, `D_4 ω_1`, `A_5 ω_3` (42 words), `D_5 ω_5` (12 words);
  `A_6 ω_2`, `D_6 ω_1`, `D_5 ω_4` with other words; `A_6 ω_3` (12 strands, 4096 vertices —
  not used to find the formula).
* **`d² e_I = 0` on random vertices of large stacks** (`sample_verify.py`), none of them used
  to find the formula: `A_7 ω_3` (15 strands, 200 vertices), `A_8 ω_3` (18, 150),
  `A_9 ω_3` (21, 120), `A_{10} ω_2` (18, 150), `D_7 ω_1` (12, 200), `D_8 ω_1` (14, 200).
  No failures. A deliberately wrong sign rule is detected by the same test.
* The diagrams do not depend on the reduced word used to draw them (`wordcheck.py`).

## 6. Beyond trees: what the true differential looks like (work in progress)

Tools: `interval.py` solves for the true differential with the complete ansatz on a
down-set of the cube (a subcomplex) of a 14–16 strand stack; `cyc_batch.py` lists minimal
cyclic configurations; `pieces/` holds 32 such local solutions for `D_6`, `E_6`, `A_7 ω_4`.

Caveat: on such a piece the correction is unique at 569 of the 750 orders solved; at the
other 181 orders the local system has a kernel and the solver picks the solution with the
fewest terms, which agrees with the tree formula. The extra terms below all arise at orders
where the solution is unique given the lower orders. Totals over the pieces: 36 659 true
terms, of which 35 861 are the tree-formula terms (all present, all with the right
coefficient) and 798 are new.

Findings **[verified on those local pieces]**:

* all tree terms are present with the predicted coefficients; nothing else changes;
* each cyclic configuration contributes extra terms **with one dot**, some with
  coefficient 2, plus dot-free terms with one more power of `r`;
* for a configuration whose link graph `G` is a single 4-cycle through the mover, the extra
  terms are exactly

      Σ_{e on the cycle}  sign(G−e) · mono(G−e) · ( t_{c_x c_y} · [dot on x] + t_{c_y c_x} · [dot on y] ) · β

  i.e. a sum over the spanning trees `G − e`, the left-over link `e = (x, y)` being replaced
  by its "bigon polynomial". Each pair on the cycle receives one dot on each of its two
  strands, the mover two dots. The position of a dot along its strand is fixed by a
  time-ordering rule (`cycle3.py`: draw the events on the right of the red strand from the
  inside out and on the left from the outside in, exchange the two events of `e`, and put the
  dots where the two strands touch). This reproduces all such cases in `D_6`, `E_6`, `A_7`
  exactly;
* equivalently the dotted part is the sum over orientations of `G` in which every pair has
  at least one outgoing link, with weight `∏ t_{tail→head}` and one dot for each surplus
  outgoing link — which suggests the general shape
  `∏_p (r_p ∂_p) ∏_{links} (t_{xy} x_x + t_{yx} x_y)` with Demazure operators `∂_p`;
* in every single-cycle configuration examined, the dotted part is consistent with the same
  spanning-tree expression, with the tree signs `sign(G−e)` of section 2. A pair `X` on the
  cycle receives two dots (one from each of its cycle links). If `X` is a source or a sink of
  the cycle for the order by `b` (it has the smaller `b`, resp. the larger `b`, in both
  links) the two tree signs agree and the dots sit on its left and on its right strand. If
  `X` lies between its two neighbours the tree signs are opposite and both dots sit on the
  same strand, so that only their `r`-correction survives. In all examples the surviving term is a *hand-over*: `r` times the
  diagram in which the strand of `X` does not cross an unaffected strand of its own colour lying
  inside the interval of `X` — on the left if that strand is on the left, on the right if it
  is on the right (checked by moving such a strand across, piece `E6 … I_6-7-12-13-14`).
  The mover always gets one dot at the source and one at the target;
* **additional term type, not yet understood**: dot-free terms in which the strand that
  crosses the red strand is not the one from the mover's slot — the moving strand "hands
  over" to a strand of its own colour instead of crossing it (on the right, on the left, or
  both), so that the bijection of slots is no longer a product of transpositions. They occur
  only next to cyclic configurations, and some of them connect vertices that no
  mover-plus-pairs configuration connects;
* **not yet tested**: link graphs with two or more independent cycles (they need `D_7`, `E_7`
  or larger type `A` pieces).

`CATALOGUE_cyclic.txt` lists every true term beyond the tree formula in the 32 local pieces
(113 families): 336 dotted terms (52 of them with coefficient 2), 99 dot-free terms in which
the mover keeps its slot, 89 in which it changes slot. `cycle3.py` implements the resulting rule for a configuration with one cycle (variant
`mix`): sum over the spanning trees `G − e` with their tree signs and monomials; the left-over
link `e = (x, y)` contributes `t_{c_x c_y}·[dot on x] + t_{c_y c_x}·[dot on y]`, where

* at a source or sink of the cycle the two links of the node put their dots on different
  sides (left strand / right strand; for the mover: source / target), at the touching points
  of the time-ordered drawing;
* at an in-between node `X` each link puts a dot on both strands of `X`, right before or right
  after the move of `X` in the canonical drawing (as the node with the larger `b`: left strand
  before, right strand after; as the node with the smaller `b`: left strand after, right
  strand before). The two links come with opposite tree signs, so only hand-over terms remain.

This reproduces 95 of the 101 components that are targets of a single-cycle configuration in
the 33 local pieces: all 17 in `A_7 ω_4`, all 40 in `D_6`, 38 of 44 in `E_6`. The 6 others
(one piece and its mirror image, cycle not through the mover) involve the hand-over terms
below. About 70 further components (out of about 33 000 compared) differ only by hand-over
terms that land in vertices which are not the target of any mover-plus-pairs configuration.

So: no complete closed formula yet outside the cycle-free range; the data indicate that one
exists (unique solutions, small integer coefficients, clear graph-theoretic structure).
