# Closed form for the differential of a stable envelope (stack of U-branes)

Status of every statement is marked: **[verified]** = checked by exact computation in the
KLRW algebra (which cases: section 5), **[derived]** = pen-and-paper argument,
**[conjecture]** = extrapolation.

Conventions are those of `klrw/` with all three scalings switched on
(`dot_scaling`, `edge_scaling`, `vertex_scaling`), base ring ℤ, parameters
`r_i` (= ħ), `t_ij`, `u_i`.

## 1. Set-up

A *stack* is a red strand `W_k` together with a word `(i_1, …, i_m)` of colours: `m` nested
U-branes around the puncture, U-brane number 1 innermost. In the application the word is a
chain of weights `ω_k, ω_k − α_{i_1}, ω_k − α_{i_1} − α_{i_2}, …` of the minuscule
representation, so its *slots* `1, …, m` are the elements of an order ideal of the minuscule
poset, listed along a linear extension, and `c(s) = i_s` is the colour of slot `s`.

Vertices of the cube are subsets `I ⊆ {1,…,m}` (the slots whose strand is to the right of the
red strand); homological degree `|I|`. The idempotent of `I` is, read from left to right,

    [slots not in I, decreasing]   W_k   [slots in I, increasing].

The undeformed differential is `δ₀ e_I = Σ_{j∈I} ± (strand j moves to the left of W_k)`.

## 2. The formula

Fix a vertex `I`. A **term** of `d e_I` is given by

* a **mover** `j ∈ I`, and
* a set `F` of pairwise disjoint **exchange pairs** `p = (a_p < b_p)` of slots with
  `c(a_p) = c(b_p)`, `a_p ∉ I`, `b_p ∈ I`, `j ∉ p`,

subject to the admissibility condition below. Its target is

    J = ( I ∖ ({j} ∪ {b_p}) ) ∪ {a_p}.

**Diagram.** `β(I; j, F) : e_I → e_J` has no dots. The mover goes from slot `j` on the right
to slot `j` on the left. For every pair, the strand in the left slot `a_p` moves outwards to
the left slot `b_p`, and the strand in the right slot `b_p` moves inwards to the right slot
`a_p`. All other strands keep their slots. So only the mover crosses the red strand, and
the two strands of a pair "exchange their labels". Strands cross minimally.

**Links.** Call a pair `p` *linked to the mover* if `a_p < j < b_p` and `c(j)` is adjacent to
`c(p)` in the Dynkin diagram. Call two pairs `p, q` *linked* if they are interlaced
(`a_p < a_q < b_p < b_q`) and their colours are adjacent.

**Admissibility.**
1. no pair of the mover's colour has `a_p < j < b_p`;
2. no two pairs of the same colour are interlaced;
3. the graph with vertices `{mover} ∪ F` and the links as edges is a **tree**.

Root the tree at the mover. Every pair `p` then has a parent, and a **connector** `x_p`: the
slot of the parent that lies strictly inside the interval `(a_p, b_p)`. Explicitly `x_p = j`
if the parent is the mover; if the parent is a pair `q` then `x_p = a_q` when
`a_p < a_q < b_p < b_q` and `x_p = b_q` when `a_q < a_p < b_q < b_p`.

**Coefficient.**

    coefficient(I; j, F)  =  ε · ∏_{p ∈ F}  r_{c(p)} · t_{c(p), c(x_p)}

    ε = κ(I; j, F) · (−1)^{ #{ p ∈ F : x_p ∈ I } }

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
slots globally, stack after stack. The differential is given by the same formula, where
pairs and links only exist inside one stack and `κ` is computed with the global numbering.
Equivalently `S_μ` is the product of its two stacks, with no cross terms.

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

## 4. What is not covered

* **`u`-corrections.** They do not occur inside a stable envelope. They can only enter through
  the gluing maps between different `S_μ` (Phase 3, not done yet).
* The formula is stated for the reduced diagram `β`. In all cases tested the element does
  not depend on the reduced word used to draw it (`wordcheck.py`); the code uses the
  basis ("lexmin") braid of `klrw`.
* Non-minuscule words are outside the scope (the toy word `(a, b, a)` happens to obey the
  formula).
* Admissible configurations whose link graph has the right number of edges but is not a
  tree do not occur in any tested case; whether they can exist in large types (two colours
  occurring ≥ 4 times each) is open.
* No proof. The formula was found from data with ≤ 10 strands and then *predicted* larger
  cases correctly (section 5).

## 5. Evidence

See `RESULTS.md` (generated) for the table of cases.

* `data/` — full-ansatz solver output (unique solutions), one stack, ≤ 6 strands.
* `data2/` — full-ansatz solver output for two-puncture stable envelopes.
* `data_fast/` — solver restricted to degree-allowed diagrams, up to 10 strands.
* `formula.py CASE.json` — compares formula and stored differential term by term.
* `verify.py` — builds `d` from the formula alone and checks homogeneity and `d² = 0`.
