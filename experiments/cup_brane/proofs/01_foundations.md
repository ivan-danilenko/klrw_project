# 01 — Foundations: presentation, basis theorem, well-definedness of the tree diagrams, intertwiners

Front: FOUNDATIONS. Scripts: `proofs/checks/01_*.py` (run from `experiments/cup_brane`;
`sage -python` for everything touching `klrw`, plain `python3` for the pure-Python ones).
Notation is that of `PROOFS.md` §1 and `NOTES.md` §2. Products are read left to right:
`a·b` = first `a`, then `b`. The sections are in the order of priority B, D, A, E, C, F;
section S is the Status.

**Headline results.**

* (B) The element `β(I; j, F)` is **not** in general independent of the reduced word: in
  words with a cyclic configuration it can depend on it (smallest example `A_2`, word
  `11221121`, §B.7). Bad triples (the only obstruction) **do** occur in admissible
  configurations of cycle-free words (smallest example `A_3`, word `12321223`, §B.5), but in
  every tested case they are *shielded* and the element is nevertheless independent of the
  word. **Proved:** independence whenever there is no bad triple (Prop. B.5), which holds for
  every word with colour multiplicities `≤ 2`, for every cycle-free word with multiplicities
  `≤ 3`, and for the minuscule words of `sl_n` with `ω_1, ω_2, ω_3` (and duals) and of
  `so_{2n}` with `ω_1` (Theorem B.6); by exact computation also for all cycle-free order
  ideals of `A_7…A_9 ω_4`, `A_{10} ω_3`, `D_5…D_8` spinors, `E_6 ω_1`, `E_7 ω_7` (§B.5). The
  definition of `β` that is safe everywhere is the **klrw basis braid** (Definition B.10); it
  is what all stored data and all `d² = 0` checks use.
* (D) All intertwiner identities hold exactly, on every colour pattern including the red
  strand, with proofs from the relations plus (F1) (Lemma D.1, Theorem D.3, Proposition D.5,
  Theorem D.6).
* (A) The presentation of `PROOFS.md` §1.1 is right, with the signs of the README; all
  relations re-verified on every colour pattern (388 checks), and independently verified for
  an explicit polynomial representation written from the formulas alone (exact identities in
  the twisted group algebra). (F1) holds for this presentation over `ℤ[r, t, u]` with
  polynomial parameters; Webster's theorem does **not** apply literally (it assumes unit
  leading coefficients), so a proof is given (Theorem A.5).
* (F) Tier 1 of `PROOFS.md` is correct and its proofs are complete; repairs: the (F1)
  reference ("Prop. 2.23" is wrong, and Webster's theorem needs unit coefficients), the exact
  form of (R2), and Prop. 4's extension to several red strands needs (F1) for that algebra.
* The draft `05_main_sliding.md`: its Straightening Lemma 1.2 and Lemma 6.2 are correct and
  agree with §B; its condition (S2) is stronger than necessary, and **its "Warning" example
  in §6 is not a failure of well-definedness** (both triples there are good; §B.9).

---

## B. The diagram of a tree term

### B.1 Strands and crossings

Fix a word `(i_1, …, i_m)`, a vertex `I`, a mover `j ∈ I` and a set `F` of pairwise disjoint
pairs `p = (a_p < b_p)` with `c(a_p) = c(b_p) =: c(p)`, `a_p ∉ I`, `b_p ∈ I`, `b_p ≠ j`
(*a configuration*; admissibility is not assumed in B.1–B.4). Put
`J = (I ∖ ({j} ∪ {b_p})) ∪ {a_p}` and let `σ` exchange `a_p ↔ b_p` for `p ∈ F` and fix all
other slots. Positions are the signed slots `pos_I(s) = ±s` of `PROOFS.md` §1.3, red = 0.
The strands of a diagram `e_I → e_J` of `σ` are

| strand | source position | target position |
|---|---|---|
| red `W` | `0` | `0` |
| mover `M` | `+j` | `−j` |
| `L_p` (left strand of `p`) | `−a_p` | `−b_p` |
| `R_p` (right strand of `p`) | `+b_p` | `+a_p` |
| unaffected `U_y` (`y ∉ {j} ∪ ⋃F`) | `pos_I(y)` | `pos_I(y)` |

(`y` keeps its side: `y ∈ J ⟺ y ∈ I`.) A word in the generators `ψ_i` from `e_I` is
*reduced* iff every two strands cross at most once; then two strands cross iff their order is
reversed, i.e. iff `(src_A − src_B)(tgt_A − tgt_B) < 0`, and the set of crossing pairs (the
inversion set) is the same for all reduced words of `σ`.

**Lemma B.2 (crossing table).** In a reduced diagram of `σ` exactly the following pairs of
strands cross:

1. `M` and `W`;
2. `M` and `U_y` iff `y < j`;
3. `M` and `L_p`, and `M` and `R_p`, iff `b_p < j`;
4. `L_p` and `U_y` iff `y ∉ I` and `a_p < y < b_p`; `R_p` and `U_y` iff `y ∈ I` and
   `a_p < y < b_p`;
5. `L_p` and `L_q`, and `R_p` and `R_q`, iff `p, q` are nested
   (`a_p < a_q < b_q < b_p` or vice versa).

No other pair crosses: not `W` with anything but `M`; not two unaffected strands; not `L_p`
with any `R_q` (including `q = p`).

*Proof.* Evaluate `(src_A − src_B)(tgt_A − tgt_B)`. `W`: a strand other than `M` keeps the
sign of its position. 2: `(j ∓ y)(−j ∓ y) < 0` iff `y < j` (both signs). 3: `(j + a_p)(b_p − j)`
and `(j − b_p)(−j − a_p)` are `< 0` iff `b_p < j`. 4: `L_p, U_y`: if `y ∈ I` both factors of
`(−a_p − y)(−b_p − y)` are negative; if `y ∉ I`, `(y − a_p)(y − b_p) < 0` iff
`a_p < y < b_p`; `R_p` symmetrically. 5: `(a_q − a_p)(b_q − b_p) < 0` iff nested, for both `L`
and `R`; `L_p, R_q`: `(−a_p − b_q)(−b_p − a_q) > 0`. Unaffected strands keep their positions. ∎

This is items 1–6 of the proof of Proposition 3 of `PROOFS.md`, and Lemma 6.1 of
`05_main_sliding.md`. *Check:* `01_triples.py` asserts the table for every configuration it
enumerates (every vertex and every admissible configuration of all `A_2` words of length
`≤ 8`, all `A_3` words of length `≤ 7`, and all words of `words.json` with `≤ 12` slots), and
`01_bad_search.py crosscheck` for all `A_2` words of length 8, `A_3` of length 6, `D_4` of
length 5, with every unaffected slot put on both sides.

### B.2 Triples

A *triple* is a set of three strands which cross pairwise. In a reduced word, the three
strands of a braid move `ψ_iψ_{i+1}ψ_i ↔ ψ_{i+1}ψ_iψ_{i+1}` form a triple, and they are at
the positions `i, i+1, i+2` in their **source order** (none of the three pairs has crossed
yet). So the *middle strand* of a triple is the middle one in the source order, and a braid
move at a triple is an instance of the cubic relation (A6) on the state formed by the colours
of (left, middle, right) strand.

**Lemma B.3 (classification of triples).** The triples of `σ` are exactly the following;
`W` lies in none of them. (Middle strand in bold.)

| type | strands (source order) | condition |
|---|---|---|
| T1 | `U_y`, **`L_p`**, `M` | `y ∉ I`, `a_p < y < b_p < j` |
| T2 | `U_y`, **`R_p`**, `M` | `y ∈ I`, `a_p < y < b_p < j` |
| T3 | `L_q`, **`L_p`**, `M` | `q ⊂ p` (nested inside), `b_p < j` |
| T4 | `R_q`, **`R_p`**, `M` | `q ⊂ p`, `b_p < j` |
| T5 | `U_y`, **`L_q`**, `L_p` | `q ⊂ p`, `y ∉ I`, `a_q < y < b_q` |
| T6 | `U_y`, **`R_q`**, `R_p` | `q ⊂ p`, `y ∈ I`, `a_q < y < b_q` |
| T7 | `L_r`, **`L_q`**, `L_p` | `r ⊂ q ⊂ p` |
| T8 | `R_r`, **`R_q`**, `R_p` | `r ⊂ q ⊂ p` |

*Proof.* `W` crosses only `M`, so it is in no triple. Two unaffected strands do not cross, and
no `L` crosses an `R`; so a triple contains at most one `U` and not both an `L` and an `R`. A
triple containing `M`: the other two cross `M` and each other: `{U_y, L_p}` (T1), `{U_y, R_p}`
(T2), `{L_p, L_q}` (T3), `{R_p, R_q}` (T4); the conditions are those of Lemma B.2 (for T3/T4,
nested and `b_p, b_q < j`, i.e. `b_{outer} < j`). A triple without `M`: `{U, L, L}` (T5),
`{U, R, R}` (T6), `{L, L, L}` (T7), `{R, R, R}` (T8), with Lemma B.2 (three pairwise nested
intervals form a chain). Middle strands: compare source positions, e.g. T1:
`−y < −a_p < +j`; T3: `−a_q < −a_p < j` (`a_p < a_q`); T4: `b_q < b_p < j`; T5:
`−y < −a_q < −a_p`; T6: `y < b_q < b_p`; T7: `−a_r < −a_q < −a_p`; T8: `b_r < b_q < b_p`. ∎

*Check:* `01_triples.py` recomputes all triples by brute force from the positions and asserts
type, condition and middle strand (8 types, all occur).

A triple is **bad** if the cubic relation on its state is not exact (§A, (A6)): the outer
strands are black of one colour `c` and the middle strand is black of a colour `c' ∼ c`
(value `r_c t_{cc'}`), or the outer strands are black of colour `k` and the middle strand is
red (value `r_k u_k`).

**Corollary B.4 (bad triples in interval form).** The second kind never occurs (`W` is in no
triple). A diagram of `σ` contains a bad triple iff one of the following holds; this does
**not** depend on `I` (T1/T2, T5/T6 differ only by the side of `y`):

* **(B1)** [T1/T2] an unaffected `y` and a pair `p` with `c(y) = c(j) ∼ c(p)`,
  `a_p < y < b_p < j`;
* **(B2)** [T3/T4] pairs `q ⊂ p` with `c(q) = c(j) ∼ c(p)` and `b_p < j`;
* **(B3)** [T5/T6] pairs `q ⊂ p` and an unaffected `y` with `c(y) = c(p) ∼ c(q)`,
  `a_q < y < b_q`;
* **(B4)** [T7/T8] pairs `r ⊂ q ⊂ p` with `c(r) = c(p) ∼ c(q)`.

*Check:* `01_bad_search.py crosscheck`: the interval form agrees with the strand-level
enumeration on 43 608 (configuration, side) cases.

**Remark.** Each condition compares only slots of equal or adjacent colours, so (exactly as in
Proposition 7 of `PROOFS.md`) the existence of a bad triple in an admissible configuration
depends only on the heap of the word, and it is inherited by every word having the given word
as a prefix.

### B.3 The criterion

**Proposition B.5.** If no diagram of `σ` contains a bad triple, all reduced words of `σ` give
the same element of `e_I A e_J`.

*Proof.* By Matsumoto (F3) two reduced words of `σ` are connected by commutation moves and
braid moves through reduced words of `σ`. A commutation move is relation (A2). A braid move
`ψ_iψ_{i+1}ψ_i ↔ ψ_{i+1}ψ_iψ_{i+1}` inside `ψ_u · (…) · ψ_v` is applied to the three strands
at the positions `i, i+1, i+2`, which form a triple of `σ` (the inversion set is the same for
all reduced words); its state is (left, middle, right) in source order. By (A6) the two sides
differ by `C·e`, and `C = 0` unless the triple is bad. ∎

So `β(I; j, F)` is well defined whenever (B1)–(B4) fail. Note that Proposition B.5 is about
*reduced* words; it does not need the stronger hypothesis (S2) of Lemma 1.2 of
`05_main_sliding.md` (which forbids every pairwise crossing triple with two outer strands of
equal colour, whatever the middle strand), see §B.9.

### B.4 When bad triples cannot occur

**Theorem B.6.** Let `(I; j, F)` be an admissible configuration (tree, no blocks).

(a) If every colour occurs at most twice in the word, `β(I; j, F)` has no bad triple.

(b) If every colour occurs at most three times, the only possible bad triples are of type
(B3), and only in the following *pattern S*: the mover has colour `c'`, `F` contains pairs
`q` (colour `c'`), `x` (colour `d ∼ c'`, `d ≠ c`) and `p` (colour `c ∼ c'`) with the tree path
`q — x — M — p`, `q ⊂ p`, and an unaffected `y` of colour `c` with `a_q < y < b_q` (so the three
slots of `c'` are `a_q, b_q, j` and the three slots of `c` are `a_p, y, b_p`). A word
containing pattern S has a cyclic configuration. Hence: **a cycle-free word with colour
multiplicities `≤ 3` has no bad triple in any tree term.**

(c) For the minuscule words of `sl_n`, `Λ^k ℂ^n`, `k ∈ {1, 2, 3, n−3, n−2, n−1}` (any order
ideal, any linear extension), and of `so_{2n}` with `ω_1`, and of `D_4`, no tree term has a
bad triple.

All parts use only that `Γ` is a forest.

*Proof.* Throughout, a *node* is the mover or a pair; linked nodes have adjacent colours; the
link graph is a tree; along a path in it the colours form a walk in `Γ`.

*(B4) needs two pairs of colour `c`, i.e. four slots of colour `c`;* impossible in (a), (b).
*(B3) needs the three slots `a_p, y, b_p` of colour `c`;* impossible in (a).

*(B1) under multiplicities `≤ 3`.* The slots `y, j` have colour `c`, so there is at most one
further slot of colour `c` and no pair of colour `c`; every colour carries at most one pair.
So the nodes have pairwise different colours. `p` is not linked to the mover (`b_p < j`). The
tree path `p = v_0, v_1, …, v_l = M` has `l ≥ 2` and pairwise different colours, so
`c(p), c(v_1), …, c(M) = c` is a path of length `≥ 2` in `Γ` between the adjacent vertices
`c(p)` and `c`; with the edge `c(p) — c` it closes a cycle in `Γ`. Contradiction.

*(B2) under multiplicities `≤ 3`.* The slots `a_q, b_q, j` are the three slots of colour `c`;
every colour carries at most one pair; the only repeated colour among nodes is `c` (`q` and
`M`). The tree path from `p` to `M` has length `≥ 2` (`b_p < j`). If it avoids `q`, its colours
are pairwise different and give a path of length `≥ 2` in `Γ` from `c(p)` to `c ∼ c(p)`:
cycle. If it passes through `q = v_a`, then `a ≥ 2` (`q ⊂ p` are nested, not linked) and the
part `v_0, …, v_a` has pairwise different colours from `c(p)` to `c`: cycle. Contradiction.

*(B3) under multiplicities `≤ 3`.* Now `a_p, y, b_p` are the three slots of colour `c`, so
`c(j) ≠ c`, every colour carries at most one pair, and the only colour that can repeat among
nodes is `c(j)` (the mover and at most one pair `π` of that colour). Let
`q = v_0, v_1, …, v_l = p` be the tree path (`l ≥ 2`, nested pairs are not linked). If the
colours along it are pairwise different we get a cycle as before. Otherwise
`c(v_a) = c(v_b) = c(j)` for exactly one `a < b`, `{v_a, v_b} = {M, π}`; deleting `v_{a+1}, …, v_b`
leaves a walk without repetitions from `c(q) = c'` to `c`, i.e. a path; it must have length 1
(else cycle), so `a + (l − b) = 1`. `b = l` is impossible (`c(p) = c ≠ c(j)`), so `a = 0`,
`b = l − 1`: `c(q) = c(j) = c'`, `π = q`, `v_{l−1} = M`, and `p` is linked to the mover. The
closed walk `c', c(v_1), …, c(v_{l−2}), c'` has pairwise different inner vertices (other than
`c'` and `c`); `l − 2 = 0` would link `q` to the mover although they have the same colour, and
`l − 2 ≥ 2` would be a cycle in `Γ`; so `l = 3` and the path is `q — x — M — p` with
`c(x) = d ∼ c'`, `d ∉ {c, c'}`. This is pattern S.

*Pattern S gives a cyclic configuration.* Admissibility says `q` (colour of the mover) does
not contain `j`; `p` contains `j` (linked to `M`). If `j < a_q` put `p* = (a_p, y)`, if
`j > b_q` put `p* = (y, b_p)`; in both cases `p*` is a pair of colour `c` containing `j`. Then
the configuration with mover `j` and pairs `{q, x, p*}` has the links `M — x`, `x — q`
(unchanged), `q — p*` (`a_p < a_q < y < b_q`, resp. `a_q < y < b_q < b_p`: interlaced,
`c ∼ c'`), `p* — M` (`j ∈ p*`, `c ∼ c'`); it has no blocks (the three pairs have three
different colours, `q` does not contain `j`, and `p*`, `x` are not of the mover's colour); it is
connected with 4 links and 3 pairs: cyclic.

*(c)* `Λ^1, Λ^2` (and duals), `so_{2n}` with `ω_1`, and `D_4` have multiplicities `≤ 2`: (a).
For `Λ^3 ℂ^n` (and dually `Λ^{n−3}`) the slots are the boxes `(r, s)`, `1 ≤ r ≤ 3`, of a
Young diagram, box `(r,s)` of colour `3 + s − r`; boxes of equal or adjacent colour are ordered
componentwise and every word lists them in that order; every colour occurs `≤ 3` times, so by
(b) only pattern S can occur, and it needs two adjacent colours `c, c'` with three boxes each,
i.e. full diagonals `d_r = (r, r+δ)` (colour `c`) and `e_r` (colour `c'`), `r = 1, 2, 3`. For
`c' = c + 1`, `e_r = (r, r+δ+1)` and the order is `d_1 < e_1 < d_2 < e_2 < d_3 < e_3`; for
`c' = c − 1` it is `e_1 < d_1 < e_2 < d_2 < e_3 < d_3`. Pattern S needs
`a_p = d_1 < a_q < y = d_2 < b_q < b_p = d_3` and the third `c'`-slot `j` strictly between `d_1`
and `d_3`: in the first case `a_q = e_1, b_q = e_2, j = e_3 > d_3`, in the second
`a_q = e_2, b_q = e_3, j = e_1 < d_1`. Impossible. ∎

*Checks.* `01_mult3.py`: exhaustive over all words with multiplicities `≤ 3` of length `≤ 9`
(`A_3`, 5 235 words; `D_4`, 150 076 words) and `≤ 10` (`A_4`, 368 476 words): the only bad
triples are of type B3 (160, 912, 5 496 configurations), as (b) says; multiplicities `≤ 2`
(`A_3`, length `≤ 8`): none. The smallest pattern-S example is `A_3`, `12132213`, mover 6,
`F = {(1,7), (2,5), (4,8)}`, `y = 3` (a cyclic word).

### B.5 Exhaustive and random searches (task (i)–(iii))

`01_bad_search.py exhaustive|random`, `01_minuscule_scan.py`; "cyclic" = has a cyclic
configuration (`cyclefree.find_cycle`, exhaustive: `max_pairs = m // 2`).

| words | cycle-free words | of which with a bad triple | cyclic words | of which with a bad triple |
|---|---|---|---|---|
| `A_2`, all 504 words of length 3–8 | 468 | 0 | 36 | 8 |
| `A_2`, all 4 088 words of length 3–11 | 2 174 | **0** | 1 914 | 1 274 |
| `A_3`, all 3 267 words of length 3–7 | 3 243 | 0 | 24 | 0 |
| `A_3`, all, length 8 | 6 225 | **8** | 336 | 68 |
| `A_3`, all, length 9 | 17 069 | **80** | 2 614 | 1 002 |
| `A_4`, all words of length 3–8 | 86 356 | **16** (length 8) | 1 004 | 128 |
| `D_4`, all words of length 3–8 | 86 040 | **24** (length 8) | 1 320 | 180 |
| `A_3` random 3000, length 6–13 | 2 197 | 16 | 803 | 538 |
| `D_5` random 3000, length 6–12 | 2 864 | 5 | 136 | 54 |
| `E_6` random 3000, length 6–12 | 2 933 | 0 | 67 | 24 |
| minuscule, all of `words.json` with ≤ 12 slots | all | **0** | — | — |
| minuscule, every order ideal of `A_7 ω_4`, `A_8 ω_4`, `A_9 ω_4`, `A_{10} ω_3`, `D_5`, `D_6`, `D_7`, `D_8` spinors, `E_6 ω_1`, `E_7 ω_7` | 67, 116, 184, 164, 15, 28, 47, 73, 23, 33 cycle-free ideals | **0** | (1 minimal cyclic ideal each) | e.g. full `A_7 ω_4`: 2 configurations |

So: **(i)** bad triples occur in arbitrary words; **(ii)** they occur in cycle-free words
(first at length 8, always with a colour occurring at least 4 times, in accordance with
Theorem B.6(b); types B1 and B3 only); **(iii)** they occur in no cycle-free minuscule order
ideal of any case computed, and provably not in the families of Theorem B.6(c).

**Example (cycle-free, with bad triples).** `A_3`, word `12321223` (slot colours
`1,2,3,2,1,2,2,3`), red colour arbitrary. `I = {4,5,6,7,8}` (resp. `{5,6,7,8}`), mover `j = 7`
(colour 2), `F = {(1,5), (2,6), (3,8)}` (colours 1, 2, 3), tree `M — (3,8) — (2,6) — (1,5)`.
Unaffected `y = 4` of colour 2 with `1 < 4 < 5 < 7`: (B1), the triple
`U_4, R_{(1,5)}, M` (resp. `U_4, L_{(1,5)}, M`) is bad. `verify.py A 3 k 12321223` gives
`d² = 0` for `k = 1, 2, 3`.

### B.6 Shields; the computations in the algebra

**Definition B.7.** A *shield* of a bad triple `(X, Y, Z)` (outer `X, Z` of colour `c`, middle
`Y`) is a strand `S` of colour `c` that crosses `X` and `Z` but not `Y`.

By Lemma B.2 the shields are: for (B1), the strand `L_q`/`R_q` (same side as `y`) of a pair
`q ∈ F` of the mover's colour with `y ∈ (a_q, b_q)` that is interlaced with `p` (hence linked
to `p`); for (B3), the strand of a pair `s ∈ F` of colour `c(p)` with `s ⊂ p`, `y ∈ s`, `s`
interlaced with `q`; (B2): a pair `s ⊃ q` of colour `c(j)` interlaced with `p` with `b_s < j`;
(B4): a pair `s` of colour `c(p)`, `r ⊂ s ⊂ p`, interlaced with `q`. (The mover, unaffected
strands and red are never shields.) In the example above `R_{(2,6)}` shields `U_4, R_{(1,5)}, M`.

**Lemma B.8 (local shield identity).** Let `c ∼ c'` (or `c = k` and `c'` the red strand).
On each of the states `(c, c', c, c)` and `(c, c, c', c)` (four strands `X, Y, S, Z`, resp.
`X, S, Y, Z`), all six reduced words of the permutation `4231` (strand 1 ↔ strand 4, the
middle two not crossing) give the same element.

*Proof.* The reduced words of `4231` are `12321`, `13213 ~ 31213 ~ 13231 ~ 31231`, `32123`
(three commutation classes). On `(X, Y, S, Z)`: the braid move `12321 → 13231` is applied to
the triple `(X, S, Z)` (colours `c, c, c`): exact. The braid move `31213 → 32123` is applied to
`(X, Y, Z)` (colours `c, c', c`): by (A6) the difference is `± C · ψ_3 ψ_3` (the two outer
letters `3` remain, the three middle letters are smoothed), and `ψ_3ψ_3 = 0` on the state of
positions 3, 4 at that moment, which is `(Z, S)`, both of colour `c` (A5). The state
`(X, S, Y, Z)` is the mirror case (`32123`/`12321` exchange roles). ∎
*Check:* `01_local4231.py` (6 words, one element on `1 2 1 1`, `1 1 2 1`, `1 W1 1 1`,
`1 1 W1 1`; three different elements on `1 2 2 1`, where the "shield" has the wrong colour).

In a full diagram the shield's two crossings need not be adjacent to the triple, and other
strands may pass between them; Lemma B.8 is therefore not a proof of the global statement.
The global statements were tested in the algebra (`01_beta_words.py`: all terms with a bad
triple, 30–100 random reduced words each, every orientation of every bad triple reached,
elements compared exactly):

| words | terms with a bad triple | all bad triples shielded → element independent | some bad triple unshielded |
|---|---|---|---|
| all 8 cycle-free `A_3` words of length 8 with bad triples (red 2) | 16 | 16 / 16 | 0 |
| all 80 cycle-free `A_3` words of length 9 with bad triples | 488 | 488 / 488 | 0 |
| all 24 cycle-free `D_4` words of length 8 with bad triples | 48 | 48 / 48 | 0 |
| all 64 cyclic `A_2` words of length 9 with bad triples | 1 056 | 512 / 512 | 544, of which **524 depend** on the word |
| cyclic `A_2` `11221121`, `12211212`, `12211221` | 12 | 6 / 6 | 6, all depend |

And the combinatorial side (`01_shield_scan.py`; every vertex `I`): in every
cycle-free word searched (`A_2` ≤ 11, `A_3` ≤ 9 and 3000 random of length ≤ 13, `A_4`, `D_4` ≤ 8,
random `A_5`, `D_5`, `E_6` of length ≤ 12) **every bad triple is shielded**; unshielded ones
occur only in cyclic words.

**Conjecture B.9.** (a) In a cycle-free word every bad triple of every tree term is shielded.
(b) If every bad triple of `σ` is shielded, all reduced words of `σ` give the same element.
Hence `β(I; j, F)` is independent of the reduced word for every cycle-free word.
*Status:* verified as above; not proved. (b) is the local Lemma B.8 plus a global argument
that is missing (the strands passing between the two crossings of the shield produce further
triples; I did not find an argument that controls them).

### B.7 Dependence in cyclic words

`A_2`, red 1, word `11221121`, `I = {3,5,6,7,8}`, mover `j = 3`, `F = {(1,8), (2,5), (4,7)}`,
monomial `r_1² r_2 t_{12}² t_{21}`. The triple `U_6, R_{(4,7)}, R_{(1,8)}` is bad (B3) and
unshielded (`(2,5)` does not contain 6). The basis braid has the triple in the
lexicographic orientation; a reduced word with the other orientation gives the element

    β' − β  =  −r_1 t_{12} · E_{(2,1,1,W,2,1,1,2,1)} s2 s1 s4 s6 E_{(1,2,1,2,W,1,1,2,1)}

— a diagram in which the strand crossing the red strand is not the mover's: a *hand-over*
diagram in the sense of `NOTES.md` §6. (The same configuration with `(2,6)` in place of
`(2,5)` is shielded and independent.) So in cyclic words the choice of word matters at the
level of `r t × (hand-over diagram)`, exactly the kind of extra terms seen in §6 of `NOTES.md`.

### B.8 The definition of `β` to be used

**Definition B.10 (basis braid).** For states `s, s'` and a bijection carrying the strands of
`s` to those of `s'` (colour preserving), let `π(p)` be the target position of the strand at
source position `p`. The *basis braid* is the word

    w(π) = ∏_{p = 1}^{n}  ( p−1, p−2, …, p−ℓ_p ),        ℓ_p = #{p' < p : π(p') > π(p)},

the factors taken in the order `p = 1, 2, …, n` (insertion sort: the strands are processed from
left to right in their source order, and each one moves to the left past exactly those
already processed strands that must end to its right). It is reduced, and it is the word that
`klrw` uses (`stable_envelopes.braid_from_marked_state`, the "Lehmer cocode"; this is
`fast.braid_for`). **Define `β(I; j, F) := ψ_{w(π)}` for the bijection `σ`.**

For a tree term this word is: first the left strands sort themselves (the `L_p` move
outwards, past the unaffected `y ∉ I` they must cross and past the nested `L`'s); then the
right-hand strands with slot `< j` in increasing slot order (the `R_p` with `b_p < j` move
inwards); then the mover moves all the way from `+j` to `−j`; then the `R_p` with `b_p > j`.

*Which word does the formula need?* In every case where `d² = 0` has been verified
(`PROOFS.md` Tier 2, `general_words.py`) the element is independent of the word: either there
is no bad triple (all minuscule cases, Theorem B.6 / §B.5), or all bad triples are shielded
and the element was checked to be independent (all cycle-free bad words of length ≤ 9 above;
`verify.py` gives `d² = 0` for `12321223`, `21232122`). So no test can distinguish reduced
words there: `d² = 0` holds for every choice. In cyclic words `d² ≠ 0` for every choice (the
tree formula is incomplete), and the data of `NOTES.md` §6 ("all tree terms present with the
predicted coefficient") are stated for the basis braid; with another word some of the extra
hand-over terms change by the amounts of §B.7. Hence: use Definition B.10, and in proofs for
cycle-free words either (a) restrict to words without bad triples (Theorem B.6), where any
reduced word may be used, or (b) cite Conjecture B.9 explicitly.

### B.9 Comparison with `05_main_sliding.md`

* **Lemma 1.2 (Straightening).** Correct. Its hypothesis (S2) excludes every triple of
  pairwise crossing strands `A, B, C` with `A, C` black of one colour and `B` adjacent,
  *whatever the order of the three strands*; this is needed there because the lemma also
  treats non-reduced words, where the middle strand of a braid move is not determined by the
  permutation. For **reduced** words the sharp condition is Proposition B.5 (only triples whose
  *middle* strand is the adjacent one matter). Lemma 1.4 (vanishing) is correct and is the
  mechanism behind Lemma B.8.
* **Lemma 6.1** = Lemma B.2 (agrees). **Lemma 6.2** (multiplicities `≤ 2`: every reduced word
  satisfies (S2), so `β` is well defined) agrees with Theorem B.6(a); the proof there is
  correct.
* **The "Warning" after Lemma 6.2 is misleading.** For `A_3`, word `3232332`,
  `I = {5,6,7}`, mover 6, pairs `(1,5), (2,7)`, (S2) fails, but the diagram has only the two
  triples `U_3, L_{(1,5)}, M` and `U_4, L_{(1,5)}, M` (type T1), whose middle strand is
  `L_{(1,5)}` (colour 3) and whose outer strands have colours `(3, 3)` resp. `(2, 3)`. Neither is
  bad (`(3,3,3)` and `(2,3,3)` are exact cubic states), so by Proposition B.5 `β` **is** well
  defined there. (Checked: `01_bad_search.bad_triples` returns none; the word is cycle-free.)
  Genuine failures of well-definedness need a cyclic word (§B.7); genuine bad triples in
  cycle-free words need a colour of multiplicity `≥ 4` (Theorem B.6(b)).

---

## D. Intertwiners

Relations are cited by the labels (A0)–(A6) of Definition A.1 below; "position" means position
in the state on the side where an element is written. For a state `s`, `P(s)` is the
polynomial ring over `R` in the dots `x_p` of the black positions `p` of `s`. For a word `w`
(from `s` to `s'`) and `f ∈ P(s)` let `f^w ∈ P(s')` be `f` *transported along the strands*:
`x_p ↦ x_{π(p)}` with `π(p)` the target position of the strand starting at `p`.

**Definition D.0.** For a state `s` and `1 ≤ i < n` put

    φ_{i,s} = (x_i − x_{i+1}) ψ_{i,s} − r_c e(s)      if s_i = s_{i+1} = c is black,
    φ_{i,s} = ψ_{i,s}                                  otherwise (different colours, or one strand red).

`φ_{i,s}` goes from `s` to `s_i s`, like `ψ_{i,s}`. It is homogeneous: in the first case of
degree `deg x + deg ψ = (2q + δ_c) + (−2q + σ_c) = σ_c + δ_c = deg r_c`.

### D.1 Two-strand identities

**Lemma D.1.** Let `s_i = s_{i+1} = c` be black, so `s_i s = s`. Then

(a) `φ_{i,s} = −ψ_{i,s}(x_i − x_{i+1}) + r_c e(s)`;

(b) `x_i φ = φ x_{i+1}` and `x_{i+1} φ = φ x_i` (a dot follows its strand); `x_p φ = φ x_p`
for `p ∉ {i, i+1}`; hence `f φ = φ f^{s_i}` for every `f ∈ P(s)`;

(c) `φ_{i,s} φ_{i,s} = r_c² e(s)`.

If `s_i ≠ s_{i+1}`, then (b) holds with `φ = ψ` (whenever the positions are black) and
`φ φ = Q_{s_i s_{i+1}}(x_i, x_{i+1})` (A5).

*Proof.* Write `x = x_i`, `x' = x_{i+1}`, `ψ = ψ_{i,s}`, `r = r_c`. (A4) reads
`xψ = ψx' + r`, `x'ψ = ψx − r`.
(a) `(x − x')ψ = ψx' + r − ψx + r = −ψ(x − x') + 2r`; subtract `r`.
(b) `xφ = x(x − x')ψ − rx = (x − x')(ψx' + r) − rx = (x − x')ψx' − rx' = φx'`, using that
dots commute (A1) and that `r` is central. Symmetrically
`x'φ = (x − x')(ψx − r) − rx' = (x − x')ψx − rx = φx`. Far dots: (A3). The statement for
`f` follows multiplicatively.
(c) `φφ = φ(x − x')ψ − rφ = (x' − x)φψ − rφ` by (b); `φψ = (x − x')ψψ − rψ = −rψ` by (A5);
so `φφ = r(x − x')ψ − r((x − x')ψ − r) = r²`. Different colours: (A4), (A5). ∎

*Check:* `01_intertwiners.py` (D1)–(D3), all 2-strand states over `{1,2,3}` of `A_3` and
`W1`/`W2`.

### D.2 The braid relation

**Lemma D.2 (leading terms).** Let `D` be a word of `l` letters from `s`, `g ∈ P(s')` a
polynomial at its target. Then `ψ_D g = g^{D^{-1}} ψ_D + (a combination of words with fewer
than `l` letters, polynomial coefficients on the left)`, where `g^{D^{-1}} ∈ P(s)` is `g`
transported back along the strands. *Proof.* Induction on `l` with (A3), (A4): moving a dot
to the left through one crossing transports it along its strand, up to a term in which that
crossing is deleted (the `± r_c e` of (A4)). ∎

**Theorem D.3.** On every state `s` (any colours, at most one red strand) and every `i`,

    φ_i φ_{i+1} φ_i  =  φ_{i+1} φ_i φ_{i+1}

exactly (each `φ` taken at the state reached at that moment). Also `φ_iφ_j = φ_jφ_i` for
`|i − j| ≥ 2`.

*Proof.* The second statement is (A2), (A3). For the first, by (A2), (A3) we may assume
`n = 3`, `i = 1`; call the strands at the positions 1, 2, 3 of `s` `A, B, C`. Write
`φ = h·ψ + c` for each of the six factors, with `h ∈ {x_X − x_Y, 1}` and `c ∈ {−r_*, 0}`.
Expanding and moving all dots to the left (Lemma D.2),

    φ_1φ_2φ_1 = P · ψ_1ψ_2ψ_1 + (words of ≤ 2 letters),    φ_2φ_1φ_2 = P' · ψ_2ψ_1ψ_2 + (words of ≤ 2 letters),

where `P` is the product, over the pairs `{X, Y} ⊂ {A, B, C}` of equal black colour, of
`x_X − x_Y` (`X` the left one at the source), the dots read at the source. Indeed in
`φ_1φ_2φ_1` the three `h` are `x_1 − x_2` (pair `A,B`), `x_2 − x_3` at the state `(B,A,C)`,
transported back: `x_1 − x_3` (pair `A, C`), and `x_1 − x_2` at `(B,C,A)`, transported back:
`x_2 − x_3` (pair `B,C`); in `φ_2φ_1φ_2` one finds the same three differences
`x_2 − x_3`, `x_1 − x_3`, `x_1 − x_2`. So `P' = P`, and by (A6)
`P(ψ_1ψ_2ψ_1 − ψ_2ψ_1ψ_2) = P·C e(s)` has no crossing. Hence `Δ := φ_1φ_2φ_1 − φ_2φ_1φ_2`
is a combination of words with at most two letters, and by the spanning part of (F1)
(Theorem A.5(i)) `Δ = Σ_v g_v ψ_v` over basis braids `ψ_v` from `s` to `w_0 s` with
`ℓ(v) ≤ 2`, `g_v ∈ P(s)`.

On the other hand, both `φ_1φ_2φ_1` and `φ_2φ_1φ_2` move the strands by `w_0 = s_1s_2s_1`, so
by Lemma D.1(b) `f Δ = Δ f^{w_0}` for every `f ∈ P(s)`. Suppose `Δ ≠ 0` and let `v` be of
maximal length with `g_v ≠ 0`. By Lemma D.2, `ψ_v f^{w_0} = f^{v^{-1}w_0}ψ_v` + shorter, where
`f^{v^{-1}w_0}` is `f` with its variables permuted by the permutation `v^{-1}w_0 ≠ 1`. By the
uniqueness in (F1) (Theorem A.5(ii)), comparing the coefficients of `ψ_v`:
`f g_v = g_v f^{v^{-1}w_0}` for all `f`. A non-trivial permutation of three positions moves at
least two of them, hence at least one black position `p` (there is one red strand at most);
take `f = x_p`: `g_v (x_p − x_{p'}) = 0` with `p' ≠ p`, so `g_v = 0` because `P(s)` is a
domain. Contradiction; `Δ = 0`. ∎

*Direct computation without (F1), for the record.* If no two of `A, B, C` have the same black
colour, all `φ = ψ` and the triple is not bad: (A6) with `C = 0`. If `c(A) = c(C) = c` and
`B` has another colour `b` (black or red):
`φ_1φ_2φ_1 = ψ_1((x_2 − x_3)ψ_2 − r_c)ψ_1 = (x_1 − x_3)ψ_1ψ_2ψ_1 − r_c Q_{cb}(x_1, x_2)` and
`φ_2φ_1φ_2 = (x_1 − x_3)ψ_2ψ_1ψ_2 − r_c Q_{bc}(x_2, x_3)`; the difference is
`(x_1 − x_3) C(c,b,c) − r_c (Q_{cb}(x_1,x_2) − Q_{bc}(x_2,x_3))`, which is
`(x_1 − x_3) r_c t_{cb} − r_c t_{cb}(x_1 − x_3) = 0` for `b ∼ c`,
`(x_1 − x_3) r_k u_k − r_k u_k (x_1 − x_3) = 0` for `c = k`, `b = W`, and `0 − 0` otherwise.
So **the bad cubic term is exactly cancelled by the bigon of the intertwiner.** If
`c(A) = c(B) ≠ c(C)`: `φ_1φ_2φ_1 = (x_1 − x_2)ψ_1ψ_2ψ_1 − r ψ_2ψ_1` and
`φ_2φ_1φ_2 = (x_1 − x_2)ψ_2ψ_1ψ_2 − rψ_2ψ_1`, equal by the exact (A6) on `(c, c, b)`;
`c(B) = c(C) ≠ c(A)` is the mirror image. Only `(c, c, c)` needs the argument above.

*Check:* `01_intertwiners.py` (D4): all 3-strand states over `{1,2,3,Wk}`, `k = 1, 2`.

**Corollary D.4.** For a permutation `w` of the positions with a reduced word
`w = s_{i_1}⋯s_{i_l}`, the element `φ_w = φ_{i_1}⋯φ_{i_l}` does not depend on the reduced word
(Matsumoto + Theorem D.3), and `f φ_w = φ_w f^w` (Lemma D.1(b)).

(This is the analogue of Kang–Kashiwara–Kim, *Symmetric quiver Hecke algebras and R-matrices of
quantum affine algebras*, Invent. Math. 211 (2018), §1.3.1, the lemma on intertwiners (items
(i)–(v)), stated there for KLR algebras with `φ = (x_{a+1} − x_a)τ_a − 1` and "easily following
from the polynomial representation" [KL Prop. 2.3, Rouquier Prop. 3.12]. It does not apply
literally here: our parameters `r_c` are not invertible and our relations carry `r_c`; over
`ℤ[r^{±1}, …]` and with `ψ = r_c ψ'` one has `φ = r_c·((x_i − x_{i+1})ψ' − 1)`.)

### D.3 Mixed relation and naturality

**Proposition D.5.** On a state `(A, B, S)` (positions `i, i+1, i+2`)

    ψ_i φ_{i+1} φ_i  =  φ_{i+1} φ_i ψ_{i+1},

and on a state `(S, A, B)`:  `φ_i φ_{i+1} ψ_i = ψ_{i+1} φ_i φ_{i+1}`.

*Proof.* If `A, B` are not black of one colour, `ψ = φ` for the crossing of `A` and `B` and
this is Theorem D.3. Let `c(A) = c(B) = c`, `i = 1`, `D_1 = x_1 − x_2`, `D_2 = x_2 − x_3`.
First identity: Theorem D.3 on `(A, B, S)` reads
`(D_1ψ_1 − r)φ_2φ_1 = φ_2φ_1(D_2ψ_2 − r)` (the last factor is the crossing of `A, B` at the
state `(S, A, B)`). By Lemma D.1(b), `φ_2φ_1 D_2 = D_1 φ_2φ_1` (the strands `A, B` end at the
positions 2, 3). So `D_1·(ψ_1φ_2φ_1 − φ_2φ_1ψ_2) = 0`. Left multiplication by `D_1 e(s)` is
injective on `e(s) A e(s')` (Theorem E.1), hence the claim. Second identity: Theorem D.3 on
`(S, A, B)` reads `φ_1φ_2(D_1ψ_1 − r) = (D_2ψ_2 − r)φ_1φ_2`, and `φ_1φ_2 D_1 = D_2φ_1φ_2`;
so `D_2·(φ_1φ_2ψ_1 − ψ_2φ_1φ_2) = 0`; Theorem E.1 again. ∎

*Check:* `01_intertwiners.py` (D5), all 3-strand states.

**Setting for naturality.** Let `t` range over the states of a *block* of `n` strands (black
strands and at most one red strand), and let `S` be one more strand of colour `c` (black, or
red if the block has no red strand). Juxtaposition `a ↦ a ⊗ 1` (the block at the positions
`1, …, n`, `S` straight at `n+1`) and `a ↦ 1 ⊗ a` (`S` at position 1, the block at
`2, …, n+1`) are well-defined `R`-linear maps from the algebra of the block, multiplicative
and compatible with idempotents, because every relation of the block algebra is a relation of
the big algebra (with all positions shifted in the second case). Put

    Φ(t)  =  φ_n φ_{n−1} ⋯ φ_1  ∈  e(t, c) A e(c, t)        (S moves from the far right to the far left),
    Φ'(t) =  φ_1 φ_2 ⋯ φ_n      ∈  e(c, t) A e(t, c)        (S moves from the far left to the far right).

These words are the unique reduced words of their permutations (one strand moves), so there is
no choice. `Φ(t)` is the elementary move of `S` with every crossing of `S` with a strand of its
own colour replaced by the intertwiner; it is homogeneous of degree
`deg(ψ-word) + N(2q + δ_c)`, `N` = number of strands of colour `c` in the block.

**Theorem D.6 (naturality).** For every element `a ∈ e(t) A_{block} e(t')`:

    (a ⊗ 1) · Φ(t')  =  Φ(t) · (1 ⊗ a),        (1 ⊗ a) · Φ'(t')  =  Φ'(t) · (a ⊗ 1).

*Proof.* Both sides are `R`-linear in `a`, and if the identity holds for `a` and for `b` it
holds for `ab`: `(ab ⊗ 1)Φ = (a⊗1)(b⊗1)Φ = (a⊗1)Φ(1⊗b) = Φ(1⊗a)(1⊗b) = Φ(1⊗ab)`. So it
suffices to take `a` a generator. Idempotents: trivial. A dot `x_p` (`p ≤ n`, black): it
commutes with `φ_l`, `l ∉ {p−1, p}` (A3); in `Φ(t)` the strand at `p` is crossed by `S` in the
factor `φ_p`, where it moves to `p+1` (Lemma D.1(b): `x_pφ_p = φ_p x_{p+1}`), and afterwards it
commutes with `φ_{p−1}, …, φ_1`. So `x_pΦ(t) = Φ(t)x_{p+1}`, which is the claim. A crossing
`ψ_i` (`i < n`): write `Φ(t) = (φ_n⋯φ_{i+2})(φ_{i+1}φ_i)(φ_{i−1}⋯φ_1)`. `ψ_i` commutes with
the first group (positions `≥ i+2`; (A2), (A3)), then `ψ_iφ_{i+1}φ_i = φ_{i+1}φ_iψ_{i+1}`
(Proposition D.5 on the strands at `i, i+1` and `S` at `i+2`), and `ψ_{i+1}` commutes with the
last group (positions `≤ i`). So `ψ_iΦ(t') = Φ(t)ψ_{i+1}` with the appropriate states, i.e.
`(ψ_i ⊗ 1)Φ = Φ(1 ⊗ ψ_i)`. The second identity is the mirror argument with the second half of
Proposition D.5. ∎

*Check:* `01_intertwiners.py` (D6): 450 random elements (random words with random dots) on
blocks `(1,2,1)`, `(1,1,2)`, `(1,W1,1)`, `(2,1,W1)`, `(1,1,1)`, `(W2,2,1)` and `S` of colours
1, 2, 3; all equal. Negative control: with `ψ` in place of `φ` 41 of 180 fail
(`01_intertwiners.py negative`).

**Remark D.7 (what this gives for the cone, README §"Intertwiners").** In the cone
decomposition `w = w'·c`, take the block = all strands of `w'` together with the red strand and
`S` = the strand of the last slot `m`. Then `Φ` is the map called `Φ` in the README, and
Theorem D.6 is exactly `(a ⊗ 1)·Φ = Φ·(1 ⊗ a)` for every `a` in the algebra of `w'`, including
the red strand and including states where `S` crosses the red strand (there `φ = ψ`). So `Φ` is a
chain map `C' ⊗ 1 → 1 ⊗ C'` as soon as `d'² = 0`, with the degree stated in the README. That
`z = ∏_{y<m, c(y)=c}(x_m − x_y)` is a non-zero-divisor on each `e A e'` is Theorem E.1.

---

## A. The presentation and the basis theorem

### A.1 The presentation

**Definition A.1.** *Data.* A finite simply-laced Dynkin diagram `Γ` (a finite graph without
loops or multiple edges which is a forest; `c ∼ c'` = adjacent), a colour `k ∈ Γ` of the red
strand, and a finite multiset `ν` of colours (the colours of the word). *Base ring*

    R = ℤ[ r_c (c ∈ Γ),  t_{cc'} (ordered pairs c ∼ c'),  u_c (c ∈ Γ) ]     (polynomial ring).

*States.* `Seq` = the set of sequences `s = (s_1, …, s_n)`, `n = |ν| + 1`, in `Γ ⊔ {W}` that
contain `W` once and every `c ∈ Γ` with multiplicity `ν(c)`. Position `p` is *black* if
`s_p ≠ W`. `s_i s` = `s` with the entries `i, i+1` exchanged.

*Generators* (each with a source and a target state): `e(s): s → s`; `x_{p,s}: s → s` for `p`
black; `ψ_{i,s}: s → s_i s` for `1 ≤ i ≤ n−1`.

`A` is the `R`-algebra spanned by composable products of generators (a product of two elements
whose target and source differ is `0`; `a·b` = first `a`, then `b`), modulo the relations below.
The subscript `s` is the state at the place where a generator is written, and is often
omitted. `P(s) = R[x_{p,s} : p black]`.

* **(A0)** `e(s)` is the identity at `s` (`e(s)a = a` for `a` with source `s`, etc.);
  `Σ_s e(s) = 1`.
* **(A1)** `x_{p,s} x_{q,s} = x_{q,s} x_{p,s}`.
* **(A2)** `ψ_{i,s} ψ_{j, s_i s} = ψ_{j,s} ψ_{i, s_j s}` for `|i − j| ≥ 2`.
* **(A3)** `x_{p,s} ψ_{i,s} = ψ_{i,s} x_{p, s_i s}` for `p ∉ {i, i+1}`.
* **(A4)** If `s_i ≠ s_{i+1}`: `x_{i,s} ψ_{i,s} = ψ_{i,s} x_{i+1, s_i s}` (if `s_i ≠ W`) and
  `x_{i+1,s} ψ_{i,s} = ψ_{i,s} x_{i, s_i s}` (if `s_{i+1} ≠ W`).
  If `s_i = s_{i+1} = c ∈ Γ` (so `s_i s = s`):

      x_{i} ψ_{i} − ψ_{i} x_{i+1} = r_c e(s),        ψ_{i} x_{i} − x_{i+1} ψ_{i} = r_c e(s).

* **(A5)** `ψ_{i,s} ψ_{i, s_i s} = Q_{s_i s_{i+1}}(x_{i,s}, x_{i+1,s})`, where

  | `(s_i, s_{i+1})` | `Q` |
  |---|---|
  | `(c, c)` | `0` |
  | `(c, c')`, `c ∼ c'` | `t_{cc'} x_i + t_{c'c} x_{i+1}` |
  | `(c, c')`, `c ≠ c'` not adjacent | `1` |
  | `(k, W)` / `(W, k)` | `u_k x_i` / `u_k x_{i+1}` (the dot of the black strand) |
  | `(c, W)` / `(W, c)`, `c ≠ k` | `1` |

* **(A6)** On a state with `(s_i, s_{i+1}, s_{i+2}) = (a, b, a')`:

      ψ_i ψ_{i+1} ψ_i − ψ_{i+1} ψ_i ψ_{i+1} = C(a, b, a') e(s),

  `C(c, c', c) = r_c t_{cc'}` if `c' ∼ c` (black); `C(k, W, k) = r_k u_k`; `C = 0` otherwise.
  (Each `ψ` at the state reached; both sides go from `s` to `(a', b, a)`.)

*Grading* (F2 of `PROOFS.md`): `deg e = 0`, `deg x_{p,s} = 2q + δ_{s_p}`,
`deg ψ_{i,s}` = the crossing degree of `(s_i, s_{i+1})` (`−2q + σ_c`, `q + ε_{cc'}`,
`q + ε_{kW}`, or `0`), `deg r_c = σ_c + δ_c`, `deg t_{cc'} = 2ε_{cc'} − δ_c`,
`deg u_c = 2ε_{cW} − δ_c`. Every relation is homogeneous (`PROOFS.md` F2, re-checked:
(A4) `σ_c + δ_c` on both sides; (A5) `2q + 2ε_{cc'} = (2ε_{cc'} − δ_c) + (2q + δ_c)`;
(A6) `σ_c + 2ε_{cc'}` and for the red case `σ_k + 2ε_{kW}`).

Remarks. (1) There is no generator for a dot on the red strand and no relation for two red
strands (there is one). (2) The two dot relations of (A4) for equal colours imply
`x_{i+1}ψ_i = ψ_i x_i − r_c` and `ψ_i x_{i+1} = x_iψ_i − r_c`, so
`ψ(x_i − x_{i+1}) + (x_i − x_{i+1})ψ = 2r_c` (README). (3) `r_c` is central, as is every
element of `R`.

### A.2 Re-verification in the implementation

`01_relations.py` builds, for **every** sequence of length 2 and 3 over `{1, 2, 3, Wk}` of
`A_3` (`1 ∼ 2 ∼ 3`, `1 ≁ 3`) with at most one red strand, `k = 1, 2`, the algebra of
`klrw` (all three scalings, base `ℤ`, polynomial parameters) and checks (A4) (both
equations, and the two "wrong-strand" versions with `−r_c`), (A5), (A6), (A3), and (A2) on
sample 4-strand states: **388 checks, 0 failures**. The non-zero cubic values found are
exactly `C(1,2,1) = r_1t_{12}`, `C(2,1,2) = r_2t_{21}`, `C(2,3,2) = r_2t_{23}`,
`C(3,2,3) = r_3t_{32}`, `C(1,W1,1) = r_1u_1`, `C(2,W2,2) = r_2u_2`. In particular the README's
signs are confirmed, including: the braid relation is exact on `(c,c,c')`, `(c',c,c)`,
`(W,k,k)`, `(k,k,W)`, `(c, W_k, c)` (`c ≠ k`), `(c, c'', c)` (non-adjacent), and on all states
with three different colours. Every 3-strand colour pattern of any simply-laced `Γ` (with at
most one red strand) is covered, because the relations on it depend only on which of the
strands have equal colours and on the relation (adjacent / non-adjacent / red of that colour /
red of another colour) between the colours involved, and on three different colours the
braid relation is exact for any adjacencies (§A.3, cocycle argument).

Convention of the implementation (`klrw`): an element is `Σ coeff · E_{left} s_{i_1} ⋯`, the
coefficient a polynomial in the parameters and in the dots of the **left** state
(`x_{c}_{n}` = dot on the `n`-th strand of colour `c` counted from the left), placed before the
crossings; basis braids are the words of Definition B.10.

### A.3 The polynomial representation

**Definition A.2.** `V = ⊕_s V_s`, `V_s = R[y_p : p black in s]`. Order the colours by any total
order with `W` last (we use `1 < 2 < … < W`). A **right** action of the generators: for
`f ∈ V_s`,

* `f · x_{p,s} = y_p f`;
* `f · ψ_{i,s} = r_c ∂_i f`, `∂_i f = (f − s_i f)/(y_i − y_{i+1})`, if `s_i = s_{i+1} = c`;
* `f · ψ_{i,s} = s_i f` if `s_i < s_{i+1}`;
* `f · ψ_{i,s} = Q_{t_i t_{i+1}}(y_i, y_{i+1}) · s_i f` if `s_i > s_{i+1}`, where `t = s_i s` is
  the target state and `Q` is the table of (A5) for `t`;

with `s_i f` = exchange of `y_i, y_{i+1}` (a black variable may move to the former red
position; the red position never carries a variable).

**Proposition A.3.** These operators satisfy (A0)–(A6); hence `V` is a right `A`-module.

*Proof.* (A0)–(A3): clear (disjoint variables). (A4), same colour: `∂_i(y_i f) − y_{i+1}∂_i f
= f` and `y_i ∂_i f − ∂_i(y_{i+1}f) = f` (direct), times `r_c`; different colours: `s_i` moves
`y_i` to `y_{i+1}` and the factor `Q` commutes with multiplication. (A5): same colour,
`∂_i^2 = 0` (`∂_i f` is `s_i`-invariant); different colours, if `a = s_i < b = s_{i+1}`,
`f ↦ s_i f ↦ Q_{ab}(y_i, y_{i+1}) s_i s_i f = Q_{ab}(y_i, y_{i+1}) f`, and if `a > b`,
`f ↦ Q_{ba}(y_i, y_{i+1}) s_i f ↦ s_i(Q_{ba}(y_i, y_{i+1}) s_i f) = Q_{ba}(y_{i+1}, y_i) f
= Q_{ab}(y_i, y_{i+1}) f` (by the table, `Q_{ba}(u, v) = Q_{ab}(v, u)`). (A6): write each
operator as an element `Σ_w c_w w` of the twisted group algebra `K ⋊ S_n`,
`K = Frac R[y]` (`w` acting by permuting the variables, `(Σ c_w w)(f) = Σ c_w w(f)`):
`ψ_i ↦ (r_c/(y_i − y_{i+1}))(1 − s_i)`, resp. `s_i`, resp. `Q·s_i`. If the three strands have
pairwise different colours (or one is red), both sides of (A6) equal
`(∏ of the three transported factors) · w_0`, because each pair of strands crosses once, with
the same strand on the left, in both words: the braid relation is exact, `C = 0`. The
remaining patterns `(c,c,c)`, `(c,c,b)`, `(b,c,c)`, `(c,b,c)` depend only on the relation of `b`
to `c` (adjacent, non-adjacent, red with `c = k`, red with `c ≠ k`), and for each of them both
sides were expanded exactly in `K ⋊ S_3` with symbolic parameters (`01_polyrep.py twisted`:
**276 identities, all exact**, including (A4) and (A5) once more). This is the computation of
Khovanov–Lauda [KL, Prop. 2.3] with the factor `r_c` on equal-colour crossings. ∎

*Further checks:* `01_polyrep.py relations` (3 746 relation instances on all monomials of
degree ≤ 3); `01_polyrep.py hom`: for 1 500 random pairs `a, b` of basis elements of the
`klrw` implementation, `ρ(a·b) = ρ(b)∘ρ(a)` on test polynomials — i.e. the implementation's
multiplication **is** the composition in this representation.

### A.4 The basis theorem (F1)

For states `s, s'` let `S(s, s')` be the set of permutations `w` of the positions with
`s'_{w(p)} = s_p` for all `p` (the strand at `p` ends at `w(p)`). For each `w` fix a reduced
word and let `ψ_w ∈ e(s) A e(s')` be the product of the corresponding `ψ`'s.

**Theorem A.5 (F1).** (i) `e(s) A e(s')` is spanned over `R` by `{x^α ψ_w : α, w ∈ S(s,s')}`
(`x^α` a monomial in the dots of `s`), and more precisely the span of all products with at most
`l` crossings equals `span{x^α ψ_w : ℓ(w) ≤ l}`.
(ii) These elements are linearly independent over `R`. Hence `e(s) A e(s')` is a **free left
`P(s)`-module with basis `{ψ_w}`**, for **any** choice of the reduced words.
(iii) `{ψ_w x^β : w, β}` (`x^β` monomials in the dots of `s'`) is also an `R`-basis: `e(s) A e(s')`
is a **free right `P(s')`-module with basis `{ψ_w}`**.
(iv) For every commutative `R`-algebra `S` (e.g. a specialisation of the parameters, `r = 0`
included), `A_S = A ⊗_R S` is presented by the same generators and relations over `S`, and
(i)–(iii) hold over `S`.
(v) The representation `V` is faithful.

*Proof.* (i) Let `A_{≤l}` be the span of the elements `f ψ_D`, `f ∈ P(s)`, `D` a word with
`≤ l` letters. *Dots to the left:* by (A3), (A4) `ψ_{i} x_p = x_{p'} ψ_i` or
`= x_{p'}ψ_i ∓ r_c e`; so a product of generators with `l` crossings lies in `A_{≤ l}`
(induction on `l` and on the number of dots to the right of crossings). *Words:* let `D` have
`l` letters. If `D` is reduced, with permutation `w`, then `ψ_D ≡ ψ_w mod A_{≤ l−1}`:
connect `D` to the fixed word by Matsumoto moves; (A2) is exact and (A6) changes the element by
`C ψ_u ψ_v`, a word of `l−3` letters. If `D` is not reduced, write `D = D_0 · (i) · D_1` with
`D_0` reduced and `D_0 · (i)` not reduced; then the permutation of `D_0` has a reduced word
`E · (i)`, so `ψ_{D_0} ≡ ψ_E ψ_i mod A_{≤ |D_0|−1}` and
`ψ_{D_0}ψ_i ≡ ψ_E ψ_iψ_i = ψ_E Q mod A_{≤|D_0|}`, and `ψ_E Q ∈ A_{≤ |E|}` (dots to the left);
so `ψ_{D_0 i} ∈ A_{≤ |D_0|}` and `ψ_D ∈ A_{≤ l−1}` (appending letters adds at most their
number). Induction on `l` gives (i).

(ii) Let `Σ c_{α,w} x^α ψ_w = 0` (finite sum, `c ∈ R`). Apply it to `V_s` and extend scalars to
the field `K = Frac R[y]` (`R` is a domain). In `K ⋊ S_n` the operator of one letter is
`c_1·1 + c_2·s_i` with `c_2 ≠ 0` (`−r_c/(y_i − y_{i+1})`, `1`, or `Q ≠ 0`: here we use that
`r_c, t_{cc'}, u_k` are non-zero in `R`); the operator of `ψ_w` (letters applied in order) is
therefore `c_w · ŵ + Σ (terms whose permutation is a product of fewer than ℓ(w) simple
reflections)`, `c_w ≠ 0`, `ŵ` the permutation of the variables induced by `w` (distinct `w`
give distinct `ŵ`). The operator of `x^α ψ_w` is `c_w ŵ(y^α) ŵ + (shorter)`. Let `w` be of
maximal length with `Σ_α c_{α,w} y^α ≠ 0`; the coefficient of `ŵ` in the total operator is
`c_w ŵ(Σ_α c_{α,w} y^α) ≠ 0`, because no other term contains `ŵ`. By Dedekind's lemma
(distinct automorphisms of `K` are `K`-linearly independent) the operator is non-zero —
contradiction. (This is the argument of [KL, Thm. 2.5], [Rouquier, Prop. 3.12], [Webster,
Thm. 2.8]; it needs only that the leading coefficients are non-zero in a domain, not that they
are units.)

(iii) The *flip* `ι` — `ι(e(s)) = e(s)`, `ι(x_{p,s}) = x_{p,s}`, `ι(ψ_{i,s}) = ψ_{i, s_i s}` (the
reflected diagram, from `s_i s` to `s`) — extends to an `R`-linear anti-automorphism of `A`
(`ι(ab) = ι(b)ι(a)`), because it maps every relation to a relation: (A0)–(A3), (A5) to
themselves, the two equations of (A4) for equal colours to each other, and (A6) on `s` to (A6)
on `w_0 s` (palindromic words; `C(a, b, a)` is symmetric). `ι(x^α ψ_w) = ψ'_{w^{-1}} x^α`, with
`ψ'_{w^{-1}}` the reversed word (a reduced word of `w^{-1}`). Apply (i), (ii) (valid for any
choice of reduced words) to `e(s') A e(s)` and then `ι`.

(iv) Presentations commute with base change (right exactness of `⊗`); a basis of a free
module stays a basis.

(v) By (i) every element is `Σ c x^αψ_w`, and the proof of (ii) shows that its operator is
non-zero unless all `c` vanish. ∎

**Literature, and what has to be assumed to cite it.**

* Khovanov–Lauda, *A diagrammatic approach to categorification of quantum groups I*,
  Represent. Theory 13 (2009): polynomial representation (Prop. 2.3) and basis (Thm. 2.5),
  for KLR algebras over a field.
* Rouquier, *2-Kac-Moody algebras*, arXiv:0812.5023: Thm. 3.7 (the PBW basis holds over any
  commutative ring `k`, for any `Q` with `Q_{ii} = 0`, iff `Q_{ij}(u,v) = Q_{ji}(v,u)`),
  Prop. 3.12 (faithful polynomial representation).
* Webster, *Weighted Khovanov–Lauda–Rouquier algebras*, Doc. Math. 24 (2019) 209–250,
  arXiv:1209.2463v4: Prop. 2.7 (polynomial representation), Thm. 2.8 (basis), numbering of
  the arXiv v4 source (labels `prop:action`, `th:basis`); §3.1 (the Crawley–Boevey quiver
  `Γ_λ` and the *reduced* algebra obtained by killing the dots on the strand of the new
  vertex `0`, the red strand). **Standing assumption there:** "`Q_e` has coefficients before
  the pure monomials in `u` and `v` which are units" — false for our polynomial `t_{cc'}`,
  `u_k`. (`PROOFS.md` cites "Prop. 2.23"; there is no such statement in that role — the basis
  theorem is Thm. 2.8.)

How our `A` relates to them: over `R' = R[r_c^{-1}]`, put `ψ' = r_c^{-1}ψ` on equal-colour
crossings. Then (A4) becomes `x_iψ' − ψ'x_{i+1} = 1`, (A5) is unchanged, and (A6) becomes
the standard cubic relation `(Q_{ab}(x_i, x_{i+1}) − Q_{ab}(x_{i+2}, x_{i+1}))/(x_i − x_{i+2})`
(`= t_{ab}`, resp. `u_k`) — so `A ⊗ R'` is the KLR algebra of the Crawley–Boevey quiver
`Γ ∪ {W}` (`W` joined to `k`) with `Q_{cc'} = t_{cc'}u + t_{c'c}v`, `Q_{kW}(u, v) = u_k u + v`,
`Q_{cW} = 1` (`c ≠ k`), modulo the dot `y` of the red strand (which is central there: it slides
through every crossing, all of which are of different colours), up to the sign conventions of
the respective paper. Rouquier's Thm. 3.7 then gives the basis of `A ⊗ R'` (the quotient by the
central `y` of a module that is free over `k[y]` with the `y`-free basis), with no unit
assumption; and the basis statement over `R` follows because `R → R'` is injective and the
spanning set is the same. Webster's theorem would require inverting `t`, `u` as well. Since the
sign conventions of these papers were not matched line by line, **Theorem A.5 is proved above
directly** for our presentation; the literature serves as a cross-check.

*Numerical check of (F1)* (`01_polyrep.py rank`, ~8 min): for all pairs of states of
`(1,1,2)`, `(1,W1,1)`, `(1,1,1)` and samples of `(1,2,1,W1)`, `(1,1,W1,2)` (31 pairs), the
elements `x^α ψ_w` (all basis braids of `klrw`, all `|α| ≤ 2`) act on `V` by operators of full
rank over `GF(1000003)` after a random specialisation of the parameters (rank 12–60 = number
of elements in every case). A full-rank specialisation implies linear independence over
`Frac R` (all maximal minors cannot vanish identically), hence over `R`. Together with the
homomorphism check, this certifies the `klrw` implementation on these states.

---

## E. Torsion-freeness

**Theorem E.1.** Let `S` be a commutative `R`-algebra (e.g. `R` itself, or any specialisation
of the parameters), `s, s'` states, `A_S = A ⊗_R S`, `P_S(s) = S[x_p : p black in s]`.

(a) If `f ∈ P_S(s)` is a non-zero-divisor of `P_S(s)`, then `a ↦ f·a` is injective on
`e(s) A_S e(s')`.
(b) If `g ∈ P_S(s')` is a non-zero-divisor of `P_S(s')`, then `a ↦ a·g` is injective on
`e(s) A_S e(s')`.
(c) Hence `f·a·g = 0` implies `a = 0` for such `f, g`. Over `S = R` (a domain), every non-zero
polynomial in the dots is a non-zero-divisor, e.g. `x_p − x_q` (`p ≠ q`) and
`z = ∏_{y < m, c(y) = c}(x_m − x_y)`.

*Proof.* (a) By Theorem A.5(ii),(iv) every `a` is uniquely `Σ_w a_w ψ_w` with `a_w ∈ P_S(s)`;
`f a = Σ_w (f a_w) ψ_w`, and this is `0` iff all `f a_w = 0` iff all `a_w = 0`. (b) The same
with the right basis of Theorem A.5(iii): `a = Σ_w ψ_w b_w`, `b_w ∈ P_S(s')`,
`a g = Σ ψ_w (b_w g)`. (c) Compose. ∎

*What is not true / what to watch.* The dots of the two ends are different variables:
`f(x)·a` with `f` read at the source and `a·f(x)` with `f` read at the target are different
elements, and (a), (b) concern them separately. A statement like "`f·a = a·f'` determines `a`"
is not implied. When a map is written as right multiplication by `z` (`PROOFS.md` §1.3
convention: components of maps are right multiplications), cancelling `z` uses (b): if
`F·z = F'·z` with `z ∈ P(s')` a non-zero-divisor, then `F = F'`. Over a specialisation `S` in
which the relevant polynomial becomes a zero-divisor (e.g. coefficients in `ℤ/N`), the
statement fails as it must; for `S = R/(r_c)` etc. `P_S` is still a polynomial ring over a
domain and everything holds.

---

## C. Elementary moves and products of two of them

**Lemma C.1.** `m_j(I)` has exactly one reduced word, namely
`(P−1, P−2, …, P')` where `P`, `P'` are the source and target positions of the strand of `j`.
*Proof.* Only that strand changes its order with other strands (Lemma 1 of `PROOFS.md`); a
letter of a reduced word crosses an inverted pair, so every letter crosses the moving strand
with its left neighbour at that moment; the word is forced. ∎
So `m_j(I)` (and `δ₀`) involve no choice, and `m_j(I) = β(I; j, ∅)` for every convention.

**Lemma C.2.** Let `i < j` in `I`. The words `m_i(I)·m_j(I∖i)` and `m_j(I)·m_i(I∖j)` are
reduced words of the same bijection (all slots fixed, `e_I → e_{I∖{i,j}}`). The crossing pairs
are `{i, W}`, `{j, W}`, `{i, x}` (`x < i`), `{j, x}` (`x < j`, including `x = i`). The triples are
exactly `{W, i, j}` and `{x, i, j}` (`x < i`), and in each the middle strand is `i`. So the only
possible bad triples are `(x, i, j)` with `x < i`, `c(x) = c(j) ∼ c(i)`; the red strand is never
the middle strand. None of them has a shield (Definition B.7).

*Proof.* First product: the strand of `i` goes from `+i` to `−i`, crossing `W` and the strands
of the slots `< i`; then the strand of `j` goes from `+j` to `−j`, crossing `W` and all slots
`< j`, including `i` (now at `−i > −j`). Second product: `j` first, crossing `W` and the slots
`< j` (including `i` at `+i`); then `i` from `+i` to `−i`, crossing `W` and the slots `< i`, but
not `j` (at `−j < −i`). Each pair crosses at most once: reduced; both have the listed
inversion set, so the same bijection. Triples: two non-moving strands never cross, so a
triple contains `i` and `j` and a third strand crossing both: `W` or a slot `x < i`. Source
order: `W(0) < +i < +j`, resp. `±x < +i < +j`: middle `i`. A shield would be a strand crossing
`x` and `j` but not `i`; strands crossing the non-moving `x` are `i, j` only. ∎

**Corollary C.3.** `m_i(I) m_j(I∖i) = m_j(I) m_i(I∖j)` whenever no slot `x < i` has
`c(x) = c(j) ∼ c(i)` (Proposition B.5 for these two words). In particular for pairwise distinct
colours (Theorem 5 of `PROOFS.md`).

*Check:* `01_moves.py 300`: Lemma C.1 on 532 moves (all reduced words enumerated), Lemma C.2
on 266 products (reducedness, inversion set, triples and middle strand asserted). In the
algebra: the two products are equal in all 229 cases without a bad triple and **different in
all 37 cases with one** (so the converse of Corollary C.3 holds in all tested cases; it is not
proved here, and it is the reason why `δ₀² ≠ 0` and corrections are needed).

**Review of the proof of Theorem 5.** Line by line: the sign computation `ε_2 = −ε_1` is
correct (`#{x ∈ I∖j : x > i} = #{x ∈ I : x > i} − 1`, `#{x ∈ I∖i : x > j} = #{x ∈ I : x > j}`
since `i < j`). "In both products every pair crosses at most once, so both are reduced diagrams
of the same bijection": correct (Lemma C.2). "A bad triple needs two black strands of the same
colour": correct for both kinds of bad triples (bad-2 needs two black strands of colour `k`);
the triple `{W, i, j}` that the proof does not mention is harmless anyway (Lemma C.2: `W` is an
outer strand). The appeal to (F3) is legitimate because every braid move between reduced words
of a fixed permutation is applied to one of its triples. **The proof is complete.**

---

## F. Review of Tier 1 of `PROOFS.md`

| statement | verdict | remarks / repairs |
|---|---|---|
| §1.1 (R2) "`xψ − ψx' = ± r_c`" | imprecise | exact form: (A4) of Definition A.1 (both equations, sign `+r_c` with the dot on the strand *before* minus *after* for the strand going right, and *after* minus *before* for the strand going left). |
| §1.2 (F1) | statement right, reference wrong | "Webster, Prop. 2.23" is not the basis theorem; Webster's Thm. 2.8 (arXiv v4) assumes unit leading coefficients and does not apply to polynomial `t, u`. Use Theorem A.5 (proved here, over `ℤ[r,t,u]` and every base change; also the right-module version). |
| Lemma 1 | complete, correct | add: `m_j(I)` has a unique reduced word (Lemma C.1). |
| Prop. 2 | complete, correct | "well defined" is justified by Lemma C.1. |
| Prop. 3 | complete, correct | implicitly uses that the degree of a reduced diagram is the sum over its inversion pairs of the crossing degrees of their colours (true: each crossing's degree depends only on the two colours). Items 1–6 = Lemma B.2. Independent check `01_prop3.py 40000`: 40 000 random configurations (`A_2…A_4`, `D_4`, `D_5`, `E_6`, arbitrary `F`, not only admissible), own grading group and own reduced words: all satisfy the identity. |
| Cor. 3.1 | complete, correct | the comparison of coefficients is legitimate (`G` free on the listed generators); the `q`-equation is implied by the `δ`-equations. `01_prop3.py`: the tree monomial has exactly the required degree for 16 785 admissible configurations. |
| Prop. 4 | complete, correct, **given F1 over polynomial parameters** (now Theorem A.5) | uses: the basis consists of homogeneous elements, so a homogeneous `z` has only basis components of its degree; each basis braid is reduced, so each black strand crosses `W` at most once (`N = ρ + λ`); `l_k ≥ 0` (polynomial `u`). The extension to several red strands needs (F1) for that algebra, which is **not** covered by Theorem A.5 (one red strand). |
| Thm. 5 | complete, correct | see §C. |
| Prop. 6 | complete, correct | uses only that juxtaposition is multiplicative (relations are local) and (A2)/(A3) for far-apart pieces; (F1) is not needed. `D` must be the sum of the per-stack `D`'s (as stated). |
| Prop. 7 | complete, correct | holds for arbitrary words. The same argument shows that the existence of a bad triple (Cor. B.4) is a heap property and inherited by longer words. |
| Prop. 8 | complete, correct | checked: all boxes lie in `(4,4,4,3)`, the six comparisons hold componentwise, the colours `k−1, k+1` are not adjacent, `p` does not contain the mover; needs `k ≥ 4`, `n − k ≥ 4` so that the colours `k−3…k+3` exist (automatic when the diagram fits). |
| `NOTES.md` §2 "in all tested cases the element does not depend on the reduced word" | **true as stated, false in general** | false for cyclic words (§B.7); in cycle-free words it is proved without bad triples (Thm. B.6) and otherwise Conjecture B.9. |

---

## S. Status

**Proved completely** (from the relations (A0)–(A6), Matsumoto (F3), and — where stated — the
basis theorem, which is itself proved here):

* Definition A.1 is the presentation implemented in `klrw` (re-verified, §A.2), and the
  polynomial representation of Definition A.2 satisfies it (Prop. A.3; the (A6) part by exact
  symbolic computation in the twisted group algebra, `01_polyrep.py twisted`).
* (F1) for this presentation over `ℤ[r, t, u]` (polynomial parameters) and over every base
  change: spanning, left basis, right basis, faithfulness of `V` (Theorem A.5).
* Torsion-freeness on both sides (Theorem E.1).
* Crossing table, classification of triples (T1–T8 with their middle strands), the red strand
  is in no triple, interval form of bad triples (Lemma B.2, B.3, Cor. B.4).
* Proposition B.5: no bad triple ⇒ `β(I; j, F)` independent of the reduced word.
* Theorem B.6: no bad triple for multiplicities `≤ 2`; for multiplicities `≤ 3` only pattern S,
  which forces a cyclic configuration; none for the minuscule words of `Λ^{1,2,3}ℂ^n` (and
  duals), the vector representation of `so_{2n}`, and `D_4`.
* The local shield identity, Lemma B.8.
* Intertwiners: Lemma D.1 (dot sliding, `φ² = r_c²` resp. `Q`), Theorem D.3 (exact braid
  relation on every state; uses (F1)), Corollary D.4, Proposition D.5 (mixed relation; uses
  Theorem E.1), Theorem D.6 (naturality `(a⊗1)Φ = Φ(1⊗a)` for all `a`, both directions).
* Elementary moves: Lemma C.1, Lemma C.2, Corollary C.3; Theorem 5 of `PROOFS.md` (its proof
  is complete).
* Review of Tier 1 (§F): all statements correct and their proofs complete, with the repairs
  listed (references for (F1); Prop. 4's several-red-strands extension needs (F1) for that
  algebra).

**Verified only by computation** (exact, in the stated ranges):

* Bad triples in cycle-free words: exist (multiplicity `≥ 4`), always shielded (§B.5, §B.6).
* `β` independent of the reduced word for every tree term with a bad triple in cycle-free
  words tested (552 terms), and dependent for most unshielded bad triples in cyclic words
  (530 of 550): Conjecture B.9.
* No bad triple in any cycle-free order ideal of `A_7…A_9 ω_4`, `A_{10} ω_3`, `D_5…D_8`
  spinors, `E_6 ω_1`, `E_7 ω_7`, and in no word of `words.json` with `≤ 12` slots.
* (F1) linear independence numerically on 31 pairs of states; the `klrw` multiplication equals
  the composition in the polynomial representation (1 500 instances).
* The converse of Corollary C.3 (bad triple ⇒ the two products differ), 37/37 cases.

**Not proved / open**

* Conjecture B.9 (shielded bad triples are harmless globally; cycle-free ⇒ shielded). The
  local Lemma B.8 is the mechanism; the global argument fails as stated because other strands
  may pass between the two crossings of the shield, producing further triples that I could not
  control. For general cycle-free words with a colour of multiplicity `≥ 4` the
  well-definedness of `β` is therefore only computational; proofs in that range should use
  Definition B.10 (basis braid) explicitly or cite Conjecture B.9.
* A proof that cycle-free minuscule order ideals have no bad triples in general (it is proved
  for multiplicities `≤ 3` and `Λ^3`, computed for the other listed cases; for `Λ^k`, `k ≥ 4`,
  and the spinors only up to the computed ranks).

**Failed attempts.** (1) "Shortening" a bad triple to a cyclic configuration directly (replace
the mover by `y`, or `p` by `(a_p, y)`) works for pattern S but not in general: in cyclic words
the naive construction finds no cycle in about half of the unshielded cases
(`01_bad_search.py`, column "shortening construction found no cycle"), so a proof of
Conjecture B.9(a) needs a different construction. (2) A degree argument for the vanishing of
the smoothed diagram of a shielded bad triple: the multidegree does not exclude basis
elements with other bijections (as for the degree principle in `NOTES.md` §3).

**For other provers.**

1. Use the presentation of Definition A.1 (labels (A0)–(A6)); (F1) = Theorem A.5 is available
   with polynomial parameters and on both sides; cancel non-zero polynomials in dots on the
   left (source dots) or right (target dots) by Theorem E.1.
2. `β(I; j, F)` := the basis braid (Definition B.10). For words with multiplicities `≤ 2`, for
   cycle-free words with multiplicities `≤ 3`, and for the minuscule cases of Theorem B.6(c)
   any reduced word may be used (Prop. B.5 + Thm. B.6). For other cycle-free words, the
   independence of the word is Conjecture B.9 (verified), not a theorem. In cyclic words `β`
   genuinely depends on the word (§B.7) — differences are `r t ×` hand-over diagrams.
3. When comparing products of diagrams by braid moves, the only dangerous triples are those of
   Corollary B.4 (and, for products of two elementary moves, `(x, i, j)` with
   `c(x) = c(j) ∼ c(i)`); the red strand is never the middle strand of a triple in any of these
   diagrams.
4. Intertwiners: `φ = (x_i − x_{i+1})ψ − r_c` for equal colours, `φ = ψ` otherwise; exact braid
   relation on every state, dot sliding, `φ² = r_c²`/`Q`, and naturality of the intertwiner
   crossing of a strand with an arbitrary block (Theorem D.6). This makes the README's `Φ` a
   chain map `C' ⊗ 1 → 1 ⊗ C'` whenever `d'² = 0`.
5. `05_main_sliding.md`: Lemma 1.2 and Lemma 6.2 are fine; the "Warning" example in its §6 is
   not a failure of well-definedness (§B.9); its (S2) can be weakened to "no *bad* triple"
   (middle strand of adjacent colour) for reduced words.

**Scripts** (all in `proofs/checks/`, run from `experiments/cup_brane`):
`01_relations.py` (§A.2), `01_polyrep.py relations|hom|rank|twisted` (§A.3–A.4),
`01_triples.py` (Lemmas B.2, B.3), `01_bad_search.py crosscheck|exhaustive|random` (Cor. B.4,
§B.5), `01_mult3.py` (Thm. B.6), `01_minuscule_scan.py` (§B.5, minuscule ideals),
`01_shield_scan.py` (§B.6), `01_local4231.py` (Lemma B.8), `01_beta_words.py WORD | tally …`
(§B.6, §B.7: dependence on the reduced word in the algebra), `01_intertwiners.py [negative]`
(§D), `01_moves.py` (§C), `01_prop3.py` (§F, Prop. 3 / Cor. 3.1 / Lemma 1).
