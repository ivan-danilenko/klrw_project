# Referee report on `proofs/03_cyclic_combinatorics.md` (the cyclic condition)

Referee's scripts: `proofs/checks/R03_*.py` (plain `python3`, written from `PROOFS.md` Cor. 3.1 /
Prop. 7, `formula.py`, `cyclefree.py` and the text of the draft; the author's `03_lib.py` is
imported only in `R03_compare_author.py`, to compare answers). Logs: `proofs/checks/R03_logs/`.

## Summary

**I found no fatal error.** I read every proof line by line. The two load-bearing results,
the cycle-shortening Lemma 2.7 and the pattern Theorem 2.8, are correct and complete as written,
and the patterns T, B, N of §1.5 are precise enough to decide by machine; the theorem is literally
true for them. Theorems A and D are correct and cover all Young diagrams and all strict partitions,
in both directions. Theorem E and every number I re-derived agree with my own code. The problems I
did find are presentational, or concern inputs taken from the literature that the draft states
honestly and that could be removed altogether:

* **repairable**: Lemmas 3.1(c) and 4.2(c) and "weight order = reverse inclusion" (used in
  Cor. 3.4(c), Cor. 4.4(b) and Theorem U) are cited from Proctor and Stembridge without saying
  which results. In the explicit models they have short self-contained proofs (Issue 2).
* **cosmetic**: quantifiers in the pattern definitions; wording in the proof of Theorem D;
  the `S`-form of Theorem A for `k ≤ 3`; an over-claim of "independent" checking; the citation for
  positivity in Lemma 7.2; a "brief" claim that I could not locate (Issues 1, 3–8).

Every independent test agreed with the draft. I brute-forced 216 765 words: all words of length
≤ 9 over `A_2`, `A_3`, all of length ≤ 8 over `A_4`, `D_4` and a disconnected forest, plus random
words of length 11–14 in `A_2…A_5`, `D_4`, `E_6`. On every one of them, brute force over **all**
configurations agrees with my T/B/N detector: 0 disagreements, 6 008 of the words cyclic. Lemma 2.7
held in 3 970 surgeries on chordless cycles of length 6, 8, 10 and 12, with 0 failures. Theorem A
holds on all 924 diagrams in the 6×6 box and all 3 432 in the 7×7 box, and Theorem D
on all strict partitions with parts ≤ 8. For `E_6` and `E_7` I re-derived 3/27 and 22/56 cyclic
ideals, unique minimal ideals of 14 and 15 slots, and the B/N table. My own Hecke-algebra code
confirms the Deodhar statements of §7.

---

## 1. Verdict per statement

| statement | verdict |
|---|---|
| Lemma 1.1 (interlacing ⇔ one slot inside) | correct and complete |
| §1.3 definitions (configuration, blocks, cyclic; `I` irrelevant) | agrees with `PROOFS.md` Cor. 3.1, `formula.py` `analyse`, `cyclefree.py` `links` (I checked the code: same mover block/link test, same interlacing test, pairs disjoint and avoiding `j`) |
| §1.4 inheritance by super-words (not only prefixes) | correct |
| §1.5 patterns T, B, N | precise enough to decide by machine; quantifiers should be made explicit (Issue 1). With the existential reading Theorem 2.8 is literally true (tested) |
| Lemma 2.1 (colour map; bipartite; turning point) | correct and complete |
| **Prop. 2.2** (multiplicity ≤ 2 ⇒ no block, link graph a forest) | correct and complete |
| **Lemma 2.3** (a)–(e) (minimal = unblocked tadpole from `M`; tree + one pair) | correct and complete. (∗) and both directions of (a) checked step by step. Also tested: all 6, 32, 32, 414 minimal configurations of `4^4`, `5^4`, `6 5^3`, `5^5` are tadpoles from `M`, and every cyclic configuration of these diagrams that is a tadpole from `M` is minimal |
| Lemma 2.4 (mover = chord to `+∞`) | correct and complete |
| Lemma 2.5 (replacement) | correct and complete (chordlessness is not even needed for the isomorphism). Tested on 29 776 chordless cycles |
| Cor. 2.6 | correct |
| **Lemma 2.7 (shortening)** | **correct and complete** — see §2 below |
| **Theorem 2.8 (pattern theorem)** | **correct and complete** — see §2 below |
| Cor. 2.9 | correct (trivial from 2.8) |
| **Cor. 2.10** (mult. ≤ 3: T or B; minimal configurations are 3-pair T/B) | correct and complete; tested on 133 cyclic words with multiplicities ≤ 3 (all their minimal configurations) |
| Remark on "pattern S" of `01_foundations.md` | not checked (outside this file's scope) |
| Prop. 2.11(a) | correct (wording, Issue 8); (b), (c) confirmed by my code (exact counts below) |
| **Lemma 2.12** (two-neighbour property) | correct and complete; also tested on every weight of `A_9ω_5`, `A_11ω_6`, `D_9ω_9`, `D_9ω_1`, `E_6ω_{1,6}`, `E_7ω_7` |
| Cor. 2.13 | correct and complete; tested on the same data (0 disagreements) |
| Lemma 2.14 | correct (standard facts on minuscule orbits are correctly stated) |
| Lemma 3.1(a), (b) | correct and complete (the box counts `min(r,s)−1`, `min(r−1,s)`, `min(r,s−1)` are right and stay inside `R`) |
| Lemma 3.1(c) | correct; cited (Issue 2, with a self-contained proof) |
| Lemma 3.2 | correct and complete |
| **Theorem A** | **correct and complete**, both directions, every `λ ⊆ k×(n−k)`, every `n` (for all stacks of a weight: modulo 3.1(c)). `S`-form: Issue 4 |
| Symmetry remark ((4,4,4,3) self-conjugate) | correct |
| Prop. 3.3(a) | correct and complete. In the four listed examples the configuration is in fact the **only** cyclic configuration (brute force) |
| Prop. 3.3(b), census | correct; census confirmed exactly: `4^4`: 6 minimal; `5^4`: 32 (cycles 4, 6; tails 0–3); `5^5`: 414 (cycles 4, 6, 8; tails 0–4) |
| Cor. 3.4 (a), (b) | correct and complete; (c) correct modulo 3.1(c) and weight order (Issue 2) |
| **Prop. 4.1**, Lemma 4.0 | correct and complete |
| Lemma 4.2(a), (b) | correct and complete (I re-did every count); (c) cited (Issue 2). The labelling is right (§5.3) |
| Lemma 4.3 | correct and complete |
| **Theorem D** | **correct and complete**, both directions, every strict partition, every `n ≥ 4`. Wording: Issue 3 |
| Cor. 4.4 | (a), (c) correct; (b) correct, and the `ε`-coordinates follow from the sign-vector description of §5.3 (no need for "computed") |
| **Theorem E** | confirmed by an independent exhaustive computation: brute force over all configurations, connected-growth search and pattern detector (§5.5) |
| **Theorem U** | correct for A and D, modulo the weight-order input (Issue 2, which is elementary in the models); E by computation (confirmed) |
| Cor. 6.1 | correct |
| Prop. 7.1 | correct and complete given Billey–Warrington Thm. 1, whose quote I checked against the arXiv text (§4) |
| Lemma 7.2 | correct (citation for positivity: Issue 6) |
| Prop. 7.3 | correct, given the Hecke computation, which I reproduced with my own code (13-slot `D_6` element is not Deodhar) |
| §7 computed claims (`D_6`, `D_7`, `E_6`, `E_7` Deodhar vs cyclic) | confirmed by my own code (§5.6) |
| §5 "max_pairs = 3 already exhaustive" | correct: `find_cycle` grows all connected unblocked configurations, and `M — q — p — q'` stays connected when grown in the order `q, p, q'` |

---

## 2. The two load-bearing proofs, line by line

### Lemma 2.7

The setting is a chordless cycle `C = (v_1,…,v_L)`, `L ≥ 6`, in `G(F)` with `F` unblocked. The
turning point is `z = v_i` with `c(x) = c(y) = d`, `x = v_{i−1}`, `y = v_{i+1}`; then
`u' = v_{i−2}` and `w' = v_{i+2}`.

1. *`s`, `t` well defined.* `u'` is linked to `x`, so by Lemma 1.1 exactly one slot of `x` lies in
   `u'`. Hence exactly one slot of `x` has `χ_{u'} = 1 − κ`, whatever `κ ∈ {0,1}` is. The same
   holds for `t`. ✓
2. *`u'` is not interlaced with `y`.* Their indices differ by 3, and `L ≥ 6` makes the other
   distance `L − 3 ≥ 3`, so they are not consecutive on `C`. As `C` is chordless they are not
   linked, and because `c(u') ∼ d = c(y)` they are not interlaced. So `χ_{u'}(a_y) = χ_{u'}(b_y) = κ`,
   hence `χ_{u'}(s) + χ_{u'}(t) = 1`, and `u'`, `x*` are interlaced and linked. The four slots are
   distinct because `F` is a family. ✓ The argument for `w'` is symmetric. ✓
3. *No block.* Only `x*` is new, and a block needs a pair `r ∈ C∖{x,y,z}` of colour `d`.
   * `r` is not interlaced with `x` or `y`: equal colour, and `F` is unblocked. So `χ_r` is
     constant on the slots of `x` and on those of `y`.
   * `r` is not a `C`-neighbour of `z`. By chordlessness and `c(r) ∼ c(z)`, `χ_r` is constant on
     the slots of `z`.
   * If `ρ_x = 1`, the slot of `z` inside `x` is inside `r`, so both slots of `z` are inside `r`.
     If moreover `ρ_y = 0`, then `y` and `r` are either disjoint intervals or `y ⊃ r`. I checked
     that these are the only two possibilities when both slots of `y` lie outside `r` and the pairs
     are not interlaced. In both cases `z` is not interlaced with `y`: a contradiction. ✓
   * The same argument with `x` and `y` exchanged gives the converse. So `χ_r(s) + χ_r(t) ∈ {0, 2}`,
     and `r` is not interlaced with `x*`. ✓
4. *Cycle.* The vertices `v_{i+2}, …, v_{i−2}` are `L − 3` consecutive vertices of `C`, and `x*`
   is linked to both ends. ✓ (`G(F')` may contain further edges; this is irrelevant, because the
   theorem uses only "contains a cycle".)

The shortened family really is a family: `x*` uses one slot of `x` and one of `y`, both disjoint
from the remaining pairs. It is unblocked by step 3. It need not be connected and contains no
mover, and neither is needed. The connection to a mover is made only at the very end, by
Lemma 2.5. The proof is complete. **Independent test** (`R03_lemma27.py`): random unblocked
families in random words (types `A_3…A_6`, `D_4`, `D_5`, `E_6`) and in words of 5×5, 6×6 and 7×7
boxes and their subwords. Every chordless cycle of length 6–12 was enumerated, and the surgery was
applied literally at every turning point, checking every conclusion. Over 1 672 cycles (lengths 6:
1 259, 8: 303, 10: 84, 12: 26) there were 3 970 surgeries and **0 failures**.

### Theorem 2.8

* (i) ⇒ (ii):
  * Lemma 2.4 turns the mover into the chord `(j, m+1)` of `w⁺`. Lemma 2.7, iterated on a
    shortest (hence chordless) cycle, strictly decreases the even cycle length down to 4.
  * Lemma 2.5 then turns the 4-cycle back into a configuration whose mover is `a_p`, where `p` is
    the pair with the largest right slot.
  * If slot `m+1` is used at all, it is `b_p`. So the mover and the three remaining pairs lie in
    `w`, and the relations there are the same. ✓
* (ii) ⇒ (iii): a bipartite graph on 4 vertices with at least 4 edges is `K_{2,2}`. The turning
  point gives `c(j) = c(p)` or `c(q) = c(q')`. The cases are exhaustive, and in each the slot
  inequalities follow exactly as written. In the case `c(q) = c(q')` the pairs `q`, `q'` both
  contain `j`, have equal colour and are not interlaced, so they are nested. ✓
* (iii) ⇒ (ii): I checked every relation of the attached configurations, including the
  non-adjacency of two distinct neighbours of a node (forest, no triangles). ✓

The conditions needed for "contains" are all in §1.5: distinct colours make the slots of
T and B automatically distinct, and `j ∉ {a, b}` is stated for N. No further disjointness or
membership condition is needed. **Independent test** (`R03_test_patterns.py`,
`R03_test_patterns_extra.py`), every word compared by three methods:
* brute force over all configurations, with no bound on `|F|`;
* exhaustive growth of connected unblocked configurations (this does not use Lemmas 2.3–2.8);
* my T/B/N detector, written from §1.5.

In addition I checked that **every** cyclic 3-pair configuration (20 207 of them) is attached to a
pattern occurrence and that every attached configuration is cyclic. Results: §5.1.

---

## 3. Numbered issues

**1 (cosmetic; matters for formalisation). Quantifiers in §1.5.** The preamble reads "In the
following, `c' ≠ c''` are two distinct neighbours of a colour `c`, and `e ∼ d ∼ c` (possibly
`e = c`)". This should be part of each definition, for example:
* "*`w` contains T* if there exist a colour `c`, two distinct neighbours `c', c''` of `c`, slots
  `x_1 < x_2 < x_3` of colour `c` and, for each `γ ∈ {c', c''}`, a pair `(y_γ, z_γ)` of colour `γ`
  with …";
* likewise "there exist colours `d ∼ c`, `e ∼ d` …" for N.

With this reading (the one my detector implements) Theorem 2.8 is literally true on all tested
words.

**2 (repairable; citation that can be eliminated). Lemmas 3.1(c), 4.2(c) and the weight order.**
Cor. 3.4(c), Cor. 4.4(b) and Theorem U for types A and D use three facts:
* every chain word `ω → μ` is a linear extension of the model ideal of `μ`;
* ideals correspond to weights;
* the weight order is reverse inclusion.

The draft cites [Proctor 1984] and [Stembridge 2001] without theorem numbers. The citations are
appropriate:
* Stembridge proves that `λ`-minuscule elements are fully commutative;
* Proctor shows that `W^J` is the distributive lattice of ideals of the minuscule poset.

But the statement actually used must be assembled from them: that all chain words to `μ` give the
**same** element needs the minimal-coset-representative argument. The draft is honest that this is
cited, and it checks the cases computed. Two simple repairs remove the dependence:

* *Weight order = reverse inclusion, in the models, is elementary.* In type A the boxes of colour
  `k+d` in a diagram form an initial segment of the diagonal `D_d`, so `λ ⊇ λ'` iff
  `|λ ∩ D_d| ≥ |λ' ∩ D_d|` for all `d`. These numbers are the root coordinates of `ω_k − μ`. In
  type D, colour `n − r` (`r ≥ 2`) is the diagonal `v − u = r`, again an initial segment. The
  diagonal `v − u = 1` is split by parity of `u` into colours `n` and `n−1`. Its initial segment
  of length `t` has `⌈t/2⌉` boxes of colour `n` and `⌊t/2⌋` of colour `n−1`, and so is determined
  by their sum. In both models the ideal is therefore determined by, and monotone in, the colour
  content, which is the dominance order.
* *Lemma 3.1(c) by induction on the word.* Let `u` be a chain word that is a linear extension of
  `λ`, and let `c = k+d`, `a_e = |λ ∩ D_e|`. Then `⟨μ_λ, α_c^∨⟩ = δ_{d,0} + a_{d−1} + a_{d+1} − 2a_d`.
  * For a Young diagram this equals `ε_− − ε_+` with `ε_± ∈ {0,1}`. For `d ≥ 1` the quantities are
    `ε_+ = a_d − a_{d+1}` and `ε_− = a_{d−1} − a_d`, with the obvious shifts for `d ≤ 0`.
  * It equals 1 iff the next box `y` of `D_d` is addable. Its left neighbour is the
    `(a_d+1)`-st box of `D_{d−1}` and its upper neighbour the `a_d`-th box of `D_{d+1}`. When
    `ε_− = 1`, `y` lies in `R`.
  * Appending `c` therefore means adding `y`. In `λ ∪ {y}` the box `y` is maximal and comparable
    with every box of equal or adjacent colour, so the heap of `u·c` is `λ ∪ {y}` with the
    componentwise order.
  * Induction then gives that every chain word is a linear extension of a diagram, and that the
    diagram is determined by the weight (first bullet).

  The same argument works in type D with the sign-vector description of §5.3. My `R03_typeA.py`,
  `R03_typeD.py` and `R03_typeE.py` verify all three facts directly for `A_11 ω_6` (924 weights),
  `D_4…D_9` and `E_6`, `E_7`. They do so by a DP over **all** chain words, without counting:
  every weight has exactly one commutation class of chain words.

**3 (cosmetic). Proof of Theorem D, "⇒".** "The boxes strictly between `x_1` and `x_2` are
`(1, 2+r)` … and `(2, 1+r)`" is false as written: in the row-major word, the rest of row 1 and the
start of row 2 lie between them. It should read "the boxes **of colours adjacent to `c`** lying
between `x_1` and `x_2` (in the heap, equivalently in any linear extension)". The computation
that follows is right: `(1, 2+r)` is the only box of `D_{r+1}` and `(2, 1+r)` the only box of
`D_{r−1}` in the interval.

**4 (cosmetic). Theorem A, `S`-form.** "`s_{k−2} ≥ k+2` and `s_{k−3} ≥ k`" is undefined for
`k ≤ 3`. Either restrict to `k ≥ 4` or set `s_i := i` for `i ≤ 0`. Both give "not cyclic" for
`k ≤ 3`, as Cor. 3.4(a) says.

**5 (cosmetic). Over-claim of independence (§5, "What the computer proof assumes (1)").** The
definitions in `03_lib` are said to be "checked against an independent re-implementation, the
pattern detector". Both are by the same author, from the same reading. They are now confirmed by
genuinely independent code: `R03_compare_author.py` compares on 3 000 random words, 462 cyclic,
with 0 disagreements, and all of §5.

**6 (cosmetic). Lemma 7.2, positivity.** "Positivity (Kazhdan–Lusztig; for Weyl groups via
intersection cohomology)" is correct for Weyl groups but imprecise as a citation. It is the
non-negativity of the structure constants of the KL basis, from the geometric interpretation
(Springer, Lusztig; for all Coxeter groups, Elias–Williamson 2014). The proof needs less: write
`Π(aa') = Π(a)·C'_{s'_1}⋯C'_{s'_r}` and multiply by one `C'_s` at a time. That uses only
`C'_y C'_s = C'_{ys} + Σ μ(z,y) C'_z` or `(v+v^{-1}) C'_y`, i.e. `μ ≥ 0`, which follows from
non-negativity of KL polynomials. The argument is correct as it stands. The same holds for the
`H_w`-coefficient: a product `H_x H_y` with `x ≤ u`, `y ≤ u'`, not both maximal, only involves
elements of length `< ℓ(w)`.

**7 (cosmetic). §8 item 1.** The "brief's claim that `(4,4,4,3)` is not transposition-symmetric"
is not in `README_FOR_PROVERS.md`, nor in any `.md` file of the project (`grep`); it presumably
comes from the draft author's own task prompt. The mathematical content of the correction is
right: `(4,4,4,3)' = (4,4,4,3)`, and transposition with `c ↦ n − c` is a symmetry of the problem.

**8 (cosmetic). Prop. 2.11(a), proof.** "So only T, B, N could occur" is a non sequitur: by
Theorem 2.8 the question is always T, B or N. The case analysis that follows is complete and right.
(I re-checked it: for `d = 1`, the only choice is `y_1,y_2,z_2,z_1 = 1,3,5,7`, `j = 4`, and
`p ⊆ {2,6}` has no slot in `(3,5)`.)

**9 (suggestion).** Prop. 3.3(a) could say more: by brute force, in `(4,4,4,3)`, `(5,4,4,3)`,
`(4,4,4,3,1)` and `(5,5,4,3)` the minimal configuration is the only cyclic configuration at all.

No issue is **fatal**. No statement is **wrong**.

---

## 4. Inputs from the literature

* **Billey–Warrington**, *Kazhdan–Lusztig polynomials for 321-hexagon-avoiding permutations*,
  arXiv:math/0005052. I extracted the text of v1. The quote of **Theorem 1** in §7 is accurate in
  substance:
  * the six conditions, for `w ∈ S_n` and a fixed reduced expression `a`: 321-hexagon-avoiding;
    the mask/defect formula for `P_{x,w}`; the Poincaré polynomial `(1+q)^{l(w)}`;
    `C'_w = C'_{s_{i_1}}⋯C'_{s_{i_r}}`; the Bott–Samelson resolution is small;
    `IH^*(X_w) ≅ H^*(Y)`;
  * the four hexagon patterns, their (7): `[4,6,7,1,8,2,3,5], [4,6,7,8,1,2,3,5], [5,6,7,1,8,2,3,4],
    [5,6,7,8,1,2,3,4]`;
  * Remark 2 ("tight", Lusztig, Fan–Green).

  BW work with `T_w` and `q^{1/2}`. Condition 4 does not depend on the normalisation. The
  pattern analysis in the proof of Prop. 7.1 is correct: one descent, the inverse-closure of the
  pattern set, and the translation into `λ_3 ≥ 4`, `λ_4 ≥ 3`.
* **Billey–Jones**, arXiv:math/0612043. The abstract confirms that they classify the `w` with
  `C'_w` a monomial in the `C'_s` ("Deodhar elements"), extending BW and Fan–Green. The draft does
  not rely on their Thm. 2.3, 5.12 or Fig. 1 for any proof. It says so, and it says correctly that
  the converse of Prop. 7.3 for all `n` would need their type-D list. Not re-checked by me.
* **Proctor 1984** (Eur. J. Combin. 5) and **Stembridge 2001** (J. Algebra 235). These are
  appropriate for the facts used (see Issue 2): Stembridge's paper on `λ`-minuscule elements shows
  that they are fully commutative and develops Peterson's theory; Proctor 1984 is the Bruhat
  lattice paper. No specific theorem numbers are given. Issue 2 gives proofs that make them
  unnecessary in types A and D.
* **KL positivity** (Lemma 7.2): correct for Weyl groups (Issue 6).

---

## 5. Independent checks (referee's code, plain `python3`)

All definitions are in `R03_core.py`:
* `analyse`: a literal transcription of the definition.
* `Tables`: a table-driven version, cross-checked against `analyse` on 3 000 random
  configurations with 0 mismatches.
* `brute`: every mover and every family, i.e. the product over colours of all partial matchings
  avoiding the mover, with no bound.
* `conn_search_fast`: all connected unblocked configurations, grown one pair at a time with a
  seen-set. It is exhaustive because a BFS order from the mover keeps every prefix connected, and
  sub-configurations of unblocked ones are unblocked. It does not use the draft's lemmas.
* `patterns`: T/B/N from the text of §1.5.
* `attached_type`: decides whether a given 3-pair configuration is attached to T, B or N.
* `all_weights_with_heaps`: BFS over weights with the set of heaps of **all** chain words.

### 5.1 Theorem 2.8 (`R03_test_patterns.py`, `R03_test_patterns_extra.py`)

| family | words | cyclic | T / B / N | disagreements (brute vs pattern vs connected) | cyclic 3-pair configurations, all attached |
|---|---|---|---|---|---|
| `A_2`, all words, length ≤ 8 | 510 | 36 | 0/0/36 | 0 | 68 |
| `A_2`, all words, length 9 | 512 | 144 | 0/0/144 | 0 | 576 |
| `A_3`, all words, length ≤ 8 | 9 840 | 360 | 64/64/244 | 0 | 600 |
| `A_3`, all words, length 9 | 19 683 | 2 614 | 512/512/1 838 | 0 | 7 776 |
| `A_4`, all words, length ≤ 7 | 21 844 | 44 | 8/8/28 | 0 | 44 |
| `A_4`, all words, length 8 | 65 536 | 960 | 184/184/616 | 0 | 1 408 |
| `D_4`, all words, length ≤ 7 | 21 844 | 60 | 12/12/36 | 0 | 60 |
| `D_4`, all words, length 8 | 65 536 | 1 260 | 276/276/744 | 0 | 1 920 |
| forest `A_2 ⊔ A_1`, all words, length ≤ 8 | 9 840 | 68 | 0/0/68 | 0 | 100 |
| random, length 11–12, `A_2…A_5`, `D_4`, `E_6` (uniform and biased) | 1 320 | 452 | — | 0 | 7 632 |
| random subwords (length 12–14) of the 6×6 stack | 300 | 10 | 4/4/4 | 0 | 23 |

Totals: 216 765 words, 6 008 cyclic, 0 disagreements; 20 207 cyclic 3-pair configurations, every
one attached to a pattern occurrence; every attached configuration cyclic. **Cor. 2.10**: on the
133 cyclic words with multiplicities ≤ 3, every inclusion-minimal cyclic configuration is a 3-pair
T or B configuration. Prop. 2.11(a) (`1212121`): not cyclic; cyclic ignoring blocks (mover 4, with
two blocks). README example `1221122`: cyclic, pattern N.

### 5.2 Type A (`R03_typeA.py`, `R03_misc.py`, `R03_census55.py`)

Model: `k = 6`, `n = 12`, box `(r,s)` has colour `k + s − r`, the word is row-major.
* All **924** diagrams of the 6×6 box. Each row-major word is a chain word. Exhaustive search:
  **362 cyclic, exactly those `⊇ (4,4,4,3)`**, with 0 disagreements against the criterion, the
  pattern detector, and the column-major linear extension. Lemma 3.2 holds.
* The census of the draft is exact: all 362 contain T, 262 contain B, 162 contain N.
* `A_11 ω_6`: each of the 924 weights has exactly one commutation class of chain words, equal to
  the heap of its row-major diagram word. The diagrams hit every weight exactly once. Dominance
  order = reverse inclusion (850 k pairs, 0 mismatches).
* 7×7 box (`A_13 ω_7`, 3 432 diagrams, run `R03_typeA.main(7)`, log `typeA7.log`): **2 116
  cyclic, exactly those `⊇ (4,4,4,3)`**, 0 disagreements against the criterion, the patterns
  and the column-major word, Lemma 3.2 holds (205 s). This agrees with the draft's 2 116 / 1 316.
* Prop. 3.3(a): `(4,4,4,3)`, `(5,4,4,3)`, `(4,4,4,3,1)`, `(5,5,4,3)` each have exactly **one**
  cyclic configuration, the stated one.
* Minimal configurations, brute force over all configurations:
  * `4^4`: 6 (2 with tail 0, 4 with tail 1);
  * `5^4`: 32 (cycle length 4 with tails 0–3; length 6 with tails 0–1);
  * `5^5`: **414** (cycle lengths 4, 6, 8; tails 0–4).

  All are tadpoles from `M`. The 6-cycle example of §3.3 is a chordless 6-cycle through the
  mover, unblocked.
* Prop. 2.11(b): the 3×5 rectangle (word `321432543654765`) is not cyclic, but is cyclic
  ignoring blocks (mover slot 8 inside `(4,12)`). Prop. 2.11(c): `sl_10 ω_5` has 42 cyclic ideals
  and 54 cyclic ignoring blocks.
* Cor. 3.4 counts: 2, 9, 25, 42.

### 5.3 Type D (`R03_typeD.py`)

**Labelling derived independently.** A weight of `V(ω_n)` is `½(σ_1,…,σ_n)` with an even number
of minus signs. Then `⟨μ, α_i^∨⟩ = 1` iff `(σ_i, σ_{i+1}) = (+,−)` (`i < n`), resp.
`(σ_{n−1}, σ_n) = (+,+)` (`i = n`). In the full chain, row `u` brings the `u`-th minus sign to
position `u`:
* for odd `u` it is created by `s_n` (together with a partner at `n`);
* for even `u` the partner is moved from `n` by `s_{n−1}`;
* then `s_{n−2}, …, s_u` move it left.

So box `(u,v)` has colour `n − (v−u)` for `v − u ≥ 2`, `n` for `v = u+1` with `u` odd, and `n−1`
for `v = u+1` with `u` even. This is the draft's labelling. The ideal `I_ℓ` has minus signs at
`{n − ℓ_u}`, plus `n` if `ℓ` has an odd number of parts. This gives Cor. 4.4(b) by hand: for
`(5,4,3,1)` the signs are `½(+,…,+,−,−,−,+,−,+)`.

Cross-checks:
* The full row-major word equals `words.json` (`cupbrane.minuscule_word` cache) for `D_4…D_8`.
* For `D_4…D_9` every weight has exactly one commutation class of chain words, equal to the heap
  of the model word. The model words hit all `2^{n−1}` weights bijectively, and dominance order
  is reverse inclusion.

**Theorem D**, all strict partitions with parts ≤ `n−1`, `n = 4…9` (parts ≤ 8 for `D_9`):
* exhaustive search ⇔ `ℓ_3 ≥ 3 ∧ ℓ_4 ≥ 1`, with 0 disagreements, and the pattern detector agrees;
* for `n ≤ 8` the brute force over all configurations finds nothing in any non-cyclic ideal;
* non-cyclic ideals have multiplicities ≤ 3;
* cyclic counts are 0, 0, 3, 16, 54, 148, with a unique minimal ideal `(5,4,3,1)` (13 slots).

`μ_c(D_6) = −ω_3 + ω_4 − ω_5`, `ε`-signs `(−,−,−,+,−,+)`, content `(1,2,3,3,2,2)`. The criterion
gives 361, 820, 1780 for `D_10…D_12`, as the draft says. Vector representation `D_4…D_9`: maximal
multiplicity 2, nothing cyclic. `D_4 ω_3, ω_4`: multiplicity ≤ 2. Prop. 2.11(c) for `D_6`: the
ideals that are cyclic ignoring blocks but not cyclic are `(4,3,2,1)`, `(5,3,2,1)`, `(5,4,2,1)`,
`(5,4,3)` (10, 11, 12, 12 slots).

### 5.4 Lemma 2.12, Cor. 2.13, Lemma 2.14 (`R03_misc.py`)

On every weight of `A_9 ω_5`, `A_11 ω_6`, `D_9 ω_9`, `D_9 ω_1`, `E_6 ω_1`, `E_6 ω_6` and
`E_7 ω_7`:
* 0 violations of the two-neighbour property;
* 0 of the multiplicity bound;
* no B occurrence with `x_1` at a first occurrence;
* the Cor. 2.13 criterion agrees with exhaustive search.

### 5.5 Type E (`R03_typeE.py`)

Weights are generated by BFS from `ω_k`, with a DP over **all** chain words; each weight has one
commutation class. Brute force over all configurations, connected search and patterns agree on
every ideal.

| case | cyclic / weights | minimal cyclic ideal | `μ_c` (fund.) | `ω − μ_c` | non-cyclic max mult. |
|---|---|---|---|---|---|
| `E_6 ω_1` | **3/27** | unique, 14 slots, `13425431654234` | `(0,0,0,−1,1,0)` | `(2,2,3,4,2,1)` | 3 |
| `E_6 ω_6` | **3/27** | unique, 14 slots, `65423143542654` | `(0,0,1,−1,0,0)` | `(1,2,2,4,3,2)` | 3 |
| `E_7 ω_7` | **22/56** | unique, 15 slots, `765423143542654` | `(0,0,1,−1,0,0,1)` | `(1,2,2,4,3,2,1)` | 3 |

In each case the cyclic set is exactly the up-set (chain reachability) of the minimal ideal, and
reachability equals dominance order. Each `I_c` has 5 cyclic configurations, of which exactly 2
are inclusion-minimal. They are exactly the B and N configurations in the table of Theorem E(d),
slot for slot, and there is no T. The 13-slot `E_6` ideal `1342543165423` is **not** cyclic, by
brute force over all its configurations. The ideals of `E_6 ω_1` that are cyclic ignoring blocks
but not cyclic have 11, 12 and 13 slots.

### 5.6 Deodhar property (`R03_deodhar.py`, own Hecke algebra)

The code uses Soergel's normalisation and the Weyl group acting on simple roots. It tests whether
every non-leading coefficient of `C'_{s_1}⋯C'_{s_l}` lies in `vℤ[v]`.

| case | non-Deodhar ideals | cyclic ideals | differ |
|---|---|---|---|
| `A_7 ω_4` (sanity check against BW) | 15, 16 slots | 15, 16 | none |
| `A_7 ω_3`, `D_6 ω_1` | none | none | none |
| `D_6` half-spin | 13, 14, 15 | 13, 14, 15 | none: the 13-slot element is **non-Deodhar** (the input of Prop. 7.3) |
| `D_7` half-spin | 16 ideals | the same 16 | none |
| `E_6 ω_1` | 13, 14, 15, 16 | 14, 15, 16 | the 13-slot ideal only |
| `E_7 ω_7` | 25 ideals | 22 | 3 ideals, of 14, 15, 16 slots |

### 5.7 Lemma 2.7 (`R03_lemma27.py`) and the comparison with the author (`R03_compare_author.py`)

* Lemma 2.7: see §2 (3 970 surgeries, 0 failures). Lemma 2.5: tested on 29 776 chordless cycles,
  0 failures.
* My exhaustive search and pattern detector agree with the author's `Word.search` and
  `03_patterns.patterns` on 3 000 random words (462 cyclic).

---

### 5.8 Scripts (run from `proofs/checks/`)

| script | content | runtime here |
|---|---|---|
| `R03_core.py` | definitions, brute force, connected search, pattern detector, chain words, heaps | — |
| `R03_test_patterns.py` | Theorem 2.8, Cor. 2.10: all words of length ≤ 8 (`A_2`, `A_3`), ≤ 7 (`A_4`, `D_4`), random words of length 11–12 | ~3 min |
| `R03_test_patterns_extra.py` | length 9 (`A_2`, `A_3`), length 8 (`A_4`, `D_4`), forest, 6×6 subwords | ~4 min |
| `R03_lemma27.py N [grid]` | Lemmas 2.5, 2.7 on random unblocked families | < 1 min |
| `R03_typeA.py [B]` | Theorem A and the model in the `B×B` box; Prop. 3.3; Prop. 2.11(b),(c); Cor. 3.4 | 16 s (B = 6), ~4 min (B = 7) |
| `R03_typeD.py [N]` | Theorem D, Lemma 4.3, Cor. 4.4, vector representation, `D_6` ignoring blocks | 2 s |
| `R03_typeE.py` | Theorem E | 1 min |
| `R03_deodhar.py [E7]` | Deodhar test in the Hecke algebra | 20 s (+ 2 min `E_7`) |
| `R03_misc.py` | tadpoles (Lemma 2.3(a)), 6-cycle example, T/B/N census, Lemmas 2.12–2.14 | 10 s |
| `R03_census55.py` | census of minimal configurations of `5^5` | 16 s |
| `R03_compare_author.py` | output comparison with `03_lib` / `03_patterns` | < 1 min |

## 6. The proposed corrections to `NOTES.md` §4 (draft §8)

1. **Range `4 ≤ k ≤ n−4`** (first table, row "`sl_n, ω_k`, `4 ≤ k ≤ n−3`"): **the correction is
   right.** For `sl_8` the row would include `ω_5`. Its poset is the 5×3 rectangle, which has only
   3 columns, so `λ_3 ≤ 3` and it is not cyclic (Cor. 3.4(a); confirmed in the 6×6 run, which
   contains all `λ ⊆ 5×3`). The second table of `NOTES.md` already has `4 ≤ k ≤ n−4`.
2. **"The minimal cyclic configuration always has the same shape" is false in general: the
   correction is right.** At the `E_6` and `E_7` thresholds `I_c` has two inclusion-minimal
   cyclic configurations:
   * one of shape B, which is the "(or below `x_1`) … (or between `x_2` and `x_3`)" alternative of
     `NOTES.md`, with three consecutive occurrences of colour 4;
   * one of shape **N**, which does not have that shape (§5.5).

   Further points in the correction are also confirmed:
   * mixing the two alternatives never gives a cyclic configuration, so the alternatives are
     coupled: with mover `x_3` and pair `(x_1, x_2)`, a pair ending in `(x_2, x_3)` is not linked
     to the mover; with mover `x_1`, a pair starting in `(x_1, x_2)` is not linked to it; and
     mover `x_2` with pair `(x_1, x_3)` is blocked;
   * in larger ideals the minimal configurations are tadpoles with cycles of length 4, 6, 8 and
     the mover off the cycle (`5^4`, `5^5`, §5.2);
   * pattern N occurs in arbitrary words, e.g. `1221122`.

   In types A and D at the threshold the description of `NOTES.md` is correct: pattern T, unique.
3. **Pattern N**: confirmed as above. It is genuinely needed: 162 of the 362 cyclic 6×6 diagrams
   contain N, though all of them also contain T. In `A_2`, where T and B are impossible, every
   cyclic word is N.

The other items of §8 are also right:
* item 4: with `max_pairs ≥ 3` the old searches were exhaustive;
* item 5: proved for all `n`;
* item 6: the 13-slot `E_6` ideal is not cyclic;
* item 7: inheritance by super-words;
* item 8: multiplicities ≤ 3 in non-cyclic minuscule stacks of types A and D.

Item 1 is right in content (Issue 7). Item 9: I also found no error in `PROOFS.md` Prop. 8 or
Cor. 3.1.

## 7. Recommendation

Accept after cosmetic revision. Make the quantifiers of §1.5 explicit (Issue 1) and fix the
wording in Theorem D's proof (Issue 3). Optionally, replace the Proctor/Stembridge citations in
types A and D by the short arguments of Issue 2. Then Theorems A, D and U in types A and D become
proved from the definitions alone. That makes them ready for formalisation: they need only finite
combinatorics plus the chain-word computation of Lemmas 3.1(b) and 4.2(b).
