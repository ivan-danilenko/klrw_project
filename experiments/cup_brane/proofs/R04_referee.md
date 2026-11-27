# R04 — Referee report on `04_first_order.md`

*(The resume notes written under the usage-limit warning are closed: everything they listed as
remaining is done and incorporated below.)*

Referee scripts:

* `proofs/checks/R04_comb.py` (pure Python);
* `R04_alg.py`, `R04_gauge.py`, `R04_cyc7.py`, `R04_cyc7b.py`, `R04_gauge_h.py`
  (`sage -python`);
* logs in `proofs/checks/R04_logs/`, data in `proofs/checks/R04_data/`.

None of the author's `04_*.py` or `04_lib.py` is imported. My scripts have their own Dynkin
diagrams, their own grading group (PROOFS F2, coordinates `q, σ_c, δ_c, ε_{cc'}, ε_{cW}`), their own enumeration of homogeneous
elements (a direct coefficient comparison, not `degree_model.py` and not Prop. 7.1), their own
strand simulation and their own reduced words. The reduced words come from a bubble sort,
*not* the klrw insertion sort. From the project I use only `formula.Stack` (the tree formula and its
signs), `checks/alg.py` (products in `klrw`), and, for Theorem 9.1 only, `fast.braid_for`,
because the theorem is stated for the klrw basis braid.

---

## 1. Summary verdict

| statement | verdict |
|---|---|
| Lemma 1.1 (working modulo `(r)^n`) | correct and complete ((b) uses F1) |
| Lemma 2.1, Lemma 2.2 (normal form mod `r`) | correct and complete (uses F3 and Tits' word property F3′, both declared) |
| Theorem A | correct and complete |
| Lemmas 4.1–4.6, Theorem B | correct and complete; all signs right. The "Observation (Tier 2, not used)" is in fact provable in two lines (§3, item I1) |
| Cor. B.2 | correct and complete |
| Cor. B.3 | correct and complete (uses F1, declared) |
| Lemma 5.1 | correct and complete |
| Lemma 5.2, Prop. 5.3, Cor. 5.4 | correct; case list complete; the proof of (A1)/(B1) is terse (gap G5, cosmetic) |
| Theorem C | correct and complete, including the four-term cancellation of the dotted terms and every sign |
| Remark 6.1 | correct (the "groups of four" is the explicit identity in the last bullet of the proof) |
| Lemma 7.0, Prop. 7.1 | correct and complete |
| Cor. 7.2 | correct; two cosmetic defects (G6) |
| Prop. 7.4 | correct and complete |
| Theorem 8.1 | correct and complete, but F1 is used and not declared (G3); the §0 table row overstates (c) (G4) |
| Example 8.3 | correct (re-derived from scratch). It does **not** contradict NOTES §3 as NOTES states it; it does refute the arbitrary-word reading of PROOFS C2 (G1) |
| Theorem 8.4, Lemma 8.5 | correct and complete (F1 undeclared, G3) |
| Comp. 8.5, Remarks 8.6, 8.7 | consistent with my spot checks; Remark 8.6 "excludes exactly" is unproved wording (G7) |
| Theorem 9.1 | (a), (b), (d) reproduced exactly; "no dot-free completion with `ε ∈ (r)³`" is correct; **"no dot-free completion with `ε ∈ (r)²`" is WRONG** (G0). A dot-free, monotone differential `d_F = d_T + κ* + ε₄` exists, and it is **gauge-equivalent** to the dotted one: `d_F = (1+h)(d_T+X)(1−h)` with `h` dot-free, monotone, of order `r_2²t_21²`. The author's script drops all products `δ₁·κ`. "The true differential" and the basis dependence of "dot-free" need fixing (G2) |
| §10 | formulation only; item 1 simplifies by I1 |

**One fatal error, in the Tier-2 part:** the second half of Theorem 9.1(c), and with it the
headline "the tree formula cannot be completed by dot-free terms" (§0, §9, §11), is false (G0).
It comes from a bug in the author's script. Worse for the narrative of §9 and NOTES §6: the
dotted cyclic terms of this example are a gauge artefact.

Every statement labelled "proved" (Tier 1) in the Status section is proved by the argument
given, under the declared standing facts plus (F1), which is used in some places without being
declared. Beyond G0, the problems are of attribution and wording, plus places where the draft
claims more than it shows (G1, G2).

---

## 2. Line-by-line notes on the proofs

**Lemma 1.1.** (a) is fine. (b): `A = ⊕_b R b` gives `(r)^n A = ⊕_b (r)^n b`. Since `A` is free,
hence flat, `(r)A/(r)²A ≅ (r)/(r)² ⊗_R A ≅ ⊕_c r_c A/(r)A`. Correct.

**Lemma 2.2.** The induction is right. Tits' word property in the form "a non-reduced word can be
brought by braid and commutation moves to a word containing `pp`" (Björner–Brenti Thm 3.3.1(i))
is exactly what is used. Braid moves preserve the multiset of crossing pairs. Deleting `pp`
leaves the strands seen by `w_2` unchanged. On every state (A6) holds modulo `(r)` (the two
exceptional values `r_c t`, `r_k u_k` lie in `(r)`), and dots slide modulo `(r)` (Lemma 2.1(a)).
Correct, including pairs with the red strand (`Q_{uW} = u_k x_u` or `1`).

**Theorem A, Lemmas 4.1–4.6, Theorem B.** I re-derived every state in Lemma 4.3 by hand. The
moves past letters in Lemma 4.4 are legitimate: `ψ_p` against letters `ψ_q` with `q ≥ p+2`, and
`ψ_{p+1}` against `ψ_q` with `q ≤ p−1`. In Lemma 4.5 the step from "`D_s − D_{s−1} = P_s(ψψψ − ψψψ)S_s`"
to "`= C·P_s S_s`" is legitimate because the source and target states of the triple coincide as
colour sequences. (A6) is `C·e(s)` with `s` the source, and the target `(j,i,y)` is the same colour
sequence since `c(y) = c(j)`. So `S_s` is literally the same element read from either state. The
case `y = W` is exact, because the state `(W, c(i), c(j))` has unequal outer strands. The
sign `ε(I;i,j) = (−1)^{n_I(i)+n_I(j)}` is right. Theorem B is exact as stated.

**Cor. B.3.** Correct. Note that it proves the converse of Corollary C.3 of `01_foundations.md`
("bad triple ⇒ the two products differ"), which `01` lists as verified only (37/37 cases). It
should be cross-referenced there.

**Prop. 5.3.** I enumerated all products of `δ₀δ₁` and `δ₁δ₀`: `x ∈ I` together with every
configuration at `I∖x` (forcing `x ∉ {j,b}` and `a ∉ I` or `a = x`), and every configuration at
`I` together with every `x ∈ J`. This is exactly the case list (A1)–(A3), (B1)–(B3), so the
list is complete. Every pair of strands was checked:

* the mover crosses everything at most once;
* no strand crosses `W` twice;
* equal-colour double crossings occur exactly in (A3)/(B3) with `c(x) = c(p)`, and then the
  composite vanishes modulo `r`;
* bigons with the red strand never occur.

**Theorem C.** Sorting by target is legitimate: (A1)/(B1) land in `K ⊆ I`, while (A2,3)/(B2,3)
land in `K ∋ a ∉ I`. Within a target, sorting by bijection is legitimate: `(I, K, (a b))`
determines `p` and the unordered pair `{x, j}`, namely the two fixed slots in `I∖K`. I recomputed
`s_A s_B = (−1)^{1+[a<x<b]+[x<b]+[x<a]} = −1` in all three cases, and `s_A s_{A'} = −1`. The
four-term identity is correct, and its dot variables are legitimately identified: `x_a` in (A3)
and in (A'3) is the dot on the same strand (slot `a` at `−a`), placed at the left end of an
element with the same source `e_I`, and modulo `r` dots are strand variables (Lemma 2.1). The
proof needs no (F1) at all, only Lemma 2.2. **Complete.**

**Prop. 7.1 / Prop. 7.4.** I redid the four-case table of `f`. In Prop. 7.4, besides the
`f(z,A)`, `f(z,B)` for `u < z < v` given in the text, the cases `z < u` and `z > v` also give
`0`: this needs monotonicity, which the text uses silently ("`f = 0` unless `u < z < v`"). The
needed facts are `[u ∉ I, v ∈ J] = 0` for `(z, A)`, `z < u`, and `[z ∉ I, z ∈ J] = 0` for `z > v`.
This is fine, but worth one line.

**Theorem 8.1, Lemma 8.5, Theorem 8.4.** The arguments are correct. In Lemma 8.5 the
coefficient of `r t β_τ` comes only from the single (B1) product with `a' = a`, and (A1) products
cannot have bijection `τ` because `a ∉ I`.

---

## 3. Errors, gaps, improvements (numbered; severity in brackets)

**G0 [FATAL for the second half of Theorem 9.1(c) and the headline "the tree formula cannot be
completed by dot-free terms" (§0 table, §9, §11 Status)].**

*The claim* (9.1(c)). "No differential `d_T + ε` with `ε` a combination of dot-free terms of
`r`-degree ≥ 2: neither with `ε ∈ (r)³` …, nor with `ε ∈ (r)²`: … the `r³`-parts of
`[δ₀ + δ₁, κ]` span a 4-dimensional space not containing `(d_T²)₃` (rank 4 → 5)."

*The bug.* `checks/04_cyclic7_dotfree.py`, lines 66–74, assembles `(δ₀+δ₁)κ + κ(δ₀+δ₁)` for the
kernel vectors `κ`. The left products are taken only from predecessors `Ip = I | {x}`
(line 70). That is right for `δ₀`. But a component of `δ₁` that ends in `I` starts at
`I∖{a}∪{j,b}`, never at `I ∪ {x}`. So the test `if I in dd[Ip]` (line 71) is always false for
`dd = d1`, and **every product `δ₁·κ` is dropped**; only `κ·δ₁` is kept.

* With all products, the rank stays 4 → 4 (`R04_cyc7.py c`).
* Dropping `δ₁·κ` in my own code reproduces the author's 4 → 5 (`R04_cyc7.py c2`).

*Exact statement of what holds* for `1221122`, red 2. Throughout: "dot-free" with respect to the
klrw basis (Definition B.10), maps homogeneous of homological degree −1, coefficients in
`ℤ[r, t, u]`.

1. **(true, as in the draft)** There is no differential `d_T + ε` with `ε ∈ (r)³` dot-free.
   Equivalently, the tree formula cannot be completed by dot-free terms **without changing its
   `r`-degree-2 part**. This is (a) + (b): there are no dot-free elements of `r`-degree 3, and
   `(d_T²)₃ ≠ 0`. It also holds with my own reduced words.
2. **(false in the draft)** There **is** a differential `d_T + ε` with `ε ∈ (r)²` dot-free:
   `d_F = d_T + κ* + ε₄`.
   * `κ*`: 8 terms `±r_2²t_21² β_σ`. Their components are `{4,6,7}→{1,3}`, `{4,6,7}→{2,3}`,
     `{1,4,6,7}→{1,2,3}`, `{2,4,6,7}→{1,2,3}`, with `σ = (1 5 4)(2 7)(3 6)`, and
     `{4,5,6,7}→{1,3,5}`, `{4,5,6,7}→{2,3,5}`, `{1,4,5,6,7}→{1,2,3,5}`,
     `{2,4,5,6,7}→{1,2,3,5}`, with `σ = (1 4 5)(2 7)(3 6)`. Signs are in §4.6.
   * `ε₄`: 4 terms in `I₀→J₀`:
     * `+r_1r_2³t_12t_21³ β_{σ'}`, `−r_1r_2³t_12t_21³ β_{σ''}`, `−r_1²r_2²t_12²t_21² β_{σ'''}`
       (exactly three of the author's `X_r` terms);
     * `+r_1²r_2²t_12²t_21² β_σ` with `σ = (1 4 5)(2 7)(3 6)`.
   * None of the 12 extra terms has the bijection of a tree term. So `d_F` contains **all 680
     tree-formula terms with their coefficients**, plus 12 dot-free terms (692 terms).
   * `d_F` is monotone: no strand crosses `W` from left to right, in any term. So `d_F`
     satisfies (H1), (H2) and (M₁).
3. **Verification of 2**, airtight relative to the `klrw` implementation:
   * `d_F` is saved as `checks/R04_data/cyclic7_1221122_W2_dotfree.json`, with the basis
     convention written into the file;
   * `R04_cyc7b.py verify` rebuilds it **from the JSON alone** and checks: `d_F² = 0` exactly
     on the full cube; every component homogeneous of degree `D(I) − D(J)` for **klrw's own**
     grading (`Cube.eq_degree`, independent of my grading code); no dot variables; no `u`;
     `r`-degree ≤ 1 part equal to `δ₀ + δ₁`.
   * The same checks pass for the dotted `d_X = d_T + X` of 9.1(d)
     (`…_dotted.json`, 690 terms).
4. **Gauge equivalence.** The two completions are gauge-equivalent:

       d_F = (1 + h) d_X (1 − h),     h = −r_2²t_21² ( β_{{4,6,7}→{1,2,3}} + β_{{4,5,6,7}→{1,2,3,5}} ),

   * the bijections are `(1 5 4)(2 7)(3 6)` and `(1 4 5)(2 7)(3 6)`;
   * `h` is homogeneous of homological degree 0, dot-free, and monotone (no strand crosses `W`
     at all);
   * `h² = 0`, so `(1+h)^{-1} = 1 − h`;
   * checked exactly in `R04_gauge_h.py`, which reads only the JSON files; `h` is saved as
     `…_gauge_h.json`.
   * The gauge was found by solving `g d_X = d_F g` order by order (`R04_cyc7b.py gauge_full`).
     At orders `r¹, r²` the system for `g = 1 + h₁ + h₂` (290 + 8 candidates, dots allowed)
     has full rank 298. So the `r¹` and `r²` parts of any gauge `g ≡ 1 mod (r)` from `d_X` to
     `d_F` are **forced**; higher parts could only differ by elements commuting with the
     differentials. With `g = 1 + h`, `g d_X − d_F g = 0` already holds exactly.

*Consequences.*

* (d) is correct but says less than the draft suggests. Kernel 0 at the four orders of `X`
  means: `X` is the unique completion *keeping the tree formula's `r²`-part*. The dotted terms
  `X_dot` are **not gauge-invariant**: the monotone, dot-free gauge `h` of order `r_2²t_21²`
  turns them into the 8 hand-over-type `r²`-terms `κ*` (and changes the `r⁴`-part).
* Hence §0 "the true differential = tree formula + 10 terms … exactly the spanning-tree × bigon
  rule" and §9 "What the example adds", points (1)–(3), describe one normal form of a gauge
  class, chosen by the normalisation "tree formula untouched at `r²`". They do not describe an
  invariant.
* For NOTES §6 / PROOFS C4: in this example the dotted cyclic terms are a normal-form
  phenomenon, not forced. Whether this persists in the 32 local pieces of NOTES §6, where the
  solver also keeps the tree formula fixed, is an open question that the maintainers of C4
  should be told about.
* The Status line "Theorem 9.1 … no dot-free completion of the tree formula by terms of
  `r`-degree ≥ 2" must be replaced by 1 and 2 above.
* *What the "Not shown" remark should say:* a dot-free differential `δ₀ + ε`, `ε ∈ (r)`, does
  exist (`d_F`); it even keeps `δ₁`.

**G1 [repairable; attribution and wording].** *"Corrections to other files", item 1 (and Ex. 8.3):
"NOTES §3 / PROOFS Tier 2 … is false for arbitrary words".*

* What NOTES §3 actually claims: kernel 0 "in every case with ≤ 6 strands" of its data (the
  seven minuscule words of `data/`: `A_3` `2132`, `2312`; `A_4` `213243`, `234123`; `D_4`
  `123421`, `321423`, `421324`), at the orders where `d²` fails. `1212` is not a minuscule word
  (for `A_2`, `ω_2` the chains have length ≤ 2). So Example 8.3 does not contradict NOTES §3 as
  stated.
* I checked that it cannot: for each of those seven words there is **no** homogeneous basis
  element of homological degree 0 and `r`-degree ≥ 1 at all (dots allowed;
  `R04_comb.py gauge`). Hence `V^0 = 0` and there is no gauge freedom of this kind in the NOTES
  cases.
* What Example 8.3 does refute:
  * the extrapolation "there is no gauge freedom" to arbitrary words;
  * **PROOFS Conjecture C2** in the arbitrary-word reading that README's C1′ suggests. `1212`
    is cycle-free (every colour twice), and `d'` is a degree-preserving differential of the
    form `δ₀ + (terms divisible by r)`, different from the tree formula.

**Repair:** say "C2 is false for arbitrary cycle-free words (Ex. 8.3); in the minuscule cases of
NOTES §3 there are no gauges at all; for arbitrary words uniqueness can hold at best modulo
gauge or under (M₁)". This is the most consequential point of the draft and should be stated
against C2, not NOTES §3.

*Detail worth adding to Ex. 8.3 (checked).* The gauge **removes** both one-pair terms of the
configuration (mover 3, pair (2,4)): in `{3,4}→{2}` and in `{1,3,4}→{1,2}` the tree formula has
`+r_2t_21 β` and `d' − d = −r_2t_21 β`. Instead `d'` acquires two non-monotone terms
`±r_2t_21 β` in `{3,4}→{1}` and `{2,3,4}→{1,2}`, in which the strand of slot 1 crosses `W` from
left to right. `d'` satisfies `d'² = 0`, homogeneity, (H1), (H2); it is dot-free and `u`-free; it
violates exactly (M₁). This is consistent with Theorem 8.4.

**G2 [repairable; overstatement and precision] Theorem 9.1.**

* (i) §0 and the heading say "the true differential = tree formula + 10 terms". Part (d) shows
  only that `d_T + X` is *a* differential, and that `X` is unique at its four orders *given the
  tree formula at all other orders*. The draft itself reports a 106-dimensional space of
  dot-free first-order cycles for this word, mostly gauge directions (see §4.6 for my count).
  So other differentials exist, and "the" should be "the completion of the tree formula
  keeping its `r²`-part". G0 makes this concrete: a different, dot-free, gauge-equivalent
  completion exists.
* (ii) "Dot-free" in (a), (c) is relative to the klrw basis (Definition B.10). Changing the
  reduced words changes basis elements by `r`-multiples of shorter, possibly dotted, words, and
  §1.2's sentence "all statements below are modulo ideals and therefore independent of these
  choices" is false for §9. The basis must be named in the statement. (For this word my
  recomputation of (b) with my own bubble-sort words gives the same `r`-parts, so nothing changes
  numerically.)
* (iii) The *reasoning* of (c) is right: `ε_3 = 0` by (a), and the `r³`-equation reduces to a
  span condition. The *computation* of that span condition is wrong (G0).

**G3 [cosmetic; undeclared input].** (F1) is used and not declared:

* Theorem 8.1: that `ε₁ ∈ V^{−1}_1`, and the nilpotency of `h` in (c), both need the basis
  expansion;
* Lemma 8.5: the conclusion `μ = 0` uses Lemma 1.1(b);
* Theorem 8.4;
* Prop. 7.4 and Cor. 7.2, which speak about basis elements;
* Remark 8.7.

The §0 table lists F1 only for Cor. B.3. (F1) is Theorem A.5 of `01_foundations.md`, which is
itself unrefereed, so the dependence should be explicit. Theorems A, B, C need no (F1); say so,
it is a strength.

**G4 [cosmetic].** §0, row Thm 8.1: "`δ₁ + [h,δ₀]` occurs for every first-order gauge `h`" is
true only when a differential with first-order part `δ₁` exists (8.1(c) needs a differential
`d`). For arbitrary words this is Conjecture C1′, and it fails for the tree formula itself in
cyclic words. Also define `[h,δ₀] := hδ₀ − δ₀h`.

**G5 [cosmetic].** Prop. 5.3, proof of (A1)/(B1): only the pairs (moving pair strand, unaffected
strand) are checked. One line is needed for the pairs (`a`-strand, mover), (`a`-strand,
`b`-strand) and (`a`-strand, `W`). All are crossed 0, 0 and 1 times; I checked them.

**G6 [cosmetic].** Cor. 7.2:

* "every `r_c` must be accompanied by some `t_{cc'}` or by a dot deficit" is misleading. The
  condition is `Σ_{c'} m_{cc'} = a_c + n_c ≥ n_c`, and dots cannot substitute for `t`'s.
* The `q`-equation is omitted. It is implied by the `δ_c` and `ε_{kW}` equations:
  `2Σa_c = ΣÊ − 2ΣŜ − 2#LR_k`. Say so.
* In 8.1, "the monomial and the dots are fixed up to finitely many solutions" should read:
  "for a fixed basis element `x^α β_σ` the monomial is unique (forest argument of PROOFS
  Cor. 3.1)". This is what makes the coefficients integers.

**G7 [cosmetic].** Remark 8.6 "(M₁) … excludes exactly the non-monotone gauges": no statement of
that kind is proved. What is proved is Theorem 8.4, and for multiplicity ≤ 2, Prop. 7.4(b):
every first-order gauge is non-monotone.

**G8 [cosmetic].** Numbering: "Lemma 8.5" and "Computation 8.5" clash. The script docstrings
cite "Lemma 6 / Prop. 7" for Prop. 5.3 and "Theorem 8.1" for Theorem 9.1 (in
`04_cyclic7_dotfree.py`).

**I1 [improvement: the Observation is a theorem].** *Claim:* `X_y = β_{τ_y}` exactly, for
**every** choice of reduced word of `τ_y` (in particular the klrw basis braid). *Proof:* by
Lemma 4.5 every inversion of `τ_y` joins one of the strands `y, i, j` to a strand outside
`{y, i, j}`, and `y, i, j` do not cross each other. So the inversion graph is bipartite and has
no triangle. A braid move needs three pairwise crossing strands, so by Matsumoto (F3) any two
reduced words of `τ_y` are related by commutations (A2) alone. ∎ (`R04_comb.py notriple`:
64 690 cases, bipartite in all; `R04_alg.py thmB`: `X_y` equals my bubble-sort word exactly in
all 3 643 cases.)

*Consequences:*

* Theorem B holds exactly with `β_{τ_y}`: `δ₀² = Σ ε r t β_{τ_y}`, with no `r`-degree ≥ 2 terms
  in any basis;
* Cor. B.3 gives the exact coefficient;
* in §10 item 1 the contribution `Σ ε r t (X_y − β_{τ_y})` vanishes identically.

---

## 4. Independent checks (all 0 failures unless stated)

### 4.1 Theorem B, entrywise and exact (`R04_alg.py thmB`)

`δ₀²` is computed in `klrw` from my own elementary-move words and the `δ₀` signs. It is
compared **exactly** with `ε r_{c(j)}t_{c(j)c(i)} Σ_y X_y`, where `X_y` is built from the verbal
description of Lemma 4.5 by my own strand simulation. I asserted at each slide that the crossed
strand is the left neighbour, that `y, i, j` become consecutive, that the bijection is `τ_y`,
and that the word is reduced. The result is also compared modulo `(r)²` with my own reduced
word of `τ_y`.

Words used:

* types `A_4`, `D_4`, `D_5`, `E_6`, `A_2`, random words of length 4–8, half of them with a
  colour of multiplicity ≥ 3;
* hand-picked trivalent-node words `D_4` `2134221` (red 2), `1213242`; `E_6` `4235442` (red 4),
  `2423454` (red 2);
* the cyclic `1221122` and `1121121`;
* random red colours.

Results:

* seed 0: 36 words, 6 770 components, 1 411 bad triples;
* seed 7: 66 words, 9 851 components, 2 232 bad triples.

All exact.

### 4.2 Theorem C (`R04_alg.py thmC`)

`(δ₀+δ₁)²` is computed from `formula.Stack` (terms with ≤ 1 pair, formula signs). The one-pair
diagrams are my **own** bubble-sort reduced words, not the basis braid; this also tests the
claimed independence of choices modulo `(r)²`. I checked that every coefficient of the result
has `r`-degree ≥ 2.

Words: types `A_2`, `A_3`, `A_4`, `D_4`, `D_5`, `E_6`; random words of length 4–8 including
multiplicity ≥ 3–4 and equal neighbours; the trivalent words above; `12321223`, `242242`
(`D_4`, red 4), and `1221122` (both reds).

Results:

* seed 0: 44 words, 1 235 vertices, 17 728 products, 30 of the words with `δ₀² ≢ 0 mod (r)²`:
  **0 failures**;
* second run: see §4.7.
* **Negative control:** with the one-pair signs flipped, i.e. dropping the factor
  `(−1)^{[x_p∈I]}`, 17 of 20 words fail.

### 4.3 Prop. 5.3 and Cor. 5.4

* Crossing numbers of all concatenations (`R04_comb.py prop53`, pure Python, inversion sets of
  the two factors). Types `A_2`, `A_3`, `A_4`, `D_4`, `D_5`, `E_6`, random words of length 3–8
  including equal neighbours, all vertices:
  * (A1) 19 988, (A2) 22 814, (A3) 23 391 (11 194 adjacent / 4 232 far / 7 965 same colour);
  * (B1) 19 988, (B2) 22 814, (B3) 23 391.

  Exactly the predicted pair crosses twice. `W` is never crossed twice, the mover is never in a
  bigon, and the targets of (A) and (B) agree.
* Values modulo `r` (`R04_alg.py vals`): the `r`-degree-0 part of every composite equals the
  table of Cor. 5.4 (bigon polynomial with my own dot bookkeeping). 816 + 2 012 composites in
  `A_2`, `A_4`, `D_4`, `D_5`, `E_6`.

### 4.4 Signs (`R04_comb.py signs`)

* Lemma 5.1 against `formula.py`: 7 530 one-pair terms.
* The set of one-pair terms of `formula.py` equals the set of one-pair configurations.
* (C1), (C2), (C3) of Theorem C recomputed from `formula.py`'s own signs, not from the closed
  forms of the proof: 15 060 bad-triple instances, 20 628 `(x, j, p)` pairs, 4 114 four-term
  groups.

All in `A_4`, `D_4`, `D_5`, `E_6`.

### 4.5 Degree calculus and section 8

* **Prop. 7.1** against a direct computation of `D(I) − D(J) − deg β_σ` in my grading group:
  20 000 random `(type, word, red, I, J, σ)` in `A_2`, `A_4`, `D_4`, `D_5`, `E_6`.
* **Prop. 7.4** (`R04_comb.py prop74`):
  * words of multiplicity ≤ 2 in `A_3`, `A_4`, `D_4`, `D_5`, `E_6`, every red colour, all
    `I`, `J`, `σ`, homogeneity by my own coefficient comparison;
  * (a): 602 monotone first-order elements, all one-pair terms with the stated monomial and no
    dots, and every one-pair configuration occurs;
  * (b): no monotone first-order element in homological degree 0 (526 non-monotone ones exist).
* **Lemma 8.5** as a rank computation (`R04_alg.py inj`): full rank on 7 words with one-pair
  terms (4–48 terms each), including `11211`, `12212`, `213242` (`D_4`), `123212` (`A_3`).
* **Ex. 8.3 from scratch** (`R04_alg.py ex83`, `R04_comb.py gauge 1212 2`):
  * homological degree 0 contains exactly one homogeneous family at all `r`-degrees ≥ 1 (dots
    allowed): `{3,4}→{1,2}`, `σ = (2 4)`, `r_2t_21`, word `ψ_2ψ_3ψ_2`;
  * `d² = 0`, `h² = 0`, `d'² = 0`;
  * (H1) holds, and `d' − d` is non-zero modulo `(r)²` in exactly the four stated components
    (details in G1).
* **Section 8 dimensions** (`R04_gauge.py`, own enumeration with dots, own words):

  | word, red | dim `Z₁` = dim `B₁` | monotone | `r`-free `t`-gauges |
  |---|---|---|---|
  | `1212`, 2 | 1 | 0 | 0 |
  | `1121`, 2 | 1 | 0 | **8** (Remark 8.7 ✓) |
  | `1321` (`A_3`), 1 and 2 | 1 | 0 | 0 |
  | `12112`, 2 | 8 | **2** | 0 |
  | `2132`, `2312` (`A_3`), 2 | 0 | 0 | 0 |

  I checked that the tree formula of `12112` (red 2) satisfies `d² = 0`. Together with the
  monotone `B₁ ≠ 0`, this makes the claim "Theorem 8.4 fails without the multiplicity
  hypothesis" a genuine counterexample: monotone `h` gives a monotone `d'`.
* **§7 examples** (`1121`/2 `{3,4}→{1}` `σ=(2 4)`; `11121`/1 `{4,5}→{1}` `σ=(1 3 2 5)`):
  homogeneous, dot-free, monomial `r_1t_12`.

### 4.6 Theorem 9.1 (`R04_comb.py cyc7`, `R04_cyc7.py b|c|d|z1`)

* **Cyclic configurations of `1221122`** (own exhaustive enumeration): exactly one, mover 5
  with pairs `(1,4)`, `(2,7)`, `(3,6)`.
* **(a)** Dot-free homogeneous elements of homological degree −1, by `r`-degree:
  `0: 2447, 1: 1464, 2: 204, 3: 0, 4: 7`. Identical to the draft, by an independent
  enumeration.
* **(b)**
  * 680 tree terms;
  * `d_T²` has no coefficient of `r`-degree ≤ 2;
  * 28 coefficients of `r`-degree 3, all with exactly one dot, in exactly the components
    `{1,4,5,6,7}→{1,2,3}`, `{2,4,5,6,7}→{1,2,3}`, `I₀→{1,3}`, `I₀→{2,3}`;
  * also 14 coefficients of `r`-degree 4 (not mentioned in the draft);
  * the same numbers with my own reduced words in place of the klrw basis braid.
* **The sign of `X_dot` against NOTES §6** (by hand from `formula.py`'s rule):
  * all four spanning trees of the 4-cycle have monomial `r_1r_2²t_12t_21²`;
  * each has two connectors in `I`;
  * so all four have the same sign `κ(I₀;5,F) = −1`, which is the sign of `X_dot`.
* **(c)** (`R04_cyc7.py c`, then `c2`; logs `cyc7_c.log`, `cyc7_c2.log`):
  * 204 dot-free `r`-degree-2 candidates;
  * the kernel of `[δ₀,−]` modulo `(r)³` has dim 4 ✓;
  * **but** the `r³`-parts of `[δ₀+δ₁, κ]` have rank 4, and **adding `(d_T²)₃` leaves the rank
    at 4**. The obstruction is removable, contrary to the draft;
  * dropping the products `δ₁·κ`, as `04_cyclic7_dotfree.py` does, reproduces the draft's
    4 → 5;
  * the solution `κ*` (coefficients `(1, 0, 0, −1)` on the kernel basis; the `r³` map on the
    kernel is injective, so `κ*` is unique) is `r_2²t_21²` times:
    * `+β`: `{4,6,7}→{1,3}`, `{4,5,6,7}→{2,3,5}`, `{1,4,5,6,7}→{1,2,3,5}`,
      `{2,4,5,6,7}→{1,2,3,5}`;
    * `−β`: `{4,6,7}→{2,3}`, `{1,4,6,7}→{1,2,3}`, `{2,4,6,7}→{1,2,3}`, `{4,5,6,7}→{1,3,5}`;

    with `σ = (1 5 4)(2 7)(3 6)` in the components without slot 5, and `(1 4 5)(2 7)(3 6)` in
    those with it. All are monotone, and none is the bijection of a tree term;
  * direct multiplication: `(d_T + κ*)²` has only 18 coefficients, all of `r`-degree 4;
  * at `r⁴` the 7 dot-free candidates have rank 7 and absorb the obstruction (rank 7 → 7),
    giving `ε₄` (4 terms in `I₀→J₀`, listed in G0);
  * the resulting `d_F = d_T + κ* + ε₄` satisfies **`d² = 0` exactly**
    (`R04_cyc7b.py build`, log `cyc7b_build.log`).
* **`d_F` as data:** `R04_data/cyclic7_1221122_W2_dotfree.json` (692 terms); the dotted
  `d_X = d_T + X` is `…_dotted.json` (690 terms). `R04_cyc7b.py verify` (log
  `cyc7b_verify.log`) re-checks both from the JSON alone:
  * `d² = 0`;
  * homogeneity under klrw's own grading;
  * no `u`;
  * `r`-degree ≤ 1 part = `δ₀ + δ₁`;
  * dot-free (`d_F`) / dotted (`d_X`).
* **Gauge** (`R04_cyc7b.py gauge`, `gauge_full`, `R04_gauge_h.py`; logs `cyc7b_gauge*.log`,
  `gauge_h.log`):
  * `r`-free homogeneous elements of homological degree 0: 1 896 (952 with a `t`);
  * at `r`-degrees 1 and 2: 290 and 8 elements (dots allowed);
  * the system for `g = 1 + h₁ + h₂` with `g d_X ≡ d_F g` modulo `(r)³` has rank 298 = number
    of unknowns, so its solution is unique;
  * that solution has `h₁ = 0`, `h₂ = h` (2 terms, see G0), and gives `g d_X = d_F g`
    **exactly**;
  * `h² = 0`, and `(1+h)d_X(1−h) = d_F` is checked again from the JSON files;
  * `h` is saved as `…_gauge_h.json`.
* **(d)** (`cyc7_d.log`):
  * the 10 terms of `X`, typed in from the text, give `(d_T + X)² = 0` exactly;
  * with `X_dot` alone, 6 components of `d²` stay non-zero;
  * `d_T` has no term in `I₀→J₀`;
  * complete dotted ansatz at the four orders: 8, 6, 4, 2 elements, ranks 8, 6, 4, 2, so
    kernel 0 ✓.
* **z1** (`cyc7_z1.log`): dot-free first-order cycles: dim **106** ✓ (as stated). The monotone
  ones have dim 56, which confirms "not forced under (M₁)".

### 4.7 Second Theorem C run

`R04_alg.py thmC 5 11` (log `thmC_5_11.log`): 68 words, 1 848 vertices, 23 007 products,
0 failures. Negative control: 23 of 32 words fail.

---

## 5. Dependence on unproved input

| input | where used | declared? |
|---|---|---|
| (F3) Matsumoto | Lemma 2.2, Thm A, Thm B (and I1) | yes |
| (F3′) Tits' word property (Björner–Brenti 3.3.1) | Lemma 2.2 only | yes (new standing fact; correct reference) |
| (F1) basis theorem (= `01` Thm A.5, itself a draft) | Lemma 1.1(b), Cor. B.3, Thm 8.1 (`ε₁ ∈ V`, nilpotency), Lemma 8.5, Thm 8.4, Prop. 7.4/Cor. 7.2 ("basis elements"), Rem. 8.7, Thm 9.1 | only for 1.1(b), B.3 (G3) |
| (F2) grading | §7, §8 | yes (standing) |
| PROOFS Lemma 1, Prop. 3 / `01` Lemma B.2 (crossing table) | Lemma 5.2, Prop. 5.3 | yes |
| independence of `β(I;j,F)` of the reduced word | not needed for A, B, C, 7, 8 (everything modulo `(r)²`, or exact with no triples); needed for §9, which fixes the basis braid | implicit; should be stated in §9 (G2) |
| `klrw` implementation and solver | Ex. 8.3, Rem. 8.7, Thm 9.1, all of §4 above | yes (Tier 2) |
| `klrw` implementation (products, basis expansion, grading) | **my G0** (`d_F`, the gauge `h`) | Tier 2 as well. It needs no solver and no degree enumeration: `d_F` and `h` are explicit data, re-checked from JSON by direct multiplication |

Theorems A, B, C, Prop. 5.3, Prop. 7.1 use only the relations, (F3), (F3′) and the crossing
table. Theorem 8.4 additionally uses (F1) and (F2).
