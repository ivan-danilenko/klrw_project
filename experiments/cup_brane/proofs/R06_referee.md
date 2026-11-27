# R06 — Referee report on `06_main_intertwiner.md` (sections 1–6 and 9; 7–8 spot-checked)

## RESUME NOTES — closed (report complete)

* Scripts: `checks/R06_lib.py` (own library), `R06_basic.py`, `R06_natural.py`,
  `R06_main.py`, `R06_classes.py`, `R06_sec9.py`, `R06_compare_author.py`; all logs in
  `checks/R06_logs/`.

---

## A. Verdict in brief

* **Theorem 6.2 and Corollary 6.3 are proved.** I found no error in §§1–6. The proof of
  Theorem 6.2 (§6.3–6.6) was redone by hand step by step: the case split of `d_Φ G − G d`
  into the classes T0, T1, T2 is exhaustive and disjoint, the expansions of the tree terms
  in the labelled calculus are right, the coefficients, the `q_{ab}` of doubly inverted
  pairs and the signs are right, and the three polynomial identities (P0)–(P2) reduce
  correctly to the tree lemma. Independently, with my own code, `d_Φ G = G d_tree` holds
  exactly for **all** words of length ≤ 6 with multiplicities ≤ 2 in `A_3` (810 cases) and
  `D_4` (9 296 cases), for every red colour, and for 24 random 7–9-strand words in `A_5`,
  `D_5`, `E_6`; perturbing `G` makes it fail; the same `G` fails for every word with a
  colour three times that I tried; and **each class T0/T1/T2 vanishes separately** in the
  algebra (so the *structure* of the proof, not only the statement, is confirmed).
* §§2–5 (labelled calculus, naturality, `d_Φ² = 0`, reduction Props 5.1/5.2) are correct
  and complete; §9 (Props 9.1–9.3, Cor. 9.4) is correct and complete, and says accurately
  what it proves.
* Problems: no fatal one; one repairable imprecision (Remark 5.3 / Prop. 5.1: order for a
  factorised `G`), the rest cosmetic (§D).

---

## B. Verdict per statement

| statement | verdict | remarks |
|---|---|---|
| §1.1 conventions | correct | (S), (D), (B) agree with the README and with `klrw` (re-checked in `R06_basic.py`: `ψ²` on `(1,2)` is `t_{12}x_1 + t_{21}x_2`; (S) both equations). The list "F1 is used only in Prop. 5.1 and Lemma 6.5" is incomplete: also Prop. 4.4, 5.2 (injectivity of `ζ_m`) and §9 (cosmetic). |
| Lemma 1.1 (`ψ f = (s f)ψ + r ∂f`) | correct and complete | base cases are the two equations of (S); the Leibniz rule `∂(fg) = (∂f)g + (sf)(∂g)` is what the induction uses. |
| Lemma 2.1 | correct and complete | identical to Lemma D.1 of `01`. |
| **Lemma 2.3** (exact braid relations for `g`) | correct and complete | The five patterns of a 3-strand state with ≤ 1 red strand are exhaustive. Recomputed every case: case 2/3 dot transport `ψ_2ψ_1(x_2−x_3) = (x_1−x_2)ψ_2ψ_1` and `ψ_1ψ_2(x_1−x_2) = (x_2−x_3)ψ_1ψ_2`, (B) exact on `(c,c,T)`, `(T,c,c)` including `T = W` of any colour; case 4: the defect `(x_1−x_3)·C(c,T,c)` is cancelled by `r(ψ_1² − ψ_2²)`, both for `T ∼ c` and for `T = W_k`, `c = k`; case 5: the expansions of `φ_2φ_1`, `φ_2φ_1ψ_2`, `ψ_1φ_2φ_1` line by line (`a_2b` symmetric in `x_1,x_2`; `ψ_1 b = a_2ψ_1 + r`, `ψ_1 a_2 = bψ_1 − r`). This proof does not use F1 (unlike Thm D.3 of `01`). |
| Prop. 3.2 | correct and complete | commutation moves: `g_i` involves only `ψ_i`, the dots at `i, i+1` and the idempotent, and the type of `g_j` is unchanged by `g_i` for `|i−j| ≥ 2`; braid moves: Lemma 2.3. |
| Prop. 3.3 | correct and complete | |
| Def. 3.4, Prop. 3.5 (product rule) | correct and complete | `q_{ab}` is symmetric (`Q_{cc′}(x,y) = Q_{c′c}(y,x)`), so defined on unordered pairs; the induction is right in both cases `{a,b} ∈ inv(s,s′)` / `∉`. |
| Def. 4.1 | correct | `m_j(I)` has a unique reduced word (Lemma C.1 of `01`). |
| **Prop. 4.2** (naturality) | correct and complete | for `ψ_i` on `(X, Y, S)` the three cases are exhaustive: `X, Y` not black of one colour ⇒ `ψ_i = g_i`, Lemma 2.3; same colour `≠ c(m)` ⇒ (B) exact on `(c,c,T)`; all three of one colour ⇒ second statement of Lemma 2.3. Red strand in the block and dots on all strands are covered. |
| **Prop. 4.3** (`d_Φ² = 0`) | correct and complete | `{i, j}` is inverted in neither of the two paths, so both are `𝒟(std I → std(I∖{i,j}))` by Prop. 3.5 (no doubly inverted pair); signs opposite. |
| Prop. 4.4 | correct and complete | `ζ_m` is well defined as a positional polynomial because it is symmetric in the other dots of colour `c(m)`; commutation with equal-colour crossings of `w′` by Lemma 1.1; injectivity is Thm E.1. |
| **Prop. 5.1** | correct and complete | `G d² = d_Φ G d = d_Φ² G = 0`; induction along `≺` (a strict order on a finite set); cancellation of `ζ_I` is *left* multiplication by a polynomial at the source of `(d²)_{IK}`, Theorem E.1(a) — the correct side. Valid verbatim for any well-founded order (see D1). |
| **Prop. 5.2** | correct and complete | with the cone differential `[[−d′⊗1, Φ],[0, 1⊗d′]]` the `R→L` component of `D_Φ Γ = Γ D` is exactly (★); `R→R` is Prop. 4.4; injectivity of left multiplication by `ζ_m` gives `(d′⊗1)F = F(1⊗d′)`. |
| Remark 5.3 | correct as stated (it claims only the factorisation and the diagonal) | the extension `1 ⊗ Γ ⊗ 1` commutes with the later `Φ_j` by Prop. 4.2; the iterated-cone signs agree with `(−1)^{#{i∈I: i>j}}` because the Koszul sign of a tree term of `w_{≤k}` changes by `(−1)^{#{i∈I: i>k}}`. Triangularity of the factorised `G` holds for the colex order, not for `≺` (D1). |
| Lemma 6.1 (Λ1)–(Λ4) | correct and complete | (Λ3): exponent `#{i>a} + #{a<i<b} ≡ 1 + #{i>b}`; (Λ2) for `j = b_p` both sides vanish. The identifications `κ(I;j,F) = ⟨ξ_J, ι_jθ_Fξ_I⟩` and `(−1)^{#{i>j}} = ⟨ξ_{I∖j}, ι_jξ_I⟩` agree with `formula.py` (`koszul`). |
| Def. 6.2, triangularity | correct | `I_F ≺ I` for `F ≠ ∅`; `[σ_F]` is dot-free and contains only crossings of different colours — I checked that it **equals the klrw basis braid** in every case of the class checks. |
| Lemma 6.3 | correct and complete | the colour map is injective on `{ρ} ⊔ 𝔉` (one pair per colour; the pair of colour `c(ρ)` contains `ρ`); (ii) by uniqueness of paths in a tree. |
| Lemma 6.4 | correct and complete | (ii) the only same-colour crossing of a tree term is mover/twin, twin `a < j`, `a` left of `j` before the crossing; (iii) a strand crossing both `a` and `j` is a strand of some `q ∈ F` with `a_q < a < b_q < j`, not linked to `j`, hence (Lemma 6.3(ii), `F` connected) of a colour not adjacent to `c(j)`. |
| **Lemma 6.5** | correct and complete | `z_p` slides to the crossing through crossings of different colours; `(x_a − x_j)ψ = φ + r` on the state `(…a, j…)`; `D_2′` is `D_2` relabelled (`a ↔ j`), hence reduced; a pair doubly inverted in `D_1, D_2′` is `{a,y}` or `{j,y}` with `y` crossing both in `β`, so `q = 1` by 6.4(iii); `W` cannot be such a `y`. Word-independence by left injectivity of `z_p` (correct side). Re-checked in the algebra for every twin term of the class checks. |
| **Lemma 6.6** (a)–(f) | correct and complete | all six tables recomputed from the (start, middle, end) positions; in particular nested pairs never give a doubly inverted pair, pairs on opposite sides of `W` in a factor are never inverted by it, and `W` is inverted only by the mover/`Φ`. The total bijections of (a)/(b), (c)/(d), (e)/(f) and the targets agree. |
| Lemma 6.7 | correct and complete | removing a leaf: the factor of `ℓ` sums to `A_ℓ + B_ℓ = M_ℓ` or is `M_ℓ`; the other factors do not see `ℓ`. |
| Lemma 6.8 | correct and complete | `Q(x,y) − Q(x′,y) = t(x − x′)`; the signs `ε_q` agree with `formula.py` (connector in the source vertex of the term ⇒ `−1`; the connector `b_π` of a parent pair `π < q` and the mover lie in `I′`, `a_π` does not). |
| Cor. 6.9 | correct and complete | a link inside `F′` is either an edge of `T` with both ends outside `S` (factor `M`) or a link outside `T` (constant); every link between two vertices of `T` is an edge of `T` (component of a forest). |
| **Theorem 6.2** (proof §6.6) | **correct and complete** | see §C below for the walk-through. |
| **Corollary 6.3** | **correct and complete** | Thm 6.2 + Prop. 5.1 with `ζ_I = ∏_{b_p∈I} z_p ≠ 0`. The examples (`sl_n` `ω_1, ω_2`; vector of `so_{2n}`; `D_4`) do have multiplicities ≤ 2. |
| Remark after §6.6 | heuristic | "contains a 4-cycle as soon as the colour has two neighbours" should be "*can* contain": words with a colour three times can be cycle-free (`111`, `12121`, …), see D5. |
| **Prop. 9.1** | correct and complete | `ζ_I` is a product of distinct same-colour position differences, hence divides `√Δ`; a triangular matrix with invertible diagonal (finite, well-founded order) is invertible. Notation `G ⊗ ℤ[Δ^{-1}]` means localisation at the central element `Δ` (cosmetic, D3). |
| Prop. 9.2 | correct and complete | rank one: `f𝒟(t→std I) = 0 ⇒ f·∏q = 0 ⇒ f = 0` (F1 is not even needed); the pairs inverted by `Φ_j(I)` and between `t` and `std I` are the `y ∈ {W, 1..j−1}` with `j` left of `y` in `t`, independent of `I`. Checked: `R06_sec9.py`, 1 536 identities. |
| Prop. 9.3 | correct and complete | over `Pol[Δ^{-1}]` each `x_i − x_{i+1}` (equal colours) is a unit, `ψ = (x_i − x_{i+1})^{-1}(φ + r)`; products of `g` along arbitrary words are polynomial multiples of one `𝒟` (Prop. 3.5 letter by letter + relabelling invariance); `n` generators of a free module of rank `n` over a commutative ring are a basis. |
| Cor. 9.4 | correct and complete (for every word with a `G` as in 9.1, in particular multiplicity ≤ 2) | `f_j(t)` is a unit or a unit times a monic polynomial in `x_j` over `R′[x_{<j}]`, so `f_1,…,f_m` is regular (or the Koszul complex is contractible); localisation is exact. **What is proved:** the higher homology of the tree complex becomes 0, and `H_0` becomes `⊕_t Pol′[Δ^{-1}]/(f(t))`, after inverting `Δ` *and all parameters*. Not proved (and not claimed): absence of `Δ`-torsion, any statement over `R`. The paragraph "What this says and does not say" is accurate. |
| §7, §8 | nothing claimed as proved, apart from the implication "Conjecture H ⇒ C1′" (correct: Prop. 5.2, induction on the length, cycle-freeness passes to prefixes by Prop. 7 of `PROOFS.md`, and the cone structure of the tree formula) | one overstatement in the Summary (D4). |

---

## C. The proof of Theorem 6.2 — what I checked

**Setting.** With multiplicities ≤ 2 there is at most one pair per colour; `𝔓(I′) = 𝔓(I)∖F′`
for `I′ = I_{F′}` (pairs not in `F′` have their slots untouched), so `F′ ∩ F″ = ∅` in every
right term, and the tree terms at `I′` are exactly the `(j, F″)` with `{j} ∪ F″` connected in
`Λ_j` (Lemma 6.3(i)).

**Exhaustiveness and disjointness of the classes.** Every term is a pair (factor of `d_Φ`
or `G`, factor of `G` or `d`):

* left terms `(j, 𝔉′)`, `𝔉′ ⊆ 𝔓(I∖j)`: either `j` in no pair of `𝔉′` (then `𝔉′ ⊆ 𝔓(I)`)
  → T0`(j, 𝔉′)`; or `j = a_p`, `p ∈ 𝔉′` (`j = b_p` impossible), `b_p ∈ I` → T1`(p, 𝔉′∖p)`.
  The map is a bijection onto the index sets of T0 and T1.
* right terms `(F′, j, F″)`: `j ∈ I′`; either `j ∈ I` (then `j` in no pair of `F′`) or
  `j = a_p`, `p ∈ F′`. Main parts: `j ∈ I` → T0`(j, F′⊔F″)`; `j = a_p` → T2`(p, F′∖p ⊔ F″)`(i)
  (a first occurrence has no twin below it, so there is no resolved part). Resolved parts
  exist iff `j ∈ I` has a twin `a < j`; `p = (a,j) ∉ F′ ∪ F″`; `a ∈ I` → T1`(p, F′⊔F″)`,
  `a ∉ I` → T2`(p, F′⊔F″)`(ii).
* Within each class all the listed terms are present (G sums over **all** `F`, `d` over all
  admissible `F″`), and each term is assigned to exactly one class. No product vanishes
  (brackets times non-zero polynomials; `θ_F ξ_I ≠ 0` because `F ⊆ 𝔓(I)`), so there are no
  hidden zero terms to account for. Since each class is shown to sum to zero, it does not
  matter that classes of different kinds can share a target bracket.

**Division by `z`.** No division of algebra elements is needed: when the mover `j` has a twin
`a < j`, `z_{(a,j)}` is a factor of the polynomial `γ_{F′}(I)` (`(a,j) ∉ F′`, `j ∈ I`), its dots
lie on strands not moved by `[σ_{F′}]`, so it slides (Prop. 3.3) to the source of `β` and
Lemma 6.5(ii) is applied to `z_p β`. Torsion-freeness is used only for the word-independence
in Lemma 6.5 and in Prop. 5.1, both times as *left* multiplication by a polynomial at the
source (Theorem E.1(a)).

**Dots.** All polynomials are written at the source `I` of the whole product, with `x_s` the
dot on the strand that *starts* in slot `s`. In left terms `γ(I∖j)` transports to `I` with the
same names (`Φ_j` is the labelled identity); in right terms `z_{(a,j)}` sits on unmoved strands;
the `q_{ab}` of (6.1) are in labelled variables at the source. In T2(i) the mover is the
strand *labelled `b`* (at slot `a` of `I′`); the table (e) and the rows `y = x_a`, `y = x_b` of
Lemma 6.8 are used with the right labels.

**`q_{ab}` of doubly inverted pairs.** Same-colour pairs never occur (one pair per colour; the
twin of the mover is unaffected and is never doubly inverted: in (a), (b) it moves in no
factor), `W` never occurs, so every `q` is `1` or a `Q`. The twin case is handled in
Lemma 6.5 (doubly inverted pairs there have `q = 1`).

**Signs.** (Λ1), (Λ2) in T0; (Λ3) in T1 gives the relative sign `−1` between left and right;
(Λ4) in T2 gives equal signs for (i) and (ii), so "(i) + (ii) = 0" is the right claim; the
`ε_q` are absorbed into `B_q = −ε_q t_q z_q`.

**(P0), (P1), (P2).** Recomputed: in each, the factors attached to links with the root are
the row-3 `A`'s with the right `y` (`x_j`; `x_b` or `x_a` according to type (α)/(β) in T1;
`x_a` or `x_b` according to (γ)/(β) in T2-left and `x_b` in T2-right), pairs not linked to the
root but adjacent to `c_p` are outside the component (Lemma 6.3(ii)) and give the constant
factors, and Cor. 6.9 turns each side into `∏_{links in 𝔉} Q^{RL}` times the same root factors.

**Conclusion.** The proof is complete. The only facts used after §3 are
`(x_a − x_b)ψ = φ + r`, the product rule and Theorem E.1, as the draft says.

---

## D. Problems (numbered; none fatal)

1. **(repairable, does not affect Cor. 6.3) Order in Prop. 5.1 vs. factorised `G`.** The
   components `h : I∪{k} → J` of a one-step map need not satisfy `J ≺ I∪{k}` for `≺` = "same
   size, smaller sum": e.g. `I∪{k} = {1,5}`, `J = {3,4}`. So a factorised
   `G = Γ^{(1)}⋯Γ^{(m)}` (Remark 5.3, §7, §8.2 "is triangular", Prop. 9.1 "as in Remark 5.3")
   is triangular for the colex order (compare the largest element of the symmetric
   difference; `h` removes `k` and adds only slots `< k`), not necessarily for `≺`.
   *Repair:* state Prop. 5.1 for an arbitrary well-founded order on the vertices of each
   size (the proof is unchanged), and say "colex" where factorised maps are used. For
   Definition 6.2 the order `≺` is fine.
2. **(cosmetic)** §1.1 lists Prop. 5.1 and Lemma 6.5 as the only uses of F1; also Props 4.4,
   5.2 and §9.
3. **(cosmetic)** §9: `G ⊗ ℤ[Δ^{-1}]`, `C[Δ^{-1}]`, `H_i(C) ⊗_R R′[Δ^{-1}]`: `Δ` is a central
   element of `A`, not of `ℤ` or `R`; write "localisation at `Δ`" (and `⊗_R R′`).
4. **(cosmetic, presentation)** Summary item 5: "*Iterated division* (§8.5) starting from
   `d_Φ` thus defines the differential without reference to the tree formula, as long as the
   prefix is cycle-free" reads as a general statement; it is computed on five words and is
   Conjecture H⁺. Likewise §8.2 "the intertwiner description extends to the cyclic case" is
   one example. Both should carry "[computed]"/"conjecturally".
5. **(cosmetic)** Remark after §6.6: "which contains a 4-cycle as soon as the colour has two
   neighbours" → "which can contain…"; cycle-free words with a colour three times exist.
6. **(cosmetic)** §10 "Every one of these statements is also checked by an exact computation
   (scripts named next to the statements)": no script is named in §5, §9 or for Lemma 6.7.
   (Prop. 9.2 is now checked by `R06_sec9.py`; the class-by-class check `R06_classes.py`
   covers the assembly of §6.6.)

No statement of the brief, `PROOFS.md` or `NOTES.md` was found to be contradicted.

---

## E. Independent checks (own code; `checks/R06_lib.py`, nothing imported from `06_lib.py` except in `R06_compare_author.py`)

`R06_lib.py` builds from the text only: `g_i` (φ for equal black colours, ψ otherwise),
`𝒟(s → s′)` along a fixed or random reduced word, `Φ_j(I)`, `d_Φ`, `G` of Definition 6.2
literally (sum over all `F ⊆ 𝔓(I)`, `ε(I,F)`, `∏(−r_p)`, `∏ z_p`, `∏ Q(x_{b_p}, x_{a_q})`,
brackets `[σ_F]` computed as `𝒟`), and the tree differential from `formula.Stack` +
`sample_verify.element` (klrw basis braids). Run from `proofs/checks` with `sage -python`.

| script / log | what | result |
|---|---|---|
| `R06_basic.py` → `R06_logs/basic.log` | conventions; (S); Lemma 1.1 on a cubic `f`; Lemma 2.1; Lemma 2.3 on **every** labelled state of 3 black strands + `W` for 13 colour words in `A_3`, `D_4`, every relevant red colour (1 960 identities, incl. `ψ_iφφ = φφψ_{i+1}`); negative control; Props 3.2/3.3/3.5 on 48 random triples of labelled states (incl. `q_{aW} = u_k x_a`) | 43/43 ok |
| `R06_natural.py` → `natural.log` | Props 4.2, 4.4 on 150 random elements (sums of 3 random words of length 5 with random dots and coefficients) on 10 words (`A_2`, `A_3`, `D_4`; red = moving colour / adjacent / far; equal neighbours and the red strand in the block) | naturality 0/150 failures; `ζ_m` 0/300; negative control (ψ instead of φ): 138/150 differ |
| `R06_main.py all A 3 6` → `all_A3_*.log` | **all** 270 words of length ≤ 6 over `A_3` with multiplicities ≤ 2, **every** red colour (810 cases): `d_Φ² = 0`, `d_tree² = 0`, `d_Φ G = G d_tree` in every component | 0 failures |
| `R06_main.py all D 4 6` → `all_D4_*.log` | **all** 2 324 words of length ≤ 6 over `D_4`, multiplicities ≤ 2, every red colour (9 296 cases) | 0 failures |
| `R06_main.py random A 5 8 7 9 11`, `D 5 … 12`, `E 6 … 13` → `rand_*.log` | 24 random words with 7–9 strands, multiplicities ≤ 2, random red colour | 0 failures (up to 4 625 components per word) |
| `R06_main.py word …` → `targeted.log` | red equal to / adjacent to / far from the doubled colours, equal neighbours: `A_5`: `3324241` (red 3), `3344225` (red 1), `12123` (red 5), `21324354` (red 4), `33224411` (red 3); `D_5`: `53214325` (red 5), `45321354` (red 4); `E_6`: `1342431` (red 6), `12345621` (red 1) | 0 failures (9 words) |
| `R06_main.py word … text noQ LL RR plusr nosign` | power: `1212` (A_2 red 1), `213213` (A_3 red 2) with `G` perturbed (drop `Q`; `Q(x_{a_p},x_{a_q})`; `Q(x_{b_p},x_{b_q})`; `+r`; drop `ε(I,F)`) | text: 0/40, 0/300 failing components; perturbed: 4, 4, 4, 22, 14 of 40 and 42, 42, 42, 170, 117 of 300 fail |
| multiplicity 3, `G` of Def. 6.2 literally (all same-colour pairs) | `111, 1121, 1211, 12121, 11211` (A_2 red 1), `123131` (A_3 red 1), `21322` (A_3 red 2), `342312453` (A_5 red 3) | `d_Φ G ≠ G d` in 5/14, 12/42, 12/42, 33/126, 61/138, 60/322, 28/112, 840/5040 components, while `d_Φ² = 0`, `d_tree² = 0` hold: the formula is specific to multiplicity ≤ 2, as the draft says |
| `R06_classes.py` → `classes_*.log` | **structure of §6.6**: all terms of `d_Φ G` and `G d` in the algebra, right terms split into main/resolved parts (`z_p` removed from `γ`, then `[σ]`, `r_p[σ(a j)]`), split compared with the actual term, each piece assigned to T0/T1/T2 by the rules of §6.6, **each class checked to vanish separately**; Lemma 6.5 at every twin term; `[σ_F]` = klrw basis braid | all 270 `A_3` words of length ≤ 6 × every red (810 cases) and all 600 `D_4` words of length 5 × every red (2 400 cases): **every class vanishes**, no split / Lemma 6.5 / basis-braid error. Negative control (`R06_MISASSIGN=1`: resolved parts put into the wrong class T1↔T2): `213213` gives 300 non-vanishing classes |
| `R06_compare_author.py` | my `G` vs `06_lib.G_formula_row` (`213213`, `2132`) | identical |
| `R06_sec9.py` → `sec9.log` | Prop. 9.2 identity for 30 random labelled `t`, all `I`, `j`; `Δ` central | 1 536 identities, 0 failures |

**Re-runs of the author's experiments in §7–8** (logs `R06_logs/rerun_*.log`):

* `06_check_factorisation.py A 3 2 213213 1 132132` (words not used by the author):
  composite of the one-step maps = Definition 6.2: True, True.
* `06_check_step_formula.py A 2 1 11211 1 12121 2 2112`: all steps hold; `A 3 1 1322312`
  (cyclic): `YYYYYYN` — fails at the last step only, as stated in §7.2.
* `06_divide.py A 2 1 1221122 plus` (§8.4): divisible in all 153 components, the quotient
  equals the stored true differential, differs from the tree formula in exactly one
  component with the four bijections and coefficients of the table in §8.1; `d² = 0`.
  As stated.

---

## F. Overall judgement

**Corollary 6.3 is proved**: for every word in which each colour occurs at most twice, and
every red colour, the tree formula satisfies `d² = 0`. The proof (Lemma 2.3 → labelled
calculus → naturality and `d_Φ² = 0` → Prop. 5.1 → explicit `G` and Theorem 6.2) is complete
and correct; it is independent of `05_main_sliding.md`. The problems found are one
repairable imprecision that does not touch the corollary (D1) and cosmetic points.
