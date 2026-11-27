# R05 — Referee report on `05_main_sliding.md`

## RESUME NOTES (for whoever continues this report)

* **Finished:** reading of all of `05_main_sliding.md` (§§1–11), verdicts below; reading of the old
  scripts `R05_{lib,basic,tools,d2,reduction,redlin,sec8,thmC}.py` (they test what they say, see
  "Independent checks"); re-runs `R05_basic.py 80`, `R05_tools.py 3000 21`, new `R05b_d2.py all A 3 2 6`.
* **Running (background, logs in `proofs/checks/R05_logs/rerun/`):** `b_d2_D4_le7.log` (all T1 words of
  `D_4` of length ≤ 7, every red colour, basis braid), `sec8_stress.log` (`run_sec8.sh`: section-8
  assertions on k = 2 words etc.), `reduction.log` (`run_red.sh`: perturbed `F`, rank test).
* **Remains:** collect those logs; sampled 9–11-strand runs of `R05b_d2.py` (A_5, A_6, D_5, D_6, E_6,
  patterns nb/red/twinred/branch); `R05_thmC.py 102` with a new seed; final verdict section.
* **New scripts of this referee:** `R05b_d2.py` (d² with `formula.Stack` + klrw basis braid; word
  generators with stress patterns), `R05b_sec8.py` (driver for `R05_sec8.check_word`).

Status of this report: **in progress** (written section by section while reading; a previous
referee's 15-line preliminary note in this file has been replaced — its scripts are re-used
only after being read and re-run, see the section "Independent checks").

## Sections 1–4 (general tools, arbitrary words)

**Lemma 1.1 (word property).** Correct and complete. Let `D_0` be the longest reduced prefix
(so `D_0·(i_t)` is not reduced). The two strands `A, B` met by `i_t` then satisfy
`n_{D_0}(A,B) = 1` (≥ 1 since `D_0·(i_t)` is not reduced, ≤ 1 since `D_0` is), i.e. `(A,B)` is
an inversion of `σ_{D_0}` at the adjacent positions `i_t, i_t+1`, so `ℓ(σ_{D_0}s_{i_t}) =
ℓ(σ_{D_0}) − 1` and `D_1·(i_t)` (any reduced `D_1` of `σ_{D_0}s_{i_t}`) is a reduced word of
`σ_{D_0}`. Matsumoto connects `D_0` with it; the moves act on the prefix only and do not change
the permutation of the prefix, so the suffix crosses the same strands and the `n_D(·,·)` of the
whole word are unchanged; then (M3) drops the length by 2. This is the standard Tits argument and
is written correctly. (Cosmetic: strands should be said to be identified by their source
position, which is what makes "`n_{D*} ≤ n_D`" meaningful along the sequence.)

**Lemma 1.2 (Straightening).** Correct. (S2) is symmetric in the order of the three strands,
which is what is needed because along a non-reduced sequence the middle strand of a braid move
is not determined by the permutation. Every exceptional cubic state of (A6) (`(c,c',c)`,
`c'∼c`; `(k,W_k,k)`) is excluded by (S2); every (M3) is on a pair with `n_D ≥ 2`, which by (S1)
has `ψ² = 1`.

**Lemma 1.3 (bigons), Lemma 1.4 (vanishing).** Correct. In 1.3 the coefficient of `x_A` in
`Q_{s_i s_{i+1}}(x_i, x_{i+1})` is `t_{c(A)c(B)}` whichever of `A, B` is on the left, so `Q_{AB}`
is well defined; the dots are carried to the source through crossings of `A` (resp. `B`) with
strands of other colours only, by the hypothesis. In 1.4 the first (M3) on a same-coloured pair
exhibits a factor `ψ_iψ_i = 0`.

**Lemma 2.1, Lemma 2.2 (Sliding).** Correct, with the stated sign: (A6) reads
`ψ_iψ_{i+1}ψ_i = ψ_{i+1}ψ_iψ_{i+1} + C·e` on `(S, S', M)`, `C = C(c(S), c(S'), c)`, which is
non-zero exactly in the two cases listed. The induction in 2.2 is right
(`E_t^{(0)}·(i_l+1) = E_t`).

**Lemma 3.1.** Correct (locality of relations; injectivity by (F1) since `ι(β_σ)` is a reduced
word of `ι(σ)` and (F1) holds for *any* choice of reduced words, Theorem A.5(ii)).

**Lemma 3.2 / Proposition 3.3 (signs).** Correct; re-derived from `formula.py`:
* case 1 (`m ∉ I`): `m` is in no pair (it could only be a `b_p`, which lies in `I`); `κ` uses the
  same sequence; connectors unchanged.
* (α): in `Stack.koszul` the list `img` gets one more entry `m` (fixed by `σ`, largest) at the end:
  the factor `(−1)^{len(img)−1−k}` changes sign and `perm_sign` is unchanged, so `κ = −κ'`;
  connectors are slots `< m`, so the connector factor is unchanged.
* (β): `j = m` is the last entry, `κ = +1`; a pair linked to `m` would need `b_p > m`, so the
  tree is `F = ∅`.
* Prop. 3.3: `(d²)_{I'∪m, K} = Σ_{m∈J}(−ι_R d')F + Σ_{m∉J} F ι_L(d') = −E_{I',K}`; the
  components with `m ∈ I, K` are `(−ι_R d')(−ι_R d') = ι_R(d'²)`; `m ∉ I` gives `ι_L(d'²)`;
  `m ∉ I, m ∈ K` is impossible. Hence the equivalence. No sign problem.

**Lemma 4.1.** Correct (pure algebra, any `F`).

**Lemma 4.2.** Correct. The crossed strands `y < a` have colours `≠ c(a)` because `a` is the first
occurrence in `w'`; the extra strand of `ι_L/ι_R` (possibly of colour `c(a)`) is never crossed
(it is at the far left/right). The word `μμ^*` has only the bigons `{a, y_i}` and no triple, so
Lemma 1.3 applies with every hypothesis verified. `P` has `r`-degree 0 and is non-zero.

**Lemma 4.3.** Correct, and the step the lead asked about is justified:
* `e'Ae_0` is a free *right* `P(e_0)`-module with basis `{ψ_w}` (Theorem A.5(iii)), and the dots
  in `P` are those of `e_0`, i.e. of the right end of `z` — consistent.
* `𝔪^{N+1}A ∩ e'Ae_0 = 𝔪^{N+1}·e'Ae_0` (apply `e'(·)e_0`, `R` is central).
* `e'Ae_0 = ⊕_w ψ_w·R[x]` with `R[x] = ℤ[r,t,u][x]` free over `R` with the monomial basis, so
  `e'Ae_0` is `R`-free with basis `{ψ_w x^β}` and `𝔪^{N+1}·e'Ae_0 = ⊕_w ψ_w·𝔪^{N+1}R[x]`
  coordinatewise. Since `𝔪^{N+1}` is a monomial ideal of the polynomial ring `ℤ[r,t,u,x]`,
  membership is decided term by term: "all components of `r`-degree `≤ N` vanish". The
  lowest-degree argument then uses only that `ℤ[r,t,u,x]` is a domain and `P` is `r`-homogeneous
  of degree 0. Part 2 is the mirror statement with the left basis. Complete.

**Lemma 4.4.** Correct.

**Proposition 4.5 (Reduction Lemma).** Correct. The induction closes: `N` is the largest
integer with all (finitely many) components in `𝔪^N A`; it exists because `⋂_N 𝔪^N A = 0`,
which is justified: `A = ⊕ e(s)Ae(s')` is `R`-free (Theorem A.5(ii)), membership in `𝔪^N A` is
coordinatewise, and `⋂_N 𝔪^N = 0` in `R` (every monomial of an element of `𝔪^N` has `r`-degree
`≥ N`). In Lemma 4.1 at `(I', K̃)` the first sum vanishes by hypothesis, the terms `a ∉ J'` vanish
by hypothesis, the `F ≠ ∅` terms of Lemma 4.4 contribute `𝔪^N·𝔪 A`, and the remaining term is
`±E_{I',K''}·ι_L(m_a(K''))`; Lemma 4.3.1 applies with `μ = ι_L(m_a(K''))` (`a` first occurrence of
its colour in `w'`; the strand of `m` sits at the far left and is not crossed). Remark (1) is
right: `F` is arbitrary (only `d'² = 0` and the shape of `d'` modulo `𝔪` are used) — this is what
test (3) below checks.

**Proposition 4.6.** Correct (same argument; in (l) `μ = ι_R(m_s(I'∪s))` ends at `e_{I'∪m}`, Lemma
4.3.2). The remark that `a` may depend on `I'` is right by induction on `|I'|`.

*Verdict on §§1–4:* no error. (Numerical confirmation: see "Independent checks".)

## Sections 5–8 (Theorem B)

**Lemma 5.1, 5.2, 5.3, 5.4.** Correct and complete. (5.4: the two colours `c_1 ≠ c_2` are
distinct because node colours are distinct; the image path avoids `c` because no node has
colour `c`; with `c_1 — c — c_2` it closes a cycle of length ≥ 3.)

**Lemma 6.1.** Correct (= Lemma B.2 of `01_foundations.md`). The direction statement ("the
strand moving towards the left") is right for (a)–(d); for (e) (not stated in the lemma, but
true and used implicitly nowhere essential): the *outer* strand `L_p` resp. `R_p` of a nested pair
moves left.

**Lemma 6.2.** Correct. All twin possibilities are listed (a pair uses both slots of its
colour, so a strand of a pair is never the twin of an unaffected strand or of the mover).

**Theorem A, Corollary 7.1.** Correct. Remark: the Convention after Lemma 3.2 (choice of reduced
words compatible with `ι_L, ι_R`) holds for the `klrw` basis braids *for every word* without any
check script: in the insertion-sort word of Definition B.10 a fixed strand at the far left has
`ℓ = 0` and shifts all other factors by one, and a fixed strand at the far right is processed
last with `ℓ = 0`; so `w(ι_L π) = ι_L w(π)`, `w(ι_R π) = ι_R w(π)` literally. (Cosmetic.)

**§8 preamble.** Cosmetic slip: "`c` is the colour of no node of a configuration of `w'`" is
false when `a` is the mover; the parenthesis says so ("it can be a mover"). It is only used for
frozen terms (`a` unaffected), where it is true. Should read "of no node of a configuration of
`w'` in which `a` is not the mover".

**Lemma 8.1, Corollary 8.2.** Correct.

**(8.1).** Correct: with `a ∉ I', K'`, the (γ)-terms `F_1;J',K'` vanish (their targets contain
`a`), all terms of `d'_{I',K'}` are frozen (`a ∉ I'` and `a` is in no pair of `w'`), and a term
of `d'` from `J' ∋ a` to `K' ∌ a` has mover `a`.

**Lemma 8.3.** Correct. (The only colour-`c` strand of the state of `w'` is `A`; `A` is
unaffected on the left, so it is crossed only by the mover (iff `a < j`) and by the `L_p` with
`a_p < a < b_p`, all moving left; `A` crosses no red strand.)

**Lemma 8.4.** Correct and complete, re-derived in full:
* 1.: a neighbour `x` of `(a,m)` is interlaced with it, and since `m` is the largest slot the only
  interlacing is `a_x < a < b_x < m` (resp. `a < j < m`); uniqueness of `N` by Lemma 5.4.
* 2.: targets: `t_1(S)` adds `a` and `t_2(S)` removes it, so `K'_U` is independent of `S`.
* 3. (surjectivity): the colour claim is right in both cases (`x` a pair: same two slots, but
  `a_p ∈ J ∌ a_q`; `x = j`: `a_q = j`, then the two colour paths from `c` to `c(j)` in the forest
  coincide; `l ≥ 2` reduces to the first case, `l = 1` gives `a < j` and `j < a`). The edge count
  (`U` connected ⇒ tree with `|F|+|F_a|` edges ⇒ no link between `({j}∪F)∖(a,m)` and `F_a`) is
  exactly what is needed to identify `S`.
* 4.: `(a,m)` is a leaf of `U` because by Lemma 5.4 only `N` has a colour adjacent to `c`.
* (L1)–(L3) follow.

**Lemma 8.5.** Correct and complete. I re-did the (S1) list from Lemma 6.1 applied to `t_2(S)`
(mover `M` at slot `a`; unaffected at `J(S)`: original `y`, `j` (left), `A` (left, slot `m`),
`L_p` (left, slot `b_p`), `R_p` (right, slot `a_p`), `p ∈ F'∪𝒯_S`): (a) `M`/red; (b) `M` with an
unaffected of slot `< a` (original, `j`, `L_p`, `R_p`); (c) `M` with `L_q, R_q`, `b_q < a`; (d)
`L_q` with a left unaffected in `(a_q,b_q)` (original, `j`, `A` impossible, `L_p`), `R_q` with a
right unaffected in `(a_q,b_q)` (original, `R_p`); (e) nested `q, q'`. The list in the draft is
exactly this. The only double crossings are `(L_q, j)` with `a_q < j < b_q` and the interlaced
`(L_q, L_p)`, `(R_q, R_p)`, all between non-linked (by (L1)), hence non-adjacent and different
(L2) colours. (S2): twins are `{A,M}`, `{L_p,R_p}`, two original unaffected, `{j, j̃}`; only the
last can cross; a strand `Y ∼ c(j)` crossing both is excluded in all cases as in the draft; the
red strand is crossed only by `j` and `M`, of different colours. So `β(t_1(S))β(t_2(S)) = B`, the
element of every reduced word of `σ_U`.

**Lemma 8.6.** Correct and complete. The description of `E_t` is right: after
`i_1…i_{t−1}`, `A, S'` sit at `i_t, i_t+1`; `(n,…,i_t+2)` brings `M` across exactly the strands
to the right of `S'` (`𝒜_R ∪ 𝒞_2`), `(i_t−1,…,1)` brings `A` across `𝒞_1 ∪ 𝒜_L`, and the shifted
suffix has `M` in the role of `A`. Crossing numbers, permutation (`σ_U` for `(j,F',∅)`), (S1)
(each node has exactly one strand crossing `A`, so `Z ≠ S'` belongs to a node `≠ N`, of colour
neither `c` nor adjacent to `c` by (L3)), (S2) (if `Y ∈ {A,M}` then `N = j`, `S' = j` and
`n(A,j) = n(M,j) = 0`; red crossed only by `j` and `M`) all check.

**Lemma 8.7 (coefficients).** Correct, re-derived against `formula.py`:
* connector of `(a,m)` with parent `N = p` (`a_p < a < b_p < m`): in `Stack.analyse` the pair
  `(a_p,b_p)` gets the entry `((a,m), b_p)`, i.e. connector `b_p` ✓ (and `j` if `N = j`);
* connector of `q_i` with parent `(a,m)` (`a_{q_i} < a < b_{q_i} < m`): `(a,m)` gets
  `(q_i, a)`, connector `a` ✓; in `t_2(S)` the connector is the mover `a` ✓; both give
  `t_{c(q_i),c}`;
* (8.2) is exactly `Stack.koszul` (replace `b_p` by `a_p` in place; remove `j` with
  `(−1)^{#entries to its right}`; sort): `∂_j` is the right contraction with sign
  `(−1)^{h−k}`;
* connector-sign bookkeeping `(−1)^{k−|S|}`, the Koszul identity
  `κ(t_1(S))κ(t_2(S))ω_{K'} = ∂_a∂_j s_{F'}s_{(a,m)}s_𝒯 ω_I`, and the case `𝒯 = ∅`
  (`κ(t_1)κ(t_2) = −κ'`, connector factor `−1`) are right.

**Proposition 8.8 and the proof of Theorem B.** Correct. Every product contributing to
`E_{I',K'}` with `a ∉ K'` is accounted for exactly once: for `a ∈ I'` there are no products
(Cor. 8.2); for `a ∉ I'` the products are exactly the pairs `(t_1,t_2)` of (8.1), which are in
bijection with (datum, subset) (Lemma 8.4.3); the frozen terms with a bad letter are in bijection
with the data with `𝒯 = ∅` (8.4.4); frozen terms without a bad letter contribute 0.

*Verdict on §§5–8 so far (reading):* **Theorem B is proved**, modulo the standing facts (F1)
(Theorem A.5 of `01_foundations.md`) and (F3). No fatal or repairable error found; cosmetic
points only (listed at the end).

## Section 9

**Corollary 9.1.** Correct (one pair `(a,b)`, linked to the mover; sign `−κ` = connector `j ∈ I`).
**Corollary 9.2.** Correct (multiplicities bounded by the coefficients of `λ − w_0λ`; the
computations of `ω_1 − w_0ω_1`, `ω_2 − w_0ω_2` for `sl_n` and `2e_1` for `so_{2n}` are right).

## Section 9A (Theorem C) — reading in progress

Checked so far: list of terms; well-definedness of `β(P_j), β(G^1_j), β(G^2_j)` (only `G^1_j`
has a same-colour crossing; (S2) holds); the case split (C1)–(C4) and that the three kinds of
targets are distinct; (C1); (C2) (equal reduced diagrams, opposite Koszul signs); (C3) for
`i = 1`, `i = 2` with `a_2 ∉ I'`, `i = 2` with `a_2 ∈ I'` (crossing lists and (S2) re-done;
coefficient computation for `m_{a_2}·G^2_j` re-done); (C4) "otherwise" case. No error so far.

Details of the re-derivation (§9A):
* *Exhaustiveness of (C1)–(C4).* Only `P_j`, `G^1_j` add `a_1`, only `m_{a_1}` removes it. First
  sum: `t' = m_j` with `j = a_1` (`a_1 ∈ I'`), `j = a_2` (`a_1 ∉ I'`), or `j ∉ {a_1,a_2}`,
  `a_1 ∉ I'`. Products (one factor of kind (γ), the other a term of `d'`): with `P_j` none; with
  `G^1_j` only `G^1_j·m_{a_1}`; with `G^2_j` and `m_{j'}`: `j' = a_1` (C2, both orders),
  `j' = a_2` (exactly one order, fixed by `a_2 ∈ I'` or not), `j' ∉ {a_1,a_2}` (C4; then
  `a_2 ∉ I'` is forced in both orders), `j' = j` impossible. For fixed `I'` the three kinds of
  `K'` are distinct, and the (C1) targets `I'∖a_i` receive no product. Complete.
* *(C3).* The unique reduced word of `m_j` crosses right slots `< j` (decreasing), red, left
  slots `< j` (increasing). If `a_2 ∈ I'`, `a_2 < j`, **both** `χ_1` (with `A_1`, left) and `χ_2`
  (with `A_2`, right) are bad, matched with `G^1_j·m_{a_1}` and `m_{a_2}·G^2_j`; the draft's case
  "`i = 1`" is stated for both positions of `a_2` and its (S2) argument covers the strand `j`
  crossing `A_2` through `χ_2` (which precedes `χ_1`). Crossing sets of `E_{χ_i}` and of the
  products re-derived in all three cases; they agree.
* *Coefficients.* `G^i_j·m_{a_i}` (`a_i ∉ I'`): Lemma 8.7 (`𝒯 = F' = ∅`, `a = a_i`) ✓;
  `m_{a_2}·G^2_j`: `∂_j s_{(a_2,m)} ∂_{a_2}(ω_{I'}∧e_m) = −κ'ω_{K'}` (the two signs from removing
  `e_{a_2}` and re-appending it cancel) ✓; (C2) and (C4)-two-term: opposite Koszul signs from
  `∂_x∂_y = −∂_y∂_x` and `s ∂_x = ∂_x s` ✓.

**Theorem C.** Correct and complete (given Theorem B for `w_{<b}`, Theorem A for the slots after
`b`, and Proposition 10.2 for the four-term case of (C4), whose hypotheses (H1), (H2) hold
there).

## Section 10

**Proposition 10.2 (four-term bigon identity).** Correct. Re-derived: in each `P_i` exactly one
pair crosses twice (`(j,A)`, `(M,j)`, `(M,j')`, `(j',A)` for `P_1, P_4, P_2, P_3`), all other
pairs at most once; no two strands of the same colour cross at all ((H1) for `A, M`, (H2) for
`j, j'`), so Lemma 1.3 applies with every hypothesis met, the dots of `Q` are those of `e_I`, and
the four formulas are right (the coefficient of `x_A` in `Q_{jA}` is `t_{ce}`). Signs: with `ε`
the sign of `−∂_{j'}∂_j s ω_I`, the four signs are `ε, −ε, ε, −ε` for `P_1, P_4, P_2, P_3`;
monomials `r_c t_{ce'}` (`P_1, P_4`), `r_c t_{ce}` (`P_2, P_3`); the sum vanishes.

*Gap (repairable, now filled): "These are the only products of two tree terms from `I` to `K`".*
The draft asserts that a product "moves `j, j'` to the left as movers" without argument. It is
true; here is a proof. Let the product be `(μ_1,F_1)·(μ_2,F_2)`. Pairs preserve the multiset of
colours on the right of the red strand and a mover removes one strand of its colour; the net
change from `I` to `K` removes colours `e, e'` (`j, j'`) and `c` (`b`) and adds `c` (`a`), so
`{c(μ_1), c(μ_2)} = {e, e'}`. If the mover of colour `e` were a slot `x ≠ j`, then `x > j` by
(H2), and `j ∈ I ∖ K` would have to leave the right side as the `b` of a pair `(z, j)` with
`z < j`, `c(z) = e` — excluded by (H2) (the other mover has colour `e' ≠ e`). Likewise the mover of
colour `e'` is `j'`. Neither `j` nor `j'` can be re-added by a pair of the second factor (it would
not be removed in total) nor removed by a pair (same reason, (H2)). The pairs therefore have the net
effect "remove `b`, add `a`, nothing else". A pair `(x,y)` of one factor can only be compensated by a
pair of the other factor sharing a slot with it, and chaining gives either the single pair `(a,b)`
in one factor, or `(a,y)`, `(y,b)` in the two factors (either order) with `c(y) = c`,
`a < y < b`, excluded by (H1). So the factors are `{m_j, (j';(a,b))}` or `{m_{j'}, (j;(a,b))}` in
either order: exactly `P_1, …, P_4`. (In Theorem C this claim is not needed, since the list
(C1)–(C4) is derived independently.)

**§10.1–10.6.** These are descriptions of data and of what remains; nothing in them is claimed as
proved beyond Proposition 10.2. Two remarks: (i) the table entry "Lemma 8.1, Corollary 8.2 hold for
every word in which `c` occurs once in `w'`" is right (the proof uses only that `A` is the only
strand of colour `c` in `w'`); (ii) in (∗) of 10.2 the bad letters with `γ = r_ku_k` (black of
colour `k` left of the red strand) never occur for tree diagrams, since no strand of a tree term
crosses the red strand from left to right — the text's description "crossings of any strand of
colour `c` standing to the left of a strand of adjacent colour" is therefore complete.

*Verdict after reading everything:* no fatal error, one under-argued step (Prop. 10.2 "only
products"; filled above), several cosmetic points.

## Independent checks (in progress)

### What the scripts are, and why I trust them

All scripts are in `proofs/checks/`, run with `sage -python` from there; logs of this pass are in
`proofs/checks/R05_logs/rerun/`. The `R05_*.py` scripts were written by an earlier referee (not by
the author); I read every one of them before using it. They do not import `05_lib.py`.

* `R05_lib.py`: an independent brute-force implementation of the tree formula from the text of
  `NOTES.md` §2 (`Setup.my_terms`: all sets of disjoint pairs, admissibility 1–3, connectors,
  `κ`, connector signs), random reduced words (`rand_reduced`), and a checker of (S1), (S2)
  (`check_S1_S2`) from their statement in Lemma 1.2. I checked `analyse` line by line against
  `formula.py`.
* `R05_basic.py`: (a) every relation with the README signs on all 2- and 3-strand states over
  `A_3` with one red strand; (b) `my_terms == formula.Stack.terms_from` on random arbitrary words.
* `R05_tools.py`: Lemmas 1.2, 1.3, 1.4 on random (mostly non-reduced) words on random states with
  the red strand anywhere; the Sliding Lemma 2.2 with the error words `E_t` and constants `γ_t`
  computed from the statement; Lemma 4.2 (`μμ^* = P e_0`, `μ^*μ = P e_1`, both sides `ι_L`/`ι_R`).
* `R05_d2.py`: `d² = 0` with `my_terms` and a fresh **random** reduced word for every use of a term.
* `R05_sec8.py`: for words `w'·c` with every colour ≤ 2 and `c` twice: every statement of Lemma 3.2,
  8.1, 8.3, 8.4 (data, families, the bijection 8.4.3 against a direct enumeration of all pairs
  `(t_1,t_2)`), 8.5 ((S1), (S2) of the concatenated words, permutation `σ_U`, product = element of
  a random reduced word of `σ_U`), 8.6 ((S1), (S2) of `E_t`, `ψ_{E_t} = B`, the slide formula),
  8.7 (the alternation `(−1)^{k−|S|}` and the case `𝒯 = ∅`), and `E_{I',K'} = 0` for all
  components, computed from the definition in §3.
* `R05_reduction.py`: Lemma 4.1 and Prop. 4.5 for `E(F+G)` with deliberately perturbed families
  (drop / sign-flip / `t_{xy} ↔ t_{yx}` of one (γ)-term, drop all of `F_1`, random elements,
  random elements concentrated on components with `a ∈ K'`, coboundaries); control with slots
  that are not first occurrences.
* `R05_redlin.py`: Prop. 4.5 as linear algebra: on an ansatz space of families (dots up to a
  degree × one reduced diagram per bijection), rank of `G ↦ E(G)` equals the rank of its
  restriction to the components with `a ∉ K'`, for every first-occurrence slot `a` (over
  `GF(32003)`; Prop. 4.5 holds over every base change by Theorem A.5(iv), so this is a valid
  test). For non-first slots the restricted rank is printed as a control.
* `R05_thmC.py`: Theorem C (full `d² = 0`, and (C1)–(C4) for the last step); Prop. 10.2 (the four
  products from the formulas, that they are the **only** products from `I` to `K`, and the sum).
* New in this pass: `R05b_d2.py` (`d² = 0` with the project's definition: `formula.Stack` and the
  klrw basis braid, via `sample_verify.check_vertex`; stress patterns for random words) and
  `R05b_sec8.py` (runs `R05_sec8.check_word` on stress words, in particular words producing data
  with `k = 2` components, which no earlier run contained).

### Results so far

| run | result |
|---|---|
| `R05_basic.py 80` | 186 relation checks exact; `my_terms == formula.Stack` on 80 random arbitrary words (22 221 terms) |
| `R05_tools.py 3000 21` (new seed) | Lemma 1.2: 1008 cases (462 non-reduced), 0 failures; hypotheses violated: 1992 words, of which 1917 differ from the reduced word (hypotheses not vacuous); Lemma 1.3: 530, Lemma 1.4: 534, Sliding Lemma: 3000 words with 1738 bad letters, Lemma 4.2: 3860 first-occurrence cases — **0 failures**; for 2140 non-first slots `μμ^* ≠ P` every time (the hypothesis matters) |
| `R05b_d2.py all A 3 2 6 6` | all 801 (word, red colour) cases of `A_3` with multiplicities ≤ 2 (this is every length, since length ≤ 6), full complexes, 219 546 products, **0 failures**; same product count as the old `R05_d2.py` run with random reduced words (log `d2_A3_all.log`, 0 failures) |
| `R05b_sec8.py D 4 60 7 11 3 k2` (words built so that data with **k = 2** components occur: two neighbour colours of `c` straddle `a`, a third neighbour after `a`) | 60 words: all assertions of Lemmas 3.2, 8.1, 8.3–8.7 and `E = 0` in all 19 680 components; data k = 0/1/2: 2880/960/**60**; products with 0/1/2 double crossings: 4438/556/**46** |
| `R05b_sec8.py D 4 40 8 12 3 k2` | 40 words, all assertions pass; data k = 0/1/2: 4576/1920/**194**; products with 2 double crossings: 168; 34 627 `E`-components, all 0 |
| `R05b_lemma43.py 1500 7` (new; Lemma 4.3 from its statement: for random `z` with random coefficients containing powers of `r`, the `r`-adic valuation of `zμ` (`μ = ι_L m_a`) resp. `μ'z` (`μ' = ι_R m_a`) equals that of `z`) | first-occurrence slot: 1966 cases, **0 failures**; control, slot not a first occurrence: valuation jumps in 449 of 1034 cases (the hypothesis is needed) |
