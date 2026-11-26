# R05 — Referee report on `05_main_sliding.md`

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

(Sections 5–10: in progress.)
