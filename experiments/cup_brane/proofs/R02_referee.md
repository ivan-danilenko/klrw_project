# R02 — Referee report on `02_omega1_ibrane.md` (the ω₁ I-brane of sl_{n+1})

Referee scripts: `proofs/checks/R02_*.py` (written from the text of `GLUING.md` §2 and of the
draft; nothing is imported from `02_*.py`). The draft itself was not edited.

## 1. Summary verdict

| claim | verdict |
|---|---|
| **Theorem A** (d² = 0 for every n, Laurent and polynomial model) | **Correct and complete.** Every step was checked line by line, including all entries and cancellations of the Clifford tables (Lemmas 5.3, 5.4), which are symbolic in the block length D and in l and therefore valid for every D. One presentation gap: items 5–6 of Lemma 4.3 are only summarised. They are true, and the computation is written out in §4.1 below. |
| **Theorem B** (homogeneity, vertex degrees, unique monomials, polynomial model) | **Correct**, with one repairable gap in the definitions (Issue 1). Cor. 6.4(c) is false under one literal reading of Def. 1.1 and true under the grading actually used (F2 of `PROOFS.md` and `klrw`). The minimality claim in Cor. 6.5 is only sketched. A complete proof is in §4.2. |
| **Theorem C** (rank-one Hom at every step; iterated cone) | **Correct and complete** as a statement about the cubes S_k with the shifts Δ. Lemma 7.1, Cor. 7.2, Thm. 7.3 (Step 1, and the "lonely path" Step 2 for d ≥ 2) and Prop. 7.4 survive line-by-line scrutiny. Caveats: (a) the coefficient ring of "free of rank one" must be stated (ℤ, the degree-0 part of R; ℚ in `standard_ebranes`), and this uses Issue 1; (b) the identification with the *output of `standard_ebranes`* relies on declared code conventions. It is now also confirmed directly for n ≤ 6 (new data point n = 6, §5). |
| **Theorem D** (resolution of a simple module) | **Correct over the Laurent ring R.** What is proved: H_i(E) = 0 for i ≠ 0 and H_0(E) = L, where e(s_B)L = R with dots acting by 0, and e(s)L = 0 otherwise. "Simple" is inaccurate over R, which is not a field (cosmetic; repair in Issue 4). Nothing is claimed or proved over R_pol. |

**No fatal error was found.** Every identity I could test independently (§5) holds, on ranges well
beyond the author's: n ≤ 10 for d² = 0, n ≤ 11 for the R-model comparison, D ≤ 12 for the block lemmas,
n ≤ 7 for the full-ansatz Hom. All mutations of the formula are detected, so the checks are not
vacuous.

## 2. Line-by-line verdicts

* **Def. 1.1 / (R1)–(R4).** These agree with `01_foundations.md` Def. A.1 and the README conventions.
  Two red strands never cross, so no relation involves both.
* **Lemma 1.4** (o_x = α_{x+1}γ_x): correct. I re-did the nine cases.
* **Lemma 2.1** (no bad triples): correct. A braid move needs three pairwise crossing strands, so at
  most one of them is red. Both bad cases need two black strands of one colour. For n = 1 there is
  only one black strand.
* **Lemma 2.2**: correct. A reduced word only crosses inverted pairs, so it never crosses W₁ with W_n.
* **Lemma 2.3** (product rule): **correct and complete.**
  * Case 1: the length goes up by one, so the concatenated word is reduced.
  * Case 2: ψ_i² on s'_1 is the bigon of {a,b}, in either order.
  * The bookkeeping of D: the first crossing of a reduced word s' → s'' inverts a pair of
    inv(s', s''), so in s'_1 that pair has its s''-order. Hence {a,b} ∉ D(s, s'_1, s''), and
    {a,b} ∈ D(s, s', s'') ⟺ Case 2.
  * Independently checked in `klrw` with random reduced words (§5, K1).
* **Lemma 2.5** (e(s)Ae(s') ≅ S·β): correct.
  * Spanning: dots slide freely, and Lemma 2.3 reduces any word to Q·β.
  * Independence: Pol_b. The module check in Remark 3.4 is complete: every factor depends only on the
    exchanged pair and its new order, and (A6) holds with C = 0.
  * There is a forward reference to Remark 3.4, but no circularity (3.4 uses Lemma 2.1, not 2.5).
  * The basis theorem is genuinely not needed.
* **Cor. 2.4**: correct (only the sufficiency direction is used).
* **Lemma 3.1** (cocycle): correct, and it is proved for all n. It is an edge-by-edge identity of
  indicator functions. I re-did all eight cases. Neither side ever contains an edge twice, so the
  multisets are sets.
* **Prop. 3.2**: correct. S = R[x] is a domain and Q_{D_b(v,w)} ≠ 0.
* **Remark 3.4**: correct. Pol_{o(s₀)} ≅ e(s₀)A because both are S-free on matching bases and the
  ψ-actions agree by Lemma 2.3.
* **Lemma 3.3** (bigon factors for b = 1): correct. I re-derived all six types from Lemma 1.4.
* **Def. 4.1 / Prop. 4.2** (R-model):
  * Monomials: correct. φ(v)/φ(w) gives q_l, 1, q_{c−1}, 1 for (i), (ii), (iii)_{c≤l} and
    (iii)_{c>k}, and q_c for int R.
  * Term shapes: correct ((iii) for d = 1 is P⁰_k).
  * The R-model has no entries without a term: checked, §5 F3.
* **Lemma 4.3** (signs):
  * Items 1–4: re-derived, correct.
  * Items 5–6: summarised in the draft. Re-derived in full in §4.1, correct.
  * ρ_c = "contract a_c, then wedge g_c" equals replacement in place up to the same parity, so the
    GLUING rule is unambiguous.
* **Lemma 5.2**: correct. The converse only needs T_{c−1}t_{c,c−1} to be a non-zero-divisor.
* **Theorem 5.1**: **correct for every block length.**
  * Case D = 1: re-derived.
  * Reduction to the block: re-derived. The decomposition E_{a,b} = (−1)^a(O + I_{a,b}). The three
    O-orders: O² = 0 and N even. [∂^out, O]₊ = −W + W_{[j,k]}. Shift invariance, because all signs
    are of the form (−1)^{a+c} or (−1)^{b−a}.
  * **Lemma 5.4**: each of the 13 table entries was recomputed with the Clifford rules, including the
    edge cases l = 1, l = D−1 and D = 2. Each listed cancellation was checked, and every product term
    is used exactly once. The remainder is W'_D ε₁ι_DΠ_{[2,D]}.
  * **Lemma 5.3**: same, all 8 entries and 4 cancellation families.
  * The tables are statements for symbolic D, l, so no "length ≤ 10" restriction is hidden.
  * Exact checks: D ≤ 12 and n ≤ 9 (§5).
* **Theorem 5.5 (A)**: correct.
* **Prop. 6.1, Thm. 6.3**: correct. I re-derived the E_x-coefficient of Δ(v) − Δ(w) − deg β per type.
* **Cor. 6.4**:
  * (a): correct.
  * (b): correct as a computation. The comparison with the shift of `standard_ebranes` is a code
    convention. It is declared, and confirmed for n ≤ 6 (shift = −Δ + const).
  * (c): see **Issue 1**.
* **Cor. 6.5**:
  * The closed forms of the rescaled coefficients are correct (τ_yπ_y = π_{y+1}).
  * Polynomiality: correct.
  * d² = 0 over R_pol: correct. It is the sufficiency direction of Cor. 2.4 over R_pol, or the
    injection R_pol[x] ⊂ R[x].
  * Minimality: only sketched. A proof is in §4.2.
* **§7.1**: correct as an identification up to vertex signs. The H¹-argument needs the two path
  products around each square to be equal non-zero elements; that holds by Lemma 2.3 and 2.5 but is
  left implicit (cosmetic). Identifying S_k with the output of `klrw.stable_envelope` is declared as
  an assumption.
* **Lemma 7.1**: **correct.**
  * Re-checked: the injection c ↦ c−1 from A_v∖A_w into F ∪ [l+1, k−1].
  * The three equality conditions.
  * The edge-by-edge analysis giving (3) ⟺ "no Z-colour C → B_R".
  * The hdeg count hdeg(w) − hdeg(v) = −1 + a − β + κ + z, also for d = 1.
  * The classification of the equality cases into (i), (ii) and (iii)_{c≤l}, (iii)_{c>k}.
* **Cor. 7.2**: **correct.**
  * Only the q-component is needed to bound the number of dots: |α| = hdeg(v) − hdeg(w) − d +
    |A_v| − |A_w| − |F| ≤ hdeg(v) − hdeg(w) − 1.
  * Dots have degree 2q + δ_c, and parameters have no q-part (F2, and `klrw/gradings.py`).
  * The uniqueness of the monomial needs Issue 1.
* **Theorem 7.3**: **correct and complete.**
  * Step 1(a): the two-path claim holds. Gluing terms move non-k colours only upwards and an internal
    move goes one colour downwards, so the internal move is on c, and the gluing term cannot involve
    c. Internal moves on a colour c ≠ k are the same in S_k and S_{k−1}, and connectivity holds.
  * Step 1(b): each e ≠ k contributes exactly one path. If e ∈ A or B_R, the path is (iii)_e and
    then the internal move on e. If e ∈ B_L or C, it is the internal move first and then (iii)_e.
    Together with (i) and (ii) this gives the bijection with the n+1 edges, and Lemma 5.2's converse
    (with ℤ ⊂ R coefficients) applies.
  * Step 2, the "lonely path": checked. Paths through intermediate envelopes do not enter, because the
    equation for h is ∂_j h + h∂_k = 0 after subtracting λΦ_{j,k}. "Internal move in S_k, then a
    length-D term" puts c in C, not B_R. Two different length-D terms out of one v have different
    targets.
  * Independently confirmed by full-ansatz linear algebra (n ≤ 7) and by `klrw`'s own Ext machinery
    (n ≤ 6), §5.
* **Prop. 7.4**: **correct.**
  * F(S_k) is, up to vertex signs, the Koszul complex on (Q_x)_{x≠k}.
  * Over R this family is a unitriangular change of the x's.
  * v₀(k) is the only vertex of minimal hdeg in S_k, and in S_{k−1}, so the induced map on H is ±id_R.
  * The projection C_{k−1} → S_{k−1} is a chain map.
* **Theorem 8.1 (D)**: **correct over R.**
  * The cube with a factor 1 in one direction is contractible.
  * The admissible b are exactly 0^z1^{n+1−z}.
  * The two-column spectral sequence has d₁ an isomorphism by Prop. 7.4.
  * 0^{n+1} is not realised, and o(s) = 1^{n+1} ⟺ s = s_B.
  * Re-checked numerically for all realised b, n ≤ 6 (§5 F7).
* **§9** (sl₂): signs re-derived with the referee implementation: B_L→A −1, (ii)₁ +1, (i)₁ +1, C→B_R +1.
  The bigons sit on edges 0 and 1, and both equal u₁x₁. Correct.
* **§10**:
  * The component count 2^{n−d−1}(n−d+4) is correct: 2·2^{n−d} terms of types (i) and (ii), plus
    (n−d)2^{n−d−1} of type (iii).
  * The count (n−1)2ⁿ+1 of the n+1-path components is correct (Σ_D (n−D+1)2^{n−D}).
  * The sentence on `PROOFS.md` "now Tier 1 for all n" needs the qualification of Issue 7.

## 3. Errors and gaps

No fatal errors.

1. **Repairable — Cor. 6.4(c), and through it Cor. 7.2 "c ∈ ℤ" and Thm. 7.3 "ℤ·Z".**
   * The problem:
     * Def. 1.1 says the grading is "(F2) with two symbols ε_{1W}, ε_{nW}", and §0 takes R = ℤ[t^{±1},
       u^{±1}, r^{±1}] without saying which u's.
     * `klrw`'s base ring (and Def. A.1 of `01_foundations.md`) contains u_c for **every** colour c.
     * The proof of independence in Cor. 6.4(c) only follows the path W₁ – 1 – … – n – W_n, so it
       ignores u_2, …, u_{n−1}.
   * Why it matters:
     * If those u_c are in R but only two ε_W symbols exist, then deg u_c = −δ_c.
     * For n ≥ 4 the monomial u_c t_{c+1,c} t_{c,c+1}^{−1} u_{c+1}^{−1} (2 ≤ c ≤ n−2) then has degree 0.
     * So the degree-0 part of R is no longer ℤ, the coefficient monomials are not unique, and "rank
       one over ℤ" fails: the Hom becomes rank one over the degree-0 subring.
   * Repair, either of:
     * (a) take R = ℤ[t_{x,x±1}^{±1}, u_1^{±1}, u_n^{±1}, r^{±1}];
     * (b) keep ε_{cW} for all c, as in F2 of `PROOFS.md` and as `klrw` does (verified:
       deg u_3 = 2ε_{W3–V3} − δ_3 in the n = 4 algebra). ε_{cW} occurs only in deg u_c, which forces
       the u_c-exponent to 0; then the draft's argument applies.

   With (b), everything matches what `standard_ebranes` computes.
2. **Repairable (presentation) — Lemma 4.3, items 5 and 6, are "summarised".** The statement is true.
   The full parity computation is in §4.1 and can be pasted in.
3. **Repairable — Cor. 6.5, minimality of f(k), "sketched".** A two-line proof is in §4.2, together
   with a computer check for n ≤ 9.
4. **Cosmetic — Theorem D, "simple".**
   * Over R = Laurent ring over ℤ, L ≅ R is free of rank one over R but not simple (2L ⊂ L).
   * Say instead "the module R concentrated at s_B, dots acting by 0".
   * It becomes simple after base change to a field k ⊇ R/𝔪 in which the parameters are invertible.
     The resolution property survives base change, because E is a bounded complex of R-free modules
     (Lemma 2.5) with R-free homology.
5. **Cosmetic — §0, Theorem C bullet.**
   * The draft says "the only homogeneous elements between different envelopes are the formula's
     dot-free terms".
   * That is true only in degree Δ(v) − Δ(w) and for an hdeg drop of 0 or 1 (Cor. 7.2); the sentence
     should say so.
   * Also state the ring of "free of rank one": ℤ, or ℚ in `standard_ebranes`.
6. **Cosmetic — numbering and order.**
   * Lemma 2.5 precedes Cor. 2.4.
   * Remark 3.4 precedes Lemma 3.3 and is used in Lemma 2.5 (a forward reference; not circular).
   * Lemma 5.2 precedes Theorem 5.1, and 5.4 is proved before 5.3.
   * Renumber before formalisation.
7. **Cosmetic — §10 vs. Status.** "= iterated cone (`PROOFS.md` §3) is now Tier 1 for all n" holds for
   the mathematically defined iterated cone of the cubes with shifts Δ. For "the output of
   `standard_ebranes`" it is Tier 1 only relative to the declared reading of the code (envelopes,
   shift sign, cone convention). The Status section says this; §10 should too.
8. **Cosmetic — implicit steps.**
   * §7.1: the H¹-argument needs equal non-zero path products around each square.
   * Prop. 7.4: "is the Koszul complex" means up to vertex signs.
   * Both are true, but should be stated.

**Remark, a strengthening.** The proof of Theorem C does not need the equivariant gradings:
* Cor. 7.2's dot count uses only the q-component.
* Steps 1 and 2 work with c_t ∈ R: R is a domain, and Lemma 5.2's converse holds over R.

So with the q-grading alone, Hom_K(S_k, C_{k−1}) in degree 0 is free of rank one over R (resp. over
R_pol in the polynomial model). This is checked in §5 F8b.

## 4. Filled-in arguments

### 4.1 Lemma 4.3, items 5 and 6

Notation of the draft:
* a₀ = |A₀|, b₀ = |B₀|, l = a₀ + b₀;
* γ = |C_v|;
* E_G, E_R, σσ = exponents mod 2 of the GLUING sign, the R-model sign and σ(v)σ(w);
* "cost" = number of generators in front.

Identities used: C(d−1,2) + C(d+1,2) ≡ 1, (d−2)(l+k) ≡ d, and Σ_{j=2}^{d−1} j = C(d,2) − 1.

*Item 5, d ≥ 2, extra c ≤ l.* Here c ∈ A₀, and w has A_w = A₀∖c, B_L(w) = B₀ ∪ c,
C_w = C ∪ [l+1, k−1].
* GLUING costs:
  * ι(a_{l+1}): l.
  * ρ_{l+j} (2 ≤ j ≤ d−1): l + k − j + γ.
  * ι(a_k): l.
  * ε(g_c): l + (d−1) + γ + |B₀|_{>c}.
  * Prefactor: l + 1 − c + d.
  * Total: E_G ≡ d + (d+1)γ + C(d,2) + 1 + c + |B₀|_{>c}.
* R-model costs:
  * ε_{l+1}: a₀.
  * ι_k: a₀ + d − 1.
  * ι_c: |A₀|_{<c}.
  * Prefactor: l − c.
  * Total: E_R ≡ d + 1 + l + c + |A₀|_{<c}.
* σσ ≡ (a₀−1)d + C(d,2) + γ(d−1) + C(d−1,2) + dl + C(d+1,2) + l + 1 + c + (b₀+1)(d−1).
* Result: E_G + σσ − E_R ≡ |B₀|_{>c} + b₀ + 1 + c + |A₀|_{<c} ≡ |B₀|_{<c} + |A₀|_{<c} + 1 + c
  = (c − 1) + 1 + c ≡ 0.

*Item 5, d ≥ 2, extra c > k.* Here c ∈ B_R(v), and C_w = C ∪ [l+1, k−1] ∪ c.
* GLUING: the core costs d + dγ + C(d,2) + 1, then ε(g_c): l + |C_{>c}|, and the prefactor is
  c − k = c − l − d. Total: E_G ≡ dγ + C(d,2) + 1 + |C_{>c}| + c.
* R-model: ε_{l+1}: a₀, then ι_k: a₀ + d − 1, then ε_c: a₀ + d − 1 + |C_{<c}|, and the prefactor is
  1 + c − l. Total: E_R ≡ a₀ + |C_{<c}| + 1 + c + l.
* σσ: the same as in item 4.
* Result: E_G + σσ ≡ |C_{>c}| + c + a₀ + 1 + l + γ, so the difference is γ + |C_{<c}| + |C_{>c}| ≡ 0
  (because c ∉ C_v).

*Item 6, d = 1.*
* c ≤ l:
  * GLUING: ι(a_k): l, then ι(g_k): l + γ, then ε(g_c): l + γ + |B₀|_{>c}, prefactor l + 2 − c. Total:
    E_G ≡ |B₀|_{>c} + c.
  * R-model: P⁰_k (sign +1), then ι_c: |A₀|_{<c}, prefactor l − c.
  * σσ ≡ a₀ − 1 + c.
  * Difference: |B₀|_{>c} + a₀ + 1 + |A₀|_{<c} + l + c ≡ |B₀|_{<c} + |A₀|_{<c} + 1 + c ≡ 0.
* c > k:
  * GLUING: E_G ≡ γ + |C_{>c}| + c + 1.
  * R-model: E_R ≡ a₀ + |C_{<c}| + 1 + c + l.
  * σσ ≡ b₀.
  * Difference: γ + |C_{<c}| + |C_{>c}| + a₀ + b₀ + l ≡ 0.

These are exactly the sign identities checked exhaustively for n ≤ 11 by `R02_formula.py rmodel`.

### 4.2 Cor. 6.5, minimality for n ≥ 2

Fix a parameter p. Feasible rescalings are exactly the exponent vectors e(·) with e(0) = 0 and
e(k) ≥ e(l) − m_t(p) for every gluing term t: S_k → S_l, where m_t(p) is the exponent of p in μ_t.
The given f satisfies these. Every exponent of the rescaled coefficients is ≥ 0.

*Minimality, by induction on k.*
* Every π_y, 1 ≤ y ≤ n+1, occurs as the rescaled coefficient of a d = 1 term out of S_k, for every
  k ≥ 1:
  * (iii)₁ with c ≤ k−1 gives π_c;
  * (i)₁ gives π_k;
  * (ii)₁ gives π_{k+1};
  * (iii)₁ with c > k gives π_{c+1}.
* For n ≥ 2, every p is missing from some π_y: t_{x,x+1} ∉ π_{x+1}, and t_{x,x−1} ∉ π_x.
* Take that term t: S_k → S_{k−1}. Then m_t(p) + f(k)_p − f(k−1)_p = [p ∈ π_y] = 0.
* So e(k) ≥ e(k−1) − m_t(p) ≥ f(k−1)_p − m_t(p) = f(k)_p, by induction.

For n = 1, u₁ lies in every π, and the least rescaling is trivial, as the draft says.

Checked (longest-path recursion): `R02_formula.py minimal`, n ≤ 9.

## 5. Independent checks

All scripts are new. "own formula" = `R02_formula.py`, implemented from the text of `GLUING.md` §2:
vertices, the term rule, the coefficients and the exterior-algebra signs (with an assertion that
each operator lands on the target monomial).

| # | what | method | range | result |
|---|---|---|---|---|
| F1 | d² = 0 via Cor. 2.4 (bigon rule) | own formula; random specialisation of **all** t, u and x mod p = 2⁶¹−1, 2 points; Laurent coefficients | n ≤ 10 (1 032 194 components at n = 10) | 0 non-zero |
| F1′ | same, exact | random rational points, Fractions | n ≤ 7 | 0 non-zero |
| F2 | Cor. 6.5 closed forms | f(k)/f(l)·μ_t vs. the stated π-products; signs of exponents | n ≤ 10 | all equal, all ≥ 0 |
| F3 | Prop. 4.2 + Lemma 4.3 | every term: GLUING sign·σσ, Q-factor, monomial·φ(v)/φ(w) vs. the R-model entry from an own Clifford implementation; no R-model entry without a term | n ≤ 11 (270 336 terms) | 0 differences, 0 extra |
| F4 | Theorem 5.1, δ² = W·H | free q₀…q_n, exact over ℤ[q] | n ≤ 9 (188 417 components) | 0 differences |
| F5 | Lemmas 5.3, 5.4 | operators on Λ(r₁…r_D), free q, exact | D ≤ 12 | 0 failures |
| F6 | Thm. 6.3, Def. 6.2, Lemma 7.1, Cor. 7.2 | own F2 grading (q, ε_x, δ_c, σ_c); vertex degrees propagated along internal and (i)₁ terms; for all pairs v ∈ S_k, w ∈ S_l (l < k) with hdeg drop 0 or 1, all (α, μ) found by solving the degree equations | n ≤ 7 | all terms homogeneous; Δ = Def. 6.2 + const; drop 0: q-count gives at most −1 dot, no elements; drop 1: the q-count alone excludes every non-term pair, and each term pair has exactly one element (dot-free) |
| F7 | Theorem 8.1 | K_b for **every** realised b (2^{n+1}−1 of them, so every state), mod p, at x = 0 and at a random x | n ≤ 6 | exact everywhere, except b = 1^{n+1} at x = 0: binom(n, i) |
| F8 | Theorem 7.3, full ansatz | unknowns = all homogeneous x^α μ β (any |α|, found by F6) S_k → S_l; homotopy unknowns likewise; equations δ_C g + g d_S in K₁ at n+3 random points mod p, which gives an upper bound for the nullity | n ≤ 7, all k | nullity 1, 0 homotopy candidates |
| F8b | same, coefficients in Frac(R) | one generic parameter point, only x varies | n ≤ 6 | nullity 1 (the remark in §3) |
| F9 | Cor. 6.5 minimality | longest-path recursion | n ≤ 9 | f is minimal for n ≥ 2 |
| M | sensitivity | mutations: sign of (i)_{d=2}; sign of (iii)_{c>k}; inverted coefficient of (iii)_{c≤l}; dropping (iii)_{c=k+1}; σ without \|B_L\|\|C\|; R-model sign of the d = 2 terms | n = 4 | every mutation detected (63–105 failing components) |
| K1 | Lemmas 2.2, 2.3 in `klrw` | own **random** reduced words; β(s,s′)β(s′,s″) vs. Q_D·β(s,s″) built from a different word; and two words for one permutation | n ≤ 6, all 27 / 300 random triples per n | 0 mismatches |
| K2 | polynomial model in the actual algebra, **non-inverted parameters** | own formula with π-coefficients; d² by multiplication in `klrw`; homogeneity in `klrw`'s grading; vertex degrees = Def. 6.2 + deg f(k) + const, using `klrw.crossing_grading` | n ≤ 7 (7168 terms, 22 273 components) | d² = 0, homogeneous, offset constant |
| K3 | Theorem 7.3 by honest linear algebra over ℚ in `klrw` (polynomial model) | candidates x^α μ β for every pair v ∈ S_k, w ∈ S_l (l < k), hdeg drop 0 or 1, every α with \|α\| ≤ 3, μ any polynomial monomial in t, u, r solving `klrw`'s degree equations; chain-map equations by multiplication in `klrw` | n ≤ 5, all k | dim cycles 1, boundaries 0 (no homotopy candidates), Hom rank 1, generated by the formula's map; no candidate with dots |
| G1 | the real gluing (`glue/glue.py` = `standard_ebranes` with `klrw`'s Hom/Ext), run by `R02_glue_run.py` into the scratchpad | compared by `R02_compare_glue.py` with the own formula | n = 1…5 (repository dumps) and **n = 6 (new)** | Ext = 1 at every step; the graded Hom⁰ has exactly the formula's terms (144, 208, 236, 248, 253, 255 at n = 6); boundaries 0; terms, signs (gauge (−1)^{k(k−1)/2+l(l−1)/2}), coefficients identical; no dots; reduced words; equivariant shift = −Δ + const |

A run for n = 7 (sl₈) was started but had not finished step 1 after 15 minutes (up to 8.6 GB of
memory). It was stopped, so G1 covers n ≤ 6 only.

**Comparison with the author's checks.** The author's 02_unique works on the formula's support (it
presupposes Cor. 7.2). F8 and K3 do not presuppose it: they find the support themselves, with dots
allowed. K3 does so with `klrw`'s degree function, and G1 with `klrw`'s ambient Hom. The author's
02_homology samples states for n = 5. F7 covers every b, hence every state, for n ≤ 6.

## 6. Dependence on unproved input

| input | used for | declared? | comment |
|---|---|---|---|
| basis theorem for two red strands | — | yes ("not needed") | correct: Lemma 2.5 proves the distinct-label case from the relations via Pol_b |
| Matsumoto (F3) | Lemma 2.2 | yes | standard |
| `klrw` implements Def. 1.1 with two red strands | only the comparison with code outputs | yes | Theorems A–D are about the presented algebra and do not need it. Evidence: 02_lemmaA (n ≤ 5), K1 (n ≤ 6), K2 (n ≤ 7) |
| envelopes of `standard_ebranes` = cubes S_k (`stable_envelope` for ω₁ is the uncorrected product of two cubes, with coefficients ±1) | Thm. C "= output of `standard_ebranes`" | yes (Status) | confirmed n ≤ 6 (G1) |
| shift of `standard_ebranes` (sign of the equivariant shift; homological shift −len(lseq)+1) | Cor. 6.4(b), Thm. C | yes ("fixed by the data, n ≤ 5") | a code convention, independent of n; confirmed n ≤ 6 |
| cone and sign convention of `klrw` (Cone(S_k[shift] → E)), Ext computed over ℚ in the full `klrw` multigrading | Thm. C | **only implicitly** | covered by "the iterated cone of `standard_ebranes` are the objects of §7.1"; should be named. The multigrading of `klrw` contains ε_{cW} for every c, which is what makes Issue 1 harmless there |
| `PROOFS.md` Thm. 5 / Prop. 6 (the product of the two cubes is a complex) | §7.1 | yes | superseded by Theorem A |

## 7. Files

* `proofs/checks/R02_formula.py`: pure Python, checks F1–F9 (modes `d2`, `d2exact`, `rmodel`, `WH`,
  `block`, `grading`, `homology`, `uniq`, `uniqR`, `minimal`).
* `proofs/checks/R02_mutations.py`: check M.
* `proofs/checks/R02_klrw.py`: *sage*, checks K1–K3 (modes `product`, `d2poly`, `hom`).
* `proofs/checks/R02_glue_run.py`: *sage*, runs `glue.run` with an explicit output path outside the
  repository.
* `proofs/checks/R02_compare_glue.py`: check G1.
