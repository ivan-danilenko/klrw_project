# R01 — Referee report on `01_foundations.md`

Referee: R01. Read: README_FOR_PROVERS, `PROOFS.md` §1–2, `NOTES.md` §2, `formula.py`,
`01_foundations.md` (all, line by line), `03` §1.3–2.4 (Thm. 2.8, Cor. 2.10), `06` §1–4
(Lemma 2.3), `05` §3–4 and the uses of (F1)/E.1 in 04–07; the arXiv sources of Webster
(wKLR v4, KIHRT), Rouquier (0812.5023), Khovanov–Lauda I, Kang–Kashiwara–Kim. All checks
below are my own scripts `proofs/checks/R01_*.py` (they import only `alg.py`; no `01_*.py`).

---

## 0. Summary

**No fatal error.** The deep input — Theorem A.5, the basis theorem over `ℤ[r,t,u]` with
polynomial parameters, both sides, all base changes — is **correct, and its proof is complete
up to one step that is a computer calculation** (the cubic relation (A6) in the polynomial
representation, Prop. A.3); I supply the hand computation below (§1.3), so this gap is closed.
I also give a second, independent proof of the linear independence via Rouquier's Theorem 3.7
with an explicit dictionary of signs (§1.5). Theorem E.1, Lemma D.1, Theorem D.3, Cor. D.4,
Prop. D.5, Theorem D.6, Lemmas B.2, B.3, Cor. B.4, Prop. B.5, Theorem B.6, Lemma B.8, §C and
the review of Tier 1 (§F) are correct. There is no circularity between A.5, E.1 and §D, and in
fact §D can be proved without (F1) at all (§2.2).

Problems found (numbered as in §6): **no fatal ones; 3 repairable** (R1 status of "A.1 is what
klrw implements"; R2 the polynomial-representation computation written out; R3 algebras without a
red strand), **6 cosmetic** (C1–C6: Dedekind vs. Artin, side of `y` in pattern S, literature
details incl. the existence of Webster's Prop. 2.23, KL over ℤ, KKK item numbering, an
overkill use of E.1). None of them affects any result of the other fronts (§6, last column).

---

## 1. §A — the presentation and the basis theorem

### 1.1 Definition A.1 — verdict: correct as a definition; "is what klrw implements" is Tier 2

The relations (A0)–(A6), their signs and the grading agree with the README, with
`PROOFS.md` §1.1 and with the implementation, and every relation is homogeneous for F2. The signs
are consistent with the anti-involution `ι` (the two equal-colour equations of (A4) have the same
right-hand side `+r_c`, so reversing words exchanges them; `C(a,b,a') = C(a',b,a)`).

What has to be true for "A.1 is the algebra of `klrw`" is: (a) in `klrw` the generators satisfy
(A0)–(A6) on **every** state, (b) the `klrw` product is associative, (c) for every basis braid of
`klrw` the letter-by-letter product of the generators along its word is that basis element (no
lower terms). Then `A → klrw` is a homomorphism which maps the spanning set `{x^α ψ_w}` of
A.5(i) onto the `klrw` basis, hence an isomorphism. The draft checks (a) on 2- and 3-strand
states and argues that this covers all colour patterns "because the relations depend only on…".
That is true of the relations of `A` (they are local), **not** of the implementation, whose
multiplication (`right_multiply_by_s`) searches globally along the braid for intersections. So
(a) on bigger states, (b), (c) are separate, empirical statements. I tested them (own code,
`checks/R01_klrw.py`, results §7): (c) for all 40 200 basis braids of seven 4–6-strand
configurations (and they are exactly the Lehmer-cocode words of Def. B.10); (a) for 8 064
instances of (A2)–(A6) at every position of 5–6-strand states, placed behind a random dotted
braid and multiplied letter by letter; (b) for 175 random triples; and the homomorphism property
`ρ(ab) = ρ(b)∘ρ(a)` for **my own** polynomial representation on 4–6-strand states with the red
strand anywhere (560 instances; negative controls with a wrong sign or exchanged `t`'s are
detected, 30/90 and 24/90 failures). Grading: the `klrw` degrees of `x, ψ, r, t, u` are exactly
the table F2, and 120 random products are homogeneous of the sum degree. **No failure.**
Verdict: the identification is well supported, but it belongs in "verified by computation"
(problem R1), not under "Proved completely" as in §S of the draft.

### 1.2 Definition A.2 / Proposition A.3 (polynomial representation) — verdict: correct; the (A6) part was only computed (repairable, filled in §1.3)

(A0)–(A3), (A4), (A5): the draft's hand verification is correct (I redid each line; the right
action with "first `a`, then `b`" is consistent, and `Q_{ab}(u,v) = Q_{ba}(v,u)` is exactly what
(A5) needs for `a > b`, including `(k,W)`/`(W,k)` where `Q_{kW}(u,v) = u_k u`).
The reduction for (A6) — pairwise different colours give one monomial operator `(∏ transported Q)·w_0`
on both sides — is correct (each pair of strands crosses once, with the same strand on the left,
in both words). The remaining patterns were only expanded by computer (`01_polyrep.py twisted`).
I re-did that computation with my own code in a stronger form (`R01_twisted.py`: `Γ = D_4`, which
has a trivalent node, red `k = 1, 2`, all 2- and 3-strand states, **three different total orders on
the colours** — natural, reversed, and `W` first — exact identities in `K ⋊ S_n`): **6 384
identities, 0 failures**; the non-zero cubic constants are exactly `C(c,c',c) = r_c t_{cc'}` and
`C(k,W,k) = r_k u_k`. Note the representation does not depend on the order chosen (so "W last"
is not needed). The hand proof:

### 1.3 Hand proof of (A6) in the polynomial representation (fills the gap of Prop. A.3)

Write `f(y_1,y_2,y_3)` for `f ∈ V_s`, `s_i f` for the exchange of the arguments `i, i+1`,
and recall `f·ψ_i = r_c (f − s_i f)/(y_i − y_{i+1})` (equal colours), `= s_i f` (`s_i < s_{i+1}`),
`= Q_{t_i t_{i+1}}(y_i, y_{i+1})·s_i f` (`s_i > s_{i+1}`, `t` the target).

* `(c,c,c)`: both sides are `r_c^3 ∂_1∂_2∂_1 f = r_c^3 ∂_2∂_1∂_2 f` (nil-Hecke braid relation), `C = 0`.
* `(c,b,c)`, `b ≠ c` black or red, `b < c`. Then
  `f·ψ_1ψ_2ψ_1 = r_c [Q_{bc}(y_2,y_1) f − Q_{bc}(y_2,y_3) f(y_3,y_2,y_1)]/(y_1 − y_3)` and
  `f·ψ_2ψ_1ψ_2 = r_c Q_{bc}(y_2,y_3) [f − f(y_3,y_2,y_1)]/(y_1 − y_3)`, so the difference is
  `r_c (Q_{bc}(y_2,y_1) − Q_{bc}(y_2,y_3))/(y_1 − y_3) · f = r_c t_{cb} f` (adjacent), `0` (`Q = 1`).
* `(c,b,c)`, `b > c` (always the case for `b = W`): the difference is
  `r_c (Q_{cb}(y_1,y_2) − Q_{cb}(y_3,y_2))/(y_1 − y_3) · f`, which is `r_c t_{cb} f` (adjacent),
  `r_k u_k f` (`b = W`, `c = k`, `Q_{kW}(u,v) = u_k u`), `0` otherwise.
* `(c,c,b)`, `b > c`: both sides equal `r_c [f(y_2,y_3,y_1) − f(y_3,y_2,y_1)]/(y_2 − y_3)`.
  For `b < c` both sides equal `Q_{bc}(y_1,y_2) Q_{bc}(y_1,y_3) · r_c [f(y_2,y_3,y_1) − f(y_3,y_2,y_1)]/(y_2 − y_3)`
  (on the right the factor `Q_{bc}(y_1,y_2)Q_{bc}(y_1,y_3)` is symmetric in `y_2, y_3` and passes
  through `∂_2`).
* `(b,c,c)`: mirror image of the previous case (same computation with the roles of positions 1 and
  3 exchanged).

(Each line is a two-step expansion of the definitions; all of them agree with `R01_twisted.py`.)
With this, Prop. A.3 is proved by hand.

### 1.4 Theorem A.5 — verdict: correct and complete (given §1.3)

I checked every step.

* **(i) Spanning.** Dots to the left: (A3), (A4) with error terms `± r_c e` (fewer crossings);
  reduced words: Matsumoto, (A2) exact, (A6) changes by `C·ψ_uψ_v` with `C ∈ {0, r t, r u}`, a word
  of `l − 3` letters; non-reduced words: exchange property (`ψ_{D_0} ≡ ψ_Eψ_i`), `ψ_iψ_i = Q`, and
  `A_{≤l'}·ψ_i ⊆ A_{≤ l'+1}`. The induction is sound and gives the refined statement
  "products with ≤ l crossings span exactly `span{x^αψ_w : ℓ(w) ≤ l}`", for **any** choice of
  reduced words. Nothing in it needs invertible parameters; the error terms `r t`, `r u` and the
  dot–crossing terms `± r_c` are all of lower crossing number.
* **(ii) Independence over `R = ℤ[r,t,u]` itself.** The parameters are inverted only inside the
  field `K = Frac R[y]` used to *test* an element; the conclusion is about coefficients in `R`. The
  leading coefficient of one letter is `−r_c/(y_i − y_{i+1})`, `1` or `Q ≠ 0` — non-zero because
  `R` is a domain and `r_c, t, u ≠ 0` in `R`, not because they are units. The leading-term argument
  (permutation `ŵ` of length `ℓ(w)`, all other terms shorter, distinct `w` give distinct `ŵ`
  because the red position is fixed) is correct. One wording point (C1): the operator is only known
  to vanish on `R[y]`, not on `K`, so "Dedekind's lemma for automorphisms of `K`" should be Artin's
  lemma for the distinct characters `w|: R[y]∖0 → K^×` of the multiplicative monoid; the conclusion
  is the same.
* **(iii) Right basis via the flip.** It does need an anti-automorphism, and `ι` exists for these
  signs: I checked every relation (A0)–(A6) under `ι(ab) = ι(b)ι(a)`, `ι(ψ_{i,s}) = ψ_{i,s_is}`;
  the two equal-colour equations of (A4) are exchanged (both have right-hand side `+r_c`), (A2) at
  `s` goes to (A2) at `s_is_js`, (A6) on `s` goes to (A6) on `w_0 s`. (With the other common sign
  convention `ψx_i − x_{i+1}ψ = −r` one would need `ι(ψ) = −ψ`.) Alternatively (iii) follows from
  (i), (ii) without `ι`: by Lemma D.2, `ψ_w x^β = x^{w(β)}ψ_w + (shorter)`, a unitriangular change of
  basis in each (permutation-length, degree) block.
* **(iv) Base change, `r = 0` included.** Correct: `A` is a free `R`-module (a direct sum, so `⊗`
  commutes), `A ⊗ S` has the same presentation (right exactness), and `e(s)Ae(s') ≅ ⊕_w P(s)ψ_w`
  stays an isomorphism after `⊗_R S`. **The statement is true for `r = 0`** (and for `t = 0`,
  `u = 0`, `ℤ/N`, …). It cannot be obtained from the polynomial representation there, which stops
  being faithful (my `R01_rank.py` part (4): at `r_1 = 0`, at `t_{12} = t_{21} = 0` and at `u_1 = 0`
  the rank of the claimed basis in `V` drops); the draft correctly does not claim (v) over `S`.
* **(v) Faithfulness of `V` over `R`.** Correct (same argument).

### 1.5 Literature (checked in the arXiv sources) and an independent second proof of A.5(ii)

* **Webster, Weighted KLR algebras, arXiv:1209.2463v4.** Standing assumption (§2): "*We will always
  assume that `Q_e` has coefficients before the pure monomials in `u` and `v` which are units*" —
  confirmed. Prop. 2.7 = polynomial representation, Thm. 2.8 = basis — confirmed (numbering of v4,
  computed from the source). **But there is a Proposition 2.23 in v4**: "*This deformation is free
  (and thus flat) over `S`*", about the *canonical deformation* `Q̃_e = Q_e + Σ z_{e,a,b} u^a v^b`,
  `S = 𝕂[z]`. So the `PROOFS.md` citation "flatness of the deformation, Prop. 2.23" points to a real
  statement; it is just not about our deformation (ours scales the nil-Hecke relation by `r` and the
  leading coefficients by non-units `t`, `u`, and keeps the unit assumption out). The draft's
  sentence "there is no such statement in that role" should be corrected to this (C3). §3.1 is the
  Crawley–Boevey trick and the reduced quotient (Thm. 3.12: `T̃^λ` is the reduced quotient) — as cited.
* **Rouquier, 2-Kac–Moody algebras, arXiv:0812.5023, Theorem 3.7** (computed number; label
  `th:PBWHecke`): for any commutative ring `k` and any `Q` with `Q_ii = 0`, `H_n(Q)` is free with
  basis `S = {τ_w x^a}` iff `Q_{ij}(u,v) = Q_{ji}(v,u)`. **No unit assumption** — confirmed.
  *Dictionary* (I checked it relation by relation): over `R' = R[r^{-1}]` put `τ = −r_c^{-1}ψ` on
  equal-colour crossings and `τ = ψ` otherwise; add a vertex `W` with `Q_{kW}(u,v) = u_k u + v`,
  `Q_{cW} = 1` (`c ≠ k`). Then (A4) becomes Rouquier's `τx_i − x_{i+1}τ = −1`, `τx_{i+1} − x_iτ = 1`
  (his products are compositions), and (A6) on `(c,b,c)` becomes
  `τ_2τ_1τ_2 − τ_1τ_2τ_1 = (Q_{cb}(x_3,x_2) − Q_{cb}(x_1,x_2))/(x_3 − x_1) = t_{cb}` resp. `u_k`, exactly
  his cubic relation (one equal-colour letter per word, so one factor `−r_c`). Killing the dot `y` of
  the single `W` strand (it slides through every crossing, so `(y)` is spanned by the `S`-elements
  with a positive `y`-exponent) gives `A ⊗ R'` with basis `{x^αψ_w}`. Since `R → R'` is injective
  and A.5(i) holds over `R`, **this is a second proof of A.5(ii) over `R`**, independent of Prop. A.3.
* **Khovanov–Lauda I, arXiv:0803.4121**: Prop. 2.3 (action on `Pol_ν`), Thm. 2.5 (basis), Prop. 2.7
  (free on both sides) — confirmed, but Thm. 2.5 is stated over `ℤ` ("free graded abelian group"),
  not "over a field" as the draft says (C4).
* **Kang–Kashiwara–Kim, arXiv:1304.0323 (Invent. Math. 211)**, §1.3.1, Lemma 1.3.1: six items
  (i)–(vi), not five; `φ_a = (x_{a+1} − x_a)τ_a − 1` — confirmed. Item (v)
  (`φ_wτ_k = τ_{w(k)}φ_w` if `w(k+1) = w(k)+1`) is the analogue of Prop. D.5 / Theorem D.6 (C5).
* **Webster, Knot invariants and higher representation theory, arXiv:1309.3796**, Prop. 4.16 ("The
  set B is a basis of `T̃`"): the unreduced tensor-product algebra, undeformed nil-Hecke part; it
  does not cover polynomial `r, t, u` either.

Conclusion: the draft's decision to prove A.5 directly is right, and its use of the literature as a
cross-check is legitimate (now with an exact dictionary for Rouquier).

---

## 2. §E and §D

### 2.1 Theorem E.1 — verdict: correct and complete

(a) from the left basis over `P_S(s)`, (b) from the right basis over `P_S(s')`, (c) composition.
The remarks are right: (a) and (b) are about different variables; over `S = R/(r_c)` etc. `P_S` is
still a polynomial ring over a domain; `x_p − x_q` is a non-zero-divisor in `S[x]` for **every**
`S` (it is monic in `x_p`), so for this element E.1 holds over every base change.

### 2.2 Intertwiners — verdicts

* **Lemma D.1** correct (each line re-derived; `φ(x−x') = (x'−x)φ`, `φψ = −rψ`).
* **Lemma D.2** correct.
* **Theorem D.3** correct. The (F1) argument is valid: `Δ ∈ A_{≤2}` (the three transported
  differences give the same `P` in both words, I recomputed them), `fΔ = Δf^{w_0}`, leading
  coefficient comparison in the **left** basis (it needs the refined spanning of A.5(i) for the
  "shorter" terms, which is available), and `v^{-1}w_0` is a non-trivial colour-preserving
  permutation, which moves a black position. The "direct computation for the record" is also
  correct (I redid the cases `(c,b,c)` with `b` adjacent/red/other and `(c,c,b)`, `(b,c,c)`).
  The reduction "we may assume `n = 3`" passes through an algebra that may have **no red strand**,
  which is not an instance of Def. A.1 (R3); the fix is to run the same argument in the big algebra
  (it only uses positions `i, i+1, i+2`), or to note that A.5 holds verbatim without red strand.
* **Comparison with `06_main_intertwiner.md` Lemma 2.3.** I checked its case 5 (`(c,c,c)`) line by
  line (`ψ_2a_1 = bψ_2 − r`, `ψ_1 b = a_2ψ_1 + r`, `ψ_1a_2 = bψ_1 − r`, `a_2 b` symmetric in
  `x_1, x_2`): it proves both `φ_1φ_2φ_1 = φ_2φ_1φ_2` and `ψ_1φ_2φ_1 = φ_2φ_1ψ_2` on `(c,c,c)` from the
  relations alone. With 01's direct computation for the other patterns this gives **D.3 without
  (F1)**, over every base ring.
* **Prop. D.5** correct (the colour sequences `(A,B,S)` and `(B,A,S)` coincide, so the same `φ_2φ_1`
  occurs on both sides; then `D_1·(ψ_1Φ − Φψ_2) = 0` and E.1(a)). Two remarks (C6): if `c(S) ≠ c` the
  identity *is* the exact braid relation on `(c,c,c')`, no E.1 needed; and the second identity follows
  from the first by the flip `ι`, since `ι(φ) = ψ(x_i − x_{i+1}) − r = −φ` (Lemma D.1(a)) and the two
  signs cancel. Together with 06 Lemma 2.3, **D.5 holds without (F1)/E.1**.
* **Theorem D.6** correct (multiplicativity reduces to generators; dots by D.1(b); a crossing `ψ_i`
  commutes with `φ_n…φ_{i+2}` and `φ_{i−1}…φ_1` by (A2)/(A3) and passes `φ_{i+1}φ_i` by D.5; the
  `φ`-type of the commuted factors is unchanged because their positions are untouched). The
  remark D.7 is right with the sign conventions of the README: the cone needs
  `(d'⊗1)F = F(1⊗d')` when `C_R` carries `−d'`, which is exactly the form of D.6.
* **No circularity.** A.5 uses only Prop. A.3 (or Rouquier) and Matsumoto; E.1 uses A.5; D.3 uses
  A.5, D.5 uses D.3 and E.1, D.6 uses D.1 and D.5. Nothing in §A or §E uses §D. And by the two
  remarks above, §D can be made independent of (F1).

My checks (`R01_intertwiners.py`, klrw, own code): D.1 (60 identities), D.3 on all 108 three-strand
states over `{1,2,3,W_k}` and 96 instances embedded in 5-strand states behind a dotted braid, D.5
(216), D.6 both directions on 8 blocks of 3–4 strands (with and without red; `S` of every colour,
red `S` for red-free blocks), random **non-reduced** dotted words `a`: 720 identities, 0 failures;
negative control (`ψ` for `φ`): 63/240 fail.

---

## 3. §B — diagrams of tree terms

* **Lemma B.2** correct (each sign product re-evaluated).
* **Lemma B.3** correct and **exhaustive**: `W` crosses only `M`; no `L` crosses an `R`; two
  unaffected strands do not cross; so a triple is `{M, X, Y}` with `X, Y` from `U, L, R`, or one of
  `{U,L,L}, {U,R,R}, {L,L,L}, {R,R,R}`; pairwise nested intervals form a chain. The middle strands
  are right. The key point "the three strands of a braid move are at positions `i, i+1, i+2` in their
  source order" holds for braid moves inside reduced words (none of the three pairs has crossed yet).
* **Cor. B.4** correct, including "does not depend on the sides of unaffected slots".
* **Prop. B.5** correct.
* **Theorem B.6** correct. (a) and (b) re-derived step by step: (B4) needs four slots, (B3) three;
  (B1), (B2) by the colour walk in the forest `Γ`; (B3) leads to the closed walk
  `c', c(v_1), …, c(v_{l−2}), c'` and forces `l = 3`, pattern S. **"Pattern S forces a cyclic
  configuration" is proved**, with one omission (C2): the new pair `p* = (a_p, y)` resp. `(y, b_p)`
  requires `y ∈ I` resp. `y ∉ I`; since `y` is unaffected one may move it (vertex
  `I ∪ {y}` resp. `I ∖ {y}`), and cyclicity is a property of the word anyway (03 §1.3). Cross-check
  with `03_cyclic_combinatorics.md` Theorem 2.8 / Cor. 2.10: pattern S is an instance of pattern
  **B** if `j < a_q` (`x_1,x_2,x_3 = j, a_q, b_q`; the pairs `(a_p, y)` and `x` have one slot below
  `j` and one in `(a_q, b_q)`) and of pattern **T** if `j > b_q` (`x_1,x_2,x_3 = a_q, b_q, j`; the
  pairs `(y, b_p)` and `x` have one slot in `(a_q,b_q)` and one above `j`), as 03 says; consistent.
  (c): `Λ^1, Λ^2`, duals, `so_{2n}` `ω_1`, `D_4` have multiplicities `≤ 2`; the `Λ^3` argument (full
  diagonals, order `d_1<e_1<d_2<e_2<d_3<e_3` resp. `e_1<d_1<…`, the third `c'`-slot outside
  `(d_1,d_3)`) is correct, and `Λ^{n−3}` is its transpose (Dynkin automorphism). **B.6(c) is a
  proof**; the larger minuscule families of §B.5 (`A_7…A_9 ω_4`, `A_10 ω_3`, spinors, `E_6`, `E_7`)
  are computations only, and the draft labels them so.
* **Lemma B.8** correct (the two braid moves and the vanishing `ψ_3ψ_3 = 0` on `(Z,S)`).
* **Conjecture B.9, Def. B.10**: B.10 is exactly what `klrw` uses (checked on 40 200 braids), and the
  description of the word for a tree term is right.
* **§B.9** (comparison with 05): the "Warning" example `3232332`, `I = {5,6,7}`, mover 6, pairs
  `(1,5), (2,7)` has exactly the two triples `U_3, L_{(1,5)}, M` (colours 3,3,3) and
  `U_4, L_{(1,5)}, M` (colours 2,3,3), neither bad; the word is cycle-free (my brute force). So the
  draft's correction of 05 is right. I did not re-referee 05's Lemmas 1.2/1.4 (that is R05's file).
* **§B.7** reproduced exactly: 81 reduced words of `σ` for `11221121` give 2 elements, the
  difference to the basis braid is `−r_1t_{12}·E_{(2,1,1,W,2,1,1,2,1)} s2 s1 s4 s6`; with `(2,6)`
  in place of `(2,5)`, one element.

My checks (`R01_triples.py`, pure Python, own enumeration): Lemma B.2, B.3 (types and middle
strands), Cor. B.4 (type by type) on 4 000 random configurations with arbitrary `F` (types `A_2…A_4`,
`D_4`, `D_5`, `E_6`), 213 of them with bad triples; Theorem B.6(a),(b) exhaustive for `A_3` (≤ 9
letters, 5 235 words: 160 configurations with a bad triple — the author's number), `D_4` (≤ 8, 52 636
words: 48), `A_4` (60 000 sampled words ≤ 9: 262): **every** bad triple is of type B3 in pattern S and
the proof's configuration `(j, {q, x, p*})` is cyclic; multiplicities `≤ 2`: none; `Λ^3ℂ^n`,
`n ≤ 10`, every Young diagram: no bad triple. `R01_words.py` (klrw): Lemma B.8 (one element on the
states `1 2 1 1`, `1 1 2 1`, `2 1 2 2`, `2 3 2 2`, `1 W1 1 1`, `1 1 W1 1`; three on `1 2 2 1`), the shielded example
`12321223` (61 words, one element, red 1, 2, 3, both vertices).

---

## 4. §C — verdict: correct and complete

Lemma C.1 (every letter moves the moving strand to the left; the word is forced), Lemma C.2
(inversion sets of both products; a triple contains `i`, `j` and a strand crossing both; middle `i`;
no shield because only `i, j` cross a non-moving strand), Cor. C.3, and the review of Theorem 5 are
correct. The converse of C.3 (computed here 10/10, author 37/37) is proved in `04_first_order.md`
Cor. B.3 (with F1). `R01_words.py c3`: 39 cases without, 10 with a bad `(x,i,j)`, no exception.

## 5. §F — review of Tier 1 of `PROOFS.md` — verdict: agree

I re-read Lemma 1, Props. 2, 3, Cor. 3.1, Props. 4, 6, 7, 8 and Theorem 5 and agree with the
draft's table, with two amendments: the reference repair should say that Webster's Prop. 2.23
exists but concerns another deformation (C3); and Prop. 4 uses, besides (F1), that `R` is
`G`-graded with homogeneous monomials, so a homogeneous `z` is a `ℤ`-combination of basis elements
`μ x^αβ_σ` of its degree (true; stated implicitly by the draft). The extension of Prop. 4 to several
red strands is, as the draft says, not covered by A.5 as stated; the proof of A.5 does go through
verbatim for several red strands that never cross each other (no variables on red strands, the
same leading-term argument), but this has to be written down before it is used.

---

## 6. List of problems

| # | where | severity | problem | repair | affected results of other fronts |
|---|---|---|---|---|---|
| R1 | §A.2, §S | repairable (status) | "Definition A.1 is the presentation implemented in klrw" is listed as *proved*; the 2-/3-strand checks only verify the relations of `A`, not the global multiplication of the implementation | move to "verified by computation"; state the criterion (relations on all states + associativity + basis words = products of generators); my tests on 4–6 strands (§1.1) support it | none of the proofs (they use A.1, not klrw); every Tier-2 computation of the project relies on it (as it always did) |
| R2 | Prop. A.3 | repairable (filled) | (A6) in the polynomial representation only by computer | hand computation §1.3; independent second proof of A.5(ii) via Rouquier Thm. 3.7, §1.5 | none (A.5 stands) |
| R3 | D.3 ("assume `n = 3`"), D.5, D.6 (block algebras), E.1 used on 3-strand states | repairable | algebras with no red strand are not instances of Def. A.1, so A.5/E.1 are not literally available there | argue in the big algebra (with its red strand) — the arguments only use positions `i,i+1,i+2`; or note A.5 holds verbatim with 0 red strands; or use the F1-free route (06 Lemma 2.3 + `ι`) | none (06 Prop. 4.2 and 05 use the algebra with the red strand) |
| C1 | A.5(ii) | cosmetic | "Dedekind's lemma for automorphisms of `K`" — the operator is only known to vanish on `R[y]` | Artin's lemma for characters of the monoid `R[y]∖0` | none |
| C2 | B.6(b), pattern S ⇒ cyclic | cosmetic | `p*` needs `y` on the appropriate side | take `I ∪ {y}` / `I ∖ {y}` | none |
| C3 | §A.4 literature, §F | cosmetic | Webster v4 **has** a Prop. 2.23 ("this deformation is free (and thus flat)"), for `Q_e + z_e`; not our deformation | correct the sentence | `PROOFS.md` rewrite (STATUS item 2) |
| C4 | §A.4 literature | cosmetic | KL I Thm. 2.5 is over `ℤ`, not over a field | — | none |
| C5 | Cor. D.4 remark | cosmetic | KKK Lemma 1.3.1 has items (i)–(vi); (v) is the analogue of D.5/D.6 | — | none |
| C6 | Prop. D.5 | cosmetic | E.1 not needed when `c(S) ≠ c`; second identity = `ι`(first) | — | none |

**Effect on the other fronts.** The other fronts use from 01: A.5 in left and right form and for
any choice of reduced words (05 Lemma 3.1, Lemma 4.3; 04 Lemma 1.1(b), Cor. 7.2, Prop. 7.4, Thm. 8.1,
8.4, Lemma 8.5; 06 §5, §6, §9; 07), E.1 (06 Props. 4.4, 5.1, Lemma 6.5; 07), Def. B.10 (04, 05, 07),
Thm. B.6 (05/06 for multiplicities ≤ 2), D.3–D.6 (06 re-proves them). All of these are within what
01 proves, with one red strand; every use I looked at (05 Lemma 3.1: basis of `A'` maps into a basis of
`A` because A.5 allows any reduced words; 05 Lemma 4.3: `𝔪^{N+1}e'Ae_0 = ⊕ b_σ 𝔪^{N+1}R[x]` from the
right basis; 05 Prop. 4.5: `⋂𝔪^N A = 0` from freeness; 04 Lemma 1.1(b): `(r)^nA = ⊕(r)^n b`) is
legitimate. 02 explicitly avoids (F1) (two red strands). **No result of another front is affected.**

---

## 7. Independent checks (own code, `proofs/checks/R01_*.py`)

| script | what | result |
|---|---|---|
| `R01_twisted.py` (sage, ~5 min) | Prop. A.3 as exact identities in `K ⋊ S_n`, `Γ = D_4`, red 1, 2, three colour orders | 6 384 identities, 0 failures (`R01_twisted.log`) |
| `R01_klrw.py basis` | each klrw basis braid = product of its letters; = Lehmer word of B.10 | 40 200 / 40 200, 40 200 / 40 200 |
| `R01_klrw.py hom` | `ρ(ab) = ρ(b)∘ρ(a)`, own polynomial representation, 4–6 strands | 560 instances, 0 failures; negative controls detected (`R01_hom_negative.py`) |
| `R01_klrw.py assoc` / `embedded` / `grading` | associativity; (A2)–(A6) at every position behind a dotted braid; F2 degrees, homogeneity | 175 / 8 064 / 120 instances, 0 failures (`R01_klrw.log`) |
| `R01_rank.py` (sage, no klrw, ~25 min) | A.5(ii),(iii) through own `V`: left and right bases, Lehmer and random reduced words; spanning degree by degree; degenerate parameters | 126 + 73 full-rank tests, 150/150 in span; rank drops at `r = 0`, `t = 0`, `u = 0` (below) |
| `R01_intertwiners.py` | D.1, D.3, D.5, D.6 (+ negative control) | all hold; control 63/240 fail |
| `R01_triples.py` (python3) | B.2, B.3, B.4 random; B.6(a),(b) exhaustive; B.6(c) `Λ^3` | all confirmed (`R01_triples.log`) |
| `R01_words.py` | B.7, B.5 example, B.8, C.3 | all as claimed |

`R01_rank.py` (`R01_rank.log`; parameters random non-zero in `GF(1000003)`): (1)/(2) 126 rank
tests on 4- and 5-strand states (`1 1 2 W1`, `1 W1 1 1`, `1 2 1 W1 1`, `A_3` `1 2 W2 2 3`, `D_4`
`2 1 W2 2 3`, `1 1 1 W1`; `|S(s,s')|` up to 6, up to 128 elements per test): the left basis
`x^αψ_w` with the Lehmer words **and** with random reduced words, and the right basis `ψ_w x^β`,
are independent in every case (rank = number of elements); 73 further tests with all monomials as
test vectors in `R01_rank_highdeg_partial.log`, same result. (3) spanning degree by degree: 150/150
random non-reduced dotted words lie in the span of the claimed basis elements of their `q`-degree
(so the image of each graded piece has exactly the claimed dimension). (4) degenerate parameters,
`1 1 2 W1`, 16 pairs, 128 elements: rank 128 generic, **64 at `r_1 = 0`**, 84 at `t_{12} = t_{21} = 0`,
84 at `u_1 = 0` — `V` is not faithful there, so (F1) at `r = 0` rests on the base-change argument
A.5(iv) (which is correct), not on `V`.

---

## 8. Verdicts per statement

| statement | verdict |
|---|---|
| Def. A.1 | correct; identification with klrw: verified by computation (R1) |
| Def. A.2, Prop. A.3 | correct; (A6) part now proved by hand (§1.3) |
| Thm. A.5 (i)–(v) | correct and complete (given §1.3); true for `r = 0` and every base change; second proof of (ii) via Rouquier |
| Thm. E.1 | correct and complete |
| Lemma D.1, D.2, Thm. D.3, Cor. D.4, Prop. D.5, Thm. D.6, Rem. D.7 | correct and complete (R3, C6 cosmetic); can be made F1-free |
| Lemma B.2, Lemma B.3, Cor. B.4, Prop. B.5 | correct and complete |
| Thm. B.6 (a), (b), (c) | correct and complete (C2) |
| Def. B.7, Lemma B.8 | correct |
| Conj. B.9 | conjecture, evidence reproduced |
| Def. B.10 | correct; equals klrw's word |
| §B.7 example | reproduced exactly |
| Lemma C.1, C.2, Cor. C.3, review of Thm. 5 | correct and complete |
| §F (Tier 1) | agree, with C3 |

## 9. How to re-run (from `experiments/cup_brane`)

    sage -python proofs/checks/R01_twisted.py              # Prop. A.3 exactly, ~5 min
    sage -python proofs/checks/R01_klrw.py all 1           # basis words, hom, assoc, embedded, grading
    sage -python proofs/checks/R01_hom_negative.py         # negative controls for 'hom'
    sage -python proofs/checks/R01_rank.py                 # A.5 through V, ~25 min
    sage -python proofs/checks/R01_intertwiners.py all     # D.1, D.3, D.5, D.6
    sage -python proofs/checks/R01_words.py all            # B.5 example, B.7, B.8, C.3
    python3      proofs/checks/R01_triples.py all          # B.2-B.4, B.6 (exhaustive), ~30 min

Logs: `proofs/checks/R01_{twisted,klrw,rank,rank_highdeg_partial,intertwiners,triples}.log`.
