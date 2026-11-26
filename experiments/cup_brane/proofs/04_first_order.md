# 04 — First-order theory and forcedness, for arbitrary words

Front: the cube differential `d = δ₀ + δ₁ + δ₂ + …` (tree formula, `δ_n` = the tree terms
with `n` exchange pairs) modulo `(r)` and modulo `(r)²`, **for every word** (no minuscule
hypothesis, equal neighbours allowed); what a first-order correction can be; the smallest
cyclic example. Notation and conventions are those of `../PROOFS.md` §1 and of
`README_FOR_PROVERS.md` (product `ab` = first `a`, then `b`; signs of (R2)–(R4) as listed
there). Scripts: `checks/04_*.py` (run with `sage -python`, except where marked pure Python).

## 0. Results at a glance

| # | statement | hypotheses | status |
|---|---|---|---|
| Thm A | `δ₀² ≡ 0 mod (r)` | none | **proved** |
| Thm B | exact formula `(δ₀²)_{I,I∖{i,j}} = ε(I;i,j) r_{c(j)} t_{c(j)c(i)} Σ_{y} X_y`, sum over the bad triples `y<i<j`, `c(y)=c(j)∼c(i)`; `X_y ≡ β_{τ_y} mod (r)` | none | **proved** |
| Cor B.3 | `δ₀² = 0` iff the word has no slots `y<i<j` with `c(y)=c(j)∼c(i)`; otherwise `δ₀² ∉ (r)²`, and every differential `δ₀+ε`, `ε∈(r)`, has a non-zero first-order part | (F1) for "otherwise" | **proved** |
| Prop 5.3 | classification of the products (elementary move)·(one-pair term) and (one-pair term)·(elementary move): reduced, or exactly one bigon `{x, strand of a or b}` | none | **proved** |
| Thm C | `δ₀² + δ₀δ₁ + δ₁δ₀ ≡ 0 mod (r)²`, i.e. the tree formula is a differential to first order | none | **proved** |
| Prop 7.1 | degree defect of an *arbitrary* bijection (generalises PROOFS Prop. 3) | none | **proved** |
| Prop 7.4 | if every colour occurs ≤ 2 times: the first-order homogeneous elements in which no strand crosses `W` from left to right are exactly the one-pair terms | multiplicity ≤ 2 | **proved** |
| Thm 8.1 | any differential `δ₀+ε`, `ε ∈ (r)`, has first-order part `δ₁ + η`, `η` a first-order cycle; `δ₁ + [h,δ₀]` occurs for every first-order gauge `h` | (H1) | **proved** |
| Ex 8.3 | the first-order part is **not** unique: word `1212`, red `2` | — | **proved** (finite computation of one conjugation) |
| Thm 8.4 | `ε ≡ δ₁ mod (r)²` | (H1), (M₁), multiplicity ≤ 2 | **proved** |
| Comp 8.5 | first-order cycles = first-order gauge directions (`Z₁ = B₁`) | words of length ≤ 6 (`A_2`), ≤ 5 (`A_3`), ≤ 4 (`D_4`) | computed only |
| Thm 9.1 | `1221122`, red 2: the tree formula cannot be completed by dot-free terms; the true differential = tree formula + 10 terms in one component, exactly the spanning-tree × bigon rule of NOTES §6 | exact computation | Tier 2 |
| §10 | second order: exact formulation | — | formulation only, no proof |

**Corrections to other files (prominent):**

1. NOTES §3 / PROOFS Tier 2 ("there is no gauge freedom and the formula is not a choice of
   normal form") is **false for arbitrary words**: already for `1212` with red colour 2
   (every colour twice) there is a homogeneous first-order map `h` of homological degree 0
   such that `(1+h) d (1+h)^{-1}` is a differential satisfying all hypotheses and differing from
   the tree formula at first order (Example 8.3, `checks/04_gauge_example.py`). For the
   minuscule words tested (e.g. `2132`, `2312` in `A_3`) the first-order cycle space is 0,
   consistent with NOTES. Uniqueness can only be expected modulo gauge (Comp. 8.5) or under
   extra normalisations (Thm 8.4).
2. The hypothesis "ε vanishes when all parameters are 0" is **too weak** for any uniqueness
   statement: there are `r`-free, `t`-dependent gauges that keep the geometric part `δ₀` but
   produce `r`-free terms (Remark 8.7, word `1121`, red 2). The right hypothesis is
   `ε ≡ 0 mod (r)` (H1).
3. PROOFS §4 (C1 sketch) is confirmed: the bad triples of `δ₀²` are exactly `(y, i, j)`,
   `y < i < j`, `c(y) = c(j) ∼ c(i)`; the red strand is never the middle strand; the pattern
   `c(y) = c(i) ∼ c(j)` never gives a bad move. The worry that the smoothed diagrams might be
   non-reduced does not materialise for the path used here (Lemma 4.5); the products of
   one-pair terms with elementary moves *are* sometimes non-reduced (Prop. 5.3), and their dot
   terms cancel in groups of four.

---

## 1. Setting

### 1.1 Data
`A` is the algebra of PROOFS §1.1 over `R = ℤ[r_c, t_{cc'}, u_c]`, for one red strand of colour
`k` and a word `(c(1), …, c(m))` (an arbitrary sequence of nodes). Slots, `pos_I`, `e_I`,
`d(x)`, `D(I)`, `[c,c']`, `w_c` are as in PROOFS §1.2–1.3. For `I ⊆ {1,…,m}` and a slot `x` put

    n_I(x) = #{z ∈ I : z > x},        N_I(a,b) = #{z ∈ I : a < z < b}.

A *map of homological degree h* on the cube is a family `f = (f_{IJ})`, `f_{IJ} ∈ e_I A e_J`,
`|J| = |I| + h`, every `f_{IJ}` homogeneous of degree `D(I) − D(J)` (PROOFS §1.3; `f_{IJ}` acts
`P_I → P_J` by right multiplication). Composition: `(fg)_{IK} = Σ_J f_{IJ} g_{JK}` (first `f`,
then `g`; this is the composition of the right multiplications). `δ_n` is the map whose
`(I,J)`-component is the sum of the terms of the tree formula (`formula.py`,
`Stack.terms_from(I)`) with exactly `n` exchange pairs, each written with the basis braid of
its bijection. In particular

    δ₀_{I, I∖x} = (−1)^{n_I(x)} m_x(I)        (x ∈ I),

and `δ₁` is described in §5. A term with `n` pairs has coefficient `± ∏ r t` with exactly `n`
factors `r`.

### 1.2 Working modulo powers of (r)
Let `(r) ⊂ R` be the ideal generated by all `r_c`, and for `n ≥ 1` let `(r)^n A` be the set of
finite sums `Σ ρ_i a_i`, `ρ_i ∈ (r)^n`, `a_i ∈ A`.

**Lemma 1.1.** (a) `(r)^n A` is a two-sided ideal of `A`, and `(r)^a A · (r)^b A ⊆ (r)^{a+b} A`.
(b) Assume (F1). An element `z = Σ_b z_b · b` (finite sum over the basis `b = x^α β_σ` of (F1),
`z_b ∈ R`) lies in `(r)^n A` iff every `z_b ∈ (r)^n`. Hence `A/(r)^n A ≅ A ⊗_R R/(r)^n` is free over
`R/(r)^n` on the images of the basis, and `(r)A/(r)²A ≅ ⊕_c r_c · (A/(r)A)` is free over `R/(r)`.

*Proof.* (a) `R` is central in `A` (the relations are `R`-linear and `R` is the coefficient
ring), so `ρ a · a' = a ρ a' = ρ (a a')`. (b) `A = ⊕_b R b` (F1), so `(r)^n A = ⊕_b (r)^n b`.
`(r)/(r)²` is free over `R/(r)` on the classes of the `r_c` (they are distinct variables of a
polynomial ring). ∎

So "modulo terms of order ≥ 2 in `r`" means: in `A/(r)²A`, which is legitimate because (r)² is
a two-sided ideal (no freeness needed for statements of the form "`z ∈ (r)²A`"), and loses no
information on the first-order part because of (b). Note that the *`r`-degree-1 part* of an
element, read off in the basis, depends on the choice of the braids `β_σ`; the class modulo
`(r)²` does not. All statements below are modulo ideals and therefore independent of these
choices (by Lemma 2.2, two reduced words of the same bijection agree modulo `(r)`, so a term
`r t β_σ` is well defined modulo `(r)²`).

Throughout, "`z ≡ z' mod (r)^n`" means `z − z' ∈ (r)^n A`, and for maps, componentwise.

---

## 2. Diagrams modulo (r)

A *word* `w = (p_1, …, p_ℓ)` read from a state `s` is the product `ψ_{p_1} ⋯ ψ_{p_ℓ}` starting
at `e(s)` (`checks/alg.py` convention). Its *strands* are labelled by their starting position
(for the cube: by slot numbers and `W`). For two strands `u, v` let `n_w(u,v)` be the number of
letters of `w` that cross `u` and `v`. The *bijection* of `w` is the map (start ↦ end) of
strands. `w` is reduced iff `n_w(u,v) ≤ 1` for all pairs (the length of a permutation is its
number of inversions, and a pair crosses an odd number of times iff it is an inversion).

**Standing fact (F3′) (Tits' word property).** Every word in the generators `s_p` of the
symmetric group can be transformed into a reduced word by a sequence of braid moves
(`s_p s_{p+1} s_p ↔ s_{p+1} s_p s_{p+1}`, `s_p s_q ↔ s_q s_p` for `|p−q| ≥ 2`) and deletions of
factors `s_p s_p` [Tits 1969; Björner–Brenti, *Combinatorics of Coxeter groups*, Thm 3.3.1].
This is the companion of Matsumoto's theorem (F3) and is used only in Lemma 2.2.

**Lemma 2.1 (relations modulo r).** In `A₀ := A/(r)A`:
(a) a dot slides through every crossing along its strand (`x ψ = ψ x'`);
(b) every braid relation `ψ_pψ_{p+1}ψ_p = ψ_{p+1}ψ_pψ_{p+1}` holds, on every state;
(c) `ψ_p² e(s) = Q_{uv} e(s)` for the two strands `u, v` at `p, p+1`, where `Q_{uv} = 0` if `u, v`
are black of the same colour, `t_{cc'}x_u + t_{c'c}x_v` if black of adjacent colours `c, c'`,
`1` if black of distinct non-adjacent colours, `u_k x_u` if `u` is black of colour `k` and `v`
is the red strand, `1` if `u` is black of colour `≠ k` and `v` red.

*Proof.* (a) (R2): `x_1ψ − ψx_2 = r_c` and `ψx_1 − x_2ψ = r_c` for equal colours, exact sliding
otherwise. (b) (R4): the two exceptions differ by `r_c t_{cc'} e` and `r_k u_k e`, both in
`(r)A`. (c) is (R3). ∎

Consequently, in `A₀` a polynomial in dots placed anywhere on the strands of a diagram is
well defined as a polynomial in *strand variables* `x_u`.

**Lemma 2.2 (normal form modulo r).** Let `w` be a word from `e(s)` with bijection `σ`. Then

    w ≡ ( ∏_{pairs {u,v}} Q_{uv}^{⌊n_w(u,v)/2⌋} ) · β_σ      mod (r),

product over unordered pairs of strands (the red strand included), `β_σ` any reduced word of
`σ`, dots placed at the left end. In particular: two reduced words of the same bijection are
equal modulo `(r)`; and `w ≡ 0 mod (r)` if two black strands of the same colour cross at least
twice.

*Proof.* Induction on `ℓ(w)`. If `w` is reduced, (F3) connects it to `β_σ` by braid moves and
commutations; commutations are (R1), braid moves hold in `A₀` (Lemma 2.1(b)). If `w` is not
reduced, by (F3′) there is a sequence of braid moves and commutations from `w` to a word
`w_1 p p w_2` (the first deletion in the sequence of (F3′) is preceded only by braid moves and
commutations). These moves hold in `A₀` and preserve every `n(u,v)` (a braid move on strands
`a,b,c` replaces the crossings `ab, ac, bc` by `bc, ac, ab`). The two letters `pp` are a double
crossing of two strands `u, v`; by Lemma 2.1(c) and (a), `w ≡ Q_{uv} · w_1 w_2`, the strands of
`w_1w_2` being those of `w` (the strand at position `p` after `w_1` simply stays there), with
`n(u,v)` decreased by 2 and all other `n` unchanged. Induction; at the end every pair crosses
`n(u,v) mod 2` times, i.e. `⌊n(u,v)/2⌋` bigons have been removed for each pair. ∎

---

## 3. Order 0

**Theorem A.** `δ₀² ≡ 0 mod (r)`, for every word and every colour of the red strand.

*Proof.* `δ₀²` has components only `I → I∖{i,j}` (`i<j` in `I`), equal to
`ε(I;i,j)·(P_1 − P_2)` with `P_1 = m_i(I) m_j(I∖i)`, `P_2 = m_j(I) m_i(I∖j)` and
`ε(I;i,j) = (−1)^{n_I(i) + n_I(j)}` (proof of PROOFS Thm 5: the two sign products are opposite).
`P_1` and `P_2` are reduced words of the same bijection (the identity of slots; every pair of
strands crosses at most once, PROOFS Thm 5), hence equal modulo `(r)` by Lemma 2.2. ∎

`checks/04_order01.py`, test (T0): 0 failures on all words of length ≤ 7 (`A_2`), ≤ 6
(`A_3`), ≤ 5 (`D_4`), 300 random words of length 6 (`D_4`), every colour of the red strand.

---

## 4. The exact first-order part of δ₀²

Fix `I` and slots `i < j` in `I`. Put

    Z_R = {z ∈ I : z < i} (decreasing),  Z_L = {z ∉ I : z < i} (increasing),
    (z_1, …, z_N) = (Z_R, W, Z_L)          (the red strand W is one of the z_s),
    U_R = {z ∈ I : i < z < j} (decreasing),  U_L = {z ∉ I : i < z < j} (increasing).

In `e_I` the strands are, from left to right,
`[slots ∉ I, > j]↓, U_L↓, z_N, …, z_1, i, U_R↑, j, [slots ∈ I, > j]↑` (by the definition of `e_I`;
`z_N, …, z_1` is `Z_L↓, W, Z_R↑`). For a strand `u` and strands `v_1, …, v_k` write
`[u : v_1, …, v_k]` for the word in which `u` successively crosses `v_1, …, v_k`, each time its
left neighbour (moving to the left).

**Lemma 4.1.** As words: `m_i(I) = [i : z_1..z_N]`, `m_j(I∖i) = [j : U_R, z_1..z_N, i, U_L]`,
`m_j(I) = [j : U_R, i, z_1..z_N, U_L]`, `m_i(I∖j) = [i : z_1..z_N]`.

*Proof.* In `m_x(K)` only the strand of `x` moves; it crosses `W` and exactly the strands of the
slots `< x` (PROOFS Lemma 1), which form the contiguous block to its left. Reading the block
from right to left in the states `e_I`, `e_{I∖i}` (where `i` sits between `U_L` and `Z_L`),
`e_I`, `e_{I∖j}` gives the four lists. A single strand moving through a block has a unique
word. ∎

**Definition 4.2 (the sweep).** For `0 ≤ s ≤ N` let

    D_s = [i : z_1..z_s] [j : U_R, z_1..z_s] [j : i] [j : z_{s+1}..z_N] [i : z_{s+1}..z_N] [j : U_L].

**Lemma 4.3.** Each `D_s` is a well defined word from `e_I` to `e_{I∖{i,j}}` (every listed
crossing is with the current left neighbour), with the identity bijection of slots.
`D_N = m_i(I) m_j(I∖i)` as words, and `D_0 = m_j(I) m_i(I∖j)` in `A`.

*Proof.* Track the order of the strands: after `[i : z_1..z_s]`:
`…, U_L↓, z_N..z_{s+1}, i, z_s..z_1, U_R↑, j, …`; after `[j : U_R, z_1..z_s]`: `…, z_{s+1}, i, j, z_s..z_1, U_R, …`;
after `[j : i]`: `…, z_{s+1}, j, i, z_s, …`; after `[j : z_{s+1}..z_N]`: `…, U_L↓, j, z_N..z_{s+1}, i, z_s.., …`;
after `[i : z_{s+1}..z_N]`: `…, U_L↓, j, i, z_N..z_1, U_R, …`; after `[j : U_L]`:
`[slots ∉ I, >j], j, U_L↓, i, z_N..z_1, U_R↑, [slots ∈ I, >j]`, which is `e_{I∖{i,j}}`.
`D_N = [i : z_1..z_N][j : U_R, z_1..z_N, i, U_L]` is the concatenation of Lemma 4.1. In
`D_0 = [j : U_R, i, z_1..z_N][i : z_1..z_N][j : U_L]` the last two factors use letters `ψ_q`
with `q ≥ q_0 + 1` and `q ≤ q_0 − 1` respectively (`q_0` = position of `j` before them), so they
commute letter by letter (R1), and `D_0 = [j : U_R, i, z_1..z_N, U_L][i : z_1..z_N] = m_j(I)m_i(I∖j)`. ∎

**Lemma 4.4 (one braid move per step).** For `1 ≤ s ≤ N` put `y = z_s`,
`P_s = [i : z_1..z_{s−1}][j : U_R, z_1..z_{s−1}]`, and let `S_s = [j : z_{s+1}..z_N][i : z_{s+1}..z_N][j : U_L]`.
After `P_s` the strands `y, i, j` stand at consecutive positions `p, p+1, p+2`, in this order, and

    D_s = P_s · ψ_pψ_{p+1}ψ_p · S_s,        D_{s−1} = P_s · ψ_{p+1}ψ_pψ_{p+1} · S_s     (in A),

where `S_s` is read from the state after the triple, which is `(j, i, y)` in both cases.

*Proof.* After `[i : z_1..z_{s−1}]` the order is `…, y, i, z_{s−1}..z_1, U_R, j, …` with `y` at
`p`. In `D_s`, the letter `[i : z_s] = ψ_p` commutes with `[j : U_R, z_1..z_{s−1}]`, whose letters
are `ψ_q`, `q ≥ p+2` (`j` stops at `p+2`). Moving it there gives `P_s ψ_p [j : z_s][j : i] S_s =
P_s ψ_pψ_{p+1}ψ_p S_s`. In `D_{s−1} = P_s [j : i][j : z_s][j : z_{s+1}..z_N][i : z_s][i : z_{s+1}..z_N][j : U_L]`
the letter `[i : z_s] = ψ_{p+1}` commutes with `[j : z_{s+1}..z_N]` (letters `ψ_q`, `q ≤ p−1`);
moving it gives `P_s ψ_{p+1}ψ_pψ_{p+1} S_s`. ∎

**Lemma 4.5 (the smoothed word).** Let `X_y := P_s · S_s`, where now `S_s` is read from the
state after `P_s` (in which `(y, i, j)` stand at `p, p+1, p+2`): so in `S_s` the strand of `y`
performs the moves written for `j`, and the strand of `j` does not move. Then `X_y` is a
reduced word from `e_I` to `e_{I∖{i,j}}` whose bijection `τ_y` sends the strand of slot `y` to
slot `j` and the strand of slot `j` to slot `y`, and fixes all other slots. Its crossings are:
strand of `y` with `z_{s+1},…,z_N` and `U_L`; strand of `j` with `U_R` and `z_1,…,z_{s−1}`;
strand of `i` with all `z_t`, `t ≠ s`.

*Proof.* Read off from the definition; the three lists of crossed strands are lists of
non-moving strands and each strand crosses each of them at most once, and the moving strands
do not cross each other; so every pair crosses at most once. The end points: the strand of `y`
ends where `j` ends in `D_s`, the strand of `j` where `y` ends. ∎

**Lemma 4.6 (which moves are bad).** In Lemma 4.4, `ψ_pψ_{p+1}ψ_p − ψ_{p+1}ψ_pψ_{p+1}` on the local
state `(y, i, j)` equals `r_{c(j)} t_{c(j)c(i)}·e` if `y` is a black strand with
`c(y) = c(j) ∼ c(i)`, and `0` otherwise.

*Proof.* (R4) with the signs of the README: the relation fails only on `(c, c', c)`, `c ∼ c'`
(difference `+r_c t_{cc'}`) and on `(k, W, k)`. The middle strand is `i`, which is black, so the
second case never occurs; the first occurs iff the outer strands `y, j` are black of the same
colour adjacent to `c(i)`. ∎

So the triangles of `P_1, P_2` are `{z, i, j}` (`z < i` a slot, or `z = W`), the middle strand
is always `i`; the red strand is never the middle strand (PROOFS §4 sketch: correct); the
pattern `c(y) = c(i) ∼ c(j)` (state `(c, c, c')`) is exact, and so is every triangle containing `W`.

**Definition.** `Bad(i,j) = { y : 1 ≤ y < i, c(y) = c(j), c(i) ∼ c(j) }` (independent of `I`).

**Theorem B (exact first-order formula).** For every vertex `I` and `i < j` in `I`,

    (δ₀²)_{I, I∖{i,j}} = ε(I;i,j) · r_{c(j)} t_{c(j)c(i)} · Σ_{y ∈ Bad(i,j)} X_y          (exactly, in A),

with `ε(I;i,j) = (−1)^{n_I(i)+n_I(j)}` and `X_y` the reduced word of Lemma 4.5; all other
components of `δ₀²` vanish. Consequently

    δ₀²  ≡  Σ_{I} Σ_{i<j ∈ I} ε(I;i,j) r_{c(j)} t_{c(j)c(i)} Σ_{y ∈ Bad(i,j)} β_{τ_y}      mod (r)².

*Proof.* `P_1 − P_2 = D_N − D_0 = Σ_{s=1}^N (D_s − D_{s−1}) = Σ_s P_s(ψψψ − ψψψ)S_s`
(Lemmas 4.3, 4.4) `= Σ_{s : z_s ∈ Bad(i,j)} r_{c(j)}t_{c(j)c(i)} X_{z_s}` (Lemma 4.6; `R` is
central). Multiply by `ε(I;i,j)` (proof of Thm A). Modulo `(r)²`: `X_y ≡ β_{τ_y} mod (r)` by
Lemmas 4.5 and 2.2, and `r t (r)A ⊆ (r)²A`. ∎

*Checks.* `checks/04_sweep.py` builds every `D_s`, `P_s`, `S_s`, `X_y` and verifies (S1)–(S5):
the words are well defined, Lemma 4.3, Lemma 4.4 (as equalities in `A`), Lemma 4.6, Lemma 4.5
(reduced, bijection `τ_y`, `≡ β_{τ_y} mod r`) and the exact identity
`P_1 − P_2 = Σ r t X_y`; all words of length ≤ 7 (`A_2`), ≤ 5 (`A_3`), ≤ 4 (`D_4`), every red
colour: 0 failures, 94 160 bad and 618 035 good triangles. `checks/04_order01.py` (T1)
compares the right-hand side of the mod-`(r)²` formula with `δ₀²` computed directly: 0
failures on the ranges of §3. *Observation (Tier 2, not used):* in every checked case
`X_y` equals the `klrw` basis element `β_{τ_y}` exactly, so `δ₀²` has no terms of `r`-degree ≥ 2
in that basis.

**Corollary B.2.** `δ₀² = 0` (exactly) for every word without slots `y < i < j` with
`c(y) = c(j) ∼ c(i)`. This contains PROOFS Thm 5 (distinct colours) and e.g. all words
`c c c' c'`, or words in which equal colours are never separated by an adjacent colour.

**Corollary B.3 (necessity of corrections; uses F1).** If the word has slots `y < i < j` with
`c(y) = c(j) ∼ c(i)`, then for `I = {i, j}` the component `(δ₀²)_{I, ∅}` has coefficient
`± r_{c(j)} t_{c(j)c(i)}` at the basis element `β_{τ_y}` modulo `(r)²`. Hence `δ₀² ∉ (r)²`
(in particular `δ₀² ≠ 0`), and for every differential `d = δ₀ + ε` with `ε ≡ 0 mod (r)`, the
first-order part `ε mod (r)²` is non-zero (indeed `δ₀ε + εδ₀ ≡ −δ₀² ≢ 0 mod (r)²`).

*Proof.* The `y ∈ Bad(i,j)` give pairwise different bijections `τ_y`, hence different basis
elements; Theorem B and Lemma 1.1(b). For the last claim: `d² = δ₀² + δ₀ε + εδ₀ + ε²` and
`ε² ∈ (r)²`. ∎

---

## 5. One-pair terms and their products with elementary moves

A *one-pair configuration* at `I` is `(j, p)`, `p = (a, b)`, with `j, b ∈ I`, `a ∉ I`,
`c(a) = c(b) =: c(p)`, `a < j < b` and `c(j) ∼ c(p)` (for one pair, admissibility in NOTES §2
reduces to "linked to the mover"). Its target is `J = I∖{j,b} ∪ {a}`, its bijection the
transposition `(a b)` of slots, `β(I;j,p)` the basis braid, and its term in `δ₁` is
`σ₁(I;j,p) r_{c(p)} t_{c(p)c(j)} β(I;j,p)` (the connector is `x_p = j`).

**Lemma 5.1 (sign).** `σ₁(I;j,p) = −κ(I;j,{p}) = (−1)^{n_I(j) + N_I(a,b)}`.

*Proof.* `(−1)^{[x_p ∈ I]} = −1` since `x_p = j ∈ I`. In `κ`, deleting `j` gives `(−1)^{n_I(j)}`;
after replacing `b` by `a` in place, sorting moves `a` to the left past the entries of
`I∖{j}` strictly between `a` and `b`, i.e. `N_I(a,b) − 1` of them (`j` is between and deleted). ∎
(`checks/04_signs.py`, (L5.1): all one-pair terms of all words of length ≤ 8 (`A_2`), ≤ 6
(`A_3`), ≤ 5 (`D_4`), pure Python.)

**Lemma 5.2 (crossings of β(I;j,p)).** The mover `j` crosses `W` and every unaffected strand
(slot `∉ {j,a,b}`) of slot `< j`; the strand of `a` (from `−a` to `−b`) crosses the unaffected
`z ∉ I` with `a < z < b`; the strand of `b` (from `+b` to `+a`) crosses the unaffected `z ∈ I`
with `a < z < b`; there are no other crossings (the mover crosses neither pair strand since
`b > j`, and the two pair strands do not cross).

*Proof.* PROOFS Prop. 3, items 1–6. ∎

**Proposition 5.3 (classification of the composites).** Let `(j,p)`, `p = (a,b)`, be a
one-pair configuration.

(A) Let `x ∈ I∖{j,b}` and suppose `(j,p)` is a one-pair configuration at `I∖x` (i.e. `a ∉ I` or
`a = x`). The concatenated word `m_x(I)·β(I∖x; j,p)`:
* (A1) if `x = a`: is reduced; its bijection `τ` exchanges the endpoints of `a` and `b`, `j`
  moves; target `I∖{j,b}`;
* (A2) if `a ∉ I` and `x ∉ (a,b)`: is reduced;
* (A3) if `a ∉ I` and `a < x < b`: the strand of `x` and the strand of `a` cross exactly twice,
  every other pair at most once.

(B) Let `a ∉ I`, `J = I∖{j,b}∪{a}`, `x ∈ J`. The word `β(I;j,p)·m_x(J)`:
* (B1) if `x = a`: is reduced, with the same bijection `τ` and target as (A1);
* (B2) if `x ∉ [a, b]`: is reduced;
* (B3) if `a < x < b`: the strand of `x` and the strand of `b` cross exactly twice, every other
  pair at most once.

In (A2), (A3), (B2), (B3) the target is `K = I∖{x,j,b}∪{a}` and the bijection is `(a b)`, the
strands of `x` and `j` both crossing `W`; (A) and (B) give the same `(K, (a b))` for the same
`(x, j, p)`. No strand crosses `W` twice and the mover `j` is never in a bigon.

*Proof.* Combine Lemma 1 of PROOFS (in `m_x`, `x` crosses `W` and exactly the slots `< x`) with
Lemma 5.2, pair by pair. (A): `x` crosses `j` in `m_x` iff `j < x`, and in `β` (as an unaffected
strand at `−x`) iff `x < j`: once. `x` crosses the strand of `b` in `m_x` iff `b < x`, never in
`β` (the strand of `b` crosses only `z ∈ I∖x`): at most once. `x` crosses the strand of `a` in
`m_x` iff `a < x` (if `a ∉ I`), and in `β` iff `a < x < b` (`x ∉ I∖x`): twice iff `a < x < b`.
If `x = a`, the strand of `a` itself moves in `m_a` across the slots `< a`, then outwards across
the `z ∉ I`, `a<z<b`: disjoint sets. Unaffected strands cross `x` at most once (only in `m_x`)
and each other never. (B): `x` crosses `j` in `β` iff `x < j`, in `m_x` (`j` at `−j`) iff `j < x`:
once. `x` crosses the strand of `b` (in `β` iff `a < x < b`; in `m_x`, where it sits at `+a`, iff
`a < x`): twice iff `a < x < b`. `x` crosses the strand of `a` (at `−b` after `β`) only in `m_x`,
iff `b < x`. If `x = a`, the strand of `b` (now at `+a`) crosses in `m_a` the slots `< a`, after
having crossed the `z ∈ I`, `a<z<b`: disjoint. ∎

**Corollary 5.4 (values modulo r).** With strand variables `x_u` (Lemma 2.1),
`Q_{cc'}(X,Y) := 0` if `c = c'`, `t_{cc'}X + t_{c'c}Y` if `c ∼ c'`, `1` otherwise:

| case | composite mod `(r)` |
|---|---|
| (A1), (B1) | `β_τ` |
| (A2), (B2) | `β_{(a b)}` |
| (A3) | `Q_{c(x)c(p)}(x_x, x_a) · β_{(a b)}` |
| (B3) | `Q_{c(x)c(p)}(x_x, x_b) · β_{(a b)}` |

(Lemma 2.2.) In words: a product of a one-pair term with an elementary move is non-reduced
exactly when the moved slot `x` lies strictly inside the pair; the bigon is then between `x`
and the strand of `a` (move first) or of `b` (move last); it gives `0` if `c(x) = c(p)`, a
dot term if `c(x) ∼ c(p)`, and nothing if the colours are far apart.

`checks/04_order01.py`, test (T2), computes every such product exactly, compares its
`r`-degree-0 part with the table and the crossing numbers of the concatenated word with
Prop. 5.3: 0 failures; counts on all words of length ≤ 6 of `A_3` (every red colour):
(A1) 56 460, (A2) 41 112, (A3) 20 556 adjacent / 6 852 far / 13 704 same colour, and the same
numbers for (B).

---

## 6. Order 1

**Theorem C.** For every word and every colour of the red strand,

    δ₀² + δ₀δ₁ + δ₁δ₀ ≡ 0   mod (r)².

Since `δ_n ∈ (r)^n` for `n ≥ 2`, this says `d² ≡ 0 mod (r)²` for the tree formula `d`.

*Proof.* `(δ₀δ₁)_{IK} = Σ_x δ₀_{I,I∖x} δ₁_{I∖x,K}` consists of the products (A) of Prop. 5.3 with
coefficient `s_A r_{c(p)}t_{c(p)c(j)}`, `s_A = (−1)^{n_I(x)} σ₁(I∖x;j,p)`;
`(δ₁δ₀)_{IK}` of the products (B) with `s_B = σ₁(I;j,p)(−1)^{n_J(x)}`. Every term carries one
factor `r t`, so only the composites modulo `(r)` matter (Cor. 5.4). We sort by the target.

*(i) Targets `K = I∖{i,j}`.* From `δ₀²`: by Theorem B, `ε(I;i,j) r_{c(j)}t_{c(j)c(i)} β_{τ_y}`
for each `y ∈ Bad(i,j)`. From (A): a product (A) has target `I∖{x,j',b'}∪{a'}` with `a' ∉ I∖x`;
it equals `I∖{i,j}` only if `a' = x` (else `a' ∈ K∖I`), and then `{j', b'} = {i, j}`, `j' < b'`,
so `j' = i`, `b' = j`, `x = a' = y` with `y ∈ I`, `c(y) = c(j) ∼ c(i)`, `y < i`: exactly the
bad `y ∈ I`, case (A1), value `β_{τ_y}`. Likewise (B) contributes exactly the bad `y ∉ I`, case
(B1). Signs: for `y ∉ I`, with `J = I∖{i,j}∪{y}`,
`s_B = (−1)^{n_I(i)+N_I(y,j)} (−1)^{n_I(y)−2}` and `n_I(y) = N_I(y,j) + 1 + n_I(j)`, so
`s_B = −ε(I;i,j)`. For `y ∈ I`, `s_A = (−1)^{n_I(y)}(−1)^{n_{I∖y}(i) + N_{I∖y}(y,j)} =
(−1)^{n_I(y) + n_I(i) + N_I(y,j)} = −ε(I;i,j)` likewise. The monomials agree
(`r_{c(p)} t_{c(p)c(i)} = r_{c(j)}t_{c(j)c(i)}`). So every bad triple cancels.

*(ii) Targets `K = I∖{x,j,b}∪{a}` with `a ∉ I`* (these contain `a ∉ I`, so `δ₀²` does not
contribute). The triples `(x, j, p)` with `x ∈ I∖{j,b}` occur in (A) iff they occur in (B).
Signs: `n_{I∖x}(j) = n_I(j) − [x>j]`, `N_{I∖x}(a,b) = N_I(a,b) − [a<x<b]`,
`n_J(x) = n_I(x) − [j>x] − [b>x] + [a>x]`, whence
`s_A s_B = (−1)^{1 + [a<x<b] + [x<b] + [x<a]} = −1` in each of the three cases `x<a`, `a<x<b`,
`x>b`. The bijection `(a b)` (with `x, j` moving) is produced by `(x, j, p)` and, if it is also a
configuration, by `(j, x, p)` (`x` the mover, `j` the elementary move), and by nothing else
(the bijection determines `p`, and the two strands that cross `W` without changing slot are
`x, j`). `(j, x, p)` is a configuration iff `a < x < b` and `c(x) ∼ c(p)`.
* `x ∉ (a,b)`: (A2) and (B2) give `s_A r t β + s_B r t β = 0`.
* `a < x < b`, `c(x) = c(p)`: (A3), (B3) are `0 mod (r)`; `(j,x,p)` is not a configuration.
* `a < x < b`, `c(x)`, `c(p)` distinct and not adjacent: `Q = 1`, so (A3) + (B3) `= 0`; `(j,x,p)`
  is not a configuration.
* `a < x < b`, `c(x) ∼ c(p)`: four products. With `s_{A'} = (−1)^{n_I(j)}σ₁(I∖j;x,p)` one gets
  `s_A s_{A'} = (−1)^{[x>j]+[j>x]} = −1`, and `s_{B'} = −s_{A'}`. Writing `t_{uv}` for
  `t_{c(u)c(v)}`, the sum is
  `s_A r_{c(p)} [ t_{pj}(t_{xp}x_x + t_{px}x_a) − t_{pj}(t_{xp}x_x + t_{px}x_b) − t_{px}(t_{jp}x_j + t_{pj}x_a) + t_{px}(t_{jp}x_j + t_{pj}x_b) ] β = 0`.

No other products occur, so all terms of `r`-degree ≤ 1 cancel. ∎

*Checks.* `checks/04_signs.py` (pure Python) verifies all sign identities of the proof
((C1) bad triples, (C2) `s_A = −s_B`, (C3) `s_{A'} = −s_A`) against `formula.py` for all words
of length ≤ 8 (`A_2`), ≤ 6 (`A_3`), ≤ 5 (`D_4`). `checks/04_order01.py`, test (T3), computes
`δ₀² + δ₀δ₁ + δ₁δ₀` exactly in `A` and checks that it has no term of `r`-degree ≤ 1: 0 failures
on all words of length ≤ 7 of `A_2` (all red colours, 2 × 254 words), ≤ 6 of `A_3`
(3 × 1092), ≤ 5 of `D_4` (4 × 1364) and 300 random words of length 6 of `D_4` (4 × 300).

*Remark 6.1 (where the dot terms go).* The only dot terms at first order are the bigons
`t x` of the (A3)/(B3) products with `c(x) ∼ c(p)`. They cancel in the group of four products
belonging to the two ways of reading `{x, j}` as (mover, elementary move); the cancellation
uses `t_{c(p)c(j)} t_{c(p)c(x)} = t_{c(p)c(x)} t_{c(p)c(j)}`, i.e. that the coefficient of a
one-pair term depends on the mover only through `t_{c(p) c(mover)}`.

---

## 7. Degree calculus for arbitrary bijections

The following generalises PROOFS Prop. 3 (which treats `σ` = mover + disjoint pairs) to every
colour-preserving bijection `σ` of slots and every pair of vertices `I, J` (any sizes).

**Lemma 7.0.** For slots `u < v`, the strand at `u` lies to the left of the strand at `v` in
`e_I` iff `v ∈ I`; the strand at `u` lies to the left of `W` iff `u ∉ I`. Hence
`D(I) = Σ_{u<v, v∈I} [c(u),c(v)] + Σ_{v∈I} w_{c(v)}`.

*Proof.* Compare `pos_I(u) = ±u` and `pos_I(v) = ±v`; the formula is the definition of `D`. ∎

**Proposition 7.1 (degree defect).** Let `Δ(I,J,σ) = D(I) − D(J) − deg β_σ`. For a pair of
strands `A, B` with source slots `s_A < s_B` put

    f(A,B) = −2·[s_B ∉ I and σ(s_B) ∈ J]          if σ(s_A) < σ(s_B),
    f(A,B) = +1 if s_B ∈ I and σ(s_A) ∉ J,  −1 otherwise,   if σ(s_A) > σ(s_B).

Then `Δ(I,J,σ) = Σ_{pairs {A,B}} [c_A, c_B]·f(A,B) − 2 Σ_{A ∈ LR(σ)} w_{c_A}`, where `LR(σ)` is the set
of strands with `s_A ∉ I`, `σ(s_A) ∈ J` (crossing `W` from left to right).

*Proof.* Pairs of slots of the source and of the target both correspond bijectively to pairs
of strands, with the same colours. For a pair `s_A < s_B`: `D(I)` contributes
`[c,c']·[s_B ∈ I]` (Lemma 7.0), `D(J)` contributes `[c,c']·[max(σs_A, σs_B) ∈ J]`, and `β_σ`
contributes `[c,c']` iff the order is reversed. By Lemma 7.0, `A` is left of `B` at the source
iff `s_B ∈ I`; at the target iff `σ(s_B) ∈ J` when `σ(s_A) < σ(s_B)`, and iff `σ(s_A) ∉ J` when
`σ(s_A) > σ(s_B)`. Writing `α = [s_B ∈ I]`, `μ = [max ∈ J]`, the contribution is
`α − μ − [order reversed]`: in the first case `α − μ − |α − μ| = −2[α=0, μ=1]`; in the second
the four values of `(α, μ)` give `−1, −1, +1, −1` for `(0,0), (0,1), (1,0), (1,1)`. For the red
strand: `[s ∈ I] − [σ(s) ∈ J] − [s crosses W] = −2[s ∉ I, σ(s) ∈ J]`. ∎

(`checks/04_degree_defect.py defect`: 40 000 random `(word, red, I, J, σ)` in `A_3`, `D_4`,
compared with the direct count of `degree_model.py`; 0 failures. Pure Python.)

**Corollary 7.2.** A basis element `μ x^α β_σ` of `e_I A e_J`, `μ = ∏ r_c^{n_c} t_{cc'}^{m_{cc'}} u_c^{l_c}`,
`a_c` = number of dots on strands of colour `c`, is homogeneous of degree `D(I) − D(J)` iff

    n_c = Ŝ_c := Σ_{pairs of colour c} f,   l = 0,   LR(σ) contains no strand of colour k,
    m_{cc'} + m_{c'c} = Ê_{cc'}/2,  Ê_{cc'} := Σ_{pairs of colours c, c'} f,
    a_c = Σ_{c' ∼ c} m_{cc'} − n_c  ≥ 0.

In particular the `r`-degree `Σ_c Ŝ_c` of a homogeneous element is determined by `(I, J, σ)`,
and every `r_c` must be accompanied by some `t_{cc'}` or by a dot deficit
(`Σ_{c'} m_{cc'} ≥ n_c`).

*Proof.* Compare coefficients of `σ_c, ε_{cc'}, ε_{cW}, δ_c` in `Δ` (Prop. 7.1) and in
`deg μ x^α = Σ n_c(σ_c+δ_c) + Σ m_{cc'}(2ε_{cc'}−δ_c) + Σ l_c(2ε_{cW}−δ_c) + Σ a_c(2q+δ_c)`, as
in PROOFS Cor. 3.1; `ε_{kW}` gives `2l_k = −2·#(LR ∩ colour k)`, both sides `≥ 0`/`≤ 0`. ∎

*What there is at first order in general (Tier 2, `checks/04_degree.py`).* For all words of
length ≤ 6 in `A_2` (both red colours), the first-order homogeneous basis families of
homological degree −1 are: 5 828 one-pair terms; 7 010 + 880 (with dots) families in which a
strand of colour `≠ k` crosses `W` from left to right (e.g. `1121`, red 2, `{3,4} → {1}`,
`σ = (2 4)`, `r_1 t_{12}`); 10 216 + 352 (with dots) *monotone* families that are not one-pair
terms, e.g. hand-over cycles (`11121`, red 1, `{4,5} → {1}`, `σ = (1 3 2 5)`, `r_1 t_{12}`) and
dotted ones (`111211`, red 1, `r_1 t_{12} t_{21} x_2`). In homological degree 0 there are
first-order elements too (e.g. `1212`, red 2, `{3,4} → {1,2}`, `σ = (2 4)`, `r_2 t_{21}`), and
`r`-free elements with `t` (e.g. `1121`, red 2: `t_{12} x β`). So Cor. 3.1 of PROOFS (dot-free,
mover + pairs) is far from describing all homogeneous elements; the next proposition is the
case where it does.

Call a bijection (or a basis element, or a map) *monotone* if no strand crosses `W` from left
to right (`LR(σ) = ∅`).

**Proposition 7.4 (multiplicity ≤ 2).** Suppose every colour occurs at most twice in the word.
(a) The monotone homogeneous basis elements of `r`-degree 1 in `e_I A e_J`, `|J| = |I| − 1`, are
exactly the `r_{c(p)} t_{c(p)c(j)} β(I;j,p)` for the one-pair configurations `(j,p)` at `I` (no
dots, unique monomial). (b) There are no monotone homogeneous basis elements of `r`-degree 1
with `|J| = |I|`.

*Proof.* Monotone with `|J| = |I| − h'` means: exactly `h'` strands cross `W`, all from right to
left. Every colour class has ≤ 2 slots, so `σ` fixes or swaps each class. For a fixed class
`{u < v}`, `f = −2[v ∉ I, v ∈ J] = 0` (monotone: `v ∉ I ⇒ v ∉ J`). For a swapped class, `f = ±1`.
`r`-degree 1 and `n_c ≥ 0` force exactly one swapped class `{u < v}` (colour `c₀`), with `f = +1`,
i.e. `v ∈ I`, `v ∉ J`. All other strands `z` are fixed. Pairs of fixed strands: `f = −2[...] = 0`
by monotonicity. Pairs `(z, A)`, `(z, B)` (`A`, `B` the strands of `u`, `v`): `f = 0` unless
`u < z < v`, in which case `f(z,A) = +1` if `z ∈ I`, `−1` if not, and `f(z,B) = +1` if `z ∉ J`,
`−1` if not (direct from the definition). Hence `Ê_{c₀ c(z)}` receives `0` from an unaffected
`z` (`z ∈ I ⇔ z ∈ J`), `+2` from a `z` with `z ∈ I`, `z ∉ J`, `u < z < v` (a mover), and `z ∉ I`,
`z ∈ J` is excluded; all other `Ê` vanish (only `A`, `B` move, and the class of `c₀` has no third
strand). Since `n_{c₀} = 1`, Cor. 7.2 needs `Σ_{c'} m_{c₀c'} ≥ 1`, so some `Ê_{c₀c'} ≥ 2`: there
is a strand `z = j` with `j ∈ I`, `j ∉ J`, `u < j < v`, `c(j) ∼ c₀`.
(b) If `|J| = |I|`, no strand crosses `W`, so there is no such `j`: impossible.
(a) Exactly one strand crosses `W`; it is `j`, so `A` (from `u`) and `B` (from `v`) do not cross
`W`: `B` stays right, so `u ∈ J`; `A` ends at `v ∉ J`, on the left, so `u ∉ I`. Hence
`J = I∖{j,v}∪{u}`, `σ = (u v)`, `(j, (u,v))` is a one-pair configuration; `Ê_{c₀ c(j)} = 2`, so
`m_{c₀c(j)} + m_{c(j)c₀} = 1`, and `a_{c₀} = m_{c₀c(j)} − 1 ≥ 0` gives `m_{c₀c(j)} = 1`,
`a = 0`. Conversely these elements are homogeneous (PROOFS Cor. 3.1). ∎

(`checks/04_degree_defect.py mult2`: exhaustive over all words with multiplicity ≤ 2 of
length ≤ 6 in `A_2`, `A_3`, `A_4`, `D_4`, every red colour; see Status for the ranges that
finished.)

---

## 8. Forcedness at first order

### 8.1 The hypothesis
Let `d = δ₀ + ε` be a map of homological degree −1 (all components homogeneous of degree
`D(I) − D(J)`) with `d² = 0`. We use

* **(H1)** `ε ≡ 0 mod (r)` — every coefficient (in the basis F1) of every component of `ε` is
  divisible by some `r_c`.

(H1) is stronger than "ε vanishes when all parameters are 0" (H2). `u` never occurs anyway
(PROOFS Prop. 4). Under (H2) alone no uniqueness statement holds even modulo `(r)`: see
Remark 8.7. Write `ε₁` for the class of `ε` in `(r)Hom/(r)²Hom`; by Lemma 1.1 it is canonical, and
it is a `ℤ`-combination of *first-order homogeneous basis elements* `μ x^α β_σ` (`r`-degree of
`μ` equal to 1, homogeneous of the right degree) — homogeneity forces the coefficients to be
integers, since the monomial and the dots are then fixed up to the finitely many solutions of
Cor. 7.2. Let `V^h_1` be the `ℤ`-span of the first-order homogeneous basis elements of
homological degree `h`, and

    Z₁ = { η ∈ V^{−1}_1 : δ₀η + ηδ₀ ≡ 0 mod (r)² },     B₁ = { hδ₀ − δ₀h mod (r)² : h ∈ V^0_1 }.

**Theorem 8.1 (reduction).** (a) Under (H1), `ε₁ − δ₁ ∈ Z₁`. Conversely every `δ₁ + η`,
`η ∈ Z₁`, satisfies `(δ₀ + δ₁ + η)² ≡ 0 mod (r)²`. (b) `B₁ ⊆ Z₁`. (c) For `h ∈ V^0_1`, `g = 1 + h`
is invertible, and if `d` is a differential satisfying (H1) then so is `g d g^{−1}`, with
first-order part `ε₁ + (hδ₀ − δ₀h)`.

*Proof.* (a) `d² = δ₀² + δ₀ε + εδ₀ + ε²`, `ε² ∈ (r)²`; subtract Theorem C. Conversely the same
computation. (b) `δ₀(hδ₀ − δ₀h) + (hδ₀ − δ₀h)δ₀ = hδ₀² − δ₀²h ∈ (r)·(r)` by Theorem A.
(c) In a fixed component `e_I A e_{I'}` of fixed degree, the `r`-degree of a homogeneous
element is bounded (Cor. 7.2: `n_c = Ŝ_c ≤` number of pairs of colour `c`), so `h^n = 0` for
large `n` and `g^{−1} = Σ (−h)^n`. `g d g^{−1}` squares to 0, is homogeneous, has geometric part
`δ₀`, satisfies (H1) since `h ∈ (r)`, and `g d g^{−1} ≡ d + hd − dh mod (r)²`. ∎

So the first-order part of a differential is at best determined modulo `B₁`.

**Example 8.3 (B₁ ≠ 0; the first-order part is not unique).** Word `1212` (`A_2`), red colour
2. `V^0_1` is spanned by one element `h : e_{{3,4}} → e_{{1,2}}`, `σ = (2 4)`, `r_2 t_{21}`, the
basis braid `ψ_2ψ_3ψ_2` (the strand of slot 1 crosses `W` from left to right, that of slot 3
from right to left). The tree formula `d` is a differential, and `d' = (1+h) d (1−h)` (here
`h² = 0`) is a differential with `d' − δ₀ ∈ (r)A` and `d' ≢ d mod (r)²`: the first-order part
changes in the four components `{3,4}→{2}`, `{3,4}→{1}`, `{1,3,4}→{1,2}`, `{2,3,4}→{1,2}`
(`checks/04_gauge_example.py 1212 2`). The same happens for `1121` (red 2), `1321` (red 1, 2),
`12112` and many more (Comp. 8.5). Note that `d'` is not monotone.

**Theorem 8.4 (forcedness).** Suppose every colour occurs at most twice in the word. Let
`d = δ₀ + ε` be a differential satisfying (H1) and

* **(M₁)** `ε₁` is a combination of monotone first-order basis elements (no strand crosses `W`
  from left to right; by Lemma 2.2 this is a property of the class `ε₁`).

Then `ε ≡ δ₁ mod (r)²`.

*Proof.* By Prop. 7.4(a), `ε₁ = Σ λ_{(I;j,p)} r_{c(p)}t_{c(p)c(j)} β(I;j,p)` with `λ ∈ ℤ`, and the
same holds for `η := ε₁ − δ₁`, which lies in `Z₁` (Thm 8.1). By Lemma 8.5, `η = 0`. ∎

**Lemma 8.5 (injectivity; every word).** Let `η = Σ μ_{(I;j,p)} r_{c(p)}t_{c(p)c(j)} β(I;j,p)`,
sum over all vertices and one-pair configurations, `μ ∈ ℤ`. If `δ₀η + ηδ₀ ≡ 0 mod (r)²`, then
all `μ = 0`.

*Proof.* Fix `(I; j, p)`, `p = (a,b)`, put `K = I∖{j,b}` and let `τ` be the bijection of (B1). We
compute the coefficient of `r_{c(p)}t_{c(p)c(j)}β_τ` in `(δ₀η + ηδ₀)_{IK}` modulo `(r)²`, using
Cor. 5.4. `(ηδ₀)_{IK} = Σ η_{IJ'} δ₀_{J'K}` with `J' = I∖{j',b'}∪{a'}`, `K = J'∖x`; since
`a' ∉ I ⊇ K`, `x = a'`, hence `{j',b'} = {j,b}`, `j' = j`, `b' = b`, and the product (B1) has
bijection `(a' b)` — equal to `τ` iff `a' = a`. So this part contributes exactly
`± μ_{(I;j,p)} r t β_τ`. `(δ₀η)_{IK} = Σ_y δ₀_{I,I∖y}η_{I∖y,K}` needs `K = (I∖y)∖{j',b'}∪{a'}` with
`a' ∉ I∖y`, `a' ∈ K ⊆ I`, so `a' = y ∈ I`, and its bijection exchanges `y` and `b'`; it equals `τ`
only if `y = a`, impossible since `a ∉ I`. So the coefficient is `± μ_{(I;j,p)} r t`, which lies in
`(r)²` only if `μ = 0` (Lemma 1.1(b)). ∎

**Computation 8.5 (Tier 2; `checks/04_forced.py`).** The script enumerates all first-order
homogeneous basis elements (dots included) of homological degrees −1 and 0 with
`checks/04_degree.py`, computes the matrices of `η ↦ δ₀η + ηδ₀` and `h ↦ hδ₀ − δ₀h` modulo
`(r)²` exactly, and their ranks over `ℚ`.
* `A_3`, all 1089 (word, red) pairs of length ≤ 5: `dim Z₁ = dim B₁` in all cases
  (`Z₁ = 0` in 839, `Z₁ ≠ 0` in 250). The minuscule words `2132`, `2312` (red 2) have `Z₁ = 0`.
* `A_2`, length ≤ 6, and `D_4`, length ≤ 4: see Status (runs listed there).
* Restricting to monotone elements (`-M`): still `Z₁^{mon} = B₁^{mon}`, but `B₁^{mon} ≠ 0`
  occurs once a colour occurs three times (e.g. `12112`, red 2: `dim = 2`), so Theorem 8.4
  needs the multiplicity hypothesis even under (M₁).

So, in every computed case, **the first-order part of any differential satisfying (H1) is
`δ₁` up to a first-order gauge transformation**; it is `δ₁` itself when `Z₁ = 0`. A proof of
`Z₁ = B₁` in general would need a description of the first-order cohomology of
`End_{A₀}(cube)`; the degree calculus alone does not give it (the zoo of §7).

**Remark 8.6 (the role of (M₁)).** Monotonicity holds for the tree formula and for every term
found in the cyclic regime (NOTES §6, and §9 below). PROOFS Prop. 4 proves it for strands of
the red colour; for other colours it is a normalisation which excludes exactly the
non-monotone gauges of Example 8.3.

**Remark 8.7 (why (H2) is not enough).** For `1121`, red 2, there are 8 homogeneous elements
`h₀` of homological degree 0 and `r`-degree 0 containing `t` (e.g. `{3,4}→{1,2}`,
`σ: 1↦4, 2↦1, 4↦2`, `t_{12}x β`). For each of them `(1+h₀) d (1−h₀)` (tree formula `d`) is a
differential whose terms of `d' − d` all contain a parameter — so `d'` satisfies (H2) — but
`d' − δ₀ ∉ (r)A`: it has `r`-free terms `±t_{12} x β` (`checks/04_gauge_example.py 1121 2 0`).
The linearised count agrees: the `r`-free, `t`-dependent cycles of `[δ₀,·]` modulo `(r)` coincide
with the gauge directions (`dim 8 = 8` here; equality in all cases of `04_forced.py`, columns
`Z0t`, `B0t`).

---

## 9. The smallest cyclic example: 1221122, red strand of colour 2

Slots `1:c1, 2:c2, 3:c2, 4:c1, 5:c1, 6:c2, 7:c2`. The only cyclic configuration
(`cyclefree.find_cycle`, all configurations enumerated) is mover `5`, pairs
`(1,4)` (colour 1), `(2,7)`, `(3,6)` (colour 2), from `I₀ = {4,5,6,7}` to `J₀ = {1,2,3}`, with the
4-cycle of links `5 – (2,7) – (1,4) – (3,6) – 5` (`(2,7)` and `(3,6)` are nested, so not
blocked). With respect to the order by `b` (mover: `b = 5`) the nodes `5` and `(1,4)` are
sources and `(2,7)`, `(3,6)` sinks; there is no in-between node.

**Theorem 9.1 (Tier 2: exact computation).** For this stack:

(a) There is no dot-free homogeneous basis element of homological degree −1 and `r`-degree 3
in any component (dot-free homogeneous elements by `r`-degree: 2447, 1464, 204, 0, 7 for
`r`-degree 0–4).

(b) The tree formula `d_T = δ₀ + δ₁ + δ₂` satisfies `d_T² ≡ 0 mod (r)³`, and the `r`-degree-3 part
of `d_T²` is non-zero (28 coefficients, all with one dot, in the components `I₀ → {2,3}`,
`I₀ → {1,3}`, `{1}∪I₀ → J₀`, `{2}∪I₀ → J₀`; bijection = cyclic configuration followed by an
elementary move).

(c) Consequently there is **no differential `d_T + ε` with `ε` a combination of dot-free terms
of `r`-degree ≥ 2** — neither with `ε ∈ (r)³` (then the `r³`-part of `d²` is that of `d_T²`, by (a)),
nor with `ε ∈ (r)²`: the dot-free `r`-degree-2 elements `κ` with `δ₀κ + κδ₀ ≡ 0 mod (r)³` form a
4-dimensional space, and the `r³`-parts of `[δ₀ + δ₁, κ]` span a 4-dimensional space not
containing `(d_T²)_3` (rank 4 → 5).

(d) There is a differential `d_T + X` with `X` the 10 terms below, all in the single component
`I₀ → J₀`; at each of the four parameter orders of `X` the complete ansatz (all homogeneous
basis elements of that order, dots allowed) has kernel 0, so `X` is the unique correction at
these orders given the tree formula below them. All 680 tree-formula terms occur with their
coefficients.

*What the computation assumes.* The correctness of the `klrw` implementation of the algebra
(products, basis) and of the library solver; the completeness of the pure-Python degree
enumeration (`degree_model.py`, `04_degree.py`), which is a finite search over all bijections
and monomials using Cor. 7.2; (F1). Scripts: `checks/04_cyclic7_dotfree.py` (a)–(c),
`checks/04_cyclic7_solve.py` (d) (data in `checks/04_data/cyclic7_1221122_W2_tree.json`).

*Not shown:* that there is no dot-free differential `δ₀ + ε` with arbitrary `ε ∈ (r)`. The
dot-free first-order cycles form a 106-dimensional space (mostly non-monotone gauge
directions), which turns the question into a large quadratic system; under (M₁) plus
(H1) the first-order part is not forced here either (colour 1 occurs three times), so this
remains open.

**The extra (cyclic) terms.** Dot variables `x_{c,k}` = `k`-th strand of colour `c` from the left
in `e_{I₀} = (3, 2, 1, W, 4, 5, 6, 7)`; so `x_{2,1}, x_{2,2}, x_{2,3}, x_{2,4}` are the strands of
slots `3, 2, 6, 7` and `x_{1,1}, x_{1,2}, x_{1,3}` those of slots `1, 4, 5`. With
`β_cyc` = basis braid of `σ_cyc = (1 4)(2 7)(3 6)` (mover 5):

    X_dot = − r_1 r_2² t_12 t_21² · [ t_21 (x_{2,1} + x_{2,2} + x_{2,3} + x_{2,4}) + t_12 (x_{1,1} + x_{1,2} + 2 x_{1,3}) ] · β_{σ_cyc}
    X_r   = + r_1 r_2³ t_12 t_21³ β_{σ'}  − r_1 r_2³ t_12 t_21³ β_{σ''}  − r_1² r_2² t_12² t_21² β_{σ'''}
    X     = X_dot + X_r,

    σ_cyc = {1↦4, 2↦7, 3↦6, 4↦1, 6↦3, 7↦2},   σ'  = {1↦4, 2↦6, 3↦7, 4↦1, 6↦3, 7↦2},
    σ''   = {1↦4, 2↦7, 3↦6, 4↦1, 6↦2, 7↦3},   σ''' = {1↦5, 2↦7, 3↦6, 4↦1, 5↦4, 6↦3, 7↦2}

(slot `5` is fixed by `σ_cyc, σ', σ''`: its strand is the one crossing `W`; in `σ'''` the strand
of slot 5 ends in slot 4 and the strand of slot 4 in slot 1).

*Comparison with NOTES §6.* Every spanning tree `G − e` of the 4-cycle has the same monomial
`r_1 r_2² t_12 t_21²` (the two colour-2 pairs hang below colour-1 nodes, the colour-1 pair
below a colour-2 node), and the four left-over links contribute `t_12[dot on the colour-1
node] + t_21[dot on the colour-2 node]`. Summing with one common tree sign gives each
colour-2 pair one dot on each of its two strands (`x_{2,1}…x_{2,4}`), the pair `(1,4)` one dot on
each strand (`x_{1,1}, x_{1,2}`), and the mover two dots (`2x_{1,3}`) — exactly the dotted part
of `X`. The dot-free terms with an extra `r` are the corrections from moving these dots,
which sit at the touching points of the time-ordered drawing, to the left end of the basis
element (they exchange two same-colour strands: `(2,3)/(6,7)` for the colour-2 dots, `4 ↔ 5`
for the mover's dot). `sage -python cycle3.py proofs/checks/04_data/cyclic7_1221122_W2_tree.json`
(the single-cycle rule of NOTES §6, read-only) reproduces the whole differential with **0 wrong
components** (all three variants `mix`, `endL`, `roleL`; cyclic component 1/1).

*What the example adds.* It is the first cyclic stack with two pairs of the *same* colour
on the cycle (`A_2` has only one neighbour of colour 1), and it has only 7 strands, so the
full cube (128 vertices) is solved, not just an interval. Its cycle has only sources and
sinks, so it does not test the hand-over part of the rule (in-between nodes), and no
hand-over terms occur. It confirms: (1) the tree formula is untouched; (2) the cyclic
correction lives in the one component of the cyclic configuration; (3) it is the spanning-tree
× bigon sum with a common sign, one dot per strand of each pair and two on the mover.

---

## 10. Second order: what has to be proved

Write `d = δ₀ + δ₁ + δ₂ + δ_{≥3}`, `δ_{≥3} ∈ (r)³`. Modulo `(r)³`,

    d² ≡ [δ₀² + δ₀δ₁ + δ₁δ₀] + [δ₀δ₂ + δ₁δ₁ + δ₂δ₀]     mod (r)³.

The first bracket lies in `(r)²` (Theorem C); call `E₂` its class in `(r)²A/(r)³A`. The second
bracket lies in `(r)²` trivially; only its value modulo `(r)³`, i.e. the composites modulo `(r)`,
matters. The statement to prove is

    (∗)   E₂ + (δ₀δ₂ + δ₁δ₁ + δ₂δ₀ mod (r)³) = 0      in (r)²A/(r)³A ≅ ⊕_{c ≤ c'} r_c r_{c'} · A₀.

Ingredients, each a finite case analysis of the kind of §5:

1. `E₂` needs the composites of §5 **modulo `(r)²`**, not `(r)`: for each product (A), (B) and
   for `δ₀²` (Theorem B gives `δ₀² = Σ ε r t X_y` exactly, so its contribution to `E₂` is
   `Σ ε r t (X_y − β_{τ_y})`, which vanishes in the `klrw` basis in all checked cases), the
   `r`-linear difference between the concatenated word (with its bigon polynomial) and the
   basis element. Sources: bad braid moves needed to bring the concatenated word to the
   chosen basis word (each a smoothing `r t`, which exchanges two more same-coloured strands),
   and the corrections `±r` from sliding the bigon dots `x_x, x_a, x_b` to the left end.
2. `δ₁δ₁ mod (r)`: products of two one-pair terms `(j, p)` then `(j', p')`: a classification of
   their crossing numbers as in Prop. 5.3 (the second mover may lie inside the first pair, the
   pairs may share a slot: `a' = b` or `b' = a` etc.).
3. `δ₀δ₂ + δ₂δ₀ mod (r)`: products of two-pair tree terms (chains `mover – p – p'` and stars
   `p – mover – p'`) with elementary moves, classified as in Prop. 5.3.

The components of (∗) are indexed by `(I, K, bijection)`. "Two pairs of different colours"
means the bijection is `(a b)(a' b')` with `c(a) ≠ c(a')`; then `r_c r_{c'}`, `c ≠ c'`, and the
contributions come from (2) with `p, p'` the two pairs, (3) with the two-pair terms with pair
set `{p, p'}`, and (1) only through bad moves whose smoothing creates the second exchange.
This sector was **not** proved here; it is verified (as part of `d_T² = 0`) in all cases of
PROOFS Tier 2 and of `checks/general_words.py`.

---

## 11. Status

**Proved completely** (from (R1)–(R4) with the README signs, (F1) where stated, (F3) Matsumoto
and (F3′) Tits' word property):
* Lemma 1.1 (legitimacy of working modulo `(r)^n`; (b) uses F1).
* Lemma 2.2 (normal form modulo `(r)`; uses F3, F3′).
* Theorem A (`δ₀² ≡ 0 mod (r)`), every word.
* Theorem B (exact first-order formula for `δ₀²`, bad triples exactly `c(y)=c(j)∼c(i)`, red
  never the middle strand, smoothed words reduced), every word; Cor. B.2; Cor. B.3 (uses F1).
* Prop. 5.3 / Cor. 5.4 (classification of the products of one-pair terms with elementary
  moves, including all non-reduced cases and their values modulo `(r)`).
* Theorem C (`δ₀² + δ₀δ₁ + δ₁δ₀ ≡ 0 mod (r)²`), every word, every red colour. With the Theorem
  of PROOFS Prop. 3/Cor. 3.1 this proves: the tree formula is a homogeneous differential
  modulo `(r)²` for every word.
* Prop. 7.1 (degree defect of an arbitrary bijection), Cor. 7.2.
* Prop. 7.4 (multiplicity ≤ 2: the monotone first-order elements are the one-pair terms).
* Theorem 8.1 (reduction of forcedness to `Z₁`, gauge directions `B₁ ⊆ Z₁`), Lemma 8.5
  (injectivity, every word), Theorem 8.4 (multiplicity ≤ 2 + (H1) + (M₁) ⇒ `ε ≡ δ₁ mod (r)²`).

**Proved by a finite exact computation (Tier 2):**
* Example 8.3 (`1212`, red 2: a gauge-equivalent differential with a different first-order
  part) and Remark 8.7 (`1121`, red 2: `r`-free gauges under (H2)).
* Theorem 9.1 (`1221122`, red 2): no dot-free completion of the tree formula by terms of
  `r`-degree ≥ 2; the true differential, its 10 extra terms, uniqueness at their orders,
  agreement with the single-cycle rule of NOTES §6.

**Verified only by computation:**
* `Z₁ = B₁` (first-order cohomology vanishes; uniqueness up to gauge): all words of length ≤ 5
  in `A_3` (1089 cases); `A_2` length ≤ 6 and `D_4` length ≤ 4, and the monotone variants, as
  far as the runs listed below finished.
* The observation that `X_y` equals the `klrw` basis element exactly.

Runs (all 0 failures): `04_order01.py` (T0–T3) on `A_2` ≤ 7, `A_3` ≤ 6, `D_4` ≤ 5 (all words,
all red colours) and 300 random `D_4` words of length 6; `04_sweep.py` (S1–S5) on `A_2` ≤ 7,
`A_3` ≤ 5, `D_4` ≤ 4; `04_signs.py` on `A_2` ≤ 8, `A_3` ≤ 6, `D_4` ≤ 5; `04_degree_defect.py
defect` (40 000 samples); `04_degree_defect.py mult2` on `A_2` ≤ 4, `A_3` ≤ 5;
`04_forced.py` on `A_3` ≤ 5 (all); `04_gauge_example.py`; `04_cyclic7_*.py`.

**Not done / failed attempts:**
* A proof of `Z₁ = B₁` in general. The degree calculus (Prop. 7.1) shows that the first-order
  homogeneous elements form a zoo (hand-over cycles, strands crossing `W` from left to right,
  dotted elements), so a proof by classification is hopeless beyond multiplicity 2; it needs
  a cohomological argument (e.g. via the cone structure `C(w·c) = Cone(F)` of the README).
* Theorem 8.4 without the multiplicity hypothesis: false as stated (monotone gauges exist for
  `12112`, red 2); the correct general statement is presumably "unique up to monotone gauge".
* Theorem 9.1 for arbitrary dot-free `ε ∈ (r)` (open, see §9).
* Second order (§10): formulated, not proved, not even in the different-colour sector.
* `D_4` second-order / `A_7 ω_4` cyclic terms were not attempted.
