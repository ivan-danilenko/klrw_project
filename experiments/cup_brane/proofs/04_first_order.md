# 04 — First-order theory and forcedness, for arbitrary words

> revised after R04_referee.md

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
| Thm B, Lemma 4.7 | exact formula `(δ₀²)_{I,I∖{i,j}} = ε(I;i,j) r_{c(j)} t_{c(j)c(i)} Σ_{y} β_{τ_y}`, sum over the bad triples `y<i<j`, `c(y)=c(j)∼c(i)`, `β_{τ_y}` any reduced word of the exchange `τ_y` (no `r²`-terms in any basis) | none | **proved** |
| Cor B.3 | `δ₀² = 0` iff the word has no slots `y<i<j` with `c(y)=c(j)∼c(i)`; otherwise `δ₀² ∉ (r)²`, and every differential `δ₀+ε`, `ε∈(r)`, has a non-zero first-order part | (F1) for "otherwise" | **proved** |
| Prop 5.3 | classification of the products (elementary move)·(one-pair term) and (one-pair term)·(elementary move): reduced, or exactly one bigon `{x, strand of a or b}` | none | **proved** |
| Thm C | `δ₀² + δ₀δ₁ + δ₁δ₀ ≡ 0 mod (r)²`, i.e. the tree formula is a differential to first order | none | **proved** |
| Prop 7.1 | degree defect of an *arbitrary* bijection (generalises PROOFS Prop. 3) | none | **proved** |
| Prop 7.4 | if every colour occurs ≤ 2 times: the first-order homogeneous elements in which no strand crosses `W` from left to right are exactly the one-pair terms | multiplicity ≤ 2; (F1) | **proved** |
| Thm 8.1 | any differential `δ₀+ε`, `ε ∈ (r)`, has first-order part `δ₁ + η`, `η` a first-order cycle; if `d` is such a differential and `h` a first-order gauge, `(1+h)d(1+h)^{−1}` is one with first-order part changed by `[h,δ₀] := hδ₀ − δ₀h` (this needs a differential `d` to start from: for arbitrary words that is Conjecture C1′, false for the tree formula of cyclic words) | (H1); (F1) | **proved** |
| Ex 8.3 | the first-order part is **not** unique (refutes PROOFS C2 for arbitrary cycle-free words): word `1212`, red `2` | — | Tier 2 (exact computation of one conjugation) |
| Thm 8.4 | `ε ≡ δ₁ mod (r)²` | (H1), (M₁), multiplicity ≤ 2; (F1) | **proved** |
| Comp 8.6 | first-order cycles = first-order gauge directions (`Z₁ = B₁`) | words of length ≤ 6 (`A_2`), ≤ 5 (`A_3`), ≤ 4 (`D_4`) | computed only |
| Thm 9.1 | `1221122`, red 2 (klrw basis): no dot-free completion of the tree formula keeps its `r²`-part; the completion keeping the `r²`-part is tree formula + 10 dotted terms in one component, exactly the spanning-tree × bigon rule of NOTES §6; a dot-free completion (tree formula + 8 hand-over terms at `r²` + 4 at `r⁴`) also exists and is gauge-equivalent to it by an explicit dot-free gauge — the dotted terms are a normal-form artefact here | exact computation | Tier 2 |
| §10 | second order: exact formulation | — | formulation only, no proof |

(F1) is the basis theorem, Theorem A.5 of `01_foundations.md` (itself a draft). It is used only
where the table says so (and in Lemma 1.1(b), Cor. 7.2, Lemma 8.5, Remark 8.8, §9); Theorems A, B,
C, Prop. 5.3 and Prop. 7.1 use only the relations, (F3), (F3′) and the crossing table, not (F1).

**Corrections to other files (prominent):**

1. **PROOFS Conjecture C2 is false for arbitrary cycle-free words** (the reading suggested by
   README's C1′): already for `1212` with red colour 2 (cycle-free, every colour twice) there
   is a homogeneous first-order map `h` of homological degree 0 such that `(1+h) d (1−h)` (`d` =
   tree formula) is a degree-preserving differential of the form `δ₀ + (terms divisible by r)`
   that differs from the tree formula at first order (Example 8.3,
   `checks/04_gauge_example.py`). This does **not** contradict NOTES §3 as stated there: its
   uniqueness claim concerns the small minuscule cases of `data/` (`A_3` `2132`, `2312`; `A_4`
   `213243`, `234123`; `D_4` `123421`, `321423`, `421324`), and for these the referee found no
   homogeneous element of homological degree 0 and positive `r`-degree at all, so no gauge
   freedom of this kind (`R04_referee.md` G1); we found `Z₁ = 0` for `2132`, `2312`. For
   arbitrary words uniqueness can hold at best modulo gauge (Comp. 8.6) or under the extra
   normalisation (M₁) (Thm 8.4).
2. The hypothesis "ε vanishes when all parameters are 0" is **too weak** for any uniqueness
   statement: there are `r`-free, `t`-dependent gauges that keep the geometric part `δ₀` but
   produce `r`-free terms (Remark 8.8, word `1121`, red 2). The right hypothesis is
   `ε ≡ 0 mod (r)` (H1).
3. PROOFS §4 (C1 sketch) is confirmed: the bad triples of `δ₀²` are exactly `(y, i, j)`,
   `y < i < j`, `c(y) = c(j) ∼ c(i)`; the red strand is never the middle strand; the pattern
   `c(y) = c(i) ∼ c(j)` never gives a bad move. The worry that the smoothed diagrams might be
   non-reduced does not materialise for the path used here (Lemma 4.5); the products of
   one-pair terms with elementary moves *are* sometimes non-reduced (Prop. 5.3), and their dot
   terms cancel in groups of four.
4. **NOTES §6 / PROOFS C4 (dotted cyclic terms):** in the cyclic example `1221122` (red 2) the
   dotted cyclic terms are not an invariant of the differential: they belong to the normal form
   "keep the tree formula's `r²`-part"; an explicit dot-free monotone gauge turns that
   differential into a dot-free one (Theorem 9.1(e),(f), found by the referee). Whether the
   same holds for the 32 local pieces of NOTES §6 is open.

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
`(r)²` does not. The statements of §§3–8 are modulo ideals (or, as in Theorem B and
Lemma 4.7, exact with no choice involved) and therefore independent of these choices (by
Lemma 2.2, two reduced words of the same bijection agree modulo `(r)`, so a term `r t β_σ` is
well defined modulo `(r)²`). **Exception: §9**, which is stated for the `klrw` basis braid;
there "dot-free" and "`r`-degree-`k` part" depend on the basis (see the basis convention in §9).

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

**Lemma 4.7 (the smoothed word is the basis element; referee's improvement I1).** Let `β` be
any reduced word of `τ_y` from `e_I` to `e_{I∖{i,j}}` (for instance the `klrw` basis braid).
Then `X_y = β` in `A`, exactly.

*Proof.* Call two strands *inverted* if they cross in one (equivalently every) reduced word of
`τ_y`; by Lemma 4.5 the inverted pairs are the crossing pairs of `X_y`. Each of them joins one
of the three moving strands (those of `y`, `i`, `j`) to a strand that does not move (a `z_t`, an
element of `U_R ∪ U_L`, or `W`), and the moving strands are pairwise not inverted, nor are the
non-moving ones. So the graph of inverted pairs is bipartite and contains no triangle. A braid
move `ψ_pψ_{p+1}ψ_p ↔ ψ_{p+1}ψ_pψ_{p+1}` applied to a reduced word of `τ_y` acts on three
strands that pairwise cross, i.e. on a triangle of that graph; hence no braid move is ever
applicable. By (F3), `X_y` and `β` are connected by braid moves and commutations, hence by
commutations alone, which are equalities in `A` (R1). ∎

**Theorem B (exact first-order formula).** For every vertex `I` and `i < j` in `I`,

    (δ₀²)_{I, I∖{i,j}} = ε(I;i,j) · r_{c(j)} t_{c(j)c(i)} · Σ_{y ∈ Bad(i,j)} X_y
                      = ε(I;i,j) · r_{c(j)} t_{c(j)c(i)} · Σ_{y ∈ Bad(i,j)} β_{τ_y}          (exactly, in A),

with `ε(I;i,j) = (−1)^{n_I(i)+n_I(j)}`, `X_y` the reduced word of Lemma 4.5 and `β_{τ_y}` *any*
reduced word of `τ_y`; all other components of `δ₀²` vanish. In particular `δ₀²` has no terms
of `r`-degree ≥ 2, in any basis (F1) built from reduced words.

*Proof.* `P_1 − P_2 = D_N − D_0 = Σ_{s=1}^N (D_s − D_{s−1}) = Σ_s P_s(ψψψ − ψψψ)S_s`
(Lemmas 4.3, 4.4) `= Σ_{s : z_s ∈ Bad(i,j)} r_{c(j)}t_{c(j)c(i)} X_{z_s}` (Lemma 4.6; `R` is
central; the triple's source and target states `(y,i,j)`, `(j,i,y)` are the same colour sequence
when the move is bad, so `S_s` is the same element read from either). Multiply by `ε(I;i,j)`
(proof of Thm A). The second equality is Lemma 4.7. ∎

*Checks.* `checks/04_sweep.py` builds every `D_s`, `P_s`, `S_s`, `X_y` and verifies (S1)–(S5):
the words are well defined, Lemma 4.3, Lemma 4.4 (as equalities in `A`), Lemma 4.6, Lemma 4.5
(reduced, bijection `τ_y`, `≡ β_{τ_y} mod r`), Lemma 4.7 for the `klrw` basis braid (`X_y = β_{τ_y}`
exactly, in all 94 160 cases), and the exact identity `P_1 − P_2 = Σ r t X_y`; all words of
length ≤ 7 (`A_2`), ≤ 5 (`A_3`), ≤ 4 (`D_4`), every red colour: 0 failures, 94 160 bad and
618 035 good triangles. `checks/04_order01.py` (T1) compares the right-hand side with `δ₀²`
computed directly (modulo `(r)²`; it also records that `δ₀²` never has terms of `r`-degree ≥ 2):
0 failures on the ranges of §3. (The referee checked Lemma 4.7's bipartiteness in 64 690 cases
and `X_y` against bubble-sort words in 3 643 cases.)

**Corollary B.2.** `δ₀² = 0` (exactly) for every word without slots `y < i < j` with
`c(y) = c(j) ∼ c(i)`. This contains PROOFS Thm 5 (distinct colours) and e.g. all words
`c c c' c'`, or words in which equal colours are never separated by an adjacent colour.

**Corollary B.3 (necessity of corrections; uses F1).** If the word has slots `y < i < j` with
`c(y) = c(j) ∼ c(i)`, then for `I = {i, j}` the component `(δ₀²)_{I, ∅}` has coefficient
exactly `ε(I;i,j) r_{c(j)} t_{c(j)c(i)}` at the basis element `β_{τ_y}`. Hence `δ₀² ∉ (r)²`
(in particular `δ₀² ≠ 0`), and for every differential `d = δ₀ + ε` with `ε ≡ 0 mod (r)`, the
first-order part `ε mod (r)²` is non-zero (indeed `δ₀ε + εδ₀ ≡ −δ₀² ≢ 0 mod (r)²`).

*Proof.* The `y ∈ Bad(i,j)` give pairwise different bijections `τ_y`, hence different basis
elements; Theorem B and Lemma 1.1(b). For the last claim: `d² = δ₀² + δ₀ε + εδ₀ + ε²` and
`ε² ∈ (r)²`. ∎

*Cross-reference.* Cor. B.3 proves the converse of Corollary C.3 of `01_foundations.md` ("bad
triple ⇒ the two products differ"), which is listed there as verified only.

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
If `x = a` (case (A1)), the strand of `a` itself moves in `m_a` across `W` and the slots `< a`,
then (as the left pair strand of `β`) outwards across the `z ∉ I`, `a<z<b`: disjoint sets, so it
crosses each unaffected strand at most once and `W` exactly once; it crosses neither the mover
`j` nor the strand of `b` (in `m_a` both have slots `> a`; in `β` the pair strands do not cross
each other or the mover, Lemma 5.2). The strand of `b` and the mover move only in `β`, where they
cross each unaffected strand and `W` at most once. Unaffected strands cross `x` at most once
(only in `m_x`) and each other never. (B): `x` crosses `j` in `β` iff `x < j`, in `m_x` (`j` at `−j`) iff `j < x`:
once. `x` crosses the strand of `b` (in `β` iff `a < x < b`; in `m_x`, where it sits at `+a`, iff
`a < x`): twice iff `a < x < b`. `x` crosses the strand of `a` (at `−b` after `β`) only in `m_x`,
iff `b < x`. If `x = a` (case (B1)), the strand of `b` (now at `+a`) crosses in `m_a` the slots
`< a` and `W`, after having crossed in `β` the `z ∈ I`, `a<z<b`: disjoint sets; in `m_a` it does
not cross the strand of `a` (at `−b`, slot `b > a`) nor the mover (at `−j`, `j > a`), and in `β` it
crosses neither (Lemma 5.2). The strand of `a` and the mover move only in `β`. So every pair
crosses at most once. ∎

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

**Corollary 7.2 (uses F1).** A basis element `μ x^α β_σ` of `e_I A e_J`, `μ = ∏ r_c^{n_c} t_{cc'}^{m_{cc'}} u_c^{l_c}`,
`a_c` = number of dots on strands of colour `c`, is homogeneous of degree `D(I) − D(J)` iff

    n_c = Ŝ_c := Σ_{pairs of colour c} f,   l = 0,   LR(σ) contains no strand of colour k,
    m_{cc'} + m_{c'c} = Ê_{cc'}/2,  Ê_{cc'} := Σ_{pairs of colours c, c'} f,
    a_c = Σ_{c' ∼ c} m_{cc'} − n_c  ≥ 0.

In particular the `r`-degree `Σ_c Ŝ_c` of a homogeneous element is determined by `(I, J, σ)`,
and the `t`-exponents satisfy `Σ_{c'∼c} m_{cc'} = a_c + n_c ≥ n_c`: every factor `r_c` needs a
factor `t_{cc'}` (dots cannot replace `t`'s; each dot of colour `c` needs a further `t_{cc'}`).
For fixed `(σ, α)` the monomial is unique (the forest argument of PROOFS Cor. 3.1: the
`m_{cc'}` live on the oriented edges of the Dynkin forest, with prescribed sums on edges and at
vertices).

*Proof.* Compare coefficients of `σ_c, ε_{cc'}, ε_{cW}, δ_c` in `Δ` (Prop. 7.1) and in
`deg μ x^α = Σ n_c(σ_c+δ_c) + Σ m_{cc'}(2ε_{cc'}−δ_c) + Σ l_c(2ε_{cW}−δ_c) + Σ a_c(2q+δ_c)`, as
in PROOFS Cor. 3.1; `ε_{kW}` gives `2l_k = −2·#(LR ∩ colour k)`, both sides `≥ 0`/`≤ 0`. The
coefficient of `q` gives `2Σ_c a_c = Σ Ê − 2ΣŜ − 2·#(LR ∩ colour k)`, which is implied by the
`δ_c`-equations (summed over `c`) and the `ε_{kW}`-equation, so it imposes nothing new. ∎

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

**Proposition 7.4 (multiplicity ≤ 2; uses F1).** Suppose every colour occurs at most twice in the word.
(a) The monotone homogeneous basis elements of `r`-degree 1 in `e_I A e_J`, `|J| = |I| − 1`, are
exactly the `r_{c(p)} t_{c(p)c(j)} β(I;j,p)` for the one-pair configurations `(j,p)` at `I` (no
dots, unique monomial). (b) There are no monotone homogeneous basis elements of `r`-degree 1
with `|J| = |I|`.

*Proof.* Monotone with `|J| = |I| − h'` means: exactly `h'` strands cross `W`, all from right to
left. Every colour class has ≤ 2 slots, so `σ` fixes or swaps each class. For a fixed class
`{u < v}`, `f = −2[v ∉ I, v ∈ J] = 0` (monotone: `v ∉ I ⇒ v ∉ J`). For a swapped class, `f = ±1`.
`r`-degree 1 and `n_c ≥ 0` force exactly one swapped class `{u < v}` (colour `c₀`), with `f = +1`,
i.e. `v ∈ I`, `v ∉ J`. All other strands `z` are fixed. Pairs of fixed strands: `f = −2[...] = 0`
by monotonicity. Pairs `(z, A)`, `(z, B)` (`A`, `B` the strands of `u`, `v`): if `z < u` or
`z > v` the pair is not inverted and `f = 0` — for `(z, A)`, `z < u`, because `v ∉ J`; for `(z, B)`,
`z < u`, because `v ∈ I`; for `z > v` because `z ∉ I ⇒ z ∉ J` (monotonicity). If `u < z < v`, `f(z,A) = +1` if `z ∈ I`, `−1` if not, and `f(z,B) = +1` if `z ∉ J`,
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

(`checks/04_degree_defect.py mult2`, pure Python: exhaustive over all words with multiplicity
≤ 2 of length ≤ 4 in `A_2`, ≤ 6 in `A_3`, `A_4`, `D_4`, every red colour: the monotone
first-order elements of homological degree −1 coincide with the one-pair terms — 7 404 in `A_3`,
77 400 in each of `A_4`, `D_4` — and there is none of homological degree 0. The referee
re-checked it with an independent enumeration in `A_3, A_4, D_4, D_5, E_6`.)

---

## 8. Forcedness at first order

### 8.1 The hypothesis
Let `d = δ₀ + ε` be a map of homological degree −1 (all components homogeneous of degree
`D(I) − D(J)`) with `d² = 0`. We use

* **(H1)** `ε ≡ 0 mod (r)` — every coefficient (in the basis F1) of every component of `ε` is
  divisible by some `r_c`.

(H1) is stronger than "ε vanishes when all parameters are 0" (H2). `u` never occurs anyway
(PROOFS Prop. 4). Under (H2) alone no uniqueness statement holds even modulo `(r)`: see
Remark 8.8. Write `ε₁` for the class of `ε` in `(r)Hom/(r)²Hom`; by Lemma 1.1 it is canonical, and
it is a `ℤ`-combination of *first-order homogeneous basis elements* `μ x^α β_σ` (`r`-degree of
`μ` equal to 1, homogeneous of the right degree) — the coefficients are integers because, for
a fixed basis element `x^α β_σ`, the monomial `μ` making `μ x^α β_σ` homogeneous of the right
degree is unique (Cor. 7.2, forest argument of PROOFS Cor. 3.1). Let `V^h_1` be the `ℤ`-span of the first-order homogeneous basis elements of
homological degree `h`, and

    Z₁ = { η ∈ V^{−1}_1 : δ₀η + ηδ₀ ≡ 0 mod (r)² },     B₁ = { hδ₀ − δ₀h mod (r)² : h ∈ V^0_1 }.

**Theorem 8.1 (reduction; uses F1: the expansion of `ε₁` in `V^{−1}_1`, and the nilpotency of `h` in (c)).** (a) Under (H1), `ε₁ − δ₁ ∈ Z₁`. Conversely every `δ₁ + η`,
`η ∈ Z₁`, satisfies `(δ₀ + δ₁ + η)² ≡ 0 mod (r)²`. (b) `B₁ ⊆ Z₁`. (c) For `h ∈ V^0_1`, `g = 1 + h`
is invertible, and if `d` is a differential satisfying (H1) then so is `g d g^{−1}`, with
first-order part `ε₁ + [h, δ₀]`, `[h, δ₀] := hδ₀ − δ₀h`. (So `δ₁ + B₁` is realised by
differentials as soon as one differential with first-order part `δ₁` exists — for arbitrary
words this is Conjecture C1′ of the README, which fails for the tree formula itself in cyclic
words.)

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
(`checks/04_gauge_example.py 1212 2`). More precisely (observed by the referee, re-checked by
the same script): the gauge **removes both one-pair terms of the configuration (mover 3, pair
(2,4))** — in `{3,4}→{2}` and in `{1,3,4}→{1,2}` the tree formula has `+r_2t_21 β` and
`d' − d = −r_2t_21 β` — and `d'` acquires instead two non-monotone terms `±r_2t_21 β` in
`{3,4}→{1}` and `{2,3,4}→{1,2}`, in which the strand of slot 1 crosses `W` from left to right.
`d'` satisfies `d'² = 0`, homogeneity, (H1) and (H2), and is dot-free and `u`-free; it violates
exactly (M₁), consistently with Theorem 8.4. The same phenomenon (`B₁ ≠ 0`) occurs for `1121`
(red 2), `1321` (red 1, 2), `12112` and many more (Comp. 8.6). Since `1212` is cycle-free,
this refutes PROOFS C2 for arbitrary cycle-free words; it does not concern the minuscule cases
of NOTES §3 (§0, correction 1).

**Theorem 8.4 (forcedness; uses F1).** Suppose every colour occurs at most twice in the word. Let
`d = δ₀ + ε` be a differential satisfying (H1) and

* **(M₁)** `ε₁` is a combination of monotone first-order basis elements (no strand crosses `W`
  from left to right; by Lemma 2.2 this is a property of the class `ε₁`).

Then `ε ≡ δ₁ mod (r)²`.

*Proof.* By Prop. 7.4(a), `ε₁ = Σ λ_{(I;j,p)} r_{c(p)}t_{c(p)c(j)} β(I;j,p)` with `λ ∈ ℤ`, and the
same holds for `η := ε₁ − δ₁`, which lies in `Z₁` (Thm 8.1). By Lemma 8.5, `η = 0`. ∎

**Lemma 8.5 (injectivity; every word; uses F1 via Lemma 1.1(b)).** Let `η = Σ μ_{(I;j,p)} r_{c(p)}t_{c(p)c(j)} β(I;j,p)`,
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

**Computation 8.6 (Tier 2; `checks/04_forced.py`).** The script enumerates all first-order
homogeneous basis elements (dots included) of homological degrees −1 and 0 with
`checks/04_degree.py`, computes the matrices of `η ↦ δ₀η + ηδ₀` and `h ↦ hδ₀ − δ₀h` modulo
`(r)²` exactly, and their ranks over `ℚ`.
All words of the given lengths, every colour of the red strand:

| type, lengths | (word, red) pairs | `dim Z₁ = dim B₁` | `Z₁ = 0` | monotone: `Z₁^{mon} = B₁^{mon}` | monotone: `Z₁^{mon} = 0` | `Z0t = B0t` |
|---|---|---|---|---|---|---|
| `A_2`, ≤ 6 | 252 | 252 | 148 | 252 | 172 (all 80 others have a colour ≥ 3 times) | 252 |
| `A_3`, ≤ 5 | 1089 | 1089 | 839 | 1089 | 1041 (all 48 others: colour ≥ 3 times) | 1089 |
| `D_4`, ≤ 4 | 1360 | 1360 | 1270 | 1360 | 1360 | 1360 |

Without (M₁), `Z₁ ≠ 0` occurs also with multiplicity ≤ 2 (e.g. `1212`, red 2; 72 cases in
`D_4`). With (M₁) and multiplicity ≤ 2, `Z₁^{mon} = 0` in every case, as Theorem 8.4 predicts.
`B₁^{mon} ≠ 0` occurs once a colour occurs three times (e.g. `12112`, red 2: `dim = 2`; the
referee checked that the tree formula of `12112` is a differential, so monotone `h` gives a
monotone `d' ≠ d` mod `(r)²`), so Theorem 8.4 needs the multiplicity hypothesis even under
(M₁). The minuscule words `2132`, `2312` (red 2) have `Z₁ = 0`. Whenever `Z₁ = 0` the script also
confirms that the unique solution of the first-order equation is `δ₁`.

So, in every computed case, **the first-order part of any differential satisfying (H1) is
`δ₁` up to a first-order gauge transformation**; it is `δ₁` itself when `Z₁ = 0`. A proof of
`Z₁ = B₁` in general would need a description of the first-order cohomology of
`End_{A₀}(cube)`; the degree calculus alone does not give it (the zoo of §7).

**Remark 8.7 (the role of (M₁)).** Monotonicity holds for the tree formula and for every term
found in the cyclic regime (NOTES §6, and §9 below). PROOFS Prop. 4 proves it for strands of
the red colour; for other colours it is a normalisation. It excludes the gauge of
Example 8.3 (whose `d'` is not monotone), and for multiplicity ≤ 2 every first-order gauge is
non-monotone (Prop. 7.4(b)), which is how Theorem 8.4 works; no general statement of the form
"(M₁) excludes exactly the non-monotone gauges" is proved (monotone gauges exist: `12112`,
red 2, Comp. 8.6; and the gauge `h` of Theorem 9.1(f) is monotone).

**Remark 8.8 (why (H2) is not enough; uses F1 for `d' − δ₀ ∉ (r)A`).** For `1121`, red 2, there are 8 homogeneous elements
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

**Basis convention for this section.** Here "basis element", "dot-free" and "the
`r`-degree-`k` part" refer to the `klrw` basis: `β_σ` is the `klrw` basis braid of `σ`
(`fast.braid_for`, Definition B.10 of `01_foundations.md`), dots at the left end, `x_{c,k}` = dot
on the `k`-th strand of colour `c` from the left of the source state. Another choice of reduced
words changes basis elements by `r`-multiples of other, possibly dotted, elements, so
dot-freeness is basis dependent: the remark of §1.2 that all statements are independent of
the choice of reduced words does **not** apply to this section. (The referee recomputed (b)
with bubble-sort words and found the same numbers.)

**Theorem 9.1 (Tier 2: exact computation).** For this stack:

(a) There is no dot-free homogeneous basis element of homological degree −1 and `r`-degree 3
in any component (dot-free homogeneous elements by `r`-degree: 2447, 1464, 204, 0, 7 for
`r`-degree 0–4, none above).

(b) The tree formula `d_T = δ₀ + δ₁ + δ₂` satisfies `d_T² ≡ 0 mod (r)³`. The `r`-degree-3 part
of `d_T²` is non-zero: 28 coefficients, all with one dot, in the components `I₀ → {2,3}`,
`I₀ → {1,3}`, `{1}∪I₀ → J₀`, `{2}∪I₀ → J₀` (bijection = cyclic configuration followed by an
elementary move); the `r`-degree-4 part has 14 coefficients (dot-free).

(c) There is no differential `d_T + ε` with `ε ∈ (r)³` dot-free; equivalently, **the tree
formula cannot be completed by dot-free terms without changing its `r`-degree-2 part.**
(By (a), `ε` has no `r³`-part, so the `r³`-part of `d²` is that of `d_T²`, non-zero by (b).)

(d) *The completion keeping the `r²`-part.* `d_X := d_T + X`, with `X` the 10 terms listed
below, all in the single component `I₀ → J₀`, is a differential; `X` has dots. At each of the
four parameter monomials of `X` the complete ansatz (all homogeneous basis elements of that
monomial, dots allowed) has kernel 0, so, given `d_T` and no further corrections at smaller
monomials (in the solver's order), `X` is forced. All 680 tree-formula terms occur in `d_X` with
their coefficients.

(e) *The dot-free completion.* There is exactly one differential of the form
`δ₀ + δ₁ + ε` with `ε ∈ (r)²` dot-free, namely `d_F = d_T + κ* + ε₄` with
* `κ*` = 8 terms `± r_2² t_21² β_σ`: with `+` in `{4,6,7}→{1,3}`, `{4,5,6,7}→{2,3,5}`,
  `{1,4,5,6,7}→{1,2,3,5}`, `{2,4,5,6,7}→{1,2,3,5}` and with `−` in `{4,6,7}→{2,3}`,
  `{1,4,6,7}→{1,2,3}`, `{2,4,6,7}→{1,2,3}`, `{4,5,6,7}→{1,3,5}`; `σ = (1 5 4)(2 7)(3 6)` in the
  components without slot 5 and `σ = (1 4 5)(2 7)(3 6)` in those with slot 5;
* `ε₄` = 4 terms in `I₀ → J₀`: `+r_1r_2³t_12t_21³ β_{σ'}`, `−r_1r_2³t_12t_21³ β_{σ''}`,
  `−r_1²r_2²t_12²t_21² β_{σ'''}` (three of the terms of `X`, notation below) and
  `+r_1²r_2²t_12²t_21² β_{(1 4 5)(2 7)(3 6)}`.

`d_F` contains all 680 tree-formula terms with their coefficients (none of the 12 extra terms
has the bijection of a tree term); it is monotone and `u`-free. The terms of `κ*` are of
hand-over type: the strand crossing `W` changes its slot (e.g. from 7 to 2), so their bijection
is not "mover + exchange pairs". *Uniqueness:* write `ε = ε₂ + ε₃ + ε₄ + …` by `r`-degree. The
`r²`-equation forces `ε₂ − δ₂ ∈ Z₂`, the 4-dimensional space of dot-free `r`-degree-2 elements
`κ` with `δ₀κ + κδ₀ ≡ 0 mod (r)³` (`d_T²` has no `r²`-part). `ε₃ = 0` by (a). The `r³`-equation is
then linear in `κ`: `(d_T²)₃ + ((δ₀+δ₁)κ + κ(δ₀+δ₁))₃ = 0`; its matrix on `Z₂` is injective
(rank 4) and the system is consistent (rank 4 → 4), so `κ = κ*`. The `r⁴`-equation for `ε₄` has
7 unknowns and rank 7 and is consistent. There are no dot-free elements of `r`-degree ≥ 5.

(f) *Gauge equivalence.* `d_F = (1 + h) d_X (1 − h)`, where

    h = − r_2² t_21² ( β_{(1 5 4)(2 7)(3 6)} : e_{4,6,7} → e_{1,2,3}   +   β_{(1 4 5)(2 7)(3 6)} : e_{4,5,6,7} → e_{1,2,3,5} )

is homogeneous of homological degree 0, dot-free and monotone (no strand crosses `W` at all),
with `h² = 0`.

**So, in this example, the dotted cyclic terms are an artefact of the normal form.** They
appear in the completion that keeps the tree formula's `r²`-part, and they are not
gauge-invariant: the dot-free monotone gauge `h` of order `r_2²t_21²` trades them for the 8
hand-over terms `κ*` at `r²` (and changes the `r⁴`-part). Neither (H1), (H2) nor (M₁)
distinguishes `d_X` from `d_F`: both satisfy all three.

*What the computation assumes.* The correctness of the `klrw` implementation of the algebra
(products, basis expansion) and, for (d), of the library solver; the completeness of the
pure-Python degree enumeration (`degree_model.py`, `04_degree.py`), a finite search over all
bijections and monomials using Cor. 7.2; (F1) (= Theorem A.5 of `01_foundations.md`). (e) and
(f) need no solver: `d_F` and `h` are explicit, and `d_F² = 0`, `h² = 0`,
`(1+h)d_X(1−h) = d_F` are checked by direct multiplication. Scripts:
`checks/04_cyclic7_dotfree.py` for (a)–(c), (e), (f) — it builds `d_F` itself, reloads the
referee's independently computed `checks/R04_data/cyclic7_1221122_W2_dotfree.json` (equal to
ours, `d² = 0`), and checks the gauge with the referee's
`checks/R04_data/cyclic7_1221122_W2_gauge_h.json` and our `d_X`;
`checks/04_cyclic7_solve.py` for (d) (data in `checks/04_data/cyclic7_1221122_W2_tree.json`).
*Correction:* the first version of this section claimed that even `ε ∈ (r)²` dot-free is
impossible ("rank 4 → 5"). That came from a bug in `04_cyclic7_dotfree.py`, which dropped all
products `δ₁·κ` (found by the referee, G0), and was false.

*Not shown:* uniqueness of dot-free differentials `δ₀ + ε` with arbitrary `ε ∈ (r)` (changing
the first-order part as well): the dot-free first-order cycles form a 106-dimensional space
(56-dimensional for the monotone ones, referee's count), mostly gauge directions.

**The terms of `X` (the completion keeping the `r²`-part).** Dot variables `x_{c,k}` as above,
in `e_{I₀} = (3, 2, 1, W, 4, 5, 6, 7)`; so `x_{2,1}, x_{2,2}, x_{2,3}, x_{2,4}` are the strands of
slots `3, 2, 6, 7` and `x_{1,1}, x_{1,2}, x_{1,3}` those of slots `1, 4, 5`. With
`σ_cyc = (1 4)(2 7)(3 6)` (mover 5):

    X_dot = − r_1 r_2² t_12 t_21² · [ t_21 (x_{2,1} + x_{2,2} + x_{2,3} + x_{2,4}) + t_12 (x_{1,1} + x_{1,2} + 2 x_{1,3}) ] · β_{σ_cyc}
    X_r   = + r_1 r_2³ t_12 t_21³ β_{σ'}  − r_1 r_2³ t_12 t_21³ β_{σ''}  − r_1² r_2² t_12² t_21² β_{σ'''}
    X     = X_dot + X_r,

    σ_cyc = {1↦4, 2↦7, 3↦6, 4↦1, 6↦3, 7↦2},   σ'  = {1↦4, 2↦6, 3↦7, 4↦1, 6↦3, 7↦2},
    σ''   = {1↦4, 2↦7, 3↦6, 4↦1, 6↦2, 7↦3},   σ''' = {1↦5, 2↦7, 3↦6, 4↦1, 5↦4, 6↦3, 7↦2}

(slot `5` is fixed by `σ_cyc, σ', σ''`: its strand is the one crossing `W`; in `σ'''` the strand
of slot 5 ends in slot 4 and the strand of slot 4 in slot 1).

*Comparison with NOTES §6 (for the normal form `d_X`).* Every spanning tree `G − e` of the
4-cycle has the same monomial `r_1 r_2² t_12 t_21²` (the two colour-2 pairs hang below
colour-1 nodes, the colour-1 pair below a colour-2 node) and the same sign `κ(I₀; 5, F) = −1`
(each has two connectors in `I₀`; referee's hand computation), and the four left-over links
contribute `t_12[dot on the colour-1 node] + t_21[dot on the colour-2 node]`. The sum gives each
colour-2 pair one dot on each of its two strands (`x_{2,1}…x_{2,4}`), the pair `(1,4)` one dot on
each strand (`x_{1,1}, x_{1,2}`), and the mover two dots (`2x_{1,3}`) — exactly `X_dot`. `X_r`
consists of the corrections from moving these dots, which sit at the touching points of the
time-ordered drawing, to the left end of the basis element (they exchange two same-colour
strands: `(2,3)/(6,7)` for the colour-2 dots, `4 ↔ 5` for the mover's dot).
`sage -python cycle3.py proofs/checks/04_data/cyclic7_1221122_W2_tree.json` (the single-cycle
rule of NOTES §6, read-only) reproduces the whole of `d_X` with **0 wrong components** (all
three variants `mix`, `endL`, `roleL`; cyclic component 1/1).

*What the example shows.* It is a cyclic stack with only 7 strands (full cube of 128
vertices solved, not just an interval) and with two pairs of the *same* colour on the cycle;
its cycle has only sources and sinks (no in-between node, so the hand-over part of the NOTES
rule is not tested). (1) In the normal form "keep the tree formula's `r²`-part" the correction
lives in the one component of the cyclic configuration and is exactly the spanning-tree ×
bigon sum of NOTES §6. (2) That normal form is a choice: the gauge class also contains the
dot-free `d_F`, whose correction instead modifies the `r²`-part by hand-over terms outside the
cyclic component. (3) Consequence for NOTES §6 / PROOFS C4: whether the dotted cyclic terms of
the 32 local pieces of NOTES §6 (where the solver also keeps the tree formula fixed) are
likewise removable by a dot-free gauge is open, and should be checked before the dotted rule
is treated as an invariant.

---

## 10. Second order: what has to be proved

Write `d = δ₀ + δ₁ + δ₂ + δ_{≥3}`, `δ_{≥3} ∈ (r)³`. Modulo `(r)³`,

    d² ≡ [δ₀² + δ₀δ₁ + δ₁δ₀] + [δ₀δ₂ + δ₁δ₁ + δ₂δ₀]     mod (r)³.

The first bracket lies in `(r)²` (Theorem C); call `E₂` its class in `(r)²A/(r)³A`. The second
bracket lies in `(r)²` trivially; only its value modulo `(r)³`, i.e. the composites modulo `(r)`,
matters. The statement to prove is

    (∗)   E₂ + (δ₀δ₂ + δ₁δ₁ + δ₂δ₀ mod (r)³) = 0      in (r)²A/(r)³A ≅ ⊕_{c ≤ c'} r_c r_{c'} · A₀.

Ingredients, each a finite case analysis of the kind of §5:

1. `E₂` needs the composites of §5 **modulo `(r)²`**, not `(r)`. By Theorem B and Lemma 4.7,
   `δ₀² = Σ ε r t β_{τ_y}` exactly, so `δ₀²` contributes **nothing** to `E₂`, in any basis; `E₂` is
   the class of `δ₀δ₁ + δ₁δ₀` modulo `(r)³`. For each product (A), (B) one needs the `r`-linear
   difference between the concatenated word (with its bigon polynomial) and the basis element.
   Sources: bad braid moves needed to bring the concatenated word to the chosen basis word
   (each a smoothing `r t`, which exchanges two more same-coloured strands), and the corrections
   `±r` from sliding the bigon dots `x_x, x_a, x_b` to the left end. Note that `E₂` depends on
   the choice of the reduced words of the one-pair terms (changing them changes `δ₁` by
   `(r)²`-terms, i.e. moves contributions between `E₂` and `δ₂`); (∗) as a whole is a statement
   about the chosen words.
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

*Where the second-order terms live (Tier 2, `checks/04_second_order.py`, `klrw` basis; all words
of length ≤ 6 in `A_2` and ≤ 5 in `A_3`, every red colour; (∗) holds in all of them).* Both
`E₂` and the second bracket are dot-free, and their terms have bijections of three kinds: two
transpositions of different colours, two transpositions of the same colour, and 3-cycles of one
colour. `δ₀²` contributes nothing (Lemma 4.7). `δ₀δ₁ + δ₁δ₀` contributes in all three sectors;
`δ₀δ₂ + δ₂δ₀` only to the two-transposition sectors; `δ₁δ₁` to all three (in `A_3` not to the
same-colour two-transposition sector). In particular, in the different-colour sector all three
products contribute, so a proof there must control the `r`-linear parts of the composites of §5
(item 1), not only diagrams modulo `(r)`.

---

## 11. Status

**Proved completely** (from (R1)–(R4) with the README signs, (F3) Matsumoto, (F3′) Tits' word
property, and (F1) = Theorem A.5 of `01_foundations.md` **only where marked**):
* Lemma 1.1 (legitimacy of working modulo `(r)^n`; (b) uses F1).
* Lemma 2.2 (normal form modulo `(r)`; F3, F3′).
* Theorem A (`δ₀² ≡ 0 mod (r)`), every word. No F1.
* Theorem B with Lemma 4.7 (exact formula `δ₀² = Σ ε r t β_{τ_y}` in any basis built from reduced
  words; bad triples exactly `c(y)=c(j)∼c(i)`; red never the middle strand), every word. No F1.
  Cor. B.2 (no F1); Cor. B.3 (F1).
* Prop. 5.3 / Cor. 5.4 (classification of the products of one-pair terms with elementary
  moves, including all non-reduced cases and their values modulo `(r)`). No F1.
* Theorem C (`δ₀² + δ₀δ₁ + δ₁δ₀ ≡ 0 mod (r)²`), every word, every red colour. No F1. With PROOFS
  Prop. 3/Cor. 3.1: the tree formula is a homogeneous differential modulo `(r)²` for every word.
* Prop. 7.1 (degree defect of an arbitrary bijection; no F1), Cor. 7.2 (F1).
* Prop. 7.4 (multiplicity ≤ 2: the monotone first-order elements are the one-pair terms; F1).
* Theorem 8.1 (reduction of forcedness to `Z₁`, gauge directions `B₁ ⊆ Z₁`; F1), Lemma 8.5
  (injectivity, every word; F1), Theorem 8.4 (multiplicity ≤ 2 + (H1) + (M₁) ⇒
  `ε ≡ δ₁ mod (r)²`; F1).

**Proved by a finite exact computation (Tier 2; relative to the `klrw` implementation):**
* Example 8.3 (`1212`, red 2: a gauge-equivalent differential with a different first-order
  part; refutes PROOFS C2 for arbitrary cycle-free words; the gauge removes both one-pair terms
  of (mover 3, pair (2,4)) and creates two non-monotone terms) and Remark 8.8 (`1121`, red 2:
  `r`-free gauges under (H2)).
* Theorem 9.1 (`1221122`, red 2, `klrw` basis): (a)–(c) no dot-free completion of the tree
  formula keeps its `r²`-part; (d) the completion `d_X` keeping the `r²`-part (10 dotted terms in
  the cyclic component, forced at their orders, equal to the single-cycle rule of NOTES §6);
  (e) the unique dot-free differential `d_F = δ₀ + δ₁ + ε`, `ε ∈ (r)²` (all tree terms, plus 8
  hand-over terms at `r²` and 4 terms at `r⁴`); (f) the explicit dot-free monotone gauge
  `d_F = (1+h)d_X(1−h)`. So the dotted cyclic terms are a normal-form artefact in this example.
  (`d_F` and `h` were found by the referee; we rebuilt `d_F` with our own code and verified
  `d_F² = 0`, equality with the referee's data, and the gauge identity.)

**Verified only by computation:**
* `Z₁ = B₁` (uniqueness of the first-order part up to first-order gauge), also restricted to
  monotone elements, and `Z0t = B0t`: all words of length ≤ 6 in `A_2`, ≤ 5 in `A_3`, ≤ 4 in
  `D_4`, every red colour (Comp. 8.6).
* The sector structure of the second-order terms (§10).

**Corrections made in this revision (after `R04_referee.md`):**
* **Fatal error corrected:** the first version claimed in Theorem 9.1(c) (and in §0, §11) that
  the tree formula of `1221122` cannot be completed by dot-free terms even when its `r²`-part
  may change ("rank 4 → 5"). This was false; it came from a bug in `04_cyclic7_dotfree.py`
  (products `δ₁·κ` dropped). Fixed; the true statement is Theorem 9.1(c)–(f).
* "The true differential" in §9 replaced by "the completion keeping the `r²`-part"; the basis
  dependence of "dot-free" stated; §1.2 no longer claims basis independence for §9.
* Correction 1 of §0 now concerns PROOFS C2 (arbitrary cycle-free words), not NOTES §3.
* The Tier-2 observation `X_y = β_{τ_y}` is now Lemma 4.7, with proof (referee's I1).
* Uses of (F1) declared; Thm 8.1 table row, Cor. 7.2, Rem. 8.7, numbering (Comp. 8.6,
  Rem. 8.7, 8.8) and script docstrings corrected; proofs of (A1)/(B1) and of Prop. 7.4 completed.

**Runs (all 0 failures):** `04_order01.py` (T0–T3) on `A_2` ≤ 7, `A_3` ≤ 6, `D_4` ≤ 5 (all
words, all red colours) and 300 random `D_4` words of length 6; `04_sweep.py` (S1–S5, including
Lemma 4.7 for the `klrw` basis braid) on `A_2` ≤ 7, `A_3` ≤ 5, `D_4` ≤ 4; `04_signs.py` on `A_2` ≤ 8,
`A_3` ≤ 6, `D_4` ≤ 5; `04_degree_defect.py defect` (40 000 samples) and `mult2` (ranges in §7);
`04_forced.py`, with and without `-M`, on the ranges of Comp. 8.6; `04_gauge_example.py`
(`1212` red 2 order 1; `1121` red 2 order 0); `04_cyclic7_solve.py` and `cycle3.py` on its output;
`04_cyclic7_dotfree.py` (corrected); `04_second_order.py` on `A_2` ≤ 6, `A_3` ≤ 5. After the
revision the scripts were re-run on smaller ranges (`A_2` ≤ 6 for `04_order01.py`, ≤ 5 for
`04_sweep.py`, the examples of §8, all of §9): unchanged results.

**Not done / open:**
* A proof of `Z₁ = B₁` in general. The degree calculus (Prop. 7.1) shows that the first-order
  homogeneous elements form a zoo (hand-over cycles, strands crossing `W` from left to right,
  dotted elements), so a proof by classification is hopeless beyond multiplicity 2; it needs a
  cohomological argument (e.g. via the cone structure `C(w·c) = Cone(F)` of the README).
* Theorem 8.4 without the multiplicity hypothesis: false as stated (monotone gauges exist for
  `12112`, red 2); the general statement is presumably "unique up to monotone gauge".
* Uniqueness of dot-free differentials `δ₀ + ε`, `ε ∈ (r)` arbitrary, for `1221122` (§9).
* Whether the dotted cyclic terms of the 32 local pieces of NOTES §6 are removable by dot-free
  gauges, as in `1221122` (§9; consequence for PROOFS C4).
* Second order (§10): formulated, not proved, not even in the different-colour sector.
