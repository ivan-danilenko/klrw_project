# 07 — Conjecture H (the one-step homotopy): proofs for one repeated colour

## RESUME NOTES (keep at the top; updated after every result)

* **Finished (proofs written below):** §1 setting and signs; §2 tools (localisation,
  moving-strand expansion, double-crossing lemma, **compact form of h**); §3 the diagonal group;
  §4 **Theorem H1a** (one colour); §5 **Theorem H1** (one colour repeated, all other colours at
  most once, under the condition (NN)); §6 consequences (contains Theorem C of 05).
* **Verified:** every coefficient formula of §3 and §5.3 term by term (`07_formulas.py`), every
  vanishing / pairwise statement (`07_vanish.py`), the compact form (`07_compact.py`), the group
  splitting (`07_groups.py`), (★) itself (`07_star_check.py`). Logs in `checks/07_logs/`.
* **Open:** H1 without (NN) (nested families of pairs around a mover of adjacent colour; the
  identity of §5.5, computed only); H2–H4 (hanging forests; §7 lists what is missing).
* **All runs done (logs in `checks/07_logs/`, after the last edit of the scripts):**
  `formulas.log` (0 mismatches), `vanish.log` (0 failures), `compact.log`, `theta.log`,
  `groups.log` (every group zero, incl. Identity N words), `adjfine.log`, `star_small.log`,
  `star_A2.log` ((★) zero on all H1 words; non-zero on the cyclic control `1221122`),
  `groups_H2.log`, `star_H2.log` (H2 words: (★) holds; groups do not close).
* **§7 (H2–H4) and §8 (Status) written.** Next steps for a successor: (1) prove Identity N by
  extending Prop. 5.3 to nested families (the configurations `(F,a)` combine by component,
  `07_adjfine.py`); (2) find a compact form for chains *with* hanging forests (§7.2.1) — the key
  to H2; (3) combine it with the tree lemma 06 Lemma 6.7 as in the `F = ∅` case of §5.3.

---

## 0. Results

Notation of `06_main_intertwiner.md` (§§1–5, 7.2): `w = w′·c`, `m` the last slot (colour `c`),
`Same = {s < m : c(s) = c} = {s_1 < ⋯ < s_N}`, `ζ_m = ∏_{s∈Same}(x_s − x_m)`, `h` the formula of
Conjecture H, `F` the components `C_R → C_L` of the tree formula of `w`, `d′` the tree
formula of `w′`. "(★)" is the identity (1.1) below (= (★) of 06 Prop. 5.2).

**Theorem H1a [proved].** If all letters of `w` have the same colour (`w = c^{N+1}`, any `N`,
any simply-laced type, any red colour), (★) holds.

**Theorem H1 [proved].** Let `c` occur `N` times in `w′` (any `N ≥ 0`), let every other colour
occur at most once in `w`, and assume

> **(NN)** no slot `j < m` of a colour adjacent to `c` satisfies `s_2 < j < s_N`.

Then (★) holds. (For `N ≤ 2` the condition is empty.)

**Corollary H1 [proved] (§6).** Let `w` be a word in which one colour `c` occurs `n` times, at
`s_1 < ⋯ < s_n`, every other colour at most once, and no slot of a colour adjacent to `c` lies
strictly between `s_2` and `s_{n−1}`. Then the tree formula of `w` satisfies `d² = 0`. For
`n = 3` the condition is empty: **Theorem C of 05 is reproved** by a different argument; for
`n ≥ 4` this is new (e.g. `c^n`, `e c^n`, `c e c^{n−1}`, `c^{n−1} e c` for `e ∼ c`).

**Tools valid for every word** (§2): the moving-strand expansion (Lemma 2.3), the
double-crossing lemma (Lemma 2.4), and the **compact form of the chain part of h**
(Proposition 2.6): the sum of the chain terms of `h` with a fixed first element `c` is one
diagram,

    H_c = ε(I; c) · (−r_c) · LS_c · ζ̂_c · Seg_0(m → c),        ζ̂_c = ∏_{s∈Same∖c} (x_s − x_m),

(`LS_c`: the strand of `c` slides out to `−m`; `Seg_0`: the strand of `m` slides in to `+c`;
`ζ̂_c` is read between the two slides). This turns a sum over `2^{N−1}` chains into a single
term and is what makes the off-diagonal components tractable.

**What the proof is.** In the localised algebra every term of (★) is a combination of
labelled brackets `[τ]` with rational coefficients (06 §3, §9). (★) splits into *groups*
(§5.1) according to the mover of the prefix term. The diagonal group is the inclusion–exclusion
identity `∏_{b∈𝔟}(A_b − B_b) = 0` (§3); the groups of a mover of colour `c` or of a colour not
adjacent to `c` vanish by the double-crossing lemma or cancel in pairs (§5.2); the group of a
mover of adjacent colour is a bigon divided difference plus two telescoping sums (§5.3–5.4).
No step uses the tree lemma of 06 §6; the hanging forests of H2–H4 are where it will have to
enter (§7).

**What is not proved.** (★) for one colour repeated and other colours once *without* (NN):
there nested families of pairs around an adjacent mover occur, and the group identity of §5.5
is only computed. Stages H2–H4: §7.

---

## 1. Setting, signs

**1.1 The identity.** Conventions of `README_FOR_PROVERS.md` and 06 §1 (product `ab` = first
`a`; `φ = (x_1 − x_2)ψ − r`). For a vertex `I` let `R = I ∖ {m}`, `L = {1,…,m−1} ∖ I` (slots on the
left of the red strand). For `I ∋ m`, `K ∌ m`, `|K| = |I| − 1`:

    E′(I→K) :=  Φ_m(I)·[K = R]  +  Σ_{J∋m} d_w(I→J)·h(J→K)  −  Σ_{J∌m} h(I→J)·d_w(J→K)  −  ζ_m F(I→K),     (1.1)

where `d_w` is the tree formula of `w` (its components between vertices containing `m` are
`−ι_R(d′)`, those between vertices not containing `m` are `ι_L(d′)`, 05 Lemma 3.2) and
`F(I→K) = d_w(I→K)`. **(★) is the statement `E′ = 0`.** (This is exactly the equation solved in
`06_divide.py` and checked in `06_check_step_formula.py`; by 06 Prop. 5.2 it implies that `F` is
a chain map, hence `d_w² = 0` if `d′² = 0`.) The terms of `d_w` are written `X_j` (mover `j`,
no pairs) and `P(j; F)` (mover `j`, pairs `F ≠ ∅`), with the coefficients and basis braids of
`formula.py` / 01 Def. B.10.

**1.2 Signs.** With the exterior algebra calculus of 06 §6.1 (`ξ_I`, `ι_j`, `θ_p`, Lemma 6.1
(Λ1)–(Λ4)), a term of `d_w` at `V` has the sign `κ(V; j, F)·(−1)^{#{p : x_p ∈ V}}` with
`κ(V; j, F) = ⟨ξ_{V′}, ι_j θ_F ξ_V⟩`, and a term of `h` at `V` has the sign
`ε(V; 𝔠, F) = ⟨ξ_{V′}, θ_{(c_1, m)} θ_F ξ_V⟩` times `(−r_c)^s ∏ r_p t_p ε_p` (06 §7.2; `θ_{(c_1,m)}`
replaces `ξ_m` by `ξ_{c_1}` in place). For a composite the signs multiply and the operators
compose in the reverse order.

**1.3 Localisation (standing facts).** `Δ` (06 §9) is central and a non-zero-divisor (01
Theorem E.1), so `A ⊂ A[Δ^{-1}]`. For two neighbouring strands of equal colour,
`ψ = (x_i − x_{i+1})^{-1}(φ + r)` in `A[Δ^{-1}]`.

**Lemma 1.1 [proved].** Let `s` be a labelled state and `e′` a colour sequence. The brackets
`𝒟(s → t)`, `t` running over the labelled states with colour sequence `e′`, form a basis of the
left `Pol[Δ^{-1}]`-module `e(s) A[Δ^{-1}] e′`, `Pol` the polynomial ring in the labelled dots
(written at the source).

*Proof.* The proof of 06 Prop. 9.3 with source and target exchanged: every product of generators
from `e(s)` is, in `A[Δ^{-1}]`, a `Pol[Δ^{-1}]`-combination of products of the generators `g`
(replace each `ψ` of equal colours by `(x_i − x_{i+1})^{-1}(φ + r)` and move the polynomials to the
source by 06 Prop. 3.3), and a product of `g`'s along a word is a polynomial multiple of one
bracket (06 Prop. 3.5). So the brackets span; their number equals the rank of the free left
module `e(s)A[Δ^{-1}]e′` (F1), so they are a basis. ∎

Hence **an identity in `A` may be checked by comparing, for every labelled target `t`, the
rational coefficients of `𝒟(s → t)`.** We write `[τ]` for the bracket whose strand starting at
slot `y` of the source vertex ends at slot `τ(y)` of the target vertex (06 §6.1), and `x_y` for the
dot on the strand that starts at slot `y` (at the source).

**Observation 1.2 (checks only).** A homogeneous identity in `A` holds iff it holds after setting
all parameters `r, t, u` to `1`: for a fixed basis element `x^α β_σ` (F1) the parameter
monomial of a given degree is unique (the degree map on monomials in `r, t, u` is injective, as in
`PROOFS.md` Cor. 3.1). Not used in the proofs.

---

## 2. Tools

### 2.1 Expansion of a word

**Lemma 2.1 (expansion rule) [proved].** Let `E = Σ_t f_t 𝒟(s → t)` (coefficients at the source).
(a) For a generator `g_i` (06 Def. 2.2) at the state of `t`:
`𝒟(s → t) g_i = q_{t_i t_{i+1}}^{[\{t_i,t_{i+1}\} ∈ inv(s,t)]}·𝒟(s → t·i)` (06 Prop. 3.5).
(b) For a crossing `ψ_i` of two strands of equal colour `c` at the state of `t`:
`𝒟(s → t) ψ_i = (x_{t_i} − x_{t_{i+1}})^{-1} ( 𝒟(s → t) φ_i + r_c 𝒟(s → t) )`.
(c) For a polynomial `g` in the positional dots at the state of `t`:
`𝒟(s → t)·g = g(x_{t_1}, …, x_{t_n})·𝒟(s → t)`.

*Proof.* (b): definition of `φ` in `A[Δ^{-1}]`, and 06 Prop. 3.3 to move the positional dots of
the state of `t` to labelled dots at the source. (c): 06 Prop. 3.3. ∎

`checks/07_lab.py` implements exactly (a)–(c) (`mul_letter`, `mul_poly`).

**Definition 2.2.** A *moving-strand word* from a state is a word `(p−1, p−2, …, p−ℓ)`: the
strand at position `p` moves to position `p − ℓ`, passing the strands `y_1, …, y_ℓ` that stand at
the positions `p−1, …, p−ℓ` ("the moving position passes `y_1, …, y_ℓ`", in this order). Every
moving strand in this file moves leftwards.

**Lemma 2.3 (moving-strand expansion) [proved].** Let `E = 𝒟(s → t)` and let `D` be a
moving-strand word from the state of `t` in which the moving position (label `μ_0` in `t`) passes
labels `y_1, …, y_ℓ`. Let `T` be the set of `k` with `y_k` black of the colour of `μ_0`. For
`S = \{k_1 < ⋯ < k_p\} ⊆ T` put `μ_k = y_{max(S ∩ [1,k))}` (`= μ_0` if `S ∩ [1,k) = ∅`), and let `t_S` be
`t·D` with the labels `μ_0, y_{k_1}, …, y_{k_p}` cyclically moved: `μ_0` to the final position of
`y_{k_1}`, `y_{k_i}` to that of `y_{k_{i+1}}`, `y_{k_p}` to the final position of the moving strand.
Then

    𝒟(s → t)·ψ_D  =  Σ_{S ⊆ T}  r_c^{|S|} ∏_{k∈T} (x_{y_k} − x_{μ_k})^{-1} · ∏_{k ∉ S,\ \{μ_k, y_k\} ∈ inv(s,t)} q_{μ_k y_k} · 𝒟(s → t_S).

In particular there is no factor `q` if no pair `\{μ_k, y_k\}` is inverted between `s` and `t`
(e.g. if `t = s`).

*Proof.* Induction on `ℓ` with Lemma 2.1: at the letter `k` the labels at the two positions are
`(y_k, μ_k)`; for `k ∈ T` Lemma 2.1(b) gives the two branches `φ` (the label `μ_k` passes, it
remains the moving label) and `r` (no crossing: the labels stay, and the label at the moving
position is now `y_k`); for `k ∉ T` Lemma 2.1(a). In the `φ`-branch and for `k ∉ T` the factor of
Lemma 2.1(a) is `q_{μ_k y_k}` iff the pair is inverted between `s` and the current state; the earlier
letters of `D` do not change this, since the labels `μ_k, y_k` meet only at the letter `k` (`y_k` is
passed only at `k`; if `y_k` had been a moving label earlier, then `y_k = y_{k′}` for some `k′ < k`). ∎

*Words used in this file.* `X_j(V)`: the moving position starts at `+j` and passes the right slots
`< j` (decreasing), the red strand, the left slots `< j` (increasing) (`PROOFS.md` Lemma 1; unique
reduced word, 01 Lemma C.1). The chain term of `h` with chain `𝔠 = (c_1 < ⋯ < c_s)` at `V`
(forest empty): the *left slide* — a moving position starts at the slot `c_1` on the left and
passes the left slots in `(c_1, m)` (increasing), with the letters of the slots `c_2, …, c_s`
omitted — followed by `Seg_0`: the position of `m` passes the right slots in `(c_1, m)`
(decreasing) and stops at `+c_1`. (Omitting the letter at `c_i` means: the strand arriving there
stops, the strand of `c_i` continues; this realises `σ_𝔠`.) In both diagrams no two moving strands
cross, so no three strands cross pairwise, every reduced word is related to every other by
commutation moves only (F3), and the element is that of the basis braid.

### 2.2 The double-crossing lemma

**Lemma 2.4 [proved].** Let `A, B` be strands and `Y = (y_1, …, y_r)` strands such that at some
point of a word the strands `y_r, …, y_1, A, B` stand next to each other in this order, and the
word continues with

    W = [B passes A] [B passes y_1] ⋯ [B passes y_r] [A passes y_1] ⋯ [A passes y_r] [A passes B]

(no dots inside `W`). Assume that `A` and `B` are black and the colour of `A` is not adjacent to
that of `B` (equal colours allowed). Then

    ψ_W = ψ_{[A passes y_1]⋯[A passes y_r][B passes y_1]⋯[B passes y_r]} · q_{AB},

with `q_{AB} = 0` if `A, B` have the same colour and `q_{AB} = 1` otherwise (R3).

*Proof.* `W_0 = [B passes A][B passes Y][A passes Y]` and `W_0′ = [A passes Y][B passes Y][B passes A]`
are reduced words of the same permutation of these `r + 2` strands (`A`, `B` cross once; each `y_k`
is crossed once by `A` and once by `B`; the `y`'s do not cross each other). The triples of
pairwise crossing strands are `\{y_k, A, B\}`; when the three mutual crossings of such a triple
are consecutive letters of a reduced word, the triple stands in its initial order `(y_k, A, B)`.
By (F3) `W_0` and `W_0′` are connected by commutation moves (R1) and braid moves on states
`(y_k, A, B)`. Such a move is exact unless the outer strands `y_k, B` are black of one colour and
the middle strand `A` is black of an adjacent colour, or `A` is red (README, braid relation);
both are excluded (`A` is black, and not adjacent to the colour of `B`). Hence
`ψ_W = ψ_{W_0′}ψ_{[A passes B]}`, and the last letter of `W_0′` and `[A passes B]` are the same letter
on the same two strands: `ψ_i² = q_{AB}` (R3). ∎

### 2.3 The compact form of the chain part of h

**Definition 2.5.** For `V ∋ m` and `c ∈ Same ∖ V` let `LS_c` be the moving-strand word in which
the strand of `c` passes the left slots in `(c, m)` (increasing) and ends at the far left, and
`Seg_0^c` the one in which the strand of `m` passes the right slots in `(c, m)` (decreasing) and
stops at `+c`. Put

    H_c(V)  :=  ε(V; c) · (−r_c) · LS_c · ζ̂_c · Seg_0^c,      ζ̂_c := ∏_{s ∈ Same∖\{c\}} (x_s − x_m),

where `ζ̂_c` is a polynomial in the positional dots at the state between the two slides (`x_s`:
dot of the strand at the slot `s` — after `LS_c` — and `x_m` the dot of the strand `m`, which has
not moved yet), and `ε(V; c)` is the sign of the permutation sorting `V` with `m` replaced by `c`.

**Proposition 2.6 (compact form) [proved].** For every word and every `V ∋ m`, `c ∈ Same ∖ V`:
the sum of the terms of `h(V)` with chain starting at `c` **and empty forest** equals `H_c(V)`.

*Proof.* The sign `ε(V; 𝔠) = ε(V; c)` depends only on `c_1 = c`. Write `Λ = Same ∩ L ∩ (c, m)`
(`L` the left slots of `V`) and `Z_c = ∏_{s∈Same∖(Λ∪\{c\})}(x_s − x_m)`. The chain terms with `c_1 = c`
are indexed by the subsets `𝔠″ = \{c_2, …, c_s\} ⊆ Λ`, with coefficient
`ε(V;c)(−r)·Z_c·∏_{y∈Λ∖𝔠″}(x_y − x_m)·(−r)^{|𝔠″|}` (dots at the source) and diagram = the left slide
with the letters of `𝔠″` omitted, then `Seg_0^c` (§2.1). Order the word with the left slide first.
The dots of `Z_c` are on strands not involved in the left slide (and `x_m`: the strand `m` moves
only in `Seg_0`), so `Z_c` may be placed between the two slides. For `y ∈ Λ` the dots `x_y` and
`x_m` commute with the letters of the left slide before the letter at `y` (they involve other
strands), so the factor `(x_y − x_m)` may be placed immediately before that letter. Multilinear
expansion of an ordered product then gives

    Σ_{𝔠″⊆Λ} ∏_{y∈Λ∖𝔠″}(x_y − x_m) (−r)^{|𝔠″|} · (left slide, letters of 𝔠″ omitted)
       =  ∏_{letters of the left slide}  ( for y ∈ Λ: (x_y − x_m)ψ − r ;  otherwise: ψ ) ,

each factor at its place. At the letter of `y ∈ Λ` the strand `y` is the left strand before the
crossing, and (S): `x_1ψ − ψx_2 = r` reads `(x_y − x_m)ψ − r = ψ·(x_y − x_m)` with the dot of `y` read
after the crossing (`x_m` commutes). After its letter the strand `y` is not crossed again in the
left slide, so `(x_y − x_m)` moves to the end of the left slide. This gives
`LS_c · ∏_{y∈Λ}(x_y − x_m) · Z_c · Seg_0^c` with all dots read between the slides, i.e. `H_c(V)`. ∎

*Check.* `07_compact.py` (all `(V, c)` of `11111` red 1, 2; `121121`, `112121`, `1213121`, `2131212`
(`A_3`), `2132422` (`D_4`): equal in the labelled basis). A first version with `x_m` read at the
target failed (hand-overs of the strand `m` in `Seg_0` change the label there); the statement
above is the corrected one.

So for a step without hanging forests,

    h(V) = Σ_{c ∈ Same∖V} H_c(V).                                                    (2.1)

---

## 3. The diagonal group

Fix `I ∋ m`. The *diagonal group* consists of the terms of (1.1) with target `K = R = I∖m` of the
following four kinds: `Φ_m(I)`; `−ζ_m X_m(I)`; `−h_𝔠(I) X_{c_1}(J)` (`𝔠 ⊆ L ∩ Same` a chain,
`J = R ∪ \{c_1\}`, empty forest); `+X_j(I) h_𝔠(I∖j)` (`j ∈ R ∩ Same`, `𝔠 = (j, c_2, …)`, empty
forest).

*Time order.* In `X_m(I)` the moving position passes `R` (decreasing), the red strand, `L`
(increasing). Order `Same` accordingly (`Same ∩ R` decreasing, then `Same ∩ L` increasing). For a
subset `𝔟 = (b_1, …, b_p)` of `Same` in time order let `τ_𝔟` be the cycle
`m ↦ b_1 ↦ b_2 ↦ ⋯ ↦ b_p ↦ m` (as a bijection from the slots of `I` to the slots of `R`; `b_i ↦ b_{i+1}`
keeps the side of `b_{i+1}`, `b_p ↦` the left slot `m`). For `y ∈ Same` let `μ_y` be the last
element of `𝔟` before `y` in time order (`= m` if none), and

    A_y = x_y − x_m,      B_y = x_y − x_{μ_y},      N_𝔟 = r_c^{|𝔟|} ∏_{y∈Same} B_y^{-1}.

Note `B_{b_1} = A_{b_1}`.

**Proposition 3.1 (expansion of the diagonal terms) [proved].** In the labelled basis only the
brackets `[τ_𝔟]` occur, with the coefficients (contributions to (1.1)):

| term | coefficient of `[τ_𝔟]` |
|---|---|
| `Φ_m(I)` | `1` if `𝔟 = ∅`, else `0` |
| `−ζ_m X_m(I)` | `−N_𝔟 ∏_{y∈Same} A_y` |
| `−h_𝔠(I)X_{c_1}(J)` | `−(−1)^{|𝔠|} N_𝔟 ∏_{y∈𝔠} B_y ∏_{y∈Same∖𝔠} A_y` if `𝔠 ⊆ 𝔟`, else `0` |
| `+X_j(I)h_𝔠(I∖j)` | `−(−1)^{|𝔠|} N_𝔟 ∏_{y∈𝔠} B_y ∏_{y∈Same∖𝔠} A′_y` if `𝔠 ⊆ 𝔟`, else `0`; `A′_y = A_y − B_y` if `y ∈ 𝔟`, `y < j`, else `A′_y = A_y` |

*Proof.* `Φ_m(I) = 𝒟(std I → std(I∖m)) = [id]`.

`X_m(I)`: Lemma 2.3 with `s = t = std I` (no inversions), `μ_0 = m`, the passed labels in time
order; `S` = the set of hand-overs `𝔟`; the result is `[τ_𝔟]` with coefficient `N_𝔟`. Multiply by
`ζ_m = ∏ A_y` and the sign `−1`.

`h_𝔠(I)X_{c_1}(J)`: write `h_𝔠` as in §2.1 (left slide with omitted letters, then `Seg_0`), its
coefficient `ε(I;𝔠)(−r)^s ∏_{y∈Same∖𝔠} A_y` at the source. `Seg_0` followed by `X_{c_1}(J)` is one
moving-strand word for the position of `m`: it passes `R ∩ (c_1, m)`, then (as the strand at the slot
`c_1` of `J`) `R ∩ [1, c_1)`, the red strand, `L ∩ [1, c_1)`. The left slide is a second
moving-strand word on disjoint strands, starting with the label `c_1` and passing `L ∩ (c_1, m)`
with the letters at `c_2, …, c_s` omitted. Apply Lemma 2.3 to both (the pairs met are never
inverted: every pair of labels meets at most once in the whole word). The hand-over sets
`S_0` (position of `m`) and `S_i` (left slide between `c_i` and `c_{i+1}`) produce the bracket
`[τ_𝔟]` with `𝔟 = S_0 ∪ \{c_1\} ∪ S_1 ∪ \{c_2\} ∪ ⋯` in time order (the moving label after the
omitted letter at `c_i` is `c_i`), and the denominators are those of `N_𝔟` except for the slots of
`𝔠`, which are not passed (the label passing `y` is in both cases the last element of `𝔟` before
`y`). The powers of `r`: `(−r)^{|𝔠|} r^{|𝔟∖𝔠|}`. Sign: `−⟨ξ_R, ι_{c_1}θ_{(c_1,m)}ξ_I⟩ = −⟨ξ_R, ι_mξ_I⟩ = −1` by
(Λ4).

`X_j(I)h_𝔠(I∖j)`: `X_j(I)` is a moving-strand word (label `j`, passing `R ∩ [1,j)`, red, `L ∩ [1,j)`)
continued, after the vertex `I∖j`, by the left slide of `h_𝔠` (which starts at the left slot `j`,
where the moving position of `X_j` ends) — one moving-strand word for the position of `j`, with the
letters at `c_2, …` omitted; `Seg_0` of `h_𝔠` (position of `m`, passing `R ∩ (j, m)`) is a second
word on disjoint strands. Lemma 2.3 again; `𝔟 = S_0 ∪ \{j\} ∪ S_1 ∪ \{c_2\} ∪ ⋯`; denominators as
in `N_𝔟` except at the slots of `𝔠`. The coefficient of `h_𝔠` is a polynomial in the positional
dots at `std(I∖j)`; by Lemma 2.1(c) the dot at the slot `y` becomes the dot of the label standing
there after the expansion of `X_j(I)`: for a hand-over `y` of `X_j` this is `μ_y`
(Lemma 2.3: the label `μ_y` stops at the final position of `y`), and `x_{μ_y} − x_m = A_y − B_y`;
the slot `j` itself carries no factor (`j ∈ 𝔠`). The hand-overs of `X_j` are exactly the `y ∈ 𝔟`
with `y < j` that come after `j` in time order, i.e. all `y ∈ 𝔟 ∩ [1, j)`. Sign:
`⟨ξ_R, θ_{(j,m)}ι_jξ_I⟩ = −⟨ξ_R, ι_mξ_I⟩ = −1` by (Λ3), times `(−1)^{|𝔠|}` from `(−r)^{|𝔠|}`. ∎

**Proposition 3.2 (the diagonal group vanishes) [proved].** The sum of the diagonal group is
`0`, for every word.

*Proof.* `𝔟 = ∅`: `1 − N_∅ ∏A = 1 − ∏A/∏A = 0` (`μ_y = m` for all `y`). Let `𝔟 ≠ ∅`. Expand each
`A′_y = A_y − B_y` in the last row; then every contribution is `−(−1)^{|U|} N_𝔟 ∏_{y∈U}B_y ∏_{y∈𝔟∖U}A_y ∏_{y∉𝔟}A_y`
for a set `U ⊆ 𝔟`: `U = ∅` for `ζ_mX_m`; `U = 𝔠` for `h_𝔠X_{c_1}`; `U = \{j\} ∪ T ∪ 𝔠″` for
`X_jh_𝔠` with `𝔠 = (j) ∪ 𝔠″` and `T` the set of `y ∈ 𝔟 ∩ [1,j)` for which `−B_y` was chosen (sign
`(−1)^{|T|}`, so the total sign is `−(−1)^{1+|𝔠″|+|T|}`). Every non-empty `U ⊆ 𝔟` arises exactly
once: let `u` be its first element in time order. If `u ∈ L`, then `U ⊆ L` (left slots come after
right slots in time), `U` is a chain in `L ∩ Same` and arises from `h_UX_u` only. If `u ∈ R`, it
arises from `X_uh_𝔠` with `𝔠 = \{u\} ∪ (U ∩ L ∩ (u,m))` and `T = U ∖ 𝔠` — all elements of `U` after `u`
in time and in `R`, or in `L ∩ [1, u)`, are `< u` —, and from no `h`-first term. Hence the sum is

    −N_𝔟 ∏_{y∉𝔟}A_y · Σ_{U⊆𝔟} (−1)^{|U|} ∏_{y∈U}B_y ∏_{y∈𝔟∖U}A_y  =  −N_𝔟 ∏_{y∉𝔟}A_y · ∏_{b∈𝔟}(A_b − B_b)  =  0,

because `A_{b_1} = B_{b_1}`. ∎

*Check.* `07_formulas.py` compares every coefficient of every diagonal term with Proposition 3.1
(`1111`, `11111`, `1211`, `112111`, `111211`, `1121111` (`A_2`), `1121311`, `1312131`, `1213111`
(`A_3`; `1312131` is not of the type of §5 — colour 3 occurs twice — and is included because
Propositions 3.1, 3.2 hold for every word), `2132422`, `2213242` (`D_4`): 28 648 coefficients, no
mismatch; log `checks/07_logs/formulas.log`). The proposition uses nothing about the colours other than
that the terms are as listed; in words with hanging forests the diagonal *component* contains
further terms (§7).

---

## 4. One colour: Theorem H1a

Let `w = c^{m}`. Then: the tree formula of `w′` and of `w` has no pairs (a pair has to be linked
to the tree, and links need adjacent colours), so `d_w = δ_0`; `F = X_m`; `h` has no forests (a
pair of colour `c` is never linked to an arc or to another pair of colour `c`). So (2.1) holds and
the terms of (1.1) are: the diagonal group, and the terms `X_j(I)h_𝔠(I∖j)`, `h_𝔠(I)X_j(J)` with
`c_1 ≠ j`, both with target `K = (R∖\{j\}) ∪ \{a\}`, `a = c_1 ∈ L`, `j ∈ R`. (Any other composite:
`d_w(I→J)h(J→K)` has `J = I∖y`, `K = (R∖y) ∪ c_1`; `h(I→J)d_w(J→K)` has `J = R∪c_1`, `K = J∖y`; so
`K = R` or `K = (R∖j)∪a` with `j ≠ a`.)

**Lemma 4.1 [proved, every word].** Let `j ∈ R ∩ Same` and `a ∈ L ∩ Same`.
(a) If `a < j`: `h_𝔠(I)X_j(J) = 0` for every chain `𝔠` with `c_1 = a` (empty forest), and
`X_j(I) H_a(I∖j) = 0`.
(b) If `a > j`: for every chain `𝔠` with `c_1 = a`, `X_j(I)h_𝔠(I∖j)` and `−h_𝔠(I)X_j(J)` cancel.

*Proof.* (a) In `h_𝔠(I)X_j(J)` take the word "left slide, `Seg_0`, `X_j(J)`" (coefficient at the
source). `Seg_0` passes `R ∩ (a, m)` decreasing; when it reaches `j`, the strands `Y = R ∩ (a, j)`
(decreasing, `y_1 = max Y`), `j`, `m` stand next to each other. Then `Seg_0` continues: `[m passes j][m passes Y]`
and stops at `+a`; `X_j(J)` begins `[j passes Y][j passes m]` (the right slots of `J` below `j` are
`Y`, then `a`, where `m` stands). This is `W` of Lemma 2.4 with `A = j`, `B = m`, equal colours:
the product is `0`. For `X_j(I)H_a(I∖j)` use Proposition 2.6:
`H_a = ε(−r) LS_a ζ̂_a Seg_0^a`. `X_j(I)` ends with `[j passes a][j passes Y′]`, `Y′ = L ∩ (a, j)`
(increasing), and `LS_a` begins with `[a passes Y′][a passes j]`; the dots `ζ̂_a` come after `LS_a`.
Lemma 2.4 with `A = a`, `B = j` gives `0`.
(b) `j < a ≤ c_1`: the strands involved in `X_j` (`j`, slots `< j`, the red strand) and in `h_𝔠`
(slots `≥ a`, and `m`) are disjoint and no strand of one set crosses one of the other, so the two
products are the same element up to their coefficients (reduced words of the same permutation,
no triple of pairwise crossing strands: commutation moves only). Coefficients: that of `h_𝔠` in
`X_jh_𝔠` is read at `std(I∖j)`; the factor `∏_{y∈Same∩[1,j]}(x_y − x_m)` is symmetric in the dots of
the strands among which `X_j` has crossings of equal colour, so it commutes with `X_j`
(06 Lemma 1.1); all other factors are on strands not crossed by `X_j`. So both coefficients are
the same polynomial at the source. Signs: `+⟨ξ_K, θ_{(a,m)}ι_jξ_I⟩` and `−⟨ξ_K, ι_jθ_{(a,m)}ξ_I⟩`,
and `θ_{(a,m)}` commutes with `ι_j` (`j ∉ \{a, m\}`; (Λ2)). ∎

**Theorem H1a [proved].** (★) holds for `w = c^m`, every `m`, every type and red colour.

*Proof.* `E′(I → R)` is the diagonal group: 0 by Proposition 3.2. `E′(I → (R∖j)∪a)`, `a < j`:
`Σ_𝔠 h_𝔠X_j = 0` and `Σ_𝔠 X_jh_𝔠 = X_jH_a = 0` (Lemma 4.1(a), (2.1)). `a > j`: Lemma 4.1(b). ∎

---

## 5. One colour repeated, all other colours at most once

Throughout: `c` occurs `N` times in `w′`, every other colour at most once in `w`. A slot `j < m` of
colour `≠ c` is a *spectator*; it is *adjacent* if `c(j) ∼ c`, *far* otherwise.

### 5.1 The terms and the groups

Pairs have colour `c` (no other colour occurs twice). Hence: `h` has no forests (a hanging pair
must be linked to an arc or a pair, which needs an adjacent colour occurring twice), so (2.1)
holds. A term of `d_w` is `X_j` (any `j`) or `P(j; F)` with `j` adjacent and `F` a non-empty family
of pairs of colour `c` around `j` (`a_p < j < b_p`); since two pairs of equal colour may not
interlace and all contain `j`, `F` is *nested*. The terms of `F` are `X_m` and `P(j; F ∪ \{(a,m)\})`.
The terms of (1.1) fall into four groups by the mover `j` of the factor taken from `d_w`:

* **diag** (§3): `Φ_m`, `ζX_m`, `h_𝔠X_{c_1}`, `X_jh_𝔠` with `c_1 = j`;
* **same(j)**: `X_jh_𝔠`, `h_𝔠X_j` for `j` of colour `c`, `c_1 ≠ j` — treated in Lemma 4.1, whose
  proof did not use the colours of the other strands;
* **far(j)**: `X_jh_𝔠`, `h_𝔠X_j` for `j` far;
* **adj(j)**: `X_jh_𝔠`, `h_𝔠X_j`, `P(j;F)h_𝔠`, `h_𝔠P(j;F)`, `ζ P(j; F∪(a,m))` for `j` adjacent.

Every term lies in exactly one group, and (★) holds if each group vanishes component by
component. (`07_groups.py … adjj` confirms that for these words each group vanishes separately;
log `07_logs/groups.log`.)

### 5.2 Far movers

**Lemma 5.1 [proved].** Let `j ∈ R` be far, `a ∈ L ∩ Same`. (a) `a > j`: `X_jh_𝔠` and `−h_𝔠X_j`
cancel for every chain with `c_1 = a`. (b) `a < j`: `X_j(I)H_a(I∖j) − H_a(I)X_j(J) = 0`.

*Proof.* (a) As Lemma 4.1(b); `X_j` has no crossing of equal colours (the colour of `j` occurs
once), so all dots slide through it. (b) `H_a(I)X_j(J) = ε(−r)LS_a ζ̂_a Seg_0^a X_j(J)`. In
`Seg_0^a X_j(J)` Lemma 2.4 applies with `A = j`, `B = m`, `Y = R ∩ (a,j)` (`A` of a colour occurring
once, not adjacent to `c`), with `q_{jm} = 1`: the product equals the word in which `m` passes
`R ∩ (a,m) ∖ j` and `j` passes `R ∩ [1, j)` (not `m`), the red strand and `(L∖a) ∩ [1,j)`. In
`X_j(I)H_a(I∖j) = ε′(−r)X_jLS_aζ̂_aSeg_0^a` Lemma 2.4 applies with `A = a`, `B = j`,
`Y = L ∩ (a, j)`, `q = 1`: `j` passes `R ∩ [1,j)`, red, `L ∩ [1,j) ∖ a`, and `a` passes
`L ∩ (a, m) ∖ j`. In both results the three moving strands `a, j, m` cross only stationary strands,
the same ones, so both are reduced words of one permutation without triples of pairwise
crossing strands: the same element. The dots `ζ̂_a` (dots of strands of colour `c` and of `m`) are
read before `Seg_0^a` in both and slide through the crossings of `j` (different colours). The signs
are `+⟨ξ_K, θ_{(a,m)}ι_jξ_I⟩` and `−⟨ξ_K, ι_jθ_{(a,m)}ξ_I⟩`, opposite. ∎

### 5.3 Adjacent movers: the component `K = (R ∖ j) ∪ \{a\}`

Let `j ∈ R` be adjacent, of colour `e ∼ c`, and `t := t_{ce}`, `Q(x) := t_{ce}x + t_{ec}x_j`. Let
`a ∈ L ∩ Same`.

*If `a > j`*, the only terms of adj(j) with target `(R∖j) ∪ a` are `X_jH_a` and `−H_aX_j` (a pair
around `j` has its left slot `< j`); they cancel as in Lemma 4.1(b) (disjoint strands; `ζ̂_a` slides
through `X_j`, which has no crossing of equal colours).

*If `a < j`*, the terms of adj(j) with target `K = (R∖j) ∪ \{a\}` are exactly (the target forces
the pairs used to be a single pair with left slot `a`):

| | term | |
|---|---|---|
| (i) | `+X_j(I)·H_a(I∖j)` | |
| (ii) | `−H_a(I)·X_j(J_a)` | `J_a = R ∪ a` |
| (iii) | `−ζ_m·P(j; (a,m))(I)` | |
| (iv) | `+P(j;(a,b))(I)·H_b(I′)` | `b ∈ R ∩ Same ∩ (j, m)`, `I′ = (I∖\{j,b\})∪\{a\}` |
| (v) | `−H_c(I)·P(j;(a,c))(J_c)` | `c ∈ L ∩ Same ∩ (j, m)` (in `J_c` the slot `c` holds the strand `m`) |

Write `Λ_L = L ∩ Same ∩ (a, m)`, `Λ_R = R ∩ Same ∩ (a, m)`. The brackets that occur are
`[a, Y, m, Z]`: the strand `j` goes to `−j`, and the cycle `a ↦ y_1 ↦ ⋯ ↦ y_p ↦ m ↦ z_1 ↦ ⋯ ↦ z_q ↦ a`
with `Y = (y_1 < ⋯ < y_p) ⊆ Λ_L`, `Z = (z_1 > ⋯ > z_q) ⊆ Λ_R`. Put, for `y ∈ Λ_L`, `μ_y` = the last
element of `\{a\} ∪ Y` below `y`, and for `z ∈ Λ_R`, `ν_z` = the last element of `\{m\} ∪ Z` above `z`;
`A_s = x_s − x_m`, `B_y = x_y − x_{μ_y}`, `B′_z = x_z − x_{ν_z}`; `λ = max(\{a\} ∪ (Y ∩ (a,j)))` (the label
that passes `j` on the left), `ν = min(\{m\} ∪ (Z ∩ (j,m)))` (the label that passes `j` on the right);
`P_Y = ∏_{y∈Λ_L} (y ∈ Y ? A_y − B_y : A_y)`; and

    base = σ · r_c t · r_c^{|Y|+|Z|} ∏_{y∈Λ_L} B_y^{-1} ∏_{z∈Λ_R} B′_z^{-1} · ∏_{z∈Λ_R} A_z · ∏_{s∈Same, s<a} A_s,     σ = ⟨ξ_K, ι_j θ_{(a,m)} ξ_I⟩.

**Proposition 5.2 (coefficients) [proved].** The coefficient of `[a, Y, m, Z]` is `base` times:

| term | factor |
|---|---|
| (i) | `−t^{-1} Q(x_λ) · P_Y` |
| (ii) | `+t^{-1} Q(x_ν) · P_Y` |
| (iii) | `A_a ∏_{y∈Λ_L} A_y` |
| (iv) | `−B′_b · P_Y` if `b ∈ Z`, else `0` |
| (v) | `−B_c · A_a ∏_{s∈Λ_L, s<c} A_s ∏_{s∈Λ_L, s>c}(s∈Y ? A_s − B_s : A_s)` if `c ∈ Y`, else `0` |

*Proof.* All diagrams are products of moving-strand words; the strand `j` crosses no strand of its
own colour, so `X_j` and the mover part of `P(j;·)` are single brackets.

(iii) `P(j;(a,m))` (01 Def. B.10: first the left strand of the pair, then the mover, then the right
strand): the strand of `a` passes `L ∩ (a, m)`; `j` passes the unaffected strands `< j`; the strand
of `m` passes `R ∩ (a,m) ∖ j`. Lemma 2.3 for the two slides gives `[a,Y,m,Z]` with coefficient
`r^{|Y|+|Z|}∏B^{-1}∏B′^{-1}`; no bigon (the mover crosses neither strand of the pair). The term
coefficient is `κ(I;j,(a,m))·(−1)·r t` (connector `j ∈ I`), `κ = σ`, and the sign in (1.1) is `−`;
so the contribution is `σ r t ζ_m·(…)`, and `ζ_m = A_a ∏_{Λ_L}A ∏_{Λ_R}A ∏_{s∈Same, s<a}A` (every slot
of `Same` is `< m`).

(i) `X_j(I)`: one bracket; it inverts `j` with every strand it passes, among them `a` and `Y ∩ (a,j)`.
`H_a(I∖j)` (Prop. 2.6): `LS_a` passes `L ∩ (a, m)` and, between, the strand `j` (now at `−j`);
`ζ̂_a`; `Seg_0^a` passes `(R∖j) ∩ (a,m)`. Lemma 2.3 for `LS_a` and `Seg_0^a`: the label passing `j` in
`LS_a` is `λ`, and `\{λ, j\}` was inverted by `X_j`: Lemma 2.1(a) gives the factor `q_{λj} = Q(x_λ)`;
no other pair is met twice. `ζ̂_a` is read after `LS_a`: at the slot `y ∈ Λ_L` stands `μ_y` if
`y ∈ Y`, else `y` (Lemma 2.3), giving `P_Y`; the factors for `Λ_R` and for `s < a` are unaffected.
Coefficient `⟨ξ_K, θ_{(a,m)}ι_jξ_I⟩(−r) = σ(−r)` (Λ2).

(ii) `H_a(I)` then `X_j(J_a)`: `Seg_0^a` passes `R ∩ (a,m)`, including `j` with the label `ν`;
`X_j(J_a)` passes the right slots of `J_a` below `j` — the slots of `R ∩ (a,j)` and the slot `a` —, the
red strand and `(L∖a) ∩ [1,j)`. The label `ν` stands, after `Seg_0^a`, at one of these slots (at the
next hand-over below `j`, or at `a`), so `\{ν, j\}` is met a second time: factor `Q(x_ν)`; no other
pair twice. Sign `−⟨ξ_K, ι_jθ_{(a,m)}ξ_I⟩(−r) = σ r`.

(iv) `P(j;(a,b))(I)`: the strand of `a` passes `L ∩ (a,b)`; `j` passes the unaffected strands `< j`;
the strand of `b` passes `R ∩ (a,b) ∖ j`. `H_b(I′)`: `LS_b` continues the left position (the
strand now at the left slot `b`) over `L ∩ (b,m)`; `Seg_0^b`: `m` passes `R ∩ (b,m)`. So the left
position passes `Λ_L` as before, the right ones pass `R ∩ (b,m)` (label `m`) and `R ∩ (a,b)` (label
`b`): this is the pattern of (iii) with `b ∈ Z` as a hand-over that is not an expanded crossing (its
denominator `B′_b` is absent). No pair is met twice (the mover crosses only unaffected strands
`< j`, which nothing else crosses). `ζ̂_b` (read between `LS_b` and `Seg_0^b`): its factors at the
slots `\{a\} ∪ R∩(a,b)` carry, after the right slide of the pair, a permutation of the labels
`\{b\} ∪ R∩(a,b)`; their product is `∏_{z∈Λ_R ∩ [a,b]}A_z · A_b`, and together with `R∩(b,m)` this is
`∏_{Λ_R}A`; on the left the factors give `P_Y` (as in (i)); `x_a − x_m` does not occur (`a` is a
slot of the pair). Coefficient: `κ(I;j,(a,b))(−1)rt · ε(I′;b)(−r)` with
`⟨ξ_K, θ_{(b,m)}ι_jθ_{(a,b)}ξ_I⟩ = −σ` (the two orders put `ξ_a, ξ_b` into the places of `b, m`
resp. leave `ξ_b` and put `ξ_a` at `m`: one transposition).

(v) `H_c(I)`: `LS_c` passes `L ∩ (c,m)`; `ζ̂_c`; `Seg_0^c`: `m` passes `R ∩ (c, m)`. `P(j;(a,c))(J_c)`:
the strand of `a` passes `L ∩ (a,c)`; `j` passes unaffected strands `< j`; the strand at the slot
`c` (the right position of `m`, continued) passes `R ∩ (a,c) ∖ j`. So the right position passes
`Λ_R` exactly as in (iii), and the left one has `c ∈ Y` as a non-expanded hand-over (no `B_c`). No
pair twice. `ζ̂_c` is read after `LS_c`: factors `A_s − B_s` resp. `A_s` for `s ∈ Λ_L, s > c`, and
`A_s` for `s < c` (including `A_a`). Coefficient `−ε(I;c)(−r)·κ(J_c;j,(a,c))(−1)rt` with
`⟨ξ_K, ι_jθ_{(a,c)}θ_{(c,m)}ξ_I⟩ = σ` (`θ_{(a,c)}θ_{(c,m)} = θ_{(a,m)}` on `ξ_I`).

In (iv), (v) the powers of `r` are `r²t·r^{|Y|+|Z|−1}`, i.e. `r t r^{|Y|+|Z|}` as in `base`. ∎

**Proposition 5.3 [proved].** The five contributions add up to `0`.

*Proof.* Divide by `base`. (i)+(ii) `= −t^{-1}(Q(x_λ) − Q(x_ν))P_Y = −(x_λ − x_ν)P_Y` (the bigon divided
difference). The `b ∈ Z ∩ (j,m) = \{z_1 > ⋯ > z_k\}` give `−Σ_b B′_b P_Y = −(x_ν − x_m)P_Y` (telescoping:
`ν_{z_1} = m`, `ν_{z_i} = z_{i−1}`, and `ν = z_k`, or `m` if `k = 0`). So (i)+(ii)+(iv)
`= −(x_λ − x_m)P_Y = −A_λ P_Y`. For (iii)+(v) write `Y_> = Y ∩ (j,m)`; the telescoping identity
`∏_{s}A_s − ∏_{s}(s∈Y_>? A_s − B_s : A_s) = Σ_{c∈Y_>} B_c ∏_{s<c}A_s ∏_{s>c}(s∈Y? A_s − B_s : A_s)` (products over
`s ∈ Λ_L ∩ (j, m)`, the factors `s < j` being common) gives (iii)+(v)
`= A_a ∏_{s∈Λ_L, s<j}A_s · ∏_{s∈Λ_L, s>j}(s∈Y? A_s − B_s : A_s)`. Finally, for `Y ∩ (a,j) = \{y_1 < ⋯ < y_q\}`,
`A_{y_k} − B_{y_k} = x_{μ_{y_k}} − x_m = A_{y_{k−1}}` (`y_0 := a`), hence

    A_λ ∏_{s∈Λ_L, s<j}(s∈Y? A_s − B_s : A_s) = A_{y_q} · A_a A_{y_1}⋯A_{y_{q−1}} · ∏_{s∉Y}A_s = A_a ∏_{s∈Λ_L, s<j} A_s,

and the two remaining expressions are equal. ∎

*Check.* `07_formulas.py` (mode "adj0"): every coefficient of (i)–(v) against Proposition 5.2, on
the words listed in §3 (6 068 coefficients, no mismatch — including words with nested families,
where this component is still of the form (i)–(v)).

### 5.4 Adjacent movers: one pair

**Lemma 5.4 [proved].** Let `j ∈ R` be adjacent, `(a, b)` a pair of colour `c` with `a < j < b`,
`a ∈ L`, `b ∈ R`, and `c′ ∈ L ∩ Same ∖ \{a\}` with `c′ > j`. (a) If `c′ > b`, `P(j;(a,b))(I)H_{c′}(I′)`
and `−H_{c′}(I)P(j;(a,b))(J_{c′})` cancel. (b) If `j < c′ < b`, both are `0`.

*Proof.* (a) The strands involved in `P(j;(a,b))` (slots in `[a,b]`, `j`, the unaffected strands `< j`
crossed by `j`) and in `H_{c′}` (slots in `[c′, m]`) are disjoint and do not cross: equal elements up
to coefficients. `ζ̂_{c′}` contains `∏_{s∈Same∩[a,b]}(x_s − x_m)`, symmetric in the dots of the strands
among which `P(j;(a,b))` has crossings of equal colour (the two strands of the pair and the unaffected
strands of colour `c` in `(a,b)`), so it commutes with `P(j;(a,b))`; the other factors are on
strands not touched. Signs opposite (`θ_{(c′,m)}` commutes with `ι_jθ_{(a,b)}`).
(b) In `P(j;(a,b))H_{c′}`: the left part of `P` ends with `[a passes c′][a passes Y]`,
`Y = L ∩ (c′, b)` — the strand of `a` passes `L ∩ (a,b) ∋ c′` —; the mover part and the right part
of `P` involve other strands and do not cross the strands `a, c′, Y` and those of `LS_{c′}`, so they
commute past `LS_{c′}` (no triples: commutation moves only); `LS_{c′}` begins with
`[c′ passes Y][c′ passes a]` (the strand of `a` stands at the left slot `b`). Lemma 2.4 (`A = c′`,
`B = a`): `0`. In `H_{c′}P(j;(a,b))`: `Seg_0^{c′}` ends with `[m passes b][m passes Y_R]`,
`Y_R = R ∩ (c′, b)`; the left and mover parts of `P` commute to the front; the right part of `P`
begins with `[b passes Y_R][b passes m]` (the strand `m` stands at the right slot `c′`). Lemma 2.4
(`A = b`, `B = m`): `0`. ∎

### 5.5 Proof of Theorem H1, and what remains without (NN)

*Proof of Theorem H1.* By §5.1 it suffices that every group vanishes. diag: Proposition 3.2.
same: Lemma 4.1 (its proof uses only Lemma 2.4 and that `j` has colour `c`). far: Lemma 5.1. adj(j):
under (NN) every family `F` in a term of the group has at most one pair: a family of two nested
pairs around `j` needs two slots of colour `c` below `j` and one in `(j, m)`, i.e. `s_2 < j < s_N`.
The components of adj(j) are therefore: `K = (R∖j) ∪ a` (§5.3: `a > j` pairwise, `a < j`
Proposition 5.3), and `K = (R∖\{j,b\}) ∪ \{a, c′\}` reached by `P(j;(a,b))H_{c′}` and
`H_{c′}P(j;(a,b))` only (a term `(A5)`/`(iv)`/`(v)`-type with this target would need two pairs).
Under (NN) `a` is the only slot of colour `c` below `j` (since `b ∈ Same ∩ (j,m) ≠ ∅`), so `c′ > j`, and
Lemma 5.4 applies. ∎

**Without (NN)** (one colour repeated, others once, an adjacent spectator in `(s_2, s_N)`) the
group adj(j) contains nested families. The data (`07_groups.py … adjj`, words `112111`, `1121111`,
`1112111`, `1121311`, `2132422` — all violate (NN)) show that adj(j) still vanishes component by
component, but no longer configuration by configuration: e.g. for `112111` (`A_2`), `I = \{3,4,6\}`,
`K = \{1,2\}`, the terms with the single pair `(1,4)` and chain start `2` (interlaced with the arc
`(2,6)`) combine with those with the pair `(2,4)` and chain start `1` and with
`ζP(3; (1,6),(2,4))`, `H_5 P(3;(1,5),(2,4))` (`07_logs/adjfine.log`). The structure is the same as in
§5.3 with the pair `(2,4)` (resp. `(1,4)`) riding along; I expect the proof of §5.3 to go through
with `P_Y` replaced by a product over the nested pairs, but this is **not done**.

> **Identity N (open, computed).** For `w` as in Theorem H1 without (NN), every component of the
> group adj(j) vanishes. Verified on all components of `112111` (red 1, 2), `1121111`, `1112111`
> (`A_2`), `1121311`, `1213111` (`A_3`), `2132422`, `2213242` (`D_4`).

With Identity N, Theorem H1 holds without (NN).

---

## 6. Consequences

**Corollary 6.1 [proved].** Let `w` be a word in which one colour `c` occurs `n` times, at
`s_1 < ⋯ < s_n`, and every other colour at most once, and assume that no slot of a colour adjacent
to `c` lies strictly between `s_2` and `s_{n−1}`. Then the tree formula of `w` satisfies `d² = 0`.

*Proof.* Induction on the length via 06 Prop. 5.2. Let `w = w′·x`. If `x ≠ c`, the colour of the
last slot occurs once: `Same = ∅`, `ζ_m = 1`, `h = 0`, `F = X_m = Φ_m` (no crossing of equal colours,
no pair `(a, m)`): (★) holds trivially (this is Theorem A of 05). If `x = c` is the `k`-th
occurrence, (NN) for `w_{≤ s_k}` is: no adjacent slot in `(s_2, s_{k−1})` — implied by the hypothesis.
Theorem H1 gives (★). `d′² = 0` by induction (the prefix has the same form), and 06 Prop. 5.2
gives `d² = 0`. ∎

For `n = 3` the hypothesis is empty: this is **Theorem C of `05_main_sliding.md`**, reproved without
the four-term identities of 05 §10. New cases (`n ≥ 4`): all `c^n` (where `d = δ_0` and `d² = 0`
was elementary; the content here is (★), i.e. the explicit homotopy), and for instance `A_2`
`2 1^k`, `1 2 1^k`, `1^k 2 1` (the spectator `2` before `s_2` or after `s_{n−1}`), `D_4` with
`c = 2`: `1 2 3 2 2 2 4 2`. Not covered: `1 1 2 1 1 1` (`s_2 = 2 < 3 < s_4 = 5`), which needs
Identity N.

---

## 7. Beyond H1 (stages H2–H4)

### 7.1 What holds for every word

* Lemma 1.1, Lemma 2.1, Lemma 2.3 (expansion), Lemma 2.4 (double crossing): no hypothesis on
  the word.
* Proposition 2.6 (compact form of the **chain part** of `h`): every word.
* Propositions 3.1, 3.2: the four kinds of diagonal terms **with empty forests** vanish
  together, for every word (the proof uses only that no two labels meet twice in these
  composites, which holds always).
* Lemma 4.1 (a), (b) for the **chain-only** terms of `h`: every word.
* The coefficient analysis of §5.3 for a mover `j` whose colour occurs once and the terms
  (i)–(v) with empty forests: every word in which those are the only terms of the component.

### 7.2 Where H2–H4 differ

As soon as a colour `c′ ∼ c` occurs twice in `w′`, `h` has **hanging forests** (06 §7.2). Three
new phenomena appear, none of which is handled above:

1. *The compact form does not cover forests.* The admissible forests of a chain depend on its arcs
   (a pair hangs from the arc `(c_i, c_{i+1})` it is interlaced with), so the sum over the chains
   with first element `c` is not a sum over the hand-overs of one left slide with a fixed
   forest. A compact form "chains + forests" would have to produce the forest links to inner
   arcs from the expansion of `LS_c` (a hand-over of `LS_c` at `y` creates the arc ending at `y`);
   I did not find it.
2. *The groups of §5.1 no longer close.* Data (`07_groups.py … adjj`, log
   `07_logs/groups_H2.log`; numbers of components that vanish / do not vanish):

   | word | diag | same, `a<j` | same, `a>j` | far | adj(j) |
   |---|---|---|---|---|---|
   | `A_2` red 1 `12121` | 16 / 4 | 5 / 0 | 5 / 4 | – | 10 / 12 |
   | `A_2` red 2 `21212` | 16 / 4 | 5 / 0 | 5 / 4 | – | 10 / 12 |
   | `A_2` red 1 `12211` (no forest can hang) | 16 / 0 | 4 / 0 | 4 / 0 | – | 16 / 0 |
   | `A_3` red 2 `213212` | 32 / 8 | 10 / 0 | 10 / 8 | – | 40 / 24 |
   | `A_3` red 2 `1232123` | 64 / 36 | – | 0 / 36 | 48 / 8 | 26 / 80 |

   Forest terms enter the diagonal component `K = I∖m` and the components of the movers of
   colour `c` with `a > j`; (★) itself holds in all these words (`07_logs/star_H2.log`). The
   case "same, `a < j`" still vanishes by itself.
3. *Trees of the prefix with pairs of several colours* (links between pairs); this is where the
   tree lemma of 06 §6.5 (sum over connected subsets) has to be combined with the chain
   inclusion–exclusion of §3. The `F = ∅` part of §5.3 is the prototype (a bigon divided
   difference next to a chain telescoping).

### 7.3 Evidence for H2–H4 obtained here

(★) was checked with the independent engine `07_lab` (labelled calculus; it does not use the
klrw implementation of the algebra) — `07_logs/star_H2.log`; together with the 168 words of 06
§7.2 (klrw-based) this is the evidence for H2–H4. The words, their colour multiplicities and the
result are listed in §8.

### 7.4 Remaining identities (precise list)

* **Identity N** (H1 without (NN)): every component of the group adj(j) of §5.1 vanishes.
  Verified on all components of the words of §5.5.
* **Identity H** (H2–H4): (★) itself on the components containing forest terms; equivalently, by
  §7.1, (★) minus the parts proved there. Verified on the words of 06 §7.2 and of `star_H2.log`.
  A proof along the lines of this file needs (1) a compact form for chains with forests, and (2)
  the analogue of Proposition 5.3 in which the bigon factor of a single adjacent mover is replaced
  by the bigon polynomials of the linked pairs (06 Lemma 6.8) and the tree lemma 06 Lemma 6.7.

---

## 8. Status

### Proved completely (from the relations, (F1), (F3), and the results of 06 §§2–5, §9 and 05 Lemma 3.2)

* Lemma 1.1 (bracket basis of `e(s)A[Δ^{-1}]e′` with source dots), Lemma 2.1 (expansion rule),
  Lemma 2.3 (moving-strand expansion), Lemma 2.4 (double-crossing lemma) — every word.
* **Proposition 2.6 (compact form of the chain part of `h`)** — every word.
* Propositions 3.1, 3.2 (the diagonal group with empty forests vanishes:
  `∏_{b∈𝔟}(A_b − B_b) = 0`) — every word.
* Lemma 4.1 (movers of colour `c`, chain-only terms) — every word.
* **Theorem H1a**: Conjecture H for `w = c^{N+1}` (all `N`, every type, every red colour).
* Lemma 5.1, Propositions 5.2, 5.3, Lemma 5.4, and **Theorem H1**: Conjecture H for `w = w′·c`
  with `c` occurring `N` times in `w′` (any `N`), every other colour at most once, and (NN).
* **Corollary 6.1**: `d² = 0` for every word with one colour `c` occurring `n` times, all other
  colours once, and no slot of a colour adjacent to `c` strictly between `s_2` and `s_{n−1}`;
  contains Theorem C of 05 (`n = 3`).

Every identity used is checked by an exact computation (after the final edit of text and
scripts): `07_formulas.py` (all coefficients of Props. 3.1, 5.2: 30 000+ coefficients, no
mismatch), `07_vanish.py` (Lemmas 4.1, 5.1, 5.4 and the pairwise statements), `07_compact.py`
(Prop. 2.6), `07_groups.py` (the splitting of §5.1), `07_star_check.py` ((★) itself). Logs in
`checks/07_logs/`.

### Proved assuming

* Theorem H1 without (NN) assuming **Identity N** (§5.5): one colour repeated `N` times, all
  other colours once, arbitrary positions.

### Verified only by computation

* Identity N: all components of adj(j) for `112111` (red 1, 2), `1121111`, `1112111` (`A_2`),
  `1121311`, `1213111` (`A_3`), `2132422`, `2213242` (`D_4`) (`07_groups.py … adjj`,
  `07_logs/groups.log`).
* (★) with the engine of this file (independent of the klrw implementation):
  `111`, `1111` (red 1, 2), `111111` (red 1, 2), `1211`, `2112`, `112111` (red 1, 2), `121111`,
  `1211111`, `1111121`, `1121111`, `1112111` (`A_2`); `1121311` (`A_3`);
  H2-type: `12121`, `21212`, `12211` (`A_2`), `213212`, `1232123`, `2132132` (`A_3`), `2132423`
  (`D_4`). No failure; (★) fails on the cyclic word `1221122` (88 classes), as in 06.
* Conjecture H in general (H2–H4): the 168 words of 06 §7.2 and the words above.

### Not proved

* **H2** (`c` three times, other colours at most twice), **H3** (multiplicities `≤ 3`,
  non-cyclic), **H4** (all non-cyclic words): the hanging forests of `h` are not covered by the
  compact form, and the groups of §5.1 do not close (§7.2: on `12121`, `21212`, `213212`,
  `1232123` the diagonal and the same-colour groups no longer vanish separately; on `12211`,
  where no forest can hang, they do). §7.4 lists what is needed.
* Identity N.

### Failed attempts, corrections

* The first version of the compact form read the dot `x_m` of `W_c` at the target; it is wrong
  as soon as the strand `m` hands over in `Seg_0` (10 of 32 `(I,c)` of `11111`); the correct
  statement reads it before `Seg_0` (Prop. 2.6).
* The Reduction Lemma of 05 (idea (iii) of the brief) applies to the defect `E′` only if
  `d_w² = 0` is already known (then `E′` is a cocycle: `ι_R(d′)E′ − E′ι_L(d′) = −ζ_m·(d_w² defect)`);
  for the induction it does not help, and it is not used.
* "Partially intertwined" moves (idea (ii)): the telescoping
  `Φ_m = ζ_mX_m − r Σ_k ζ_{>k} P_k S_k` (time order of the same-colour crossings of `m`) is exact,
  but its terms do not match the terms of `[d′, h]` one by one; the inclusion–exclusion of §3 is
  what organises the matching. Not used.
* Lemma 1.4 of 05 (vanishing) needs (S2), which fails with adjacent spectators between the two
  crossings; Lemma 2.4 (only the triples with a fixed middle strand matter) is the version that
  works here.
* The groups "adj(j)" do not split by configuration `(F, a)` when nested families occur (§5.5);
  they split by component.

---

## Appendix: scripts (`proofs/checks/`, run with `sage -python` from there)

| script | content | used in |
|---|---|---|
| `07_lab.py` | labelled calculus over the fraction field: expansion of words (Lemma 2.1), terms of the tree formula, of `h`, `Φ`, `X`; all contributions to (1.1) | all |
| `07_star_check.py TYPE RANK RED WORD …` | (1.1) expanded, every class `(I, K, τ)` | evidence |
| `07_tabulate.py TYPE RANK RED WORD [I]` | prints every product of (1.1) by bracket | exploration |
| `07_theta.py RMAX` | the local identity behind Lemma 4.1(a) (a special case of Lemma 2.4 with the chain sum) | §2 |
| `07_compact.py …` | Proposition 2.6 | §2.3 |
| `07_groups.py … [adjj]` | splitting of (1.1) into the groups of §5.1 | §5 |
| `07_formulas.py …` | Propositions 3.1 and 5.2, coefficient by coefficient | §3, §5.3 |
| `07_vanish.py …` | Lemmas 4.1, 5.1, 5.4 and the pairwise statements | §4, §5 |
| `07_adjfine.py` | splitting of adj(j) by configuration `(F, a)`; prints the configurations that do not vanish alone (nested families) | §5.5 |

Logs (`checks/07_logs/`, all produced after the last edit of the scripts): `formulas.log`,
`vanish.log`, `compact.log`, `theta.log`, `groups.log`, `adjfine.log` (H1 words); `star_small.log`,
`star_A2.log` ((★) for H1 words, and the cyclic control `1221122`); `groups_H2.log`,
`star_H2.log` (H2-type words).
