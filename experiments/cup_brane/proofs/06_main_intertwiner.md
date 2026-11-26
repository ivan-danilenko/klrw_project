# 06 — The main theorem via intertwiners ("renormalised R-matrix")

Notation and standing facts are those of `../PROOFS.md` §1 and `README_FOR_PROVERS.md`
(product `a·b` = first `a`, then `b`; the tree formula of `../NOTES.md` §2 with the signs of
`../formula.py`). All scripts named below are in `proofs/checks/`; run them from that
directory with `sage -python 06_….py …` (only `06_check_bigons.py` is plain `python3`). The
common library is `06_lib.py`.

Every statement is marked **[proved]** (complete argument in this file, from the relations
and the standing facts F1–F3), **[computed]** (exact computation, cases listed) or
**[conjecture]**.

---

## 0. Summary

**The idea works.** Let `C_Φ` be the cube in which every elementary move has its
same-colour crossings replaced by the intertwiner `φ = (x_1 − x_2)ψ − r_c`, with differential
`d_Φ`. Then:

1. **[proved]** `d_Φ² = 0` for every word, and the moves `Φ_j` are natural (they slide through
   every element of the algebra of the earlier strands, red strand included). Both follow
   from one statement: the generators "`ψ` for different colours, `φ` for equal colours"
   satisfy the braid relations *exactly* (Lemma 2.2), so products of them depend only on the
   permutation of the *labelled* strands, and double crossings multiply by explicit
   polynomials (the *labelled calculus*, §3).
2. **[proved]** If a triangular map `G : C_Φ → C` with non-zero polynomials on the diagonal
   satisfies `d_Φ G = G d` for some family `d` of cube components, then `d² = 0`
   (Proposition 5.1); one-step (cone) version: Proposition 5.2.
3. **[proved] Theorem 6.2.** If every colour occurs at most twice in the word, then
   `d_Φ G = G d_tree` for the explicit map

       G e_I = Σ_{F ⊆ 𝔓(I)}  ε(I,F) · ∏_{p∈F} (−r_p) · ∏_{b∈I second occurrence, not exchanged} (x_{a} − x_{b})
                              · ∏_{p<q in F linked} ( t_{c_p c_q} x_{b_p} + t_{c_q c_p} x_{a_q} ) · [σ_F]

   (sum over *all* sets of exchange pairs available at `I`, no tree condition; a "mixed
   bigon polynomial" in the dot of the right strand of the earlier pair and the left strand
   of the later pair for every linked couple of pairs). Hence **the tree formula satisfies
   `d² = 0` whenever every colour occurs at most twice** (Corollary 6.3: `sl_n` with
   `ω_1, ω_2`, the vector representation of `so_{2n}`, all of `D_4`, and all non-minuscule
   words of this kind). The proof has no induction on the word and no braid-relation
   corrections: in the labelled calculus the identity splits into three families of
   polynomial identities, each of which is the same one-line recursion over a tree
   (`Q(x_a, y) − Q(x_b, y) = t·(x_a − x_b)`: "the divided difference of a bigon is `t`").
4. **[computed]** For words with a colour occurring three or four times and no cyclic
   configuration, `G` exists and, whenever the complex is not contractible, is unique in
   the complete ansatz (19 words, §7.1). Its one-step factors have a closed form
   (**Conjecture H**, §7.2: chains of same-colour slots with dot factors, and trees of
   exchange pairs hanging from the chain), verified as exact chain maps on 168 cycle-free words
   including the minuscule words of `A_5 ω_3` (9 strands) and `D_5` spinor (10 strands).
   By Proposition 5.2, Conjecture H implies Conjecture C1′ (tree formula for every
   cycle-free word).
5. **[computed]** Cyclic case (`A_2`, word `1221122`, red 1, 7 strands; §8): the true
   differential is the tree formula plus one component (dotted terms, a coefficient 2,
   hand-over terms). A comparison map `Γ : Cone(Φ_7) → (C, d_true)` **exists**, is unique up
   to homotopy (the space of chain maps modulo null-homotopic ones is 1-dimensional), and a
   sparsest representative is the formula of Conjecture H plus four terms. So the true
   differential *is* `G^{-1} d_Φ G`; but the tree-only formula for `h` does not produce it
   (the division `Φ − [d′, h]` by `z` fails in 21 components), and the general rule for the
   extra terms of `h` is open.

Conventions fixed here: `φ = (x_1 − x_2)ψ − r` with the dots *before* the crossing (README);
consequently the diagonal of `G` is `ζ_I = ∏_{x∈I} ∏_{i<x, c(i)=c(x)} (x_i − x_x)`, which is
`±` the `z_I` of the brief (`z = ∏ (x_m − x_y) = (−1)^N ζ`; replacing `φ` by `−φ` exchanges
the two). Dots in coefficients are always written at the *source* (left) end of a diagram.

---
## 1. Conventions

**1.1 Algebra.** `A` is the algebra of `PROOFS.md` §1.1 for the word `w = (i_1,…,i_m)` and
the red colour `k`, with the signs of the README:

* (S) same colour `c`, state `(c,c)`: `ψ² = 0`, `x_1ψ − ψx_2 = r_c`, `ψx_1 − x_2ψ = r_c`;
* (D) different colours or red–black: dots slide through the crossing; `ψ² = q`, where
  `q = t_{cc′}x_c + t_{c′c}x_{c′}` (`c ∼ c′`), `q = u_k x` (black of colour `k` with the red
  strand), `q = 1` otherwise;
* (B) `ψ_1ψ_2ψ_1 = ψ_2ψ_1ψ_2` on every state of three strands except
  `(c, c′, c)`, `c ∼ c′`: difference `+ r_c t_{cc′}`, and `(k, W, k)`: difference `+ r_k u_k`.

Standing facts used: (F1) basis theorem, in the form "`e A e′` is free as a left and as a
right module over the polynomial ring in the dots" (only in Proposition 5.1 and Lemma 6.5);
(F3) Matsumoto's theorem (Proposition 3.2).

**1.2 Divided differences.** On a state with two neighbouring strands of the same colour at
positions `i, i+1`, for a polynomial `f` in the dots write `s_i f` for `f` with `x_i, x_{i+1}`
exchanged and `∂_i f = (f − s_i f)/(x_i − x_{i+1})`.

**Lemma 1.1 [proved].** `ψ_i f = (s_i f) ψ_i + r_c ∂_i f` for every polynomial `f` in the
dots (on the left side `f` is read after the crossing, on the right before it).

*Proof.* Dots at other positions commute with `ψ_i` (R1). For `f = x_i`: `ψx_1 = x_2ψ + r`
(S), and `∂_i x_i = 1`. For `f = x_{i+1}`: `ψx_2 = x_1ψ − r` (S), and `∂_i x_{i+1} = −1`. If
the formula holds for `f` and `g`, then
`ψfg = (s f)ψ g + r(∂f)g = (sf)(sg)ψ + r( (sf)(∂g) + (∂f) g ) = s(fg)ψ + r∂(fg)`. ∎

In particular `ψ_i` commutes with every `f` that is symmetric in `x_i, x_{i+1}`.

---

## 2. The intertwiner

For two neighbouring black strands of the same colour `c` put

    φ = (x_1 − x_2) ψ − r_c .

**Lemma 2.1 [proved]** (README; checked in `alg.py`, `06_check_basic.py`).
(a) `φ = −ψ(x_1 − x_2) + r_c`; (b) `x_1φ = φx_2`, `x_2φ = φx_1`; (c) `φ² = r_c²`.

*Proof.* (a) `(x_1−x_2)ψ = (ψx_2 + r) − (ψx_1 − r) = −ψ(x_1−x_2) + 2r` by (S).
(b) `x_1φ = (x_1−x_2)x_1ψ − rx_1 = (x_1−x_2)(ψx_2 + r) − rx_1 = (x_1−x_2)ψx_2 − rx_2 = φx_2`;
`x_2φ = (x_1−x_2)(ψx_1 − r) − rx_2 = (x_1−x_2)ψx_1 − rx_1 = φx_1`.
(c) By (a), `φ² = ((x_1−x_2)ψ − r)(−ψ(x_1−x_2) + r) = −(x_1−x_2)ψ²(x_1−x_2) + r((x_1−x_2)ψ + ψ(x_1−x_2)) − r² = 0 + 2r² − r²`. ∎

**Definition 2.2.** On a state of the algebra, for neighbouring strands at positions
`i, i+1` (not both red) let `g_i = φ_i` if both are black of the same colour, and `g_i = ψ_i`
otherwise.

**Lemma 2.3 (exact braid relations) [proved].** On every state with three neighbouring
strands (at most one red), `g_1 g_2 g_1 = g_2 g_1 g_2`. Moreover, on a state `(c, c, c)`:
`ψ_1 φ_2 φ_1 = φ_2 φ_1 ψ_2`.

*Proof.* By the pattern of colours of the state `(X, Y, Z)`; the red strand counts as a
strand of a colour of its own, and `T` denotes a strand whose colour differs from `c`
(black of colour `c′ ≠ c`, or red).

1. *No two black strands of the same colour.* All `g` are `ψ` and (B) is exact, since both
   exceptional states have two black strands of the same colour.
2. *State `(c, c, T)`.* The claim is `φ_1ψ_2ψ_1 = ψ_2ψ_1φ_2`. The left side is
   `(x_1−x_2)ψ_1ψ_2ψ_1 − rψ_2ψ_1`. On the right, `φ_2 = (x_2−x_3)ψ_2 − r` is read on the
   state `(T, c, c)` reached by `ψ_2ψ_1`; its dots sit on the two `c`-strands, which started
   at positions 1 and 2, and slide back through the crossings with `T` (D):
   `ψ_2ψ_1(x_2−x_3) = (x_1−x_2)ψ_2ψ_1`. So the right side is
   `(x_1−x_2)ψ_2ψ_1ψ_2 − rψ_2ψ_1`, and the difference is `(x_1−x_2)(ψ_1ψ_2ψ_1 − ψ_2ψ_1ψ_2) = 0`
   because (B) is exact on `(c,c,c′)` and on `(k,k,W)`.
3. *State `(T, c, c)`.* Claim `ψ_1ψ_2φ_1 = φ_2ψ_1ψ_2`. In the same way
   `ψ_1ψ_2(x_1−x_2) = (x_2−x_3)ψ_1ψ_2`, both sides are `(x_2−x_3)·(braid word) − rψ_1ψ_2`,
   and (B) is exact on `(c′,c,c)` and `(W,k,k)`.
4. *State `(c, T, c)`.* Claim `ψ_1φ_2ψ_1 = ψ_2φ_1ψ_2`. Sliding the dots of `φ` to the
   left through the crossings with `T`:

       ψ_1φ_2ψ_1 = (x_1−x_3)ψ_1ψ_2ψ_1 − rψ_1²,      ψ_2φ_1ψ_2 = (x_1−x_3)ψ_2ψ_1ψ_2 − rψ_2² ,

   so the difference is `(x_1−x_3)(ψ_1ψ_2ψ_1 − ψ_2ψ_1ψ_2) − r(ψ_1² − ψ_2²)`.
   * `T` black, `c′ ∼ c`: (B) gives `r t_{cc′}`; `ψ_1² = t_{cc′}x_1 + t_{c′c}x_2`,
     `ψ_2² = t_{c′c}x_2 + t_{cc′}x_3`; the difference is `r t_{cc′}(x_1−x_3) − r t_{cc′}(x_1−x_3) = 0`.
   * `T` red of colour `k = c`: (B) gives `r_k u_k`; `ψ_1² = u_k x_1`, `ψ_2² = u_k x_3`; again `0`.
   * otherwise (B) is exact and `ψ_1² = ψ_2² = 1`.
5. *State `(c, c, c)`.* Put `a_1 = x_1−x_2`, `a_2 = x_2−x_3`, `b = x_1−x_3`. By Lemma 1.1,
   `ψ_2 a_1 = bψ_2 − r`, hence

       φ_2φ_1 = (a_2ψ_2 − r)(a_1ψ_1 − r) = a_2 b ψ_2ψ_1 − r b ψ_1 − r a_2 ψ_2 + r² .

   Multiply by `ψ_2` on the right (`ψ_2² = 0`): `φ_2φ_1ψ_2 = a_2 b ψ_2ψ_1ψ_2 − r b ψ_1ψ_2 + r²ψ_2`.
   Multiply by `ψ_1` on the left, using Lemma 1.1: `ψ_1 (a_2 b) = (a_2 b)ψ_1` (symmetric in
   `x_1, x_2`), `ψ_1 b = a_2ψ_1 + r`, `ψ_1 a_2 = bψ_1 − r`, and `ψ_1² = 0`:

       ψ_1φ_2φ_1 = a_2 b ψ_1ψ_2ψ_1 − r·rψ_1 − r(bψ_1ψ_2 − rψ_2) + r²ψ_1 = a_2 b ψ_1ψ_2ψ_1 − r b ψ_1ψ_2 + r²ψ_2 .

   (B) is exact on `(c,c,c)`, so `ψ_1φ_2φ_1 = φ_2φ_1ψ_2`. Finally, by Lemma 2.1(b),
   `(x_1−x_2)φ_2φ_1 = φ_2φ_1(x_2−x_3)`, so
   `φ_1φ_2φ_1 = (x_1−x_2)ψ_1φ_2φ_1 − rφ_2φ_1 = φ_2φ_1((x_2−x_3)ψ_2 − r) = φ_2φ_1φ_2`. ∎

*Check.* `06_check_basic.py` (all states of the words `121`, `1212`, `1211`, `111` in `A_2`,
red 1 or 2: 360 identities of the form of Proposition 4.2 below, which contains every case of
Lemma 2.3); `06_check_labelled.py` part (a).

---

## 3. The labelled calculus

**Definition 3.1.** A *labelled state* is an ordering `s` of the set `{1,…,m} ∪ {W}` (slots
and the red strand); its colour sequence is a state of `A`, with idempotent `e(s)`. For
`1 ≤ i ≤ m`, `s·i` is `s` with the entries at positions `i, i+1` exchanged, and
`g_i(s) = g_i e(s) ∈ e(s)Ae(s·i)` (Definition 2.2). `x_a(s)` is the dot on the strand
labelled `a`. A pair `{a, b}` of labels is *inverted* between `s` and `s′` if `a, b` stand
in opposite orders in `s` and `s′`; `inv(s, s′)` is the set of these pairs. A word
`i_1 … i_l` from `s` to `s′ = s·i_1⋯i_l` is *reduced* if `l = |inv(s, s′)|`.

**Proposition 3.2 [proved].** The product `g_{i_1}(s) g_{i_2}(s·i_1) ⋯` along a reduced word
from `s` to `s′` does not depend on the reduced word. It is denoted `𝒟(s → s′)`.

*Proof.* By (F3) two reduced words are connected by commutation moves and braid moves. A
commutation move is an instance of (R1) (`g_i` is a combination of `ψ_i`, dots at `i, i+1`
and the idempotent). A braid move is Lemma 2.3. ∎

**Proposition 3.3 (dots slide) [proved].** `x_a(s) 𝒟(s → s′) = 𝒟(s → s′) x_a(s′)` for
every label `a`.

*Proof.* For one generator: (R1) if the strand `a` is not involved, (D) if `g = ψ`, Lemma
2.1(b) if `g = φ`. ∎

So a polynomial in the dots of the *labelled* strands can be written at either end of
`𝒟(s → s′)`; we write polynomials in variables `x_1, …, x_m` (`x_a` = dot on the strand
labelled `a`) at the source end.

**Definition 3.4.** For two labels `a ≠ b` (one of them possibly `W`) let `q_{ab}` be the
polynomial of the double crossing of the two strands:

| strands | `q_{ab}` |
|---|---|
| black, `c(a) ∼ c(b)` | `Q_{c(a)c(b)}(x_a, x_b) := t_{c(a)c(b)} x_a + t_{c(b)c(a)} x_b` |
| black, `c(a) = c(b)` | `r_{c(a)}²` |
| black, other | `1` |
| `b = W`, `c(a) = k` | `u_k x_a` |
| `b = W`, `c(a) ≠ k` | `1` |

By (D) and Lemma 2.1(c), `g_i(s) g_i(s·i) = q_{ab} e(s)` if `a, b` are the labels at positions
`i, i+1` of `s`.

**Proposition 3.5 (product rule) [proved].**

    𝒟(s → s′) · 𝒟(s′ → s″) = ∏_{ {a,b} ∈ inv(s,s′) ∩ inv(s′,s″) } q_{ab} · 𝒟(s → s″).

*Proof.* Induction on `l = |inv(s′, s″)|`; `l = 0` is trivial. Let `i_1 … i_l` be a reduced
word from `s′` to `s″` and `{a, b}` the labels at positions `i_1, i_1+1` of `s′`. Exchanging
two neighbours changes the inversion status of this pair only. If `{a,b} ∉ inv(s, s′)`, a
reduced word from `s` to `s′` followed by `i_1` is reduced, so `𝒟(s→s′)g_{i_1}(s′) = 𝒟(s→s′·i_1)`.
If `{a,b} ∈ inv(s,s′)`, then `|inv(s, s′·i_1)| = |inv(s,s′)| − 1`, a reduced word from `s` to
`s′·i_1` followed by `i_1` is a reduced word from `s` to `s′`, and
`𝒟(s→s′)g_{i_1}(s′) = 𝒟(s→s′·i_1) g_{i_1}(s′·i_1) g_{i_1}(s′) = q_{ab}·𝒟(s→s′·i_1)` (Proposition 3.3
to move `q_{ab}` to the source). Now apply the induction hypothesis to
`𝒟(s → s′·i_1)·𝒟(s′·i_1 → s″)`:
`inv(s′·i_1, s″) = inv(s′,s″) ∖ {{a,b}}` and `inv(s, s′·i_1) = inv(s,s′) Δ {{a,b}}`, so the
pairs inverted in both are those of `inv(s,s′) ∩ inv(s′,s″)` other than `{a,b}`. ∎

*Check.* `06_check_labelled.py`, parts (a)–(c): random labelled states of `21321` (`A_3`),
`232434` (`A_4`), `12121` (`A_2`), `213421` (`D_4`): independence of the reduced word, the
product rule and the sliding of dots, 130 random triples of states, no failure.

*Remarks.* (1) Propositions 3.2–3.5 say that `s ↦ e(s)`, `𝒟` is a functor to `A` from the
category `Ã` with objects the labelled states and `Hom(s, s′)` free of rank one over the
polynomial ring, with composition twisted by the `q_{ab}`: the KLR-type algebra in which
all strands have pairwise different colours (strands of equal colour in `A` become "twins":
different, non-adjacent colours joined to the same neighbours, with `ψ² = r²`). There are no
exceptional triples in `Ã`. (2) This is the well-known statement that intertwiners satisfy
the braid relations (Kang–Kashiwara–Kim, *Symmetric quiver Hecke algebras and R-matrices*,
Lemma 1.3.1 for the KLR algebra); the content here is that it holds for this deformed
presentation with the signs of §1.1 and with the red strand (case 4 of Lemma 2.3 with `T`
red), which is why it was re-proved.

---

## 4. The intertwiner cube

Recall `std(I)` = (slots not in `I`, decreasing; `W`; slots of `I`, increasing), and
`pos_I(s) = ±s`, `pos(W) = 0`.

**Definition 4.1.** For `j ∈ I`, `Φ_j(I) = 𝒟(std(I) → std(I∖j)) ∈ e_I A e_{I∖j}`: the
elementary move `X_j(I) = m_j(I)` of `PROOFS.md` Lemma 1 with every crossing of two strands
of the same colour replaced by `φ`. `C_Φ` is the cube with
`d_Φ e_I = Σ_{j∈I} (−1)^{#{i∈I: i>j}} Φ_j(I)`. The inverted pairs of `Φ_j(I)` are `{j, W}` and
`{j, y}`, `y < j` (`PROOFS.md` Lemma 1).

*Degrees.* `deg φ = deg ψ + deg(dot)`, so `deg Φ_j(I) = d(j) + N_j·(2q + δ_{c(j)})` with
`N_j = #{i < j : c(i) = c(j)}`; `C_Φ` has the shifts `D_Φ(I) = D(I) + Σ_{x∈I} N_x(2q + δ_{c(x)})`,
and a component `I → J` of a degree-preserving map `C_Φ → C` has degree `D_Φ(I) − D(J)`.

**Proposition 4.2 (naturality) [proved].** Let `s` be an ordering of `{1,…,m−1} ∪ {W}` and
`Φ(s) = 𝒟((s, m) → (m, s))` (the strand `m` passes from the far right to the far left).
For every `a ∈ e(s) A′ e(s′)`, `A′` the algebra of the word `w′ = (i_1,…,i_{m−1})`,

    (a ⊗ 1) · Φ(s′) = Φ(s) · (1 ⊗ a),

where `a ⊗ 1` (`1 ⊗ a`) is `a` with an additional strand of colour `c(m)` on the far right
(far left).

*Proof.* `a ↦ a ⊗ 1` and `a ↦ 1 ⊗ a` are algebra homomorphisms (the relations are local), so
it suffices to take generators. Dots: Proposition 3.3 and (R1). A crossing `ψ_i` of the
strands at positions `i, i+1` of `s`: `Φ(s′) = g_n g_{n−1} ⋯ g_1` (`n = m`), `ψ_i` commutes
with `g_n, …, g_{i+2}` (R1), and `ψ_i g_{i+1} g_i = g_{i+1} g_i ψ_{i+1}`: if the two strands
have different colours or one is red this is Lemma 2.3 (`ψ_i = g_i`); if they have the same
colour `c ≠ c(m)` it is exactness of (B) on `(c,c,T)`; if all three have the same colour it
is the second statement of Lemma 2.3. Then `ψ_{i+1}` commutes with `g_{i−1}, …, g_1`. ∎

*Check.* `06_check_basic.py`: all orderings `s`, all generators, for `121`, `1212`, `1211`,
`111` with red 1 or 2 (`A_2`).

**Proposition 4.3 [proved].** `d_Φ² = 0` for every word and every red colour.

*Proof.* Let `i < j` in `I`. The pairs inverted by `Φ_i(I)` contain `i`, those inverted by
`Φ_j(I∖i)` contain `j`, and `{i, j}` is not inverted by `Φ_i(I)` (`j > i`). So by Proposition
3.5, `Φ_i(I)Φ_j(I∖i) = 𝒟(std I → std(I∖{i,j}))`. Likewise `{i,j}` is inverted by `Φ_j(I)` but
not by `Φ_i(I∖j)` (then `j` is on the left, further out than `i` goes), so
`Φ_j(I)Φ_i(I∖j) = 𝒟(std I → std(I∖{i,j}))` too. The two paths carry opposite signs
(`PROOFS.md`, proof of Theorem 5). ∎

*Check.* `06_check_basic.py` (`d_Phi^2 = 0: True` in all cases).

**Proposition 4.4 [proved].** Let `ζ_m = ∏_{a<m, c(a)=c(m)} (x_a − x_m)` (dots on the labelled
strands). Then `ζ_m` commutes with `a ⊗ 1` and `1 ⊗ a` for all `a ∈ A′`, and left or right
multiplication by `ζ_m` is injective on every `e A e′`.

*Proof.* As a polynomial in the dots, `ζ_m` is symmetric in the dots of the strands of
colour `c(m)` other than `m`. It commutes with dots, with crossings of different colours
(D), and with a crossing of two strands of the same colour among those of `w′` by Lemma 1.1.
Injectivity is (F1). ∎ (`06_check_basic.py`, `z central: True`.)

---

## 5. Comparison maps: the reduction

Order the subsets of `{1,…,m}` by `J ≺ I` iff `|J| = |I|` and `Σ_{x∈J} x < Σ_{x∈I} x`.

**Proposition 5.1 [proved].** Let `d = (d_{IJ})`, `d_{IJ} ∈ e_I A e_J` (`|J| = |I| − 1`), be
any family, and `G = (G_{IJ})`, `G_{IJ} ∈ e_I A e_J` (`|J| = |I|`), such that

* `G_{II} = ζ_I e_I` with `ζ_I` a non-zero polynomial in the dots, and `G_{IJ} = 0` unless `J = I` or `J ≺ I`;
* `Σ_J (d_Φ)_{IJ} G_{JK} = Σ_J G_{IJ} d_{JK}` for all `I, K`.

Then `Σ_J d_{IJ} d_{JK} = 0` for all `I, K`.

*Proof.* `G d² = d_Φ G d = d_Φ² G = 0` by Proposition 4.3, i.e.
`ζ_I (d²)_{IK} = − Σ_{J ≺ I} G_{IJ} (d²)_{JK}`. By induction on `I` along `≺` the right side
vanishes, and left multiplication by `ζ_I` is injective (F1). ∎

**Proposition 5.2 (one step) [proved].** Let `w = w′·c`, `(C′, d′)` a complex on the cube of
`w′` (`d′² = 0`), and `C_R = C′ ⊗ 1`, `C_L = 1 ⊗ C′`. Let `F, h` be families of elements
`F_{I∪m, K}`, `h_{I∪m, J} ∈ e_{I∪m} A e_•` (`K, J ⊆ {1..m−1}`, `|K| = |I|`, `|J| = |I|+1`) with

    Φ = ζ_m F + h (1 ⊗ d′) + (d′ ⊗ 1) h                                     (★)

(matrix products; `ζ_m` at the source end). Then `(d′ ⊗ 1)F = F(1 ⊗ d′)`, i.e. the cube of `w`
with differential `(−d′ ⊗ 1, F; 1 ⊗ d′)` is a complex, and `Γ = (ζ_m on C_R, 1 on C_L, h)` is
a chain map from the cone of `Φ` to it.

*Proof.* `(d′⊗1)Φ = Φ(1⊗d′)` by Proposition 4.2. Hence, using (★), `d′² = 0` and
Proposition 4.4,
`ζ_m[(d′⊗1)F − F(1⊗d′)] = (d′⊗1)(Φ − h(1⊗d′) − (d′⊗1)h) − (Φ − h(1⊗d′) − (d′⊗1)h)(1⊗d′) = 0`,
and `ζ_m` is injective. The chain map condition for `Γ` on `C_R → C_L` is (★); on
`C_R → C_R` it is Proposition 4.4. ∎

**Remark 5.3 (factorisation) [proved].** Let `C^{(k)}` be the cube with the tree formula of
`w_{≤k}` on the slots `≤ k` and the moves `Φ_j` for `j > k` (iterated cones of
`Φ_{k+1}, …, Φ_m`; a complex as soon as the tree formula of `w_{≤k}` is a differential, by
Proposition 4.2). `C^{(0)} = C_Φ`, `C^{(m)} = C`. A chain map `Γ : C^{(k−1)}_{≤k} → C^{(k)}_{≤k}`
for the word `w_{≤k}` extends to `C^{(k−1)} → C^{(k)}` by `Γ ⊗ 1` and `1 ⊗ Γ` on the later
cones, again by naturality. So `G = Γ^{(1)} Γ^{(2)} ⋯ Γ^{(m)}` (first `Γ^{(1)}`) and
the diagonal of `G` is `∏_{x∈I} ζ_x`. (`06_lib.d_mixed` implements `C^{(k)}`.)

---
## 6. Words in which every colour occurs at most twice

Throughout this section every colour occurs at most twice in `w`.

### 6.1 Notation

* `𝔓` is the set of *pairs* `p = (a_p < b_p)` of slots of the same colour `c_p`; there is at
  most one pair per colour. `a_p` is the *twin* of `b_p`. Put `r_p = r_{c_p}` and
  `z_p = x_{a_p} − x_{b_p}`. For colours `c ∼ c′`, `Q_{cc′}(x, y) = t_{cc′}x + t_{c′c}y` (first
  variable: a dot on a strand of colour `c`), and `Q_{cc′} = 1` if `c ≠ c′` are not adjacent.
* For a vertex `I`: `𝔓(I) = {p : a_p ∉ I, b_p ∈ I}`. For `F ⊆ 𝔓(I)`:
  `I_F = (I ∖ {b_p}_{p∈F}) ∪ {a_p}_{p∈F}` and `σ_F = ∏_{p∈F}(a_p b_p)`.
* *Links.* Two pairs `p, q` are interlaced if `a_p < a_q < b_p < b_q` (then we write `p < q`)
  or the same with `p, q` exchanged; *linked* if interlaced with `c_p ∼ c_q`. A pair `q` is
  linked to a slot `ρ ∉ {a_q, b_q}` if `a_q < ρ < b_q` and `c_q ∼ c(ρ)`.
* *Brackets.* For vertices `I, J` and a colour-preserving bijection `τ` of the slots,
  `[τ]_{I→J} = 𝒟(std I → τ^{-1}·std J)`, where `τ^{-1}·std J` is `std J` with every slot `t`
  replaced by the label `τ^{-1}(t)`: the strand that starts in slot `s` of `I` ends in slot
  `τ(s)` of `J`. Since `g_i(s)` depends only on the colours, `𝒟(π·s → π·s′) = 𝒟(s → s′)`
  for every colour-preserving relabelling `π`; so Proposition 3.5 reads

      [σ]_{I→J} · [τ]_{J→K} = ∏ q_{ab} · [τσ]_{I→K},                                   (6.1)

  the product over the pairs of labels (= slots of `I`, or `W`) inverted in both factors,
  where the strand labelled `s` runs `pos_I(s) → pos_J(σ s) → pos_K(τσ s)`, and two
  strands are inverted in a factor iff the differences of their positions before and after
  have opposite signs. **In all formulas below `x_s` is the dot on the strand that starts
  in slot `s` of the source vertex `I` of the whole product** (Proposition 3.3 lets us write
  all coefficients at the source).
* *Signs.* Let `Λ` be the exterior algebra over `ℤ` on `ξ_1, …, ξ_m`, `ξ_I` the increasing
  product. Let `ι_j` be the right interior product
  (`ι_j(ξ_{i_1}⋯ξ_{i_k}) = Σ_{l: i_l = j} (−1)^{k−l} ξ_{i_1}⋯\hat{ξ_{i_l}}⋯ξ_{i_k}`) and `θ_p` the
  derivation of `Λ` with `θ_p ξ_{b_p} = ξ_{a_p}`, `θ_p ξ_s = 0` for `s ≠ b_p` ("replace `b_p`
  by `a_p` in place"). Write `θ_F = ∏_{p∈F} θ_p` and `⟨ξ_J, O ξ_I⟩` for the coefficient of
  `ξ_J` in `O ξ_I`. By the definitions in `NOTES.md` §2 / `formula.py`:

      (−1)^{#{i∈I: i>j}} = ⟨ξ_{I∖j}, ι_j ξ_I⟩,      κ(I; j, F) = ⟨ξ_J, ι_j θ_F ξ_I⟩,

  and we put `ε(I, F) = ⟨ξ_{I_F}, θ_F ξ_I⟩` (the sign of the permutation sorting `I` with
  every `b_p` replaced by `a_p` in place). If a family of cube components is written as
  `a(I) ⊗ O` (meaning the entry `I → J` is `⟨ξ_J, Oξ_I⟩·a(I)`), the matrix product of
  `a_1 ⊗ O_1` followed by `a_2 ⊗ O_2` is `a_1a_2 ⊗ O_2O_1`.

**Lemma 6.1 (signs) [proved].** On monomials `ξ_I`:
(Λ1) `θ_p θ_q = θ_q θ_p`; (Λ2) `ι_j θ_p = θ_p ι_j` if `j ≠ a_p`;
(Λ3) if `a_p, b_p ∈ I`: `θ_p ι_{a_p} ξ_I = − ι_{b_p} ξ_I`;
(Λ4) if `a_p ∉ I`, `b_p ∈ I`: `ι_{a_p} θ_p ξ_I = ι_{b_p} ξ_I`.

*Proof.* (Λ1), (Λ2): replacing `ξ_{b_p}` by `ξ_{a_p}` in place commutes with another such
replacement at a different place and with deleting a factor `ξ_j`, `j ∉ {a_p, b_p}`, whose
sign `(−1)^{k−l}` only depends on its place; for `j = b_p` both sides of (Λ2) vanish on
`ξ_I` (after deleting or replacing the only `ξ_{b_p}`, none is left). (Λ3): `ι_a ξ_I` deletes `ξ_a` with
sign `(−1)^{#{i∈I: i>a}}`; `θ_p` then puts `ξ_a` at the place of `ξ_b`, and moving it back to
its sorted place passes the `#{i ∈ I: a<i<b}` factors between: total sign
`(−1)^{#{i∈I: i≥b}} = −(−1)^{#{i∈I: i>b}}`. (Λ4): `θ_pξ_I` has `ξ_a` at the place of `ξ_b`, and
`ι_a` deletes it with the sign with which `ι_b` deletes `ξ_b` from `ξ_I`. ∎

### 6.2 The map `G` and the theorem

**Definition 6.2.** For a vertex `I` and `F ⊆ 𝔓(I)` let

    γ_F(I) = ∏_{p∈F} (−r_p) · ∏_{p∈𝔓∖F : b_p∈I} z_p · ∏_{p<q in F, linked} Q_{c_p c_q}(x_{b_p}, x_{a_q}),

and `G = Σ_F γ_F(I) [σ_F]_{I→I_F} ⊗ θ_F`, i.e. `G_{I, I_F} = ε(I,F) γ_F(I) [σ_F]_{I→I_F}`.

In words: `[σ_F]` is the dot-free diagram in which, for each `p ∈ F`, the strand at slot
`a_p` on the left slides outwards to slot `b_p` and the strand at slot `b_p` on the right
slides inwards to slot `a_p` (the pure exchange `E` of the brief; no strand crosses the red
strand, no two strands of the same colour cross, so `[σ_F]` is a basis diagram with
crossings `ψ` only). `F = ∅` gives the diagonal `G_{II} = ζ_I e_I`, `ζ_I = ∏_{p∈𝔓: b_p∈I} z_p`.
`G_{I,J} ≠ 0` only for `J = I` or `J ≺ I`.

**Theorem 6.2 [proved].** If every colour occurs at most twice in `w`, then for the tree
differential `d`:  `d_Φ G = G d`.

**Corollary 6.3 [proved].** If every colour occurs at most twice in `w` (any word, any red
colour; in particular `sl_n` with `ω_1`, `ω_2`, the vector representation of `so_{2n}`, and
all minuscule representations of `D_4`), the tree formula satisfies `d² = 0`.

*Proof.* Theorem 6.2 and Proposition 5.1 (`ζ_I ≠ 0`). ∎

*Check of Theorem 6.2* (exact, all components): `06_check_G_formula.py` — 45 random words
of `A_4`, `D_4` with 3–8 strands (`… A 4 random 25 8 1`, `… D 4 random 20 8 2`), and
`06_check_minuscule.py` — the minuscule words `A_4 ω_2` (`213243`), `A_5 ω_2` (`21324354`),
`A_6 ω_2` (`2132435465`, 10 strands, 2500 components of `G`), `D_4 ω_1` (`123421`),
`D_5 ω_1` (`12345321`). No failure.

The proof occupies §6.3–6.6.

### 6.3 The tree terms in the labelled calculus

Let `(j, F)` be a term of the tree formula at a vertex `V` (mover `j ∈ V`, `F ⊆ 𝔓(V)`,
`j ∉ {a_q, b_q}` for `q ∈ F`), with target `K = (V∖j)_F`, diagram `β(V; j, F)` (a reduced
diagram in the generators `ψ` of the bijection `σ_F`), coefficient
`c(F) = ∏_{q∈F} r_q t_q ε_q`, `t_q := t_{c_q, c(x_q)}`, and sign `κ(V; j, F)`; here `x_q` is the connector of
`q` and `ε_q = −1` if `x_q` is the mover or the slot `b` of the parent pair, `ε_q = +1` if it
is the slot `a` of the parent pair.

**Lemma 6.3 (forest) [proved].** Let `ρ` be a slot and `𝔉 ⊆ 𝔓` a set of pairs none of which
contains `ρ`. The graph `Λ_ρ(𝔉)` with vertices `{ρ} ⊔ 𝔉` and the links as edges is
isomorphic to a subgraph of the Dynkin diagram, hence a forest. Consequently:
(i) for `F ⊆ 𝔉 ∩ 𝔓(V)`, `(ρ, F)` is admissible iff `{ρ} ∪ F` is connected in `Λ_ρ(𝔉)`;
(ii) if `q ∈ 𝔉` lies in the component of `ρ` and `c_q ∼ c(ρ)`, then `q` is linked to `ρ`.

*Proof.* Different pairs have different colours, and no pair of `𝔉` has the colour `c(ρ)`
(the only pair of that colour contains `ρ`); links join adjacent colours. So the colour map
is an embedding into the Dynkin diagram, which is a tree. (i) Conditions 1 and 2 of
admissibility concern a pair of the mover's colour not containing the mover, resp. two pairs
of equal colour, which do not exist; condition 3 says that the induced graph on `{ρ} ∪ F`
is a tree, i.e. (in a forest) connected. (ii) The path from `ρ` to `q` maps to a path
without repeated vertices in the Dynkin diagram between two adjacent vertices; in a tree
this is the edge. ∎

**Lemma 6.4 [proved].** In a reduced diagram of the term `(j, F)`:
(i) only the mover crosses `W`; the mover crosses an unaffected strand `y` iff `y < j`, and
both strands of `q ∈ F` if `b_q < j`, neither otherwise; a strand of `q` crosses an
unaffected strand `y` iff `a_q < y < b_q` and they are on the same side; the two strands of
a pair do not cross (`PROOFS.md`, proof of Proposition 3).
(ii) Two strands of the same colour cross iff they are the mover `j` and its twin `a < j`
(`j` a second occurrence); they cross once, and before the crossing `a` is to the left of `j`.
(iii) If `j` has a twin `a < j` and a strand `y` crosses both `a` and `j`, then `y` is black
and `c(y)` is different from and not adjacent to `c(j)`.

*Proof.* (ii) The strands of colour `c ≠ c(j)` are the two strands of the pair of colour
`c` (which do not cross, whether or not the pair is in `F`: unaffected strands do not
cross). The strands of colour `c(j)` are the mover and its twin `t`, which is unaffected
(its pair contains `j`, so is not in `F`); they cross iff `t < j`. If `a = t < j`: in
`std V`, `a` is to the left of `j` (`a ∈ V`: both on the right, increasing; `a ∉ V`: left
of `W`). (iii) `y ≠ W` since `a` does not cross `W`. `y` crosses the unaffected strand `a`,
so `y` is a strand of some `q ∈ F` with `a_q < a < b_q`, and it crosses the mover, so
`b_q < j`; in particular `q` is not linked to `j`. `c_q ≠ c(j)`. If `c_q ∼ c(j)`, Lemma
6.3(ii) (with `ρ = j`, `𝔉 = F`, connected by admissibility) would make `q` linked to `j`. ∎

**Lemma 6.5 (expansion of the tree terms) [proved].**
(i) If the mover `j` has no twin `a < j`: `β(V; j, F) = [σ_F]_{V→K}`.
(ii) If `j` has the twin `a < j`, with `p = (a, j)`:

    z_p · β(V; j, F) = [σ_F]_{V→K} + r_p · [σ_F ∘ (a j)]_{V→K}.

In particular `β(V; j, F)` does not depend on the reduced word.

*Proof.* (i) By Lemma 6.4(ii) no two strands of equal colour cross, so every `ψ` in a
reduced word of `β` is a `g`. (ii) By Lemma 6.4(ii), `β = D_1 ψ_i D_2` where `D_1`, `D_2` are
products of generators `g` and `ψ_i` crosses `a` (position `i`) and `j` (position `i+1`).
By the definition of `φ` and Proposition 3.3,

    z_p β = D_1 (x_a − x_j) ψ_i D_2 = D_1 φ_i D_2 + r_p D_1 D_2′ ,

where `D_2′` is the word of `D_2` read from the state before the crossing (the labels `a`,
`j` exchanged). `D_1 φ_i D_2 = [σ_F]_{V→K}` (a reduced word in the `g`). `D_1` and `D_2′` are
of the form `𝒟`, and the labelled strand `j` (resp. `a`) ends in `D_1 D_2′` where `a` (resp.
`j`) ends in `β`: slot `a` (resp. slot `j` on the left). By Proposition 3.5,
`D_1 D_2′ = ∏ q · [σ_F∘(a j)]_{V→K}` over the pairs inverted in `D_1` and in `D_2′`. As
`D_1ψ_iD_2` is reduced, no pair is inverted in both `D_1` and `D_2`; `D_2′` inverts `{a, y}`
(resp. `{j, y}`) iff `D_2` inverts `{j, y}` (resp. `{a,y}`), and the other pairs as `D_2`. So
a doubly inverted pair is `{a, y}` or `{j, y}` with `y` crossing both `a` and `j` in `β`,
and `q = 1` by Lemma 6.4(iii). Independence of the word: `[σ_F]`, `[σ_F(a j)]` do not depend
on it, and left multiplication by `z_p` is injective (F1). ∎

*Check.* `06_check_labelled.py` part (d): all 549 tree terms of `21321`, `232434`,
`213421`.

### 6.4 Crossing bookkeeping

**Lemma 6.6 [proved].** Let `I` be a vertex. In each of the following products of two
brackets the pairs of labels inverted in both factors are exactly the listed ones (`W` never
occurs). `F′, F″, 𝔉` denote disjoint subsets of `𝔓(I)`; for `p′ ∈ F′`, `q ∈ F″` interlaced,
the pair `{a_{p′}, a_q}` if `p′ < q` and `{b_{p′}, b_q}` if `q < p′` is called their
*interlacing pair*.

| | product | conditions | doubly inverted pairs |
|---|---|---|---|
| (a) | `[σ_{F′}]_{I→I′} [σ_{F″}]_{I′→K}`, `I′ = I_{F′}`, `K = (I′∖j)_{F″}` | `j ∈ I` in no pair of `F′ ∪ F″` | interlacing pairs of `F′ × F″`; `{b_{p′}, j}` for `p′ ∈ F′`, `a_{p′} < j < b_{p′}` |
| (b) | `Φ_j(I) [σ_𝔉]_{I∖j→K}` | `j ∈ I` in no pair of `𝔉` | `{j, a_q}` for `q ∈ 𝔉`, `a_q < j < b_q` |
| (c) | `Φ_a(I) [σ_{𝔉⊔p}]_{I∖a→K}` | `p = (a,b)`, `a, b ∈ I` | `{a, a_q}` for `q ∈ 𝔉`, `a_q < a`, `b < b_q` |
| (d) | `[σ_{F′}]_{I→I′} [σ_{F″}(a b)]_{I′→K}`, `K = (I′∖b)_{F″}` | `p = (a,b) ∉ F′∪F″`, `a, b ∈ I` | interlacing pairs; `{b_{p′}, b}` for `p′∈F′`, `a < a_{p′} < b < b_{p′}`; `{b_{p′}, a}` for `p′ ∈ F′`, `a_{p′} < a < b_{p′}` |
| (e) | `[σ_{F′⊔p}]_{I→I′} [σ_{F″}]_{I′→K}`, `I′ = I_{F′⊔p}`, `K = (I′∖a)_{F″}` | `p = (a,b) ∈ 𝔓(I)`, `p ∉ F′∪F″` | interlacing pairs of `F′ × F″`; `{a, a_q}` for `q ∈ F″`, `a < a_q < b < b_q`; `{b, b_{p′}}` for `p′ ∈ F′`, `a_{p′} < a`, `b < b_{p′}` |
| (f) | `[σ_{F′}]_{I→I′} [σ_{F″}(a b)]_{I′→K}`, `K = (I′∖b)_{F″}` | `p = (a,b) ∈ 𝔓(I)`, `p ∉ F′∪F″` | interlacing pairs; `{a_{p′}, a}` for `p′ ∈ F′`, `a_{p′} < a < b_{p′} < b`; `{b_{p′}, b}` for `p′∈F′`, `a_{p′} < b < b_{p′}` |

*Proof.* The positions (start, middle, end) of the strands, by label, are: for `p′ ∈ F′`:
`a_{p′}: (−a_{p′}, −b_{p′}, −b_{p′})`, `b_{p′}: (+b_{p′}, +a_{p′}, +a_{p′})`; for `q ∈ F″` (or `𝔉`
in (b), (c)): `a_q: (−a_q, −a_q, −b_q)`, `b_q: (+b_q, +b_q, +a_q)`; other strands, apart
from the special ones below, and `W` do not move. A doubly inverted pair must contain a
strand that moves in the first factor and a strand that moves in the second. Two strands
that are on different sides of `W` before and after a factor are not inverted by it
("sides").

(a) Special strand `j: (+j, +j, −j)`. Candidates: `X` a strand of `p′ ∈ F′`, `Y` a strand of
`q ∈ F″` or `Y = j`.
`(a_{p′}, a_q)`: first factor iff `a_{p′} < a_q < b_{p′}`, second iff `a_q < b_{p′} < b_q`; both iff `p′ < q` interlaced.
`(b_{p′}, b_q)`: first iff `a_{p′} < b_q < b_{p′}`, second iff `a_q < a_{p′} < b_q`; both iff `q < p′` interlaced.
`(a_{p′}, b_q)`, `(b_{p′}, a_q)`: sides, in both factors.
`(b_{p′}, j)`: first iff `a_{p′} < j < b_{p′}`, second iff `a_{p′} < j`.
`(a_{p′}, j)`: first: sides.

(b) Special strand `j: (+j, −j, −j)`; it is the only strand moving in the first factor.
`(j, a_q)`: first iff `a_q < j`, second iff `a_q < j < b_q`. `(j, b_q)`: second: sides.
`(j, y)`, `y` not moving in the second factor, and `(j, W)`: not inverted in the second.

(c) Special strands `a: (+a, −a, −b)`, `b: (+b, +b, +a)`; only `a` moves in the first factor,
where it is inverted with `W` and with the strands at slots `< a`.
`(a, W)`: second: no. `(a, b)`: first: no (`a` stays to the left of `b`).
`(a, y)`, `y` unaffected: second iff `y ∉ I`, `a < y < b`; first iff `y < a`: no.
`(a, a_q)`: first iff `a_q < a`; second iff `(a_q − a)(b_q − b) < 0`; both iff `a_q < a`, `b < b_q`.
`(a, b_q)`: second: sides.

(d) Special strands `a: (+a, +a, −b)`, `b: (+b, +b, +a)`. Candidates `(X, Y)` with `X` a strand
of `p′ ∈ F′`, `Y` a strand of `F″` (as in (a)) or `Y ∈ {a, b}`.
`(b_{p′}, b)`: first iff `a_{p′} < b < b_{p′}`, second iff `a < a_{p′} < b`.
`(b_{p′}, a)`: first iff `a_{p′} < a < b_{p′}`, second iff `a_{p′} < a`.
`(a_{p′}, a)`, `(a_{p′}, b)`: first: sides.

(e) Special strands `a: (−a, −b, −b)`, `b: (+b, +a, −a)`.
`(X, Y)`, `X` of `p′ ∈ F′`, `Y` of `q ∈ F″`: as in (a).
`(b_{p′}, b)`: first iff `(b_{p′} − b)(a_{p′} − a) < 0`, second iff `a_{p′} < a`; both iff `a_{p′} < a`, `b < b_{p′}`.
`(a_{p′}, b)`: first: sides. `(a_{p′}, a)`, `(b_{p′}, a)`: second: neither moves.
`(a, a_q)`: first iff `a < a_q < b`, second iff `a_q < b < b_q`. `(a, b_q)`: sides.
`(b, b_q)`: first iff `a < b_q < b`, second iff `b_q < a`: no. `(b, a_q)`: first: sides.
`(a, b)`: first: sides. `(b, y)`, `y` unaffected: first iff `y ∈ I`, `a < y < b`; second iff `y < a`: no. `(b, W)`: first: no.

(f) Special strands `a: (−a, −a, −b)`, `b: (+b, +b, −a)`. Candidates `(X, Y)`, `X` of `p′∈F′`,
`Y` of `F″` (as in (a)) or `Y ∈ {a,b}`.
`(a_{p′}, a)`: first iff `a_{p′} < a < b_{p′}`, second iff `a < b_{p′} < b`.
`(b_{p′}, b)`: first iff `a_{p′} < b < b_{p′}`, second iff `a_{p′} < b`.
`(b_{p′}, a)`, `(a_{p′}, b)`: first: sides. ∎

*Check.* `06_check_bigons.py 30000 1` (plain `python3`): the six statements, and the
equality of the total bijections of (b) and (a), of (c) and (d), of (e) and (f), on 18 136
random configurations with up to 11 slots.

### 6.5 The tree lemma

**Lemma 6.7 [proved].** Let `T` be a finite rooted tree with root `ρ` and parent function
`π`, and for every vertex `v ≠ ρ` let `A_v, B_v` be elements of a commutative ring,
`M_v = A_v + B_v`. Then

    Σ_S  ∏_{v∈S∖ρ} B_v · ∏_{v∉S, π(v)∈S} A_v · ∏_{v∉S, π(v)∉S} M_v  =  ∏_{v≠ρ} M_v ,

the sum over the connected subsets `S ∋ ρ`.

*Proof.* Induction on the number of vertices. Let `ℓ ≠ ρ` be a leaf and `T′ = T∖ℓ`. The
connected `S ∋ ρ` of `T` are the sets `S′` and, if `π(ℓ) ∈ S′`, `S′ ∪ {ℓ}`, for `S′` connected
in `T′`. For fixed `S′` the factor of `ℓ` sums to `A_ℓ + B_ℓ` if `π(ℓ) ∈ S′` and is `M_ℓ`
otherwise. ∎

**Lemma 6.8 (weights) [proved].** Let `q` be a pair and `π` either a pair linked to `q` or a
slot `ρ` to which `q` is linked. Put `B_q = −ε_q t_{c_q, c(π)} z_q`, where `ε_q = −1` if
`π = ρ` or `π` is a pair with `π < q`, and `ε_q = +1` if `π` is a pair with `q < π`
(this is the sign `ε_q` of §6.3 when `π` is the parent of `q`). Then:

| `π` | `A_q` | `M_q = A_q + B_q` |
|---|---|---|
| pair, `q < π` | `Q_{c_q c_π}(x_{a_q}, x_{a_π})` | `Q_{c_q c_π}(x_{b_q}, x_{a_π})` |
| pair, `π < q` | `Q_{c_q c_π}(x_{b_q}, x_{b_π})` | `Q_{c_q c_π}(x_{a_q}, x_{b_π})` |
| slot `ρ`, any variable `y` of a strand of colour `c(ρ)` | `Q_{c_q c(ρ)}(x_{b_q}, y)` | `Q_{c_q c(ρ)}(x_{a_q}, y)` |

*Proof.* `Q_{cc′}(x, y) − Q_{cc′}(x′, y) = t_{cc′}(x − x′)`. First row: `B_q = −t z_q = t(x_{b_q} − x_{a_q})`.
Second and third row: `B_q = +t z_q = t(x_{a_q} − x_{b_q})` (`t = t_{c_q,c(π)}`). ∎

So in the first two rows `M_q` is the factor of the link `{q, π}` in Definition 6.2 (the dot
of the right strand of the earlier pair and of the left strand of the later one).

**Corollary 6.9 [proved].** Let `ρ` be a slot, `𝔉 ⊆ 𝔓(I)` a set of pairs not containing `ρ`,
`T` the component of `ρ` in `Λ_ρ(𝔉)` rooted at `ρ`, and for every `q ∈ T∖ρ` with `π(q) = ρ`
let a variable `y_q` of a strand of colour `c(ρ)` be given. Then

    Σ_{F″}  ∏_{q∈F″} B_q · ∏_{links within F′} Q^{RL} · ∏_{q∈F′∩T, π(q)∈F″∪ρ} A_q
        =  ∏_{links within 𝔉} Q^{RL} · ∏_{q∈T, π(q)=ρ} Q_{c_q c(ρ)}(x_{a_q}, y_q),

the sum over the `F″ ⊆ 𝔉` with `{ρ} ∪ F″` connected, `F′ = 𝔉∖F″`; `Q^{RL}` of a link `p < q`
is `Q_{c_pc_q}(x_{b_p}, x_{a_q})`, and `A_q, B_q` are as in Lemma 6.8 (`y = y_q` in the third
row).

*Proof.* A link within `F′` is either an edge of `T` both of whose ends are outside
`S = {ρ}∪F″`, with factor `M_q` (`q` the child) by Lemma 6.8, or a link outside `T`; the
latter are the same for all `F″`. Apply Lemma 6.7. ∎

### 6.6 Proof of Theorem 6.2

Fix `I`. Write `Z(V; F) = ∏_{p∈𝔓∖F: b_p∈V} z_p`, so that
`γ_F(V) = ∏_F(−r_p)·Z(V;F)·∏_{links within F}Q^{RL}`.

**Left side.** `d_Φ G` at `I` is the sum over `j ∈ I` and `𝔉′ ⊆ 𝔓(I∖j)` of
`Φ_j(I)·γ_{𝔉′}(I∖j)[σ_{𝔉′}] ⊗ θ_{𝔉′}ι_j`. Either `j` is in no pair of `𝔉′` — then `𝔉′ ⊆ 𝔓(I)`;
or `j = a_p` for one `p ∈ 𝔉′` — then `b_p ∈ I`, and `𝔉′ = 𝔉 ⊔ {p}` with `𝔉 ⊆ 𝔓(I)`
(`j = b_p` is impossible, `b_p ∈ I∖j`).

**Right side.** `G d` at `I` is the sum over `F′ ⊆ 𝔓(I)` and the terms `(j, F″)` of the tree
formula at `I′ = I_{F′}` of `γ_{F′}(I)[σ_{F′}]·c(F″)β(I′; j, F″) ⊗ ι_jθ_{F″}θ_{F′}`. `F′ ∩ F″ = ∅`
(`p ∈ F′` has `a_p ∈ I′`). Expand `β` by Lemma 6.5 into its *main part* `[σ_{F″}]` (with the
factor `1/z_{(a,j)}` if `j` has a twin `a < j`) and, in that case, its *resolved part*
`(r/z)[σ_{F″}(a j)]`. For the main part, either `j` is in no pair of `F′`, or `j = a_p`
with `p ∈ F′` (`j = b_p`, `p ∈ F′`, is impossible: `b_p ∉ I′`). For the resolved part
`p = (a, j) ∉ F′ ∪ F″`, `j ∈ I`, and `a ∈ I′ ⟺ a ∈ I`. If `j` has a twin `a < j`, then
`p = (a,j) ∉ F′` and `j ∈ I`, so `z_p` divides `Z(I; F′)`; its dots are on strands not moved
by `[σ_{F′}]`, so the division is exact in the polynomial ring.

We sort all these terms into three kinds of classes and show that in each class the left
and the right terms agree.

**Class T0`(j, 𝔉)`**, `j ∈ I`, `𝔉 ⊆ 𝔓(I)`, `j` in no pair of `𝔉`: the left term `(j, 𝔉)`, and
the main parts of the right terms `(F′, j, F″)` with `F′ ⊔ F″ = 𝔉`. All are multiples of
`[σ_𝔉]_{I→K}`, `K = (I∖j)_𝔉` (Lemma 6.6: same bijection).
*Signs.* `θ_𝔉ι_j = ι_jθ_𝔉 = ι_jθ_{F″}θ_{F′}` by (Λ1), (Λ2).
*Coefficients.* By (6.1) and Lemma 6.6(b) the left coefficient is

    ∏_𝔉(−r_p) · Z(I∖j; 𝔉) · ∏_{links within 𝔉} Q^{RL} · ∏_{q∈𝔉: a_q<j<b_q} q_{j a_q} .

By (6.1) and Lemma 6.6(a) the right coefficients are

    ∏_{F′}(−r_p) · Z(I; F′)/[z_{(a,j)}] · ∏_{links within F′} Q^{RL} · ∏_{q∈F″} r_q t_q ε_q · ∏ (interlacing pairs) q · ∏_{p′∈F′: a_{p′}<j<b_{p′}} q_{b_{p′} j} ,

and `Z(I;F′) = Z(I∖j; 𝔉)·∏_{q∈F″} z_q·[z_{(a,j)}]` (the pairs with `b_p ∈ I`, `p ∉ F′` are those
of `F″`, the pair `(a, j)` if `j` has a twin `a<j`, and those with `b_p ∈ I∖j`, `p ∉ 𝔉`). So
after division by `∏_𝔉(−r_p)·Z(I∖j;𝔉)` the claim is

    ∏_{links within 𝔉} Q^{RL} · ∏_{q∈𝔉: a_q<j<b_q} q_{j a_q}
       = Σ_{F″} ∏_{q∈F″}(−ε_q t_q z_q) · ∏_{links within F′} Q^{RL} · ∏ (interlacing pairs) q · ∏_{p′∈F′: a_{p′}<j<b_{p′}} q_{b_{p′} j} .   (P0)

By Lemma 6.3(i) the sum is over the `F″` with `{j} ∪ F″` connected in `Λ_j(𝔉)`. The factors
`q` different from 1 belong to adjacent colours, i.e. to links. A linked interlacing pair
`(p′ ∈ F′, q ∈ F″)` is an edge of the component `T` of `j` with `π(p′) = q` (the parent of
`q` is in `F″ ∪ j`), and its factor is `A_{p′}` of Lemma 6.8 (rows 1, 2); `q_{b_{p′} j}` for
`p′` linked to `j` is `A_{p′}` of row 3 with `y = x_j`; `−ε_q t_q z_q = B_q`. So the right
side of (P0) is the left side of Corollary 6.9 with `y_q = x_j`, and the left side of (P0)
is its right side, since `q_{j a_q} = Q_{c_q c(j)}(x_{a_q}, x_j)` for `q` linked to `j` and `1`
otherwise.

**Class T1`(p, 𝔉)`**, `p = (a, b)` with `a, b ∈ I`, `𝔉 ⊆ 𝔓(I)`: the left term
`(j, 𝔉′) = (a, 𝔉 ⊔ p)`, and the resolved parts of the right terms `(F′, b, F″)`,
`F′ ⊔ F″ = 𝔉`. All are multiples of `[σ_𝔉(a b)]_{I→K}`, `K = (I∖b)_𝔉`.
*Signs.* Left: `θ_pθ_𝔉ι_a = θ_𝔉θ_pι_a`, and `θ_pι_aξ_I = −ι_bξ_I` (Λ3), so the left sign is
`−⟨ξ_K, θ_𝔉ι_bξ_I⟩ = −⟨ξ_K, ι_bθ_𝔉ξ_I⟩`, minus the right sign.
*Coefficients.* Left, by Lemma 6.6(c):
`(−r_p)∏_𝔉(−r_q)·Z(I∖a; 𝔉⊔p)·∏_{links within 𝔉⊔p}Q^{RL}·∏_{q∈𝔉: a_q<a, b<b_q} q_{a a_q}`.
Right, by Lemma 6.6(d): `∏_{F′}(−r_q)·Z(I;F′)(r_p/z_p)·∏_{links within F′}Q^{RL}·∏_{F″}r_qt_qε_q·∏q`,
with `Z(I;F′) = z_p∏_{F″}z_q·Z(I∖a;𝔉⊔p)`. With the signs, and after division by
`r_p∏_𝔉(−r_q)Z(I∖a;𝔉⊔p)`, the claim is

    ∏_{links within 𝔉⊔p} Q^{RL} · ∏_{q∈𝔉: a_q<a, b<b_q} q_{a a_q}
       = Σ_{F″} ∏_{F″}(−ε_qt_qz_q) ∏_{links within F′}Q^{RL} ∏(interlacing pairs) q ∏_{p′∈F′: a<a_{p′}<b<b_{p′}} q_{b_{p′} b} ∏_{p′∈F′: a_{p′}<a<b_{p′}} q_{b_{p′} a} .   (P1)

The sum is over the `F″` with `{b} ∪ F″` connected in `Λ_b(𝔉)`. Sort the `q ∈ 𝔉` with
`c_q ∼ c_p` by their position relative to `p`:
(α) `a < a_q < b < b_q`; (β) `a_q < a < b < b_q`; (γ) `a_q < a < b_q < b`; (δ) all others.
The pairs linked to the root `b` are those of types (α), (β). A pair of type (γ) or (δ) is
not linked to `b`, hence not in the component `T` of `b` (Lemma 6.3(ii)), hence in `F′` for
every `F″`. Right side: for `p′` of type (α) in `F′` the factor `q_{b_{p′} b}` is `A_{p′}` (row 3,
`y = x_b`); of type (β) in `F′`: `q_{b_{p′} a} = A_{p′}` (row 3, `y = x_a`); of type (γ):
the constant factor `q_{b_{p′} a} = Q_{c_{p′}c_p}(x_{b_{p′}}, x_a)`; type (δ) and non-adjacent
colours: `1`. By Corollary 6.9 the right side of (P1) is

    ∏_{links within 𝔉}Q^{RL} · ∏_{(α)} Q_{c_qc_p}(x_{a_q}, x_b) · ∏_{(β)} Q_{c_qc_p}(x_{a_q}, x_a) · ∏_{(γ)} Q_{c_qc_p}(x_{b_q}, x_a).

Left side: the links of `p` within `𝔉 ⊔ p` are with the `q` of type (α) (`p < q`, factor
`Q_{c_pc_q}(x_b, x_{a_q})`) and of type (γ) (`q < p`, factor `Q_{c_qc_p}(x_{b_q}, x_a)`), and
`q_{a a_q} = Q_{c_pc_q}(x_a, x_{a_q})` for type (β). The two sides agree.

**Class T2`(p, 𝔉)`**, `p = (a, b) ∈ 𝔓(I)`, `𝔉 ⊆ 𝔓(I)∖p`: (i) the main parts of the right terms
`(F′ ⊔ p, a, F″)` (the mover is the strand brought to slot `a` by the exchange of `p`; `a` is
a first occurrence, so there is no resolved part and no division), and (ii) the resolved
parts of the right terms `(F′, b, F″)`; in both `F′ ⊔ F″ = 𝔉`. No left term belongs here.
All are multiples of `[σ_𝔉(a b)]_{I→K}`, `K = (I∖b)_𝔉`.
*Signs.* (i): `ι_aθ_{F″}θ_{F′}θ_p = ι_aθ_pθ_𝔉`, and `θ_𝔉ξ_I = ±ξ_{I_𝔉}` with `a ∉ I_𝔉 ∋ b`, so by
(Λ4) the sign is `⟨ξ_K, ι_bθ_𝔉ξ_I⟩`, the sign of (ii).
*Coefficients.* (i), by Lemma 6.6(e):
`(−r_p)∏_{F′}(−r_q)·Z(I;F′⊔p)·∏_{links within F′⊔p}Q^{RL}·∏_{F″}r_qt_qε_q·∏q`, with
`Z(I;F′⊔p) = ∏_{F″}z_q·Z_0`, `Z_0 = Z(I; 𝔉⊔p)`. (ii), by Lemma 6.6(f):
`∏_{F′}(−r_q)·Z(I;F′)(r_p/z_p)∏_{links within F′}Q^{RL}∏_{F″}r_qt_qε_q∏q`, with
`Z(I;F′) = z_p∏_{F″}z_qZ_0`. After division by `r_p∏_𝔉(−r_q)Z_0` the claim "(i) + (ii) = 0" is

    Σ_{F″: {a}∪F″ connected} ∏_{F″}(−ε_qt_qz_q) ∏_{links within F′⊔p}Q^{RL} ∏(interl.) q ∏_{q∈F″: a<a_q<b<b_q} q_{a a_q} ∏_{p′∈F′: a_{p′}<a, b<b_{p′}} q_{b b_{p′}}
     = Σ_{F″: {b}∪F″ connected} ∏_{F″}(−ε_qt_qz_q) ∏_{links within F′}Q^{RL} ∏(interl.) q ∏_{p′∈F′: a_{p′}<a<b_{p′}<b} q_{a_{p′} a} ∏_{p′∈F′: a_{p′}<b<b_{p′}} q_{b_{p′} b} .   (P2)

With the types (α)–(δ) as before:
*Left side* (root `a`): the pairs linked to `a` are of types (γ), (β). A pair of type (α)
with adjacent colour is not in the component of `a`, so it is in `F′`, where its link with
`p` gives `Q_{c_pc_q}(x_b, x_{a_q})`; the factors `q_{a a_q}` (`q ∈ F″` of type (α)) are
therefore `1`. For `q` of type (γ) in `F′` the link `q < p` gives `Q_{c_qc_p}(x_{b_q}, x_a) = A_q`
(row 3, `y = x_a`); for `p′` of type (β) in `F′`, `q_{b b_{p′}} = Q_{c_{p′}c_p}(x_{b_{p′}}, x_b) = A_{p′}`
(row 3, `y = x_b`). By Corollary 6.9 the left side is

    ∏_{links within 𝔉}Q^{RL} · ∏_{(α)} Q_{c_qc_p}(x_{a_q}, x_b) · ∏_{(β)} Q_{c_qc_p}(x_{a_q}, x_b) · ∏_{(γ)} Q_{c_qc_p}(x_{a_q}, x_a).

*Right side* (root `b`): the pairs linked to `b` are of types (α), (β), with
`q_{b_{p′} b} = A_{p′}` (row 3, `y = x_b`); a pair of type (γ) with adjacent colour is in `F′`
and contributes `q_{a_{p′} a} = Q_{c_{p′}c_p}(x_{a_{p′}}, x_a)`. By Corollary 6.9 the right side
is the same product.

**Completeness.** Every left term is in exactly one class T0 or T1, every main part of a
right term in one class T0 or T2(i), every resolved part in one class T1 or T2(ii), by the
case distinctions made above. Hence `d_Φ G − G d = 0`. ∎

*Remark (where the hypothesis is used).* (1) Lemma 6.4(ii): a tree term contains at most
one crossing of strands of equal colour. (2) Lemma 6.3: the link graph is a forest, so that
the sums over admissible `F″` are the sums over connected sets of Lemma 6.7. With a colour
occurring three times, the graph of the *labelled* strands is that of the Dynkin diagram
with the colour doubled, which contains a 4-cycle as soon as the colour has two neighbours
(or one neighbour occurring twice among the pairs); this is exactly the minimal cyclic
configuration of `NOTES.md` §4.

---
## 7. Words with a colour occurring three or more times

Nothing in this section is proved beyond §§2–5 (which hold for every word).

### 7.1 Existence and uniqueness of `G` **[computed]**

`06_solve_G.py`, `06_batch_G.py` solve `d_Φ G = G d_tree` in the *complete* ansatz (all
basis elements — dots, parameters `r, t, u`, basis braids — of degree `D_Φ(I) − D(J)` in
every `e_I A e_J`, `|I| = |J|`; the homogeneous monomials are enumerated by
`06_lib.Setup.monomials_of_degree`, which does not need an ILP solver).

| words | result |
|---|---|
| `A_2` red 1: `111`, `1121`, `1211`, `12121`, `11211`; `A_3`: `123131`, `12211` (red 1), `21322`, `21211`, `21131`, `21311`, `211331`, `22112`, `22332` (red 2), `333`, `321131` (red 3), `12333` (red 1) | the space of chain maps `C_Φ → C` is **1-dimensional**: `G` exists and is unique up to a scalar; normalised by `G_∅ = 1` it has integer coefficients `±1` in the basis, and `G_{II} = ζ_I` |
| `A_3` red 3: `312131`, `33123`; `A_2` red 2: `121`, `1212` | `G` with `G_∅ = 1` exists, but the space of chain maps is large (406, 169, 6, 21) and, where this was computed (`121`, `1212`), consists of null-homotopic maps. In these words some elementary move has degree 0 and is invertible (a strand whose colour differs from, and is not adjacent to, the colours of the red strand and of all earlier strands), so the cube is contractible |
| also all words of §6 that were solved (`1212`, `2112`, `2132`, `21321`, `232434`, … red = first letter) | 1-dimensional; the solution is Definition 6.2 |

So in every non-degenerate case examined `G` is *forced*: no choice of normal form is
involved (for the words solved with `06_solve_G.py` there are no homotopies of the right
degree at all). The one-step maps `Γ^{(k)} : C^{(k−1)} → C^{(k)}` (Remark 5.3) are equally
unique (`06_solve_step.py`, words `1212`, `111`, `1211`, `12121`, `21321`, `213243`, `21342`).

### 7.2 The closed form of the one-step maps

Let `w = w′·c`, `m` the last slot, `Same = {a < m : c(a) = c}`, `ζ_m = ∏_{a∈Same}(x_a − x_m)`,
and `I ∋ m` a vertex. A *chain* is a sequence `𝔠 = (a_1 < a_2 < ⋯ < a_s)` in `Same ∖ I`,
`s ≥ 1`; its *arcs* are `(a_1, a_2), …, (a_{s−1}, a_s), (a_s, m)`. A *hanging forest* is a set
`F` of pairwise disjoint exchange pairs `p = (a_p < b_p)` (`a_p ∉ I`, `b_p ∈ I`, `b_p < m`),
disjoint from `𝔠`, such that

1. no pair of colour `c` is interlaced with an arc, and no two pairs of `F` of the same
   colour are interlaced;
2. the graph on `{root} ⊔ F` with one edge root–`p` for every arc interlaced with `p` with
   `c_p ∼ c`, and one edge `p`–`q` for every two linked pairs, is a tree.

Every `p ∈ F` then has a parent and a connector `x_p`: the slot of the parent pair, or the
end of the parent arc, lying in `(a_p, b_p)`; `ε_p = −1` if `x_p ∈ I` (it is `m` or the
slot `b` of a pair), `+1` otherwise. The bijection `σ_{𝔠,F}` sends `m ↦ a_1`,
`a_i ↦ a_{i+1}`, `a_s ↦ m` and exchanges the pairs of `F`; the target vertex is `J` = `I`
with `m` replaced by `a_1` and every `b_p` by `a_p`: the strand of `m` slides inwards on the
right to slot `a_1`, and the left strands of the chain move one step outwards each.

**Conjecture H.** For every word `w` without cyclic configuration,
`Γ = (ζ_m on C_R, 1 on C_L, h)` with

    h e_I = Σ_{𝔠, F}  ε(I; 𝔠, F) · (−r_c)^s · ∏_{a∈Same∖𝔠} (x_a − x_m) · ∏_{p∈F} r_p t_{c_p, c(x_p)} ε_p · β(I; σ_{𝔠,F})

satisfies (★) of Proposition 5.2 with `d′` the tree differential of `w′` and `F` the
components `C_R → C_L` of the tree formula of `w`. Here `ε(I; 𝔠, F)` is the sign of the
permutation sorting `I` with `m` replaced by `a_1` and `b_p` by `a_p` in place, the dots are
at the source, and `β` is the basis diagram of the bijection (the `klrw` basis braid, as in
`sample_verify.element`; it contains crossings `ψ` of strands of equal colour when a
moving strand passes an unaffected strand of its colour — e.g. the factor
`(x_{a′} − x_m)` of an `a′ ∈ I` between `a_1` and `m` multiplies a diagram in which the
strands of `m` and `a′` cross).

By Proposition 5.2 and induction on the length, **Conjecture H implies Conjecture C1′**
(and gives `G = Γ^{(1)}⋯Γ^{(m)}` in closed form).

For `s = 1` and `Same = {a}` this is the guess of the brief: `h = −κ′·r_c·(E(a, m) +` trees
hanging from the pair `(a, m))`. If every colour occurs at most twice, the composite
`Γ^{(1)}⋯Γ^{(m)}` of these maps equals `G` of Definition 6.2 [computed:
`06_check_factorisation.py`, 7 words up to 8 strands]; so for such words Conjecture H is consistent with Theorem 6.2. (I have not derived
Conjecture H from Theorem 6.2; Theorem 6.2 does not need it.)

**Evidence [computed]** (`06_check_step_formula.py`, `06_check_minuscule.py`: for every `k`
the map `Γ^{(k)}` of the formula is an exact chain map `C^{(k−1)} → C^{(k)}`, which for
`k = m` is (★)):

| sample | words | result |
|---|---|---|
| 13 chosen words, `A_3` (`111`, `1211`, `1121`, `12121`, `11211`, `121121`, `123131`, …) | 13 | all steps hold |
| `A_3`, random, ≤ 7 strands, multiplicity ≤ 3 (`… A 3 random 40 7 3 11`) | 40 | 39 hold; the cyclic word `1322312` (red 1) fails at the last step only |
| `A_2`, random, ≤ 7 strands, multiplicity ≤ 4 (`… A 2 random 30 7 4 3`) | 30 | 29 hold; the cyclic word `2211221` (red 2) fails at the last step only |
| `A_2`, random, ≤ 8 strands, multiplicity ≤ 4 (`… A 2 random 40 8 4 21`) | 40 | all hold (none cyclic) |
| `D_4`, random, ≤ 7 strands, multiplicity ≤ 3 (`… D 4 random 40 7 3 7`) | 40 | all hold (none cyclic) |
| minuscule words `A_4ω_2`, `A_5ω_2`, `A_6ω_2`, `D_4ω_1`, `D_5ω_1`, **`A_5ω_3`** (`321432543`, 9 strands, a colour three times), **`D_5ω_5`** (`5321432534`, 10 strands, a colour three times) | 7 | all steps hold |

In total 168 cycle-free words, no failure; and the formula fails for each of the three
cyclic words tested (`1322312`, `2211221`, and `1221122` red 1), exactly at the step at which
the cyclic configuration appears. The samples contain chains with `s = 1, 2, 3`, hanging
trees on chains, blocked configurations and multiplicity 4.

### 7.3 What a proof of Conjecture H needs

The method of §6 applies in principle: all objects live in the labelled calculus after
multiplying by the `z`'s of the same-colour crossings (`(x_a − x_b)ψ = φ + r`). Two things
are new. (1) A tree term or a chain term now contains several crossings of equal colour
(a moving strand passes every unaffected strand of its colour, nested pairs of equal colour
cross twice), and each has to be expanded. (2) The graph of the labelled strands is no
longer a forest (remark at the end of §6); Lemma 6.7 has to be replaced by a statement for
graphs with cycles in which the pairs on a cycle are *blocked* or *cyclic* — this is where
the cyclic terms must come from. I did not carry this out. A natural intermediate target
is `N = 2` with a single neighbouring colour.

---
