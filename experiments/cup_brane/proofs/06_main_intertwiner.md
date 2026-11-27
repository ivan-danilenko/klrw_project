# 06 — The main theorem via intertwiners ("renormalised R-matrix")

*revised after `R06_referee.md`*

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
   satisfy the braid relations *exactly* (Lemma 2.3), so products of them depend only on the
   permutation of the *labelled* strands, and double crossings multiply by explicit
   polynomials (the *labelled calculus*, §3).
2. **[proved]** If a map `G : C_Φ → C` that is triangular for *some* strict partial order on
   the vertices, with non-zero polynomials on the diagonal, satisfies `d_Φ G = G d` for some
   family `d` of cube components, then `d² = 0` (Proposition 5.1); one-step (cone) version:
   Proposition 5.2. The `G` of Definition 6.2 is triangular for the order `≺` by the sum of
   the slots and for the colex order `<_c`; a factorised `G = Γ^{(1)}⋯Γ^{(m)}` of one-step
   maps of cone form is in general triangular only for `<_c` (Lemma 5.0, Remark 5.3).
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
5. **[computed, on a few words only]** Cyclic case (§8). Everything in §8 is an exact
   computation on five cyclic 7-strand cases (and one 8-strand word); nothing there is
   claimed in general. For `1221122` (`A_2`, red 1) the differential `d_true` found by the
   order-by-order solver — *one normal form* in a gauge class, fixed by the solver's choices
   (Remark 8.0) — is the tree formula plus one component (dotted terms, a coefficient 2,
   hand-over terms). A comparison map `Γ : Cone(Φ_7) → (C, d_true)` exists and is unique up
   to homotopy (chain maps modulo null-homotopic ones: dimension 1). The formula of
   Conjecture H for `h` is not enough (the division of `Φ − [d′, h]` by `ζ_7` fails in 21
   components), but a small extension `H⁺` (Definition 8.1: a hanging pair may start at a
   chain slot) is, in this example: `F = ζ_7^{-1}(Φ − (d′⊗1)h⁺ − h⁺(1⊗d′))` is polynomial and
   equals the solver's differential in all components. For red 2 the quotient is exactly the
   *dotted* completion `d_X` of `04_first_order.md` Theorem 9.1(d), not the gauge-equivalent
   *dot-free* completion `d_F` of Theorem 9.1(e); replacing `h⁺` by `h⁺ − ζ_7 h_g` (`h_g` the
   dot-free gauge of `R04_referee.md`) gives `d_F` exactly, and the same `h_g` removes the
   dots for red 1 as well (`06r_gauge_1221122.py`, Remark 8.0). So the dotted terms are
   produced by the choice of `h⁺`; they are not an invariant of the intertwiner description.
   The division is also exact for `2132213` (`A_3`, red 2; there the comparison map is unique
   outright), `2211221` (red 2) and `1322312` (`A_3`, red 1). On these five cases, *iterated
   division* (§8.5) starting from `d_Φ` produces a differential without using the tree
   formula; that this happens whenever the prefix is cycle-free is **Conjecture H⁺**, not a
   theorem. **It fails one step later**: for the 8-strand word `12211221`, whose prefix is
   cyclic, the division at the last step is not exact with the same formula for `h`
   (18 components). What is missing: a proof; the correction of `h` after a cyclic step; a
   term-by-term description of the quotient.

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

Standing facts used:

* (F1) the basis theorem, **Theorem A.5 of `01_foundations.md`** (proved there over
  `ℤ[r, t, u]` and every base change): `e A e′` is a free left module over the polynomial ring
  in the dots of the source and a free right module over the polynomial ring in the dots of
  the target, with basis `{ψ_w}` for any choice of reduced words; and its consequence,
  torsion-freeness, **Theorem E.1 of `01_foundations.md`**: left multiplication by a non-zero
  polynomial in the dots at the source, and right multiplication by one at the target, are
  injective on `e A e′`. Used in:
  * Proposition 4.4 (injectivity of multiplication by `ζ_m`: Theorem E.1);
  * Proposition 5.1 (cancellation of `ζ_I` on the left: Theorem E.1(a));
  * Proposition 5.2 (cancellation of `ζ_m` on the left, via Proposition 4.4);
  * Lemma 6.5 (independence of the reduced word: cancellation of `z_p` on the left,
    Theorem E.1(a)); hence in Theorem 6.2 and Corollary 6.3 (which also use Proposition 5.1);
  * §9: Proposition 9.2 (rank one; only the injectivity of `Pol → e(t) A e(t)` is needed,
    which also follows from the polynomial representation, §A.3 of `01_foundations.md`),
    Proposition 9.3 (`e A e_I` is a free `Pol`-module of rank the number of colour-preserving
    bijections, Theorem A.5(ii)), hence Corollary 9.4; the remark after it (`G` injective,
    Theorem E.1).

  Lemma 2.3 and §3 do not use F1.
* (F2) the grading: only in the degree remarks of §4 and in the computations of §§5, 7, 8.
* (F3) Matsumoto's theorem (Proposition 3.2).

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
Injectivity is (F1): Theorem E.1 of `01_foundations.md`. ∎ (`06_check_basic.py`, `z central: True`.)

---

## 5. Comparison maps: the reduction

**Orders on the vertices.** A map `G : C_Φ → C` of degree 0 has components `G_{IJ}` only for
`|J| = |I|`, so only vertices of equal size have to be compared. Two orders are used:

* `J ≺ I` iff `|J| = |I|` and `Σ_{x∈J} x < Σ_{x∈I} x` (*sum order*);
* `J <_c I` iff `|J| = |I|` and `Σ_{x∈J} 2^x < Σ_{x∈I} 2^x` (*colex order*); equivalently
  `J ≠ I` and the largest element of the symmetric difference `I △ J` lies in `I`.

Both are strict partial orders (`<_c` is total on the vertices of a given size), and, the
cube being finite, every strict partial order on its vertices is well-founded.

**Lemma 5.0 [proved].** Let `|I| = |J|`, `I ≠ J`.
(a) If there is a bijection `β : I∖J → J∖I` with `β(x) < x` for all `x`, then `J ≺ I` and
`J <_c I`.
(b) If some `k ∈ I∖J` satisfies `I ∩ {k+1,…,m} = J ∩ {k+1,…,m}`, then `J <_c I`.

*Proof.* (a) `Σ J − Σ I = Σ_{x∈I∖J} (β(x) − x) < 0`, as `I∖J ≠ ∅`. Let `y = max(I △ J)`. If
`y ∈ J∖I`, then `y = β(x) < x` for some `x ∈ I∖J ⊆ I △ J`, contradicting the maximality; so
`y ∈ I`. (b) `I △ J ⊆ {1,…,k}` and `k ∈ I △ J`, so `max(I △ J) = k ∈ I`. ∎

(b) does not give `J ≺ I`: `I = {1,5}`, `J = {3,4}`, `k = 5` satisfies (b) and has the larger
sum. (`06r_triangular.py` part (0): (a), (b) and the equivalence of the two descriptions of
`<_c` on all pairs of subsets of `{1,…,9}`.)

**Proposition 5.1 [proved].** Let `◁` be a strict partial order on the set of vertices (for
instance `≺` or `<_c`). Let `d = (d_{IJ})`, `d_{IJ} ∈ e_I A e_J` (`|J| = |I| − 1`), be any
family, and `G = (G_{IJ})`, `G_{IJ} ∈ e_I A e_J` (`|J| = |I|`), such that

* `G_{II} = ζ_I e_I` with `ζ_I` a non-zero polynomial in the dots, and `G_{IJ} = 0` unless
  `J = I` or `J ◁ I` (`G` is *`◁`-triangular*);
* `Σ_J (d_Φ)_{IJ} G_{JK} = Σ_J G_{IJ} d_{JK}` for all `I, K`.

Then `Σ_J d_{IJ} d_{JK} = 0` for all `I, K`.

*Proof.* `G d² = d_Φ G d = d_Φ² G = 0` by Proposition 4.3; the `(I, K)` component reads
`ζ_I (d²)_{IK} = − Σ_{J ◁ I} G_{IJ} (d²)_{JK}`. We show `(d²)_{IK} = 0` for all `K` by
well-founded induction on `I` along `◁` (the set of vertices is finite): if
`(d²)_{JK} = 0` for all `J ◁ I` and all `K` (for a `◁`-minimal `I` the sum is empty), then
`ζ_I (d²)_{IK} = 0`, and left multiplication by the non-zero polynomial `ζ_I` at the source
is injective on `e_I A e_K` (Theorem E.1(a) of `01_foundations.md`, a consequence of F1).
Only the existence of a well-founded order for which `G` is triangular is used, not which
one. ∎

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
cones, again by naturality. So `G = Γ^{(1)} Γ^{(2)} ⋯ Γ^{(m)}` (first `Γ^{(1)}`) is a chain
map `C_Φ → C`. (`06_lib.d_mixed` implements `C^{(k)}`.)

*Triangularity and diagonal.* Suppose every `Γ^{(k)}` has the cone form of Proposition 5.2
for the word `w_{≤k}`: `Γ^{(k)} = (ζ_k` on the vertices containing `k`, `1` on the others,
`h_k)`, where `ζ_k = ∏_{a<k, c(a)=c(k)} (x_a − x_k)` and every component of `h_k` goes from a
vertex `I ∋ k` to a vertex `J ∌ k` (`|J| = |I|`) with `I ∩ {k+1,…,m} = J ∩ {k+1,…,m}` (the
extension by `⊗ 1`, `1 ⊗` does not touch the later slots). Then:

(i) *`G` is `<_c`-triangular with `G_{II} = ∏_{x∈I} ζ_x e_I`.* `G_{IJ}` is the sum, over the
sequences `I = J_0, J_1, …, J_m = J` with `J_k = J_{k−1}` (diagonal entry) or `J_{k−1} → J_k`
a component of `h_k`, of the products of the corresponding entries. A component of `h_k`
satisfies Lemma 5.0(b) (with this `k`), so `J_k <_c J_{k−1}`; hence `J = I` or `J <_c I`,
and the only sequence from `I` to `I` is the constant one, whose product is
`∏_{k∈I} ζ_k e_I` (polynomials in the dots at the same idempotent commute). The factors
`x_a − x_x` (`a < x ∈ I`, `c(a) = c(x)`) are pairwise different.

(ii) *`G` need not be `≺`-triangular*: the cone form allows components such as
`{1,5} → {3,4}` of `h_5` (Lemma 5.0), and the degree does not always exclude them: for
`11211` (`A_2`, red 1) the shapes `{1,5} → {2,4}` and `{1,3,5} → {2,3,4}` of `h_5` (equal
sums) carry non-zero homogeneous elements of the degree of a component of `Γ^{(5)}`
(20 basis elements each) [computed, `06r_triangular.py` part (3); in `12121`, `21321`,
`1212`, `312131` no such shape is allowed by the degree]. So in this generality
Proposition 5.1 is applied with `◁ = <_c`.

(iii) If every component `I → J` of every `h_k` admits a bijection `I∖J → J∖I` lowering every
slot, then `G` is also `≺`-triangular (the argument of (i), with Lemma 5.0(a) in place of
5.0(b)). This holds for the formula of
Conjecture H and for `H⁺` (§7.2, Definition 8.1): there `J` is `I` with `k` replaced by
`a_1 < k` and every `b_p` by `a_p < b_p` (all `a`'s outside `I` and pairwise different), and
for the comparison map of §8.2.

(`06r_triangular.py` parts (1), (2), log `06r_logs/triangular.log`: the `G` of
Definition 6.2 on `1212`, `213213`, `123421` (`D_4`), `213243` (`A_4`); every component of
every `h_k` of the formulas H (on `213213`, `12121`, `11211`, `123131`, `121121`, `221122`)
and `H⁺` (on `121121`, `221122`, `1221122`) is of cone form with a lowering bijection, and
the factorised `G` is `≺`- and `<_c`-triangular with diagonal `∏_{x∈I} ζ_x`; no failure.)

---
## 6. Words in which every colour occurs at most twice

Throughout this section every colour occurs at most twice in `w`. (Statements are identified
by type *and* number: Definition 6.2 and Theorem 6.2, Corollary 6.3 and Lemma 6.3 are
different statements.)

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
`G_{I,J} ≠ 0` only for `J = I_F`, and for `F ≠ ∅` the bijection `b_p ↦ a_p` (`p ∈ F`) from
`I∖I_F` to `I_F∖I` lowers every slot, so `I_F ≺ I` and `I_F <_c I` (Lemma 5.0(a)): `G` is
triangular for both orders.

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
on it, and left multiplication by `z_p` is injective (F1: Theorem E.1(a) of `01_foundations.md`). ∎

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

*Check.* `06r_tree_lemma.py 7` (added in the revision): the identity as a polynomial
identity in independent variables `A_v, B_v` over `ℤ`, for all 874 recursive trees with at
most 7 vertices (every rooted tree of that size up to relabelling); log
`06r_logs/tree_lemma.log`.

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
`c` occurring three times there are several pairs of colour `c`, the colour map of Lemma 6.3
is no longer injective, and the link graph of a configuration *can* contain a 4-cycle (two
pairs of colour `c` both linked to the same two nodes of colours adjacent to `c` — possible
when `c` has two neighbours, or one neighbour occurring twice among the pairs); a
configuration whose link graph is such a 4-cycle is the minimal cyclic configuration of
`NOTES.md` §4. It need not: words with a colour three times can be cycle-free, e.g. `111`
and `12121` (`A_2`), and the words with a colour three or four times among the cycle-free
samples of §7.2. For those words the tree formula is a differential (computed, §7), but the `G` of Definition 6.2 is not a chain map (`111`,
`12121`, …: `R06_referee.md` §E, multiplicity-3 row), because (1) fails — the mover crosses
every earlier unaffected strand of its colour — and the correct `G` has the chain terms of
§7.2.

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
## 8. The cyclic case **[computed]**

### 8.1 The example `1221122` (`A_2`, red 1)

Slots `1,4,5` have colour 1 and `2,3,6,7` colour 2; the cyclic configuration is the mover
`5` with the pairs `(1,4)`, `(2,7)`, `(3,6)` (`cyclefree.find_cycle`). The tree formula does
not square to zero. The differential produced by the order-by-order solver in the complete
ansatz (`06_cyclic_true_d.py A 2 1 1221122`, stored in `06_data/`; 690 terms; the normal
form `d_true` of Remark 8.0(a)) is the tree formula plus **one** component,
`{4,5,6,7} → {1,2,3}`:

| bijection | coefficient |
|---|---|
| `(1 4)(2 7)(3 6)` (the cyclic configuration, mover 5) | `− r_1 r_2² t_{12} t_{21}² · ( t_{12}(x_{1,1} + x_{1,2} + 2x_{1,3}) + t_{21}(x_{2,1} + x_{2,2} + x_{2,3} + x_{2,4}) )` |
| `(1 4)(2 6 3 7)` (hand-over) | `+ r_1 r_2³ t_{12} t_{21}³` |
| `(1 4)(2 7 3 6)` (hand-over) | `− r_1 r_2³ t_{12} t_{21}³` |
| `(1 5 4)(2 7)(3 6)` (hand-over, the mover changes slot) | `− r_1² r_2² t_{12}² t_{21}²` |

(`x_{c,i}` = dot on the `i`-th strand of colour `c` from the left at the source vertex.) So all
three phenomena of `NOTES.md` §6 occur *in this normal form*: dotted terms, the coefficient
2, hand-over terms. (The solver's choice is not canonical: three orders have a kernel.)

**Remark 8.0 (normal forms; the dotted terms are not gauge-invariant).**
(a) *What "the true differential" `d_true` means in this section* (and in the scripts):
a normal form fixed by the solver. `06_data/*.json` (`1221122` red 1, `1322312`, `2132213`)
are written by `06_cyclic_true_d.py` = `cupbrane.run_case`: start from `δ₀` (the elementary
moves with their signs, *not* the tree formula), then repeatedly take the smallest parameter
monomial (dots ignored) occurring in `d²` and add the correction of that monomial which
solves that order in the complete ansatz (all homogeneous elements of that monomial, dots
allowed, `klrw` basis braids, dots at the left end) with minimal `ℓ¹`-norm of its integer
coefficient vector (HiGHS). That the result contains all tree-formula terms is an outcome of
the minimisation, not an input. For `1221122` red 1 the orders `r_2t_21`, `r_2²t_21²`,
`r_1t_12` have kernels of dimension 50, 2, 90 (re-run, identical output:
`06r_logs/rerun_cyclic_true_d_1221122.log`); the dot-free `d_{F,1}` of (d) below differs
from `d_true` first at `r_2²t_21²`. `04_data/cyclic7_1221122_W2_tree.json` (red 2) is the same
solver with weight 1 on tree-formula terms and 1000 on all others: the completion `d_X` that
keeps the tree formula's `r²`-part (`04_first_order.md` Theorem 9.1(d)).
(b) *Gauge.* For `g = 1 + k` with `k` of degree 0 and `g` invertible (e.g. `k² = 0`),
`g d g^{-1}` is again a differential and `g` is an isomorphism between the two complexes. By
`R04_referee.md` (G0) and `04_first_order.md` Theorem 9.1(e),(f), for red 2 the dot-free
`d_F = (1 + h_g) d_X (1 − h_g)` is a differential, with `h_g` the two dot-free terms
`− r_2²t_21² β_{(1 5 4)(2 7)(3 6)} : {4,6,7} → {1,2,3}` and
`− r_2²t_21² β_{(1 4 5)(2 7)(3 6)} : {4,5,6,7} → {1,2,3,5}`. So the dotted terms, the
coefficient 2 and the particular hand-over terms of the table are properties of a normal
form, not invariants of the complex.
(c) *Lemma [proved].* In Proposition 5.2, if (★) holds for `(F, h)` and `k` is any family of
the shape of `h` (components `C_R → C_L`), then (★) holds for
`(F + (d′⊗1)k + k(1⊗d′), h − ζ_m k)`, and the new cube differential is `(1 + k) D (1 − k)`,
`D = (−d′⊗1, F; 1⊗d′)`. *Proof.* `(d′⊗1)ζ_m k = ζ_m (d′⊗1) k` (Proposition 4.4), so
`Φ − (d′⊗1)(h − ζ_m k) − (h − ζ_m k)(1⊗d′) = ζ_m (F + (d′⊗1)k + k(1⊗d′))`. And `k² = 0`;
in `(1+k)D(1−k) = D + kD − Dk − kDk` only `k(1⊗d′)` and `−(−d′⊗1)k` survive, since `k`
starts in `C_R` and ends in `C_L`. ∎ So the division fixes a normal form through the choice
of `h`; changing `h` by `ζ_m`-multiples changes the quotient by a gauge.
(d) *Test* (`06r_gauge_1221122.py`, log `06r_logs/gauge_1221122.log`). **Red 2:** the
division with `h⁺` (Definition 8.1) is exact in all 153 components and its quotient is
**exactly `d_X`**, the dotted completion (equal in every component to
`04_data/cyclic7_1221122_W2_tree.json` and to the referee's independent
`R04_data/cyclic7_1221122_W2_dotted.json`); it differs from the dot-free `d_F`
(`R04_data/cyclic7_1221122_W2_dotfree.json`) in 9 components. Iterated division from `d_Φ`
gives the same. With `h′ = h⁺ − ζ_7 h_g` (the lemma with `k = h_g`; `Γ′ = Γ(1 − h_g)`) the
division is again exact and its quotient is **exactly `d_F`**. So the division with `h⁺`
produces neither `d_F` nor a third normal form, but `d_X`. **Red 1** (the example above): the
stored `d_true` is the quotient of the division with `h⁺`; the same two terms `h_g` are
homogeneous of degree 0 for red 1 as well and conjugate it into a **dot-free** differential
`d_{F,1}`, which is the quotient of the division with `h⁺ − ζ_7 h_g`; its terms outside the
tree formula are exactly those of `d_F` (the 8 hand-over terms `κ*` of `04` Theorem 9.1(e)
at `r_2²t_21²` and the four terms `ε₄` in `{4,5,6,7} → {1,2,3}`). Hence for this word, for
both red colours, the dotted terms are an artefact of the normal form selected by `h⁺`.
Whether the dotted terms of `1322312` (§8.6) and `2132213` (§8.5) can be removed in the
same way was not tested.

### 8.2 A comparison map exists and is unique up to homotopy

`06_cyclic_step.py`, `06_cyclic_homotopy.py` (`A 2 1 1221122`), complete ansatz (21 858
unknowns): the space of chain maps `Cone(Φ_7 on the tree cube of 122112) → (C, d_true)` has
dimension 563, its subspace of null-homotopic maps has dimension 562. So there is, up to
homotopy and a scalar, exactly one comparison map; a sparsest representative with
`Γ_∅ = 1` has the diagonal `ζ_7` and equals the formula of Conjecture H **plus four terms**,

    {4,6,7} → {1,2,3},  {4,5,6,7} → {1,2,3,5}:   ± r_1 r_2³ t_{12} t_{21} (x_6 − x_7) · [(1 4)(2 6 3 7)],
    {5,6,7} → {1,2,3},  {4,5,6,7} → {1,2,3,4}:   ± r_1 r_2³ t_{12} t_{21} (x_6 − x_7) · [(1 5)(2 6 3 7)].

This representative has the cone form of Proposition 5.2 (diagonal `ζ_7` on `C_R`, `1` on
`C_L`, all other components `C_R → C_L`: the formula of Conjecture H differs from it only in
these four components, re-run `06r_logs/rerun_cyc_step_1221122.log`). Take it as `Γ^{(7)}`
and, for `k ≤ 6`, the formula of Conjecture H for the cycle-free prefix `122112` (a chain
map at each of these steps [computed], `06_check_step_formula.py A 2 1 1221122`: it fails
only at the step `k = 7`). Then `G = Γ^{(1)}⋯Γ^{(7)} : C_Φ → (C, d_true)` is a chain map; all
seven maps have the cone form, so `G` is `<_c`-triangular with diagonal `∏_{x∈I} ζ_x`
(Remark 5.3(i)); the four extra terms also admit lowering bijections (e.g.
`{4,6,7} → {1,2,3}`: `4 ↦ 1, 6 ↦ 2, 7 ↦ 3`), so `G` is `≺`-triangular as well (Remark
5.3(iii)). By Proposition 9.1 (with `◁ = <_c`), `d_true = G^{-1} d_Φ G` after localisation
at `Δ`. So **in this example** the intertwiner description extends to a cyclic word; this is
one computed case (and its analogues in §8.5), not a general statement. (By Remark 8.0 the
target `d_true` is itself one normal form; composing `Γ^{(7)}` with `1 − h_g` gives the
analogous statement for the dot-free `d_{F,1}`.) With the formula of Conjecture H alone the
division fails: `Φ − (d′⊗1)h − h(1⊗d′)` is not divisible by `ζ_7` in 21 components
(`06_divide.py A 2 1 1221122`).

### 8.3 The formula `H⁺`

The four extra terms are of the same kind as the others once one allows the following.
After the chain move `m → a_1`, `a_1 → a_2`, …, `a_s → m`, the left slots `a_2, …, a_s` are
occupied again (by the strands that came from `a_1, …, a_{s−1}`), so a pair `(a_i, b)`,
`i ≥ 2`, `b ∈ I`, `a_i < b < m`, of the colour of the chain can be exchanged *after* the chain.

**Definition 8.1 (formula `H⁺`).** As in Conjecture H, except that a hanging forest may also
contain pairs `p = (a_i, b)` with `a_i` a chain slot, `i ≥ 2` (at most one for each `a_i`),
`b ∈ I`, `a_i < b < m`, `c(b) = c`. Links, the tree condition, connectors, signs and the
coefficient are defined by the same rules, with the slots `(a_i, b)` of the pair (an arc
ending or starting at `a_i` is not interlaced with it). In the bijection, the strand
arriving in slot `a_i` (from `a_{i−1}`) continues to slot `b`, and the strand of `b` goes to
slot `a_i` on the right:  `m ↦ a_1`, `a_{i−1} ↦ b`, `a_i ↦ a_{i+1}`, `b ↦ a_i`.
(`06_lib.SHARED = True`; option `plus` of the scripts.)

In the example: chain `(2, 3)`, forest `{(1,4), (3,6)}` (resp. `{(1,5), (3,6)}`): the pair
`(1,4)` hangs from the arc `(3,7)` and `(3,6)` hangs from `(1,4)`; coefficient
`(−r_2)²·(x_6 − x_7)·r_1t_{12}·r_2t_{21}` — the four terms of §8.2, with their signs. Such a
forest contains a pair of the colour of the chain attached, through a pair of an adjacent
colour, to the chain: three nodes of colours `c, c′, c`.
`H⁺` differs from `H` only if the colour of the chain has at least four slots
(`a_{i−1}, a_i, b, m`).

### 8.4 Division with `H⁺` reproduces the solver's differential (one example)

`06_divide.py A 2 1 1221122 plus`: with `h⁺` of Definition 8.1 and `d′` the tree differential
of `122112`, the element `Φ − (d′⊗1)h⁺ − h⁺(1⊗d′)` **is divisible by `ζ_7` in all 153
components**, and the quotient

    F = ζ_7^{-1} ( Φ_7 − (d′⊗1) h⁺ − h⁺ (1⊗d′) )

equals the stored `d_true` in every component: the tree terms, and the component
of §8.1 with its dotted terms, the coefficient 2 and the three hand-over terms. (By
Proposition 5.2 divisibility alone implies `d² = 0`; the script checks it again.)

So in this example a differential is given by a closed formula: the tree formula for the
prefix, the explicit `h⁺`, and one division by the product `ζ_7` of differences of dots. It
is the solver's normal form; `h⁺ − ζ_7 h_g` gives the dot-free one instead (Remark 8.0(d)).

### 8.5 Iterated division: a construction of a differential that does not use the tree formula (computed on a few words)

`06_divide_all.py` starts from `d_0 = d_Φ` and for `k = 1, …, m` defines `d_k` from `d_{k−1}`
by `d_k = (Γ^{(k)})^{-1} d_{k−1} Γ^{(k)}` with `Γ^{(k)} = ζ_k + h_k⁺` — concretely only the
components `I → K` with `k ∈ I`, `k ∉ K` change, and

    ζ_k · d_k(I → K) = Σ_J d_{k−1}(I → J) Γ^{(k)}(J → K) − Σ_{J≠I} Γ^{(k)}(I → J) d_{k−1}(J → K);

the script solves this for `d_k(I → K)` in the complete ansatz and reports non-divisible
components. If all divisions are exact, the script checks `d_m² = 0` directly; `d_m` is then a
differential on the cube of `w` obtained from `d_Φ` and the explicit `h_k⁺` alone, without
any information about the tree formula (formally `d_m = G^{-1}d_ΦG`, `G = Γ^{(1)}⋯Γ^{(m)}`,
after localisation at `Δ`). Whether this happens for a given word is a computation; the
table lists every word on which it was done.

**Results** (`06_divide_all.py … plus`; "exact" = all divisions exact, and then `d_m² = 0`
was verified again):

| word | cyclic | result |
|---|---|---|
| `A_2` red 1: `12121`, `1211`; red 2: `2112` | no | exact; `d_m` = tree formula |
| `A_2` red 1 `1221122` | yes | exact; `d_m` = stored `d_true` (solver normal form) in all components (tree formula + the component of §8.1) |
| `A_2` red 2 `1221122` | yes | exact; `d_m` = the differential `04_data/cyclic7_1221122_W2_tree.json` computed by the first-order front, in all components (`06_compare_04.py`): the dotted normal form `d_X`, not the dot-free `d_F` (Remark 8.0(d), `06r_gauge_1221122.py`) |
| `A_2` red 2 `2211221` | yes | exact; `d_m² = 0`; 3 components differ from the tree formula |
| `A_3` red 1 `1322312` (one step, `06_divide.py`) | yes | exact; `d² = 0`; 3 components differ from the tree formula (§8.6) |
| `A_3` red 2 `2132213` (the minimal cyclic configuration with two different neighbouring colours: mover 5 and pair `(1,4)` of colour 2, pairs `(2,6)`, `(3,7)` of colours 1, 3; here `H⁺ = H`) | yes | exact; `d_m` = the corrected differential computed by `06_cyclic_true_d.py` (616 terms; solver normal form, Remark 8.0(a)), in all components. It is the tree formula plus the component `{4,5,6,7} → {1,2,3}`: `r_1r_2r_3t_{12}t_{32}·(t_{12}t_{23}(x_{1,1}+x_{1,2}) + t_{21}t_{23}(x_{2,1}+x_{2,2}+2x_{2,3}) + t_{21}t_{32}(x_{3,1}+x_{3,2}))·[(1 4)(2 6)(3 7)]` and the hand-over term `r_1r_2²r_3t_{12}t_{21}t_{23}t_{32}·[(1 5 4)(2 6)(3 7)]`. For this word the comparison map is unique outright: the space of chain maps `Cone(Φ_7) → (C, d_true)` is 1-dimensional and there are no homotopies (`06_cyclic_homotopy.py`) |
| `A_2` red 1 `12211221` (8 strands; the prefix `1221122` is already cyclic) | yes | steps 1–7 exact (they are the word `1221122`); **step 8 is not exact: 18 components are not divisible** (17 minutes) |
| `A_2` red 1 `12211212` (8 strands, cycle-free prefix `1221121`; one step, `06_divide.py … plus`) | yes | not completed (stopped after 27 minutes) |

On cycle-free words the formula `H⁺` passes the same tests as the formula of Conjecture H
(`06_check_step_formula.py … plus` on the samples `A 3 random 40 7 3 11`,
`A 2 random 30 7 4 3`, `D 4 random 40 7 3 7`: all cycle-free words hold), so `H⁺` can
replace `H` in Conjecture H (as far as these samples show).

**Conjecture H⁺.** If the prefix `w′` has no cyclic configuration, then
`Φ_m − (d′⊗1)h⁺ − h⁺(1⊗d′)` is divisible by `ζ_m` (`d′` the tree differential of `w′`, `h⁺` of
Definition 8.1), and the quotient `F` completes `d′` to a differential of the cube of `w`
(by Proposition 5.2 it does give a differential); it is the tree formula iff `w` has no
cyclic configuration. *Evidence:* the cycle-free samples of §7.2 and the five cyclic
7-strand cases of the table above — nothing more. In the cyclic case the conjecture
concerns *one* normal form of the differential: in the examples the quotient is the solver's
(dotted) normal form, and by Remark 8.0(c) a different `h` (`h⁺` minus a `ζ_m`-multiple)
gives a gauge-equivalent one.

**This does not extend as it stands to words with a cyclic prefix**: for `12211221` the
division at step 8, with `d_7` the (true) differential of the cyclic prefix and the same
formula `h_8⁺`, fails in 18 components. Presumably `h` needs further terms attached to the
cyclic configurations of the prefix (as `d′` then contains dotted and hand-over terms); I
did not determine them. So the statement "the differential is `G^{-1}d_ΦG`" is
computed in the cyclic case only at the first cyclic step, and only on the words of the
table; a formula for `G` beyond it is open.

The evidence in the cyclic case is the 7-strand cases above (five cases: `1221122` with
red 1 and red 2, `2211221`, `1322312`, `2132213`; four of them with a non-trivial
comparison, three of which reproduce an independently computed normal form exactly). I have no example with two independent cycles and none in
which a cyclic configuration needs a chain of length 3.

### 8.6 The example `1322312` (`A_3`, red 1)

This is the minimal cyclic configuration with two different neighbouring colours (mover
and pair of colour 2, pairs of colours 1 and 3). Here the cube is contractible (the strand
of slot 2, colour 3, crosses the red strand of colour 1 and the strand of colour 1 in
degree 0; `06_cyclic_homotopy.py`: all 522 chain maps are null-homotopic), so the stored
corrected differential (`06_data/A3__W1_1322312.json`, 601 terms, 8 with dots) is far from
unique and a comparison of individual terms with it is meaningless. But the division is a
meaningful test: `06_divide.py A 3 1 1322312` (here the colour of the last slot has only
three slots, so `H⁺ = H`: the plain formula of Conjecture H is used) — divisible in all 155 components,
`d² = 0`, and the result differs from the tree formula in three components:

    {3,5,6,7} → {1,2,4}:  (1 6)(2 5)(4 7),   r_1r_2r_3 t_{12}t_{21}t_{23}t_{32} · ( −x_{2,1} + 2x_{2,2} − x_{2,3} )
    {3,5,6,7} → {1,2,3}:  (1 6)(2 5)(3 4 7),  − r_1r_2²r_3 t_{12}t_{21}t_{23}t_{32}
    {4,5,6,7} → {1,2,4}:  (1 6)(2 5)(3 7 4),  − r_1r_2²r_3 t_{12}t_{21}t_{23}t_{32}

again a dotted term with a coefficient 2 and two hand-over terms, of the shape described in
`NOTES.md` §6. (The solver's output is another solution of `d² = 0`; I did not construct the
change of basis relating the two.)

---
## 9. Consequences for the homology

Let `Pol = R[x_1, …, x_m]`, and `Δ = ∏_c ∏_{i<j} (x_{c,i} − x_{c,j})²`, the product over the
colours and over the pairs of strands of that colour (at each state; the same symmetric
polynomial). `Δ` is central in `A`: it commutes with crossings of different colours because
dots slide and `Δ` is symmetric within each colour, and with crossings of equal colour by
Lemma 1.1.

*Localisation at `Δ`.* The powers of `Δ` form a multiplicative set `S_Δ = {Δ^n}` of central
elements of `A`. For an `A`-module, a complex of `A`-modules or a `Pol`-module `M` write
`M[Δ^{-1}] = S_Δ^{-1} M` for its *localisation at the central element `Δ`* (on `Pol` and its
modules, `Δ ∈ Pol` acts by multiplication); `A[Δ^{-1}] = S_Δ^{-1}A` is a ring, and a map of
complexes `f` induces `f[Δ^{-1}]`. Localisation is exact and commutes with homology and with
`e (−)`. (No ring of the form `ℤ[Δ^{-1}]` or `R[Δ^{-1}]` is meant: `Δ` is an element of `A`,
not of `ℤ` or `R`.)

**Proposition 9.1 [proved].** Let `G : C_Φ → C` satisfy the hypotheses of Proposition 5.1
for some strict partial order `◁` on the vertices, and let every `ζ_I` be a product of
factors `±(x_a − x_b)`, `a ≠ b`, `c(a) = c(b)`, no two of them for the same pair `{a, b}`.
This holds for the `G` of Definition 6.2 (`◁ = ≺` or `<_c`) and for a factorised
`G = Γ^{(1)}⋯Γ^{(m)}` of one-step maps of cone form (`◁ = <_c`, Remark 5.3(i)). Then the
localisation of `G` at `Δ`, `G[Δ^{-1}] : C_Φ[Δ^{-1}] → C[Δ^{-1}]`, is an isomorphism of
complexes, and `d = G^{-1} d_Φ G` with `G^{-1}` a `◁`-triangular matrix over `A[Δ^{-1}]`.

*Proof.* At the idempotent `e_I` put `δ = ∏_c ∏_{i<j} (x_{c,i} − x_{c,j})`, so `δ² = Δ`. The
factors of `ζ_I` are, up to sign, pairwise different factors of `δ`, so `ζ_I g_I = ±δ` for a
polynomial `g_I`, and `ζ_I e_I` has the inverse `±g_I δ Δ^{-1} e_I` in `e_I A[Δ^{-1}] e_I`
(polynomials in the dots at one idempotent commute). Write `G = D + N` with `D` the diagonal
and `N_{IJ} ≠ 0` only for `J ◁ I`. Then `G = D(1 + D^{-1}N)`, and the components of
`(D^{-1}N)^l` are sums over chains `I = J_0 ▷ J_1 ▷ ⋯ ▷ J_l`; a chain has at most `2^m`
elements, so `(D^{-1}N)^l = 0` for `l ≥ 2^m`, and `G^{-1} = Σ_{l<2^m} (−D^{-1}N)^l D^{-1}` is a
two-sided inverse, again `◁`-triangular. Localisation is a functor, so `G[Δ^{-1}]` is a
chain map, and its inverse is then a chain map as well. From `d_Φ G = G d`,
`d = G^{-1} d_Φ G`. ∎

**Proposition 9.2 [proved].** Let `t` be a labelled state. The left `Pol`-submodules
`Pol·𝒟(t → std I) ⊆ e(t) A e_I` are free of rank one and form a subcomplex `𝒦(t)` of
`e(t) C_Φ`, isomorphic to the Koszul complex over `Pol` of the elements

    f_j(t) = ∏ { q_{jy} :  y ∈ {W, 1, …, j−1},  j to the left of y in t },      j = 1, …, m.

*Proof.* `𝒟(t → std I)𝒟(std I → t) = ∏_{inv} q ≠ 0` (Proposition 3.5) and `e(t)Ae_I` is a
free `Pol`-module (F1: Theorem A.5(ii) of `01_foundations.md`; only `Pol ↪ e(t)Ae(t)` is needed), so the rank is one. By Proposition 3.5,
`𝒟(t → std I)Φ_j(I) = ∏ q_{jy}·𝒟(t → std(I∖j))` over the pairs `{j, y}` inverted by `Φ_j(I)`
(`y < j` or `y = W`; in `std I` these `y` are to the left of `j`) which are also inverted
between `t` and `std I`, i.e. for which `j` is to the left of `y` in `t`. This does not
depend on `I`. ∎

**Proposition 9.3 [proved].** For every state `e` of `A` and every vertex `I`,
`e A[Δ^{-1}] e_I = ⊕_t Pol[Δ^{-1}]·𝒟(t → std I)`, the sum over the labelled states `t` with
colour sequence `e`. Hence `e\,C_Φ[Δ^{-1}] = ⊕_t 𝒦(t)[Δ^{-1}]` as complexes (localisations
at `Δ`).

*Proof.* Over `Pol[Δ^{-1}]`, `ψ = (x_i − x_{i+1})^{-1}(φ + r)` for strands of equal colour, so
every product of generators of `A` from `e` to `e_I` is a combination, with coefficients
in `Pol[Δ^{-1}]` written at the source (Proposition 3.3), of products of generators `g` along
words ending in `std I`; by Proposition 3.5 (applied letter by letter) each such product
is a `Pol`-multiple of one `𝒟(t → std I)`. So the `𝒟(t → std I)` span. Their number is the
number of colour-preserving bijections, which is the rank of the free `Pol[Δ^{-1}]`-module
`eA[Δ^{-1}]e_I` (F1: Theorem A.5(ii) of `01_foundations.md`, localised at `Δ`); a spanning set of `n` elements of a free module of rank `n` over a
commutative ring is a basis. ∎

**Corollary 9.4 [proved, for every word for which a map `G` as in Proposition 9.1 exists —
by Theorem 6.2 for every word in which each colour occurs at most twice].** Let
`R′ = R[r^{-1}, t^{-1}, u^{-1}]` (all parameters inverted; `R → R′` is flat, so
`H_i(C) ⊗_R R′ = H_i(C ⊗_R R′)`), `Pol′ = R′[x_1, …, x_m]`, and let `[Δ^{-1}]` denote
localisation at the central element `Δ` as above. Then

    (H_i(C) ⊗_R R′)[Δ^{-1}] = 0   for i > 0,
    e·(H_0(C) ⊗_R R′)[Δ^{-1}] ≅ ⊕_t Pol′[Δ^{-1}] / (f_1(t), …, f_m(t)).

*Proof.* Localisation at `Δ` and `⊗_R R′` are exact, so the left sides are the homology of
`e·(C ⊗_R R′)[Δ^{-1}]`, which by Propositions 9.1 and 9.3 (over `R′`) is isomorphic to
`⊕_t 𝒦(t) ⊗_R R′` localised at `Δ`. So it suffices to show that the Koszul complex of
`(f_j(t))` over `Pol′` is exact in positive degrees (then so is its localisation). Over `R′` each `f_j(t)` is a
unit (if no factor contains a dot) or, up to a unit, a monic polynomial of positive degree
in `x_j` with coefficients in `R′[x_1,…,x_{j−1}]` (each non-constant factor is `u_k x_j` or
`t x_j + t′ x_y`, `y < j`). If some `f_j` is a unit the Koszul complex is contractible.
Otherwise `f_j` is a non-zero-divisor on `R′[x_1..x_m]/(f_1,…,f_{j−1}) = (R′[x_1..x_{j−1}]/(f_1..f_{j−1}))[x_j, …, x_m]`, so the sequence is
regular. ∎

*What this says and does not say.* The higher homology of the tree complex `C` is
`Δ`-torsion up to the parameters (it vanishes on the locus where the dots of equal colour
are distinct and the parameters are invertible), and there `H_0(C) = S = P_∅/Σ_j A·m_j` is
identified with a sum of complete intersections. Whether `C` is a resolution of `S` over
`R` itself — i.e. whether `H_i(C)`, `i > 0`, has no `Δ`-torsion — is **not** decided by this
argument; `G` is injective with torsion cokernel, so `C_Φ ⊂ C` is a subcomplex of full rank,
and `H(C)` and `H(C_Φ)` can differ by torsion.

---
## 10. Status

### Proved completely (from the relations of §1.1, F1 = Theorems A.5 and E.1 of `01_foundations.md`, and F3)

* Lemma 1.1, Lemma 2.1 (intertwiner identities), **Lemma 2.3** (the generators "`ψ` for
  different colours, `φ` for equal colours" satisfy the braid relations exactly, red strand
  included; `ψ_1φ_2φ_1 = φ_2φ_1ψ_2`).
* §3, the labelled calculus: `𝒟(s → s′)` is well defined (3.2), dots slide (3.3), the
  product rule (3.5).
* Proposition 4.2 (naturality of `Φ`, statement (i) of the brief), **Proposition 4.3**
  (`d_Φ² = 0` for every word), Proposition 4.4 (statement (ii) of the brief).
* Lemma 5.0 (the sum order `≺` and the colex order `<_c`), Proposition 5.1 (a `G` with
  `d_Φ G = G d`, triangular for *some* strict partial order on the vertices, forces
  `d² = 0`), Proposition 5.2 (the Key Lemma (★) implies that `F` is a chain map), Remark 5.3
  (factorisation of `G`; a factorised `G` of one-step maps of cone form is `<_c`-triangular
  with diagonal `∏_{x∈I} ζ_x`, and `≺`-triangular when every `h_k` lowers every slot, as for
  the formulas H and `H⁺`; not `≺`-triangular in general).
* **Theorem 6.2 and Corollary 6.3**: for every word in which each colour occurs at most
  twice, `d_Φ G = G d_tree` with `G` of Definition 6.2 (triangular for `≺` and for `<_c`);
  hence the tree formula is a differential. (Lemmas 6.1, 6.3–6.8, Corollary 6.9.) This is
  independent of the proof in `05_main_sliding.md`: it uses no sliding of strands through
  triple points; the only relations entering after §3 are `(x_a − x_b)ψ = φ + r` and the
  product rule. Confirmed by the referee (`R06_referee.md`), including an independent
  class-by-class check of §6.6.
* Propositions 9.1–9.3 and Corollary 9.4: for such words (and every word with a `G` as in
  Proposition 9.1) `C[Δ^{-1}] ≅ C_Φ[Δ^{-1}]` (localisation at the central element `Δ`), and
  after inverting the parameters and localising at `Δ` the tree complex has no higher
  homology.
* The lemma of Remark 8.0(c): changing `h` by a `ζ_m`-multiple `ζ_m k` changes the quotient
  `F` of (★) by the gauge `1 + k`.

*Which of these have a script.* Not every statement has one. With a script (exact
computation, cases listed next to the statement): Lemma 1.1 (referee: `R06_basic.py`, on
a cubic `f`); Lemmas 2.1, 2.3, Propositions 4.2–4.4 (`06_check_basic.py`; referee:
`R06_basic.py`, `R06_natural.py`); §3 and Lemma 6.5
(`06_check_labelled.py`); Lemma 5.0 and the triangularity statements of Remark 5.3,
Definition 6.2 and §8.2 (`06r_triangular.py`, added in the revision); Lemma 6.6
(`06_check_bigons.py`); Lemma 6.7 (`06r_tree_lemma.py`, added in the revision); Theorem 6.2
(`06_check_G_formula.py`, `06_check_minuscule.py`; referee: `R06_main.py` on all 810 + 9 296
cases of length ≤ 6 in `A_3`, `D_4`, and `R06_classes.py`, which checks that each class
T0/T1/T2 of §6.6 vanishes separately); Proposition 9.2 (referee: `R06_sec9.py`, 1 536
identities); the lemma of Remark 8.0(c) in the example of §8 (`06r_gauge_1221122.py`
parts (3), (4), (6)). **Without a script of their own:** Lemmas 6.1, 6.3, 6.4,
6.8 and Corollary 6.9 (they enter every check of Theorem 6.2 and the referee's class-by-class
check, but are not tested separately), Propositions 5.1 and 5.2 (formal arguments; their
conclusion `d² = 0` is re-checked whenever they are applied, e.g. in `06_divide.py`),
Propositions 9.1, 9.3 and Corollary 9.4.

### Verified only by computation

* Existence and uniqueness of `G` in the complete ansatz for 17 non-degenerate words with a
  colour occurring three times (§7.1).
* Conjecture H (closed form of the one-step maps, any multiplicity): 168 cycle-free words,
  up to 10 strands, including `A_5ω_3` and the spinor of `D_5` (§7.2).
* Cyclic case (§8), **on five cyclic 7-strand cases only** (`1221122` with red 1 and red 2,
  `2211221`, `1322312`, `2132213`): existence and uniqueness (up to homotopy) of the
  comparison map for `1221122` (red 1) and `2132213`; the formula `H⁺`; exactness of the
  division at the first cyclic step and equality of the quotient with an independently
  computed *normal form* of the differential (`1221122` red 1 and red 2, `2132213`), resp.
  `d² = 0` of the quotient (`2211221`, `1322312`). For `1221122` (red 1 and red 2) the
  quotient with `h⁺` is the dotted normal form; the dot-free gauge-equivalent differential
  is the quotient with `h⁺ − ζ_7 h_g` (Remark 8.0(d)). These are not general statements.

*Re-runs in this revision.* The author's scripts quoted in the text that finish within a few
minutes were re-run (logs `checks/06r_logs/rerun_*.log`, summary `rerun_SUMMARY.txt`); every
result agrees with the text: `06_check_basic.py` (each of `121`, `1212`, `1211`, `111` with
red 1 and red 2: 672 identities, all true), `06_check_labelled.py` (the four words, 60
random samples per part; (d) 549 tree terms), `06_check_bigons.py 30000 1` (18 136
configurations), `06_check_G_formula.py` (both random samples: 45 of 45 words),
`06_check_minuscule.py` (all seven minuscule words, including `A_6ω_2` and `D_5ω_5`),
`06_check_factorisation.py` (on `2132`, `21321`, `213213`, `132132`; the text's seven words
are not listed), `06_check_step_formula.py` (the seven named chosen words, the four random
samples with and without `plus`: as in the table of §7.2, the cyclic words failing at the
last step only; `1221122` red 1: `YYYYYYN`), `06_batch_G.py` (all 21 words of §7.1: as in the
table), `06_solve_step.py` (the seven words of §7.1: every step 1-dimensional),
`06_cyclic_step.py`, `06_cyclic_homotopy.py` (`1221122`: 563 / 562; `2132213`: 1 / 0;
`1322312`: 522 / 522), `06_divide.py` (`1221122` with H: 21 non-divisible components; with
`plus`: exact, 0 components differ from `d_true`; `1322312`: exact, 3 components differ from
the tree formula), `06_divide_all.py … plus` (`12121`, `1211`, `2112`, `1221122` red 1,
`2211221`, `2132213`: as in the table of §8.5), `06_compare_04.py` (0 components differ), and
the solver of `06_cyclic_true_d.py A 2 1 1221122` (run into a scratch directory so as not to
overwrite `06_data/`: 690 terms, identical to the stored file;
`rerun_cyclic_true_d_1221122.log`). Not re-run: `06_divide_all.py`
on `12211221` (17 minutes), `06_divide.py … 12211212 plus` (not completed by the author,
27 minutes), `06_cyclic_G.py` (not completed by the author).

### Open

* A proof of Conjecture H (hence of C1′) for words with a colour occurring three or more
  times. §7.3 says what is missing.
* Conjecture H⁺ (division at the first cyclic step), and the correct `h` after a cyclic
  step, where `H⁺` is known to be insufficient (`12211221`, §8.5); a closed formula for
  the quotient in the cyclic case (the dotted and hand-over terms are *produced* by the
  division; they are not yet *described* term by term) — and, by Remark 8.0, which normal
  form such a formula should describe.
* Whether the dotted cyclic terms of `1322312`, `2132213`, `2211221` are gauge artefacts as
  they are for `1221122` (Remark 8.0(d)).
* Whether `C` is a resolution over `R` (no `Δ`-torsion in the higher homology), §9.
* The other-direction map `G′ : C → C_Φ` (diagonal `∏_{x∉I} ζ_x`) was not studied.

### Failed attempts and corrections to the brief

* *(Corrected in the revision after `R06_referee.md`.)* The first version stated
  Proposition 5.1 for the sum order `≺` only and called a factorised `G = Γ^{(1)}⋯Γ^{(m)}`
  triangular (Remark 5.3, §8.2, Proposition 9.1) without saying for which order; for one-step
  maps of cone form it is `<_c`-triangular, not `≺`-triangular in general (Lemma 5.0,
  Remark 5.3(ii)). Corollary 6.3 was not affected (Definition 6.2 is `≺`-triangular).
* *(Corrected in the revision, after `R04_referee.md` G0.)* The first version called the
  solver's output "the true differential" and presented its dotted terms, coefficient 2 and
  hand-over terms as *the* cyclic correction, and the iterated division as a general
  definition of the differential. For `1221122` (red 1 and red 2) the dotted terms are a
  normal-form artefact: a dot-free gauge-equivalent differential exists and is produced by
  the division with `h⁺ − ζ_7 h_g` (Remark 8.0); the statements about division are
  computations on five cyclic cases (Conjecture H⁺).

* The global `G` is **not** the product of the commuting factors `z_p − r_pE_p`: for two
  linked pairs the coefficient of the double exchange is neither of the two honest bigons
  `Q(L_p, L_q)`, `Q(R_p, R_q)` but the mixed one `Q(R_p, L_q)` (Definition 6.2); equivalently
  the one-step map for the later pair has a tree term (root `q`, child `p`).
* Conjugating `z_q − r_qE_q` by the earlier one-step map does not give the later one-step
  map (it produces a non-polynomial term `r_pr_qt_{qp}z_q/z_p`).
* The group of relabellings of equal-coloured slots does not act on `C_Φ[Δ^{-1}]` by chain
  maps (the bigon factors differ on the two sides); so `C` is not obtained as a
  symmetrisation of `C_Φ` in an obvious way.
* The tree-only formula for `h` (Conjecture H) is wrong in the cyclic case (division fails
  in 21 components for `1221122`); `H⁺` repairs this at the first cyclic step in the
  examples tested, but is itself insufficient at the step after a cyclic one (`12211221`).
* In the brief, `z = ∏(x_m − x_y)` and `Fz`: with `φ = (x_1−x_2)ψ − r` the correct statement
  is `Φ = ζ_m F + h(1⊗d′) + (d′⊗1)h` with `ζ_m = ∏(x_y − x_m)` on the **source** side of `F`
  (`ζ_m` does not commute with `F`: an exchange of the pair `(a, m)` changes the sign of
  `x_a − x_m`); and `h = −κ′ r_c E(a, m) + …`.
* Words in which some elementary move is invertible (e.g. red 2, word `121`) have a
  contractible cube; there "existence of `G`" is empty and solutions are not unique. All
  uniqueness statements are for words without such a move.

---

## Appendix: scripts (`proofs/checks/`)

All take `TYPE RANK RED WORD` (e.g. `A 2 1 1212`) unless stated; run with `sage -python`
from `proofs/checks/`.

| script | what it does | used in |
|---|---|---|
| `06_lib.py` | library: labelled states, `ψ`, `φ`, `Φ_j`, `d_Φ`, tree `d`, `C^{(k)}` (`d_mixed`), complete ansatz (`basis`, own enumeration of monomials), linear systems for chain maps (`ChainMapSystem`), `𝒟`, `q_{ab}`, brackets, the formulas `G_formula_row` (Definition 6.2) and `step_formula_row` (Conjecture H; `SHARED = True` for `H⁺`) | all |
| `06_check_basic.py` | naturality of `Φ_m` on all states, `ζ` central, `d_Φ² = 0` | §2, §4 |
| `06_check_labelled.py` | independence of the reduced word, product rule, sliding of dots, expansion of the tree terms | §3, Lemma 6.5 |
| `06_check_bigons.py [N] [SEED]` (plain `python3`) | the six statements of Lemma 6.6 | §6.4 |
| `06_check_G_formula.py` (also `TYPE RANK random N MAXLEN [SEED]`) | `d_Φ G = G d_tree` for Definition 6.2 | Theorem 6.2 |
| `06_check_minuscule.py TYPE RANK RED [min\|max]` | Theorem 6.2 and Conjecture H on minuscule words | §6, §7 |
| `06_check_factorisation.py` | `G = Γ^{(1)}⋯Γ^{(m)}` | §7.2 |
| `06_solve_G.py … [print]`, `06_batch_G.py` | solve for `G` in the complete ansatz; uniqueness, homotopies | §7.1 |
| `06_solve_step.py … [print] [k …]` | solve for the one-step maps | §7.1 |
| `06_check_step_formula.py` (also `random N MAXLEN MAXMULT [SEED]`, `plus`) | Conjecture H (resp. `H⁺`): each `Γ^{(k)}` is a chain map | §7.2, §8.5 |
| `06_cyclic_true_d.py` | corrected differential in the complete ansatz, stored in `06_data/` (2 minutes for 7 strands) | §8.1 |
| `06_cyclic_step.py`, `06_cyclic_homotopy.py` | comparison map for a stored differential; chain maps modulo homotopy (2 minutes each) | §8.2 |
| `06_divide.py … [plus]` | one division `ζ F = Φ − [d′, h]`, comparison with the tree formula and a stored differential | §8.4, §8.6 |
| `06_divide_all.py … [plus] [print]` | iterated division from `d_Φ` | §8.5 |
| `06_compare_04.py [FILE]` | iterated division against the differential stored in `04_data/` | §8.5 |
| `06_cyclic_G.py` | global `G` in the complete ansatz against a stored differential — **not completed** for 7 strands (too large); superseded by `06_cyclic_step.py` | — |
| `06r_triangular.py` (revision) | Lemma 5.0 on all pairs of subsets of `{1..9}`; triangularity (`≺`, `<_c`) and diagonal of the `G` of Definition 6.2 and of factorised `G` (formulas H, `H⁺`); one-step shapes not `≺`-smaller allowed by the degree (log `06r_logs/triangular.log`, < 1 min) | §5, Remark 5.3 |
| `06r_tree_lemma.py [N]` (revision) | Lemma 6.7 as a polynomial identity on all recursive trees with ≤ `N` vertices (log `06r_logs/tree_lemma.log`) | Lemma 6.7 |
| `06r_gauge_1221122.py` (revision) | `1221122`, red 2: the quotient of the division with `h⁺` against `d_X` (`04_data`, `R04_data`) and `d_F` (`R04_data`); iterated division; division with `h⁺ − ζ_7h_g`; the gauge; red 1: the same gauge on `06_data` (log `06r_logs/gauge_1221122.log`, 3 min) | Remark 8.0 |
