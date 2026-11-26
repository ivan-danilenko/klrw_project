# 05 — The main theorem by direct diagrammatics (cone induction on the last slot)

Notation and standing facts are those of `../PROOFS.md` (section 1) and of
`README_FOR_PROVERS.md`; the tree formula is that of `../NOTES.md` section 2, with the signs of
`../formula.py`. All scripts named below are in `proofs/checks/` and are run with
`sage -python`.

## 0. Results

**Theorem B (= target (T1); section 8).** Let `Γ` be a simply-laced Dynkin diagram (only "`Γ` is
a forest" is used), `k ∈ Γ` any red colour, and `w` a word in which **every colour occurs at most
twice**. Then the tree formula for `w` satisfies `d² = 0`.
This covers `sl_n` with `ω_1`, `ω_2`, the vector representation of `so_{2n}`, all of `D_4`, and
every word, minuscule or not, with colour multiplicities `≤ 2`.

**Theorem C (section 9A).** If one colour occurs three times in `w` and every other colour
at most once, the tree formula satisfies `d² = 0`. (The first case with multiplicity three;
here dots have to be cancelled by hand, with the four-term identity of Proposition 10.2.)

**Theorem A (section 7).** For an arbitrary word `w'` and a colour `c` not occurring in `w'`:
the elementary move `X_m` of the new last slot commutes with every dot-free diagram of the
algebra of `w'`; hence if the tree formula of `w'` satisfies `d'² = 0`, so does that of `w'·c`.
(This reproves Theorem 5 of `PROOFS.md`.)

General tools, valid for **arbitrary words** (also cyclic ones):

* the Straightening Lemma 1.2 (when a non-reduced diagram equals the reduced one);
* the Sliding Lemma 2.2 with exact signs (the commutator of `X_m` with any diagram of `w'`);
* the cone decomposition with exact signs, Proposition 3.3;
* the Reduction Lemma 4.5: *if `d'² = 0`, the chain map equation for `F` has only to be
  checked on the components whose target vertex does not contain a fixed slot `a` that is the
  first occurrence of its colour*.

The Reduction Lemma is what makes Theorem B short: for colour multiplicities `≤ 2` all the
components of `d²` in which dots appear (the four-term cancellations of bigon polynomials seen
in the data) are among the components that need not be checked; on the remaining ones every
product of two tree terms is a single reduced diagram and the cancellation is in pairs
(section 8).

What is **not** proved here: Conjecture C1′ for general words with a colour occurring three
times or more (target (T2)); only the family of Theorem C is done. Section 10 says exactly which steps use "at most twice", what the data
look like beyond it, and which identities remain.

Method. Before proving anything, every product of two tree terms occurring in `d²` was
expanded in the basis for all words with colour multiplicities `≤ 2` of length `≤ 6` in
`A_3`, and for samples in `A_4`, `D_4` (`05_explore.py`, `05_explore_stats.py`):
of 73 164 products in `A_3`, 66 876 are a single basis diagram and 6 288 have two basis terms;
the basis diagrams cancel in groups of 2, of 4 and of 8, the groups of 4 and 8 being sums of
bigon polynomials `t x + t' x'` with the dots on different strands (first example: `A_3`,
word `132213`, `I = {3,4,5}`, `K = {1}`). Restricted to the components singled out by the
Reduction Lemma, all products are single diagrams without dots and all groups have size 2
(sections 8 and 10).

---

## 1. Words of crossings; the Straightening Lemma

### 1.1 Words and strands

A *state* `s` is a sequence of `n` strands, one of them red (colour `k`), the others black
with colours in `Γ`. A *word* from `s` is a sequence `D = (i_1, …, i_l)` with
`1 ≤ i_t ≤ n−1`; its element is `ψ_D = e_s ψ_{i_1} ⋯ ψ_{i_l}` (product = first factor first;
no dots). The letter `i_t` crosses the two strands which are at the positions `i_t`, `i_t+1`
after the letters `i_1, …, i_{t−1}`; we call them the *left* and the *right* strand of the
letter. For two strands `A ≠ B` let `n_D(A, B)` be the number of letters of `D` crossing `A`
and `B`. `D` is *reduced* if `n_D(A, B) ≤ 1` for all pairs; this is the usual notion for the
permutation `σ_D` of the positions, and then `n_D(A,B) = 1` exactly for the pairs whose order
is reversed (the *inversions*).

We use three moves on words:

* (M1) `(…, i, j, …) → (…, j, i, …)` for `|i − j| ≥ 2`;
* (M2) `(…, i, i+1, i, …) ↔ (…, i+1, i, i+1, …)`;
* (M3) `(…, i, i, …) → (…, …)` (deletion of two equal neighbouring letters).

(M1) and (M2) do not change the numbers `n_D(A,B)`: for (M2), if `A, B, C` are the strands at
the positions `i, i+1, i+2` before the three letters, both sides cross each of the pairs
`{A,B}`, `{A,C}`, `{B,C}` once. (M3) lowers one `n_D(A,B)` by 2. In particular the numbers
`n_D(A,B)` never increase.

**Lemma 1.1 (word property).** Every word can be transformed into a reduced word by a finite
sequence of moves (M1), (M2), (M3). Any two reduced words with the same permutation are
connected by moves (M1), (M2).

*Proof.* The second statement is (F3). For the first, let `D` be non-reduced and let
`D_0 = (i_1, …, i_{t−1})` be its longest reduced prefix, so that the strands `A, B` crossed by
`i_t` have `n_{D_0}(A,B) = 1`. Then `σ_{D_0} s_{i_t}` has one inversion less than `σ_{D_0}`.
Choose a reduced word `D_1` of `σ_{D_0} s_{i_t}`; then `D_1·(i_t)` is a reduced word of
`σ_{D_0}`, and by (F3) `D_0` is connected to it by moves (M1), (M2). Now
`D_1·(i_t)·(i_t)·(i_{t+1}, …)` admits a move (M3). The length has dropped by 2; induct on the
length. ∎

(This is Tits' solution of the word problem; it is stated because (F3) alone is about reduced
words only.)

### 1.2 The Straightening Lemma

**Lemma 1.2.** Let `D` be a word from the state `s`. Assume

* **(S1)** if `n_D(A,B) ≥ 2`, then either `A, B` are black of different, non-adjacent colours,
  or one of them is red and the other is black of a colour `≠ k`;
* **(S2)** there are no three strands `A, B, C` with `n_D(A,B), n_D(B,C), n_D(A,C) ≥ 1` such
  that `A, C` are black of the same colour and either `B` is black of an adjacent colour, or
  that colour is `k` and `B` is the red strand.

Then `ψ_D = ψ_{D'}` for **every** reduced word `D'` from `s` with `σ_{D'} = σ_D`. In
particular the element of a reduced word satisfying (S2) does not depend on the reduced word.

*Proof.* By Lemma 1.1 there is a sequence of moves from `D` to a reduced word `D_1`. Every
word `D*` in the sequence has `n_{D*} ≤ n_D`, so it satisfies (S1) and (S2). We check that no
move changes the element.
(M1) is relation (R1).
(M2) is applied to three strands `A, B, C` (at the positions `i, i+1, i+2`) which cross
pairwise in `D*`. By the list of relations in `README_FOR_PROVERS.md` the braid relation holds
on every state except `(c, c', c)` with `c ∼ c'` and `(k, W_k, k)`. In both exceptions the two
outer strands are black of the same colour and the middle one is black of an adjacent colour
or red of the same colour; this is excluded by (S2).
(M3) is applied to two strands `A, B` with `n_{D*}(A,B) ≥ 2`; by (S1) and (R3) the double
crossing equals the identity.
Hence `ψ_D = ψ_{D_1}`. If `D'` is another reduced word with the same permutation, it is
connected to `D_1` by moves (M1), (M2) (Lemma 1.1), and these do not change the element for
the same reason (`n_{D_1} ≤ n_D`). ∎

The same argument gives two variants. They are not used in the proof of Theorem B; they
describe what happens in the components of `d²` that the proof avoids (section 10).

**Lemma 1.3 (bigons).** Let `D` be a word from `s` satisfying (S2). Call a pair `{A, B}` with
`n_D(A,B) ≥ 2` *of adjacent type* if `A, B` are black of adjacent colours, or one is red and
the other black of colour `k`; put `Q_{AB} = t_{c(A)c(B)} x_A + t_{c(B)c(A)} x_B`, resp.
`Q_{AB} = u_k x_A` (`A` the black strand), polynomials in the dots of the strands at the
state `s`. Assume that no two black strands of the same colour cross twice, and that no
strand of a pair of adjacent type crosses a strand of its own colour (`n_D(Z, Z') = 0` for
`Z` in such a pair and `c(Z') = c(Z)`). Then for every reduced word `D'` with `σ_{D'} = σ_D`

    ψ_D  =  ∏_{\{A,B\} of adjacent type} Q_{AB}^{⌊n_D(A,B)/2⌋} · ψ_{D'}.

*Proof.* Follow a sequence of moves from `D` to a reduced word (Lemma 1.1); all words `D*` in
it satisfy the hypotheses. (M1) and (M2) are exact as before. A move (M3) on a pair that is
not of adjacent type is exact by (R3) (the two strands have different colours, because
same-coloured strands do not cross twice). A move (M3) on a pair `{A,B}` of adjacent type
replaces the double crossing by `Q_{AB}`, with the dots of `A` and `B` at that place (R3).
Move these dots to the source: the letters before that place cross other strands (R1), or
they cross `A` or `B` with a strand of a different colour (R2), because `A` and `B` cross no
strand of their own colour in `D*`. So `ψ_{D*} = Q_{AB} ψ_{D**}` for the word `D**` after the
move. The number of moves (M3) applied to `{A,B}` is `⌊n_D(A,B)/2⌋`. The independence of the
reduced word is Lemma 1.2. ∎

**Lemma 1.4 (vanishing).** Let `D` satisfy (S2), and assume that every pair with
`n_D(A,B) ≥ 2` either is as in (S1) or consists of two black strands of the same colour. If
there is a pair of the second kind, then `ψ_D = 0`.

*Proof.* In a sequence of moves from `D` to a reduced word, all moves (M1), (M2) and the moves
(M3) on pairs as in (S1) are exact. A pair of black strands of the same colour with
`n_D ≥ 2` has `n ≤ 1` in the final word, so at some point a move (M3) is applied to such a
pair; consider the first time. The word before this move has the element `ψ_D` and contains
`ψ_i ψ_i` on two strands of the same colour, which is `0` by (R3). ∎

*Check.* `05_straighten.py 3000 1`: among 3000 random words on random states (types `A_3`,
`A_4`, `D_4`, 3–6 strands) 1371 satisfy (S1), (S2), 863 of them non-reduced; all are equal to
two random reduced words of their permutation. Of the 1629 words violating (S1) or (S2), 1589
are different from the reduced word, so the hypotheses are not vacuous. Lemma 1.4: 381 words,
all zero. Lemma 1.3: 868 words, all equal to the predicted product.

---

## 2. The Sliding Lemma

Let `s` be a state on `n` strands and `c ∈ Γ`. Write `s·c` (resp. `c·s`) for the state with
one more black strand of colour `c` at the far right (resp. far left); call this strand `M`.
Put

    X_s = e_{s·c} ψ_n ψ_{n−1} ⋯ ψ_1  ∈ e_{s·c} A e_{c·s}

(`M` crosses all strands of `s`, from the right to the left). For a word `β` from `s` to `s'`
write `β^R` for the same word read from `s·c` and `β^L` for the word with all letters raised
by 1, read from `c·s`.

**Lemma 2.1 (one letter).** Let `1 ≤ i ≤ n−1`, let `S, S'` be the strands of `s` at the
positions `i, i+1`, and `s^i` the state after crossing them. Then

    e_{s·c} ψ_i · X_{s^i}  =  X_s · ψ_{i+1}  +  γ · e_{s·c} (ψ_n ψ_{n−1} ⋯ ψ_{i+2}) (ψ_{i−1} ψ_{i−2} ⋯ ψ_1),

where

* `γ = r_c t_{c, c(S')}` if `S` is black of colour `c` and `S'` is black with `c(S') ∼ c`;
* `γ = r_k u_k` if `c = k`, `S` is black of colour `k` and `S'` is the red strand;
* `γ = 0` otherwise.

*Proof.* `ψ_i` commutes with `ψ_n, …, ψ_{i+2}` (R1), so
`ψ_i X_{s^i} = ψ_n ⋯ ψ_{i+2} · (ψ_i ψ_{i+1} ψ_i) · ψ_{i−1} ⋯ ψ_1`. When the three middle letters
act, the strands at the positions `i, i+1, i+2` are `S, S', M` (`M` has been brought to the
position `i+2` by `ψ_n ⋯ ψ_{i+2}`). On this state
`ψ_i ψ_{i+1} ψ_i = ψ_{i+1} ψ_i ψ_{i+1} + γ' e`, where `γ' ≠ 0` only on the states
`(c, c', c)`, `c' ∼ c`, where `γ' = r_c t_{cc'}`, and `(k, W_k, k)`, where `γ' = r_k u_k`
(README, conventions). As the third strand has colour `c`, `γ' = γ`. Finally
`ψ_n ⋯ ψ_{i+2} ψ_{i+1} ψ_i ψ_{i+1} ψ_{i−1} ⋯ ψ_1 = ψ_n ⋯ ψ_1 ψ_{i+1}` because `ψ_{i+1}`
commutes with `ψ_{i−1}, …, ψ_1`. ∎

In the error term the strand `M` comes in from the far right and stops immediately to the
right of `S'`; `S'` does not move; `S` leaves to the far left. So after the error term the
strand `M` stands where `S` would stand after the crossing (to the right of `S'`), and `S`
stands where `M` would stand (far left).

**Lemma 2.2 (Sliding Lemma).** Let `β = (i_1, …, i_l)` be any word (reduced or not) from `s`
to `s'`. For `1 ≤ t ≤ l` let `γ_t` be the constant of Lemma 2.1 for the letter `i_t` (computed
from its left strand `S_t` and its right strand `S'_t`); call the letter *bad* if `γ_t ≠ 0`.
Then

    ψ_{β^R} · X_{s'}  −  X_s · ψ_{β^L}   =   Σ_{t bad}  γ_t · ψ_{E_t},

    E_t  =  (i_1, …, i_{t−1}) · (n, n−1, …, i_t+2) · (i_t−1, …, 1) · (i_{t+1}+1, …, i_l+1)

(a word from `s·c`).

*Proof.* Induction on `l`; `l = 0` is trivial. Write `β = β_0·(i_l)`, with `s''` the state
before the last letter. By Lemma 2.1 and then the induction hypothesis for `β_0`,

    ψ_{β^R} X_{s'} = ψ_{β_0^R} ( X_{s''} ψ_{i_l+1} + γ_l ψ_{(n,…,i_l+2)(i_l−1,…,1)} )
                  = ( X_s ψ_{β_0^L} + Σ_{t<l} γ_t ψ_{E_t^{(0)}} ) ψ_{i_l+1} + γ_l ψ_{E_l},

where `E_t^{(0)}` is the word `E_t` for `β_0`; `E_t^{(0)}·(i_l+1) = E_t`. ∎

*Remarks.* (1) A letter is bad only if its **left** strand is black of colour `c`. (2) Nothing
is assumed about `β`; in particular the lemma holds for the diagrams of arbitrary words `w'`.
(3) The sign is `+` in the stated form.

*Check.* `05_sliding.py 300 1`: the eight elementary cases and 300 random words (not reduced
in general, the red strand in arbitrary position, types `A_2`, `A_3`, `D_4`), 142 bad letters
in total; the identity holds exactly in all cases.

---

## 3. The cone decomposition (arbitrary words)

Let `w = w'·c` have `m` slots, `n = m − 1`; `A` is the algebra of `w`, `A'` that of `w'`
(same red colour). Vertices of the cube of `w'` are written `I', J', K' ⊆ {1,…,n}`.

**Lemma 3.1 (adding a strand).** For a diagram `x` of `A'` let `ι_R(x)` be the same diagram
with one more black strand of colour `c` at the far right, and `ι_L(x)` the same diagram with
one more strand at the far left (positions raised by 1). Then `ι_R, ι_L` are well defined
maps `e_s A' e_{s'} → e_{s·c} A e_{s'·c}`, resp. `→ e_{c·s} A e_{c·s'}`, they are
multiplicative and `R`-linear, and they are injective.

*Proof.* The defining relations (R1)–(R4) are local: each of them holds in `A` verbatim with
an additional vertical strand far to the left or right. So the maps are well defined and
multiplicative. Injectivity: by (F1), `e_s A' e_{s'}` has the basis
`x^α β_σ`. The diagram `ι(β_σ)` is reduced and realises the bijection `ι(σ)` (the new strand
is fixed), so by (F1) for `A`, with `β_{ι(σ)} := ι(β_σ)`, the elements
`ι(x^α β_σ) = x^α ι(β_σ)` are part of a basis. ∎

The idempotent of a vertex `I` of `w` is `e_{I'}·c` if `I = I' ∪ {m}` (the strand of `m` is at
the position `+m`, far right) and `c·e_{I}` if `m ∉ I` (position `−m`, far left).

**Lemma 3.2 (the terms of `d` in terms of `w'`).** Let `I` be a vertex of `w`.

1. If `m ∉ I`: `(j, F)` is admissible at `I` for `w` iff it is admissible at `I` for `w'`.
   Target, monomial and sign `ε` are the same, and if `D'` is a reduced word of the
   configuration for `w'`, then `D'^L` is one for `w`.
2. If `m ∈ I`, `I' = I ∖ {m}`, the admissible `(j, F)` at `I` are of three kinds:
   * (α) `j < m` and no pair of `F` contains `m`. These are exactly the admissible
     configurations at `I'` for `w'`; the target is `J' ∪ {m}`, the monomial is the same,
     **the sign is the opposite one**, and if `D'` is a reduced word of the configuration for
     `w'` then `D'^R` is one for `w`.
   * (β) `j = m`. Then `F = ∅`, the sign is `+1`, the target is `I'`, and the diagram is
     `X_{I'} := X_s` for `s` the state of `e_{I'}` (Lemma 1 of `PROOFS.md`).
   * (γ) `j < m` and `F` contains a pair `(a, m)`, with `c(a) = c`, `a ∉ I`. The target does
     not contain `m` and contains `a`.

*Proof.* Admissibility, links, connectors and targets are defined through the relative order
and the colours of the slots involved. 1. `m ∉ I` cannot be a mover or a `b_p`, nor an `a_p`
(`a_p < b_p ≤ m`). The sequence used for `κ` is the same. The strand of `m` stays at `−m`, to
the left of all others. 2. (α): the strand of `m` stays at `+m`. In the definition of `κ` the
sequence `I` is the sequence `I'` followed by `m`, and `m` stays in place; so there is one
more entry to the right of `j`, the sorting permutation is unchanged, and `κ = −κ'`. The
factor `(−1)^{#{p : x_p ∈ I}}` is unchanged, since connectors are slots `< m` of the
configuration. (β): a pair linked to the mover `m` would need `a_p < m < b_p`; as the link
graph is connected, `F = ∅`; `κ = (−1)^0`. (γ): the remaining case; `(a, m) ∈ F` means
`a ∉ I`, `c(a) = c(m)`, and the target contains `a` and not `m`. ∎

**Convention.** For a general word the element `β(I; j, F)` is defined through a chosen
reduced word. We assume the choice compatible with Lemma 3.2: `β_w(I; j, F) = ι_L β_{w'}(I; j, F)`
in case 1 and `= ι_R β_{w'}(I'; j, F)` in case (α). (If the element does not depend on the
reduced word — Lemma 6.2 below for colour multiplicities `≤ 2` — this is automatic; for the
basis braids of `klrw` it is checked in `05_cone.py`.)

Let `d'_{I',J'} ∈ e_{I'} A' e_{J'}` be the components of the tree differential of `w'`, and
define, for `I', K' ⊆ {1,…,n}`,

    F_{I',K'}  =  δ_{I',K'} · X_{I'}  +  F_1{}_{;I',K'}   ∈ e_{I' ∪ m} A e_{K'},

where `F_1{}_{;I',K'}` is the sum of `coefficient · β` over the terms of kind (γ) at
`I' ∪ {m}` with target `K'`, and

    E_{I',K'}  =  Σ_{J'} ι_R(d'_{I',J'}) · F_{J',K'}  −  Σ_{J'} F_{I',J'} · ι_L(d'_{J',K'}).

**Proposition 3.3 (cone).** `d² = 0` for `w` if and only if `d'² = 0` for `w'` and
`E_{I',K'} = 0` for all `I', K'`.

*Proof.* Let `I, K` be vertices of `w`. By Lemma 3.2 the component `(d²)_{I,K}` is:
`ι_L((d'²)_{I,K})` if `m ∉ I` (then `m ∉ K`); `ι_R((d'²)_{I',K'})` if `m ∈ I`, `m ∈ K`
(two factors `−ι_R`); and for `m ∈ I`, `m ∉ K`: a path `I → J → K` has `m ∈ J` (first step of
kind (α), second of kind (β) or (γ)) or `m ∉ J` (first step (β) or (γ), second as in 1.), so

    (d²)_{I' ∪ m, K}  =  −Σ_{J'} ι_R(d'_{I',J'}) F_{J',K}  +  Σ_{J'} F_{I',J'} ι_L(d'_{J',K})  =  −E_{I',K}.

`ι_L`, `ι_R` are injective and multiplicative (Lemma 3.1). ∎

*Check.* `05_cone.py A 3 6 40 1`, `05_cone.py D 4 6 30 2`: for 70 random words (arbitrary,
random red colour) all statements of Lemma 3.2, including the signs and the equality of the
diagrams (basis braid of `w'` with a strand added = basis braid of `w`, as elements) hold for
all 5062 terms.

---

## 4. The Reduction Lemma (arbitrary words)

Throughout this section `w = w'·c` is arbitrary and `d'² = 0` is assumed.

**Lemma 4.1 (the defect is a cocycle).** For all `I', K'`:

    Σ_{J'} ι_R(d'_{I',J'}) · E_{J',K'}  +  Σ_{J'} E_{I',J'} · ι_L(d'_{J',K'})  =  0.

*Proof.* Insert the definition of `E`. The two mixed sums
`∓ Σ ι_R(d') F ι_L(d')` cancel, and what remains is
`Σ ι_R((d'²)_{I',J'}) F_{J',K'} − Σ F_{I',J'} ι_L((d'²)_{J',K'}) = 0`. ∎

Let `𝔪 ⊂ R` be the ideal generated by all `r_e`, `e ∈ Γ`.

**Lemma 4.2 (a strand that goes and comes back).** Let `a` be a slot of `w'` which is the
**first occurrence of its colour** `e = c(a)` in `w'`, and let `K''` be a vertex of `w'`
containing `a`. Let `μ` be `ι_L(m_a(K''))` or `ι_R(m_a(K''))`, where `m_a(K'')` is the
elementary move of `a` (Lemma 1 of `PROOFS.md`: the strand of `a` crosses the red strand and
the strands of the slots `y < a`; its reduced word is `(p−1, p−2, …, p−l)`, `p` the position
of `a`), and let `μ^*` be the same word read backwards. Then

    μ · μ^*  =  P · e_0,        μ^* · μ  =  P · e_1,        P  =  ∏_y Q_y   ∈ R[x],

where `e_0`, `e_1` are the idempotents at which `μ` starts and ends, the product is over the
strands `y` crossed, and `Q_y = u_k x_a` if `y` is red and `e = k`;
`Q_y = t_{e,c(y)} x_a + t_{c(y),e} x_y` if `y` is black with `c(y) ∼ e`; `Q_y = 1` otherwise
(`x_a`, `x_y` are the dots on these strands at `e_0`, resp. at `e_1`). `P` is a non-zero
polynomial in which no `r` occurs.

*Proof.* Let `y_1, …, y_l` be the strands crossed, in this order. All `y_i` are red or black
of a colour `≠ e`, because `a` is the first occurrence of `e` in `w'`, and the additional
strand of `ι_L`, `ι_R` (which may have the colour `e`) is not crossed. So the word
`μ·μ^* = (p−1, …, p−l, p−l, …, p−1)` satisfies the hypotheses of Lemma 1.3: only the pairs
`{a, y_i}` cross (twice each), there are no three pairwise crossing strands, and the strand
of `a` and the `y_i` cross no strand of their own colour. Its reduced word is empty, and
Lemma 1.3 gives `μ μ^* = ∏ Q_{a y_i} e_0`; by (R3) `Q_{a y}` is the `Q_y` of the statement.
The same applies to the word `μ^*·μ = (p−l, …, p−1, p−1, …, p−l)` from `e_1`. Each `Q_y` is
a non-zero element of the domain `R[x]`. ∎

**Lemma 4.3 (injectivity modulo `𝔪`).** In the situation of Lemma 4.2 let `e'` be any
idempotent and `N ≥ 0`.

1. If `z ∈ e' A e_0` and `z μ ∈ 𝔪^{N+1} A`, then `z ∈ 𝔪^{N+1} A`.
2. If `z ∈ e_1 A e'` and `μ z ∈ 𝔪^{N+1} A`, then `z ∈ 𝔪^{N+1} A`.

*Proof.* 1. `z P = z μ μ^* ∈ 𝔪^{N+1} A`. By the standing fact quoted in the README (the
algebra is free as a right module over the polynomial ring in the dots; this is the mirror
image of (F1)), `e' A e_0` is a free right module over `R[x]` (dots at the right end); let
`(b_σ)` be a basis and write `z = Σ b_σ f_σ`. Then
`𝔪^{N+1} e' A e_0 = ⊕ b_σ 𝔪^{N+1} R[x]`, so `f_σ P ∈ 𝔪^{N+1} R[x]` for all `σ`. Grade `R[x]` by the total degree in the variables `r`; `𝔪^{N+1} R[x]` consists of the
elements whose components of degree `≤ N` vanish. If `f_σ ∉ 𝔪^{N+1} R[x]`, let `g ≠ 0` be its
component of lowest degree `d ≤ N`; as `P` has degree 0, the component of degree `d` of
`f_σ P` is `g P ≠ 0`, a contradiction.
2. `P z = μ^* μ z ∈ 𝔪^{N+1} A`, and `e_1 A e'` is a free left `R[x]`-module with basis the
`β_σ` ((F1) as stated in `PROOFS.md`); write `z = Σ f_σ β_σ` and argue as in 1. ∎

**Lemma 4.4 (the mover-`a` block).** Let `a` be any slot of `w'` and `K̃ ∌ a` a vertex. The
terms of `d'` from a vertex `J' ∋ a` to `K̃` are: the elementary move `±m_a(K̃ ∪ a)` from
`J' = K̃ ∪ {a}`, and terms with `F ≠ ∅`, whose coefficients lie in `𝔪`.

*Proof.* A term with `F = ∅` and mover `j` goes from `J'` to `J' ∖ j`; `a ∈ J'`,
`a ∉ J' ∖ j` forces `j = a`. A term with `F ≠ ∅` has a coefficient `± ∏ r t` with at least
one factor `r`. ∎

**Proposition 4.5 (Reduction Lemma).** Let `w = w'·c` be arbitrary, `d'² = 0`, and let `a`
be a slot of `w'` which is the first occurrence of its colour in `w'`. If

    E_{I',K'} = 0   for all I' and all K' with a ∉ K',

then `E = 0`, hence `d² = 0` for `w`.

*Proof.* Suppose `E ≠ 0`. `A` is a free `R`-module (F1), so `⋂_N 𝔪^N A = 0` and there is a
largest `N` with `E_{I',K'} ∈ 𝔪^N A` for all `I', K'`. Choose `(I', K'')` with
`E_{I',K''} ∉ 𝔪^{N+1} A`; then `a ∈ K''`. Put `K̃ = K'' ∖ {a}` and apply Lemma 4.1 to
`(I', K̃)`. The first sum vanishes because `a ∉ K̃`. In the second sum the terms with
`a ∉ J'` vanish, and by Lemma 4.4 the others give

    ± E_{I',K''} · ι_L(m_a(K''))  +  (an element of 𝔪 · 𝔪^N A)  =  0.

So `E_{I',K''} · ι_L(m_a(K'')) ∈ 𝔪^{N+1} A`, and Lemma 4.3.1 gives `E_{I',K''} ∈ 𝔪^{N+1} A`, a
contradiction. The last statement is Proposition 3.3. ∎

*Remarks.* (1) Nothing but `d'² = 0`, (F1) and the shape of the tree formula at lowest order
is used; in particular `F` could be any family for which `E` is defined. (2) The hypothesis
"first occurrence" cannot be dropped in this argument: if a strand of colour `c(a)` is
crossed, `μ μ^* ∈ 𝔪 A` and right multiplication by a same-colour crossing is not injective.
(3) A more flexible form, not used for Theorem B:

**Proposition 4.6 (deduction rule).** Let `w = w'·c` be arbitrary, `d'² = 0`, `N ≥ 0` such
that all `E_{I',K'}` lie in `𝔪^N A`, and let `𝒵` be a set of pairs `(I', K')` with
`E_{I',K'} ∈ 𝔪^{N+1} A`. Let `s` be a slot which is the first occurrence of its colour in
`w'`, and `(I', K')` a pair.

* (r) If `s ∈ K'`, and `(I' ∖ j, K' ∖ s) ∈ 𝒵` for all `j ∈ I'`, and
  `(I', (K' ∖ s) ∪ j) ∈ 𝒵` for all `j ∉ K'`, then `E_{I',K'} ∈ 𝔪^{N+1} A`.
* (l) If `s ∉ I'`, and `((I' ∪ s) ∖ j, K') ∈ 𝒵` for all `j ∈ I'`, and
  `(I' ∪ s, K' ∪ j) ∈ 𝒵` for all `j ∉ K'`, then `E_{I',K'} ∈ 𝔪^{N+1} A`.

Consequently, if every pair `(I', K')` can be reached from a set `𝒵_0` of pairs with
`E_{I',K'} = 0` by finitely many applications of (r) and (l), then `E = 0`.

*Proof.* (r) Lemma 4.1 at `(I', K̃)`, `K̃ = K' ∖ s`, modulo `𝔪^{N+1} A`: the terms of `d'`
with `F ≠ ∅` have coefficients in `𝔪` and are multiplied with elements of `𝔪^N A`. The terms
with `F = ∅` are `± m_j`, from a vertex `V` to `V ∖ j`. So

    Σ_{j ∈ I'} ± ι_R(m_j(I')) E_{I' ∖ j, K̃}  +  Σ_{j ∉ K̃} ± E_{I', K̃ ∪ j} ι_L(m_j(K̃ ∪ j))  ∈ 𝔪^{N+1} A.

By hypothesis all summands except the one with `j = s` in the second sum lie in
`𝔪^{N+1} A`. Hence `E_{I',K'} ι_L(m_s(K')) ∈ 𝔪^{N+1} A`, and Lemma 4.3.1 applies.
(l) Lemma 4.1 at `(I' ∪ s, K')` gives in the same way

    Σ_{j ∈ I' ∪ s} ± ι_R(m_j(I' ∪ s)) E_{(I' ∪ s) ∖ j, K'}  +  Σ_{j ∉ K'} ± E_{I' ∪ s, K' ∪ j} ι_L(m_j(K' ∪ j))  ∈ 𝔪^{N+1} A,

all summands except the one with `j = s` in the first sum lie in `𝔪^{N+1} A`, so
`ι_R(m_s(I' ∪ s)) E_{I',K'} ∈ 𝔪^{N+1} A`, and Lemma 4.3.2 applies.
For the last statement suppose `E ≠ 0`, let `N` be the largest integer with all
`E_{I',K'} ∈ 𝔪^N A`, and apply (r), (l) starting with `𝒵 = 𝒵_0`: all components lie in
`𝔪^{N+1} A`, contradicting the choice of `N`. ∎

Proposition 4.5 is the case `𝒵_0 = \{(I', K') : a ∉ K'\}`, with one application of (r) for
each remaining pair. In particular the slot `a` may be chosen depending on `I'` (by induction
on `|I'|`).

*Check.* `05_reduction.py A 3 40 6 1`, `05_reduction.py D 4 25 6 2`: for 65 random words,
`μ μ^* = P e ≠ 0` with the stated `P` in all 2428 cases where `a` is a first occurrence, and
`μ μ^* ∈ 𝔪 A` in all 2148 cases where it is not; Lemma 4.4 in all 4576 cases.

---

## 5. Words with every colour at most twice: combinatorics

From here to section 9, `w` is a word in which every colour occurs at most twice (a **T1
word**). The two slots of a colour occurring twice are called *twins*. A *configuration* at a
vertex `I` is a mover `j ∈ I` and a set `F` of pairwise disjoint pairs `p = (a_p < b_p)` with
`c(a_p) = c(b_p)`, `a_p ∉ I`, `b_p ∈ I`, `b_p ≠ j`. Its *nodes* are the mover and the pairs;
`c(j)` and `c(p)` are their colours; its *link graph* has the links of `NOTES.md` §2 as
edges (a pair `p` is linked to the mover iff `a_p < j < b_p` and `c(p) ∼ c(j)`; two pairs are
linked iff they are interlaced and their colours are adjacent).

**Lemma 5.1.** The nodes of a configuration have pairwise different colours.

*Proof.* A pair uses both slots of its colour. So two pairs of the same colour would not be
disjoint, and a pair of the colour of the mover `j` would contain `j`, which is excluded
(`b_p ≠ j`, and `a_p ∉ I ∋ j`). ∎

**Lemma 5.2.** The link graph of a configuration is a forest, and the configuration is
admissible if and only if its link graph is connected.

*Proof.* Linked nodes have adjacent colours, so by Lemma 5.1 the map "node ↦ colour" embeds
the link graph into `Γ` (injective on nodes and on edges). `Γ` is a forest. The admissibility
conditions 1 and 2 of `NOTES.md` concern two nodes of the same colour and are empty by Lemma
5.1; condition 3 says that the link graph is a tree. ∎

**Lemma 5.3.** In an admissible configuration, two nodes with adjacent colours are linked.

*Proof.* The path joining them in the link graph maps to a path in `Γ` with pairwise
different vertices (Lemma 5.1) joining two adjacent vertices of the forest `Γ`; so it is a
single edge. ∎

**Lemma 5.4.** Let `c` be a colour which is not the colour of any node of an admissible
configuration. Then at most one node has a colour adjacent to `c`.

*Proof.* If two nodes had colours `c_1 ≠ c_2` adjacent to `c`, the path joining them in the
link graph would map to a path in `Γ` from `c_1` to `c_2` avoiding `c`; together with
`c_1 — c — c_2` this is a cycle in `Γ`. ∎

## 6. The diagrams of tree terms

**Lemma 6.1 (inversions of a configuration; arbitrary words).** Let `(I; j, F)` be a
configuration with target `J`. The strands are the *mover* (from `+j` to `−j`), for each pair
`L_p` (from `−a_p` to `−b_p`) and `R_p` (from `+b_p` to `+a_p`), the red strand, and the
*unaffected* strands (fixed). The pairs of strands whose order is reversed are exactly:

* (a) mover and red strand;
* (b) mover and an unaffected `y` (on either side), iff `y < j`;
* (c) mover and `L_p`, and mover and `R_p`, iff `b_p < j`;
* (d) `L_p` and an unaffected `y ∉ I`, iff `a_p < y < b_p`; `R_p` and an unaffected `y ∈ I`,
  iff `a_p < y < b_p`;
* (e) `L_p` and `L_q`, and `R_p` and `R_q`, iff `p, q` are nested (`a_p < a_q < b_q < b_p` or
  the same with `p, q` exchanged).

In every such crossing the strand moving towards the left is the mover in (a), (b), (c), and
`L_p`, `R_p` in (d). *Proof.* Items 1–6 of the proof of Proposition 3 in `PROOFS.md`. ∎

So a *diagram of the configuration* is a reduced word from `e_I` whose inversions are (a)–(e).

**Lemma 6.2 (T1: the element of a tree term is well defined).** Let `w` be a T1 word and
`(I; j, F)` an admissible configuration. Every reduced word of the configuration satisfies
(S2) of Lemma 1.2; hence all of them have the same element `β(I; j, F)`.

*Proof.* Two black strands of the same colour belong to twin slots. Among the strands of the
configuration, twins are: `L_p, R_p` for a pair; two unaffected strands; or the mover and an
unaffected strand `j̃` (not a strand of a pair, by Lemma 5.1). By Lemma 6.1 the first two
kinds are never inverted, and the mover and `j̃` are inverted iff `j̃ < j`. Let then `B` be a
black strand inverted with both and with `c(B) ∼ c(j)`. `B` is not unaffected (unaffected
strands are not inverted with `j̃`), so `B ∈ {L_p, R_p}`, and by (c) `b_p < j`. But
`c(p) ∼ c(j)` implies by Lemma 5.3 that `p` is linked to the mover, i.e. `a_p < j < b_p`.
Contradiction. For the second clause of (S2): only the mover is inverted with the red strand.
So (S2) holds, and Lemma 1.2 applies ((S1) is empty for a reduced word). ∎

*Check.* `05_T1_check.py` (see section 8) verifies Lemmas 5.1, 5.3, (S2) and the equality of
the basis braid with two random reduced words for every tree term of every word tested
(more than 1 100 000 terms).

*Warning.* For general words (S2) can fail for a tree diagram (e.g. `A_3`, word `3232332`,
`I = {5,6,7}`, mover 6, pairs `(1,5), (2,7)`); there Lemma 1.2 does not apply as it stands.

---

## 7. Theorem A: a new colour needs no correction (arbitrary words)

**Theorem A.** Let `w'` be an arbitrary word and `c` a colour not occurring in `w'`.

1. For every word `β` in the crossings of `A'`, from a state `s` to a state `s'`:
   `ψ_{β^R} X_{s'} = X_s ψ_{β^L}`.
2. For `w = w'·c` there are no terms of kind (γ), `F = X`, and `E = 0`. Hence `d'² = 0`
   implies `d² = 0`.

*Proof.* 1. By Remark (1) after Lemma 2.2 a bad letter needs a black strand of colour `c` in
`s`; there is none. 2. A term of kind (γ) needs a slot `a < m` with `c(a) = c`. So
`F_{I',K'} = δ_{I',K'} X_{I'}` and
`E_{I',K'} = ι_R(d'_{I',K'}) X_{K'} − X_{I'} ι_L(d'_{I',K'})`, which vanishes by 1., because
`d'_{I',K'}` is a linear combination of dot-free diagrams. Conclude with Proposition 3.3. ∎

**Corollary 7.1 (= Theorem 5 of `PROOFS.md`).** If the colours of `w` are pairwise distinct,
`d = δ₀` and `δ₀² = 0`.

*Proof.* Induction on the length: no pairs exist, so `d = δ₀`; Theorem A.2 gives the
induction step; for length `≤ 1`, `d²` has no components. ∎

---

## 8. Theorem B: every colour at most twice

**Theorem B.** If every colour occurs at most twice in `w`, the tree formula satisfies
`d² = 0`.

The proof is by induction on the length `m` of `w`; for `m ≤ 1` there is nothing to prove.
Let `w = w'·c`. The prefix `w'` is a T1 word, so `d'² = 0`. If `c` does not occur in `w'`,
Theorem A applies. So assume from now on:

> `c` occurs exactly once in `w'`, at the slot `a`. `A` denotes the strand of the slot `a`
> and `M` the strand of the slot `m` (they are twins).

By Proposition 3.3 we must show `E = 0`. The slot `a` is the first occurrence of its colour
in `w'`, so by the Reduction Lemma 4.5 it suffices to show

    E_{I',K'} = 0     for all I', and all K' with a ∉ K'.                                  (8.0)

Facts used repeatedly: `c` is the colour of no node of a configuration of `w'` (the slot `a`
has no twin in `w'`, so it is in no pair of `w'`; it can be a mover). In a term of kind (γ)
the pair containing `m` is `(a, m)`, and `a ∉ I'`.

### 8.1 The components with `a ∈ I'`, `a ∉ K'`

**Lemma 8.1.** Let `a ∈ I'` and `t = (a, F_a)` an admissible configuration of `w'` at `I'`
(mover `a`), with target `K'`. Then `ι_R(β(t)) X_{K'} = X_{I'} ι_L(β(t))`.

*Proof.* Take a reduced word of `t` and apply the Sliding Lemma. A bad letter has a left
strand which is black of colour `c`, i.e. the strand `A`, which is the mover. By Lemma 6.1
(a)–(c) the mover moves towards the left in each of its crossings, so it is the right strand
of each of them. There is no bad letter. ∎

**Corollary 8.2.** `E_{I',K'} = 0` if `a ∈ I'`, `a ∉ K'`.

*Proof.* A term of kind (γ) starts at a vertex not containing `a` and ends at a vertex
containing `a`. So in `Σ ι_R(d'_{I',J'}) F_{J',K'}` only `F_{K',K'} = X_{K'}` contributes
(`F_1{}_{;J',K'} = 0` because `a ∉ K'`), and in `Σ F_{I',J'} ι_L(d'_{J',K'})` only
`F_{I',I'} = X_{I'}` (`F_1{}_{;I',J'} = 0` because `a ∈ I'`). A term of `d'` from `I' ∋ a`
to `K' ∌ a` has mover `a`, since `a` is in no pair. So
`E_{I',K'} = Σ_t coef'(t) ( ι_R(β(t)) X_{K'} − X_{I'} ι_L(β(t)) ) = 0` by Lemma 8.1. ∎

### 8.2 The components with `a ∉ I'`, `a ∉ K'`: the three kinds of contributions

Fix `I' ∌ a`, put `I = I' ∪ {m}`, and let `K' ∌ a`. As in the proof of Corollary 8.2:

* in `Σ ι_R(d'_{I',J'}) F_{J',K'}` only `J' = K'`, `F = X` contributes;
* in `Σ F_{I',J'} ι_L(d'_{J',K'})` the term `J' = I'`, `F = X` contributes, and the terms
  `F_1{}_{;I',J'} · ι_L(d'_{J',K'})`, where `a ∈ J'`, so that the term of `d'` has mover `a`.

Call a term of `d'` at `I'` *frozen* (the slot `a ∉ I'` is unaffected in it: it is neither
mover nor in a pair). Then

    E_{I',K'} =   Σ_{t' frozen, I' → K'} coef'(t') ( ι_R(β(t')) X_{K'} − X_{I'} ι_L(β(t')) )
               −  Σ_{(t_1, t_2)} coef(t_1) coef(t_2) · β(t_1) · β(t_2),                      (8.1)

the second sum over the pairs with `t_1 = (j, F)` a term of `d` (for `w`) at `I` with
`(a, m) ∈ F`, target `J`, and `t_2 = (a, F_a)` a term at `J` with mover `a` and target `K'`
(a term of `w'`, equivalently of `w`, by Lemma 3.2.1; `β(t_2)` is taken in `A`).

**Lemma 8.3 (bad letters of a frozen term).** Let `t' = (j, F')` be a frozen term at `I'`.
By Lemma 5.4 at most one node of `t'` has a colour adjacent to `c`; call it `N` if it exists.
Say that `a` is *inside* `N` if `N = j` and `a < j`, or `N = p` and `a_p < a < b_p`. A reduced
word of `t'` has exactly one bad letter if `N` exists and `a` is inside `N`, and none
otherwise. The bad letter is the crossing of `A` with the strand `S'` of `N` which crosses
`A` (the mover if `N = j`, `L_p` if `N = p`), and its constant is `γ = r_c t_{c, c(N)}`.

*Proof.* The left strand of a bad letter is black of colour `c`, hence `A`, which is
unaffected and sits at `−a`; the red strand is not crossed by `A`, so the right strand `S'`
is black with `c(S') ∼ c`. By Lemma 6.1 the strands inverted with `A` are the mover, iff
`a < j`, and the `L_p` with `a_p < a < b_p`; they move towards the left, so `A` is the left
strand of these letters. Hence the bad letters are the crossings of `A` with the strands of
this list whose colour is adjacent to `c`, and by Lemma 5.4 (applicable since `c` is not the
colour of a node of `t'`) there is at most one such node. ∎

**Lemma 8.4 (the pairs `(t_1, t_2)`).** Call *datum* an admissible configuration
`U = (j, F_U)` of `w` at `I` with `(a, m) ∈ F_U`. Remove the node `(a,m)` from its link graph
(a tree); let `{j} ∪ F'` be the component containing `j`, let `T_1, …, T_k` (`k ≥ 0`) be the
other components, and `𝒯 = T_1 ⊔ … ⊔ T_k`. We also write `U = (j, F', 𝒯)`.

1. `(j, F')` is a frozen term of `d'` at `I'`; exactly one node of `{j} ∪ F'` is linked to
   `(a,m)`; it is the node `N` of Lemma 8.3, and `a` is inside `N`. Each `T_i` contains
   exactly one pair `q_i` linked to `(a,m)`, and `a_{q_i} < a < b_{q_i}`, `c(q_i) ∼ c`.
2. For `S ⊆ {1,…,k}` put `𝒯_S = ⋃_{i ∈ S} T_i`, `𝒯_{S^c} = 𝒯 ∖ 𝒯_S`. Then
   `t_1(S) = (j, F' ∪ {(a,m)} ∪ 𝒯_S)` is admissible at `I`, and `t_2(S) = (a, 𝒯_{S^c})` is
   admissible at the target `J(S)` of `t_1(S)`; the target of `t_2(S)` is the same vertex
   `K'_U = ( I ∖ ({j, m} ∪ \{b_p\}) ) ∪ \{a_p\}` (`p` running through `F' ∪ 𝒯`) for all `S`.
3. `(U, S) ↦ (t_1(S), t_2(S))` is a bijection from the set of pairs (datum, subset) onto the
   set of pairs `(t_1, t_2)` of (8.1) (all targets `K'` together).
4. `U ↦ (j, F')` is a bijection from the data with `𝒯 = ∅` onto the frozen terms at `I'` for
   which `N` exists with `a` inside `N`; it preserves the target.

*Proof.* 1. If a node is removed from a tree, each component of the rest contains exactly one
neighbour of that node. A neighbour `x` of `(a,m)` has `c(x) ∼ c`; if `x = j` the link means
`a < j < m`; if `x = q` is a pair, it is interlaced with `(a,m)`, and as `m` is the largest
slot this means `a_q < a < b_q < m`. In both cases `a` is inside `x`. The pairs of `F'` do
not contain `m` or `a` (they are disjoint from `(a,m)`), and `j ≠ a` because `a ∉ I`; so
`(j, F')` is a configuration of `w'` at `I'` in which `a` is unaffected; its link graph is
connected, so it is admissible (Lemma 5.2), i.e. a frozen term. Its node linked to `(a,m)`
has a colour adjacent to `c`, so it is `N` (Lemma 8.3).
2. `t_1(S)`: its nodes form a connected subgraph of the tree of `U` (links between two nodes
depend only on these two nodes), and its pairs are pairs at `I`; admissible by Lemma 5.2.
`t_2(S)`: the slots of `𝒯_{S^c}` are untouched by `t_1(S)`, so its pairs are pairs at `J(S)`;
`a ∈ J(S)`; each `q_i` with `i ∉ S` is linked to the mover `a`, because
`a_{q_i} < a < b_{q_i}` and `c(q_i) ∼ c`; the links inside `T_i` are those of `U`; so the link
graph is connected and `t_2(S)` is admissible by Lemma 5.2. The target is computed from the
definitions.
3. *Injectivity:* `U = (j, F_{t_1} ∪ F_{t_2})` and `S = \{i : T_i ⊆ F_{t_1}\}` are determined by
the pair. *Surjectivity:* let `t_1 = (j, F)`, `(a,m) ∈ F`, with target `J`, and
`t_2 = (a, F_a)` at `J`.
*Claim: the nodes of `{j} ∪ F ∪ F_a` have pairwise different colours.* Inside `{j} ∪ F` and
inside `{a} ∪ F_a` this is Lemma 5.1; in particular `c(q) ≠ c` for `q ∈ F_a`. Let
`x ∈ {j} ∪ F`, `x ≠ (a,m)`, and `q ∈ F_a` have the same colour `e`.
If `x = p` is a pair, then `q` consists of the same two slots, so `a_q = a_p`; but `a_p ∈ J`
and `a_q ∉ J`.
If `x = j`: `q` consists of `j` and its twin `j̃`; as `j ∉ J`, `a_q = j`. Let
`a = q_0, q_1, …, q_l = q` be the path from the mover to `q` in the tree of `t_2`, and
`(a,m) = p_0, p_1, …, p_{l'} = j` the path from `(a,m)` to `j` in the tree of `t_1`. Their
colours form two paths in `Γ` from `c` to `e` without repeated vertices (Lemma 5.1), so they
coincide: `l = l'`, `c(p_i) = c(q_i)`. If `l ≥ 2`, `p_1` is a pair of `F` with the colour of
`q_1 ∈ F_a`, which has just been excluded. If `l = 1`, `(a,m)` is linked to `j`, so `a < j`,
and `q` is linked to the mover `a` of `t_2`, so `a_q = j < a`. Contradiction. The claim is
proved.
Consequently the pairs of `F_a` are disjoint from the slots of `F` and from `j`; their slots
are untouched by `t_1`, so they are pairs at `I`, and `U = (j, F ∪ F_a)` is a configuration at
`I`. A pair `q ∈ F_a` is linked to the mover `a` of `t_2` iff `a_q < a < b_q` and
`c(q) ∼ c`, iff it is linked to `(a,m)` in `U`. So the link graph of `U` contains the tree of
`t_1` (on `{j} ∪ F`, with `|F|` edges) and a copy of the tree of `t_2` (on
`{(a,m)} ∪ F_a`, with `|F_a|` edges); these two edge sets are disjoint, because the two node
sets share only `(a,m)`. Hence `U` is connected, so admissible (Lemma 5.2), so a tree with
`|F| + |F_a|` edges: there are **no other edges**, i.e. no link between a node of
`({j} ∪ F) ∖ (a,m)` and a pair of `F_a`. Therefore every component of `U ∖ (a,m)` lies
entirely in `{j} ∪ F` or entirely in `F_a`; the component of `j` lies in `{j} ∪ F`. With
`S = \{i : T_i ⊆ F\}` we get `F = F' ∪ {(a,m)} ∪ 𝒯_S` and `F_a = 𝒯_{S^c}`.
4. For a datum with `𝒯 = ∅`, `(j, F')` is such a frozen term by 1. Conversely, for such a
frozen term, `U = (j, F' ∪ {(a,m)})` is a configuration at `I` (`a ∉ I`, `m ∈ I`), in which
`(a,m)` is linked to `N` (if `N = j`: `a < j < m`; if `N = p`: `a_p < a < b_p < m`); so it is
connected, hence a datum, and removing `(a,m)` leaves only `{j} ∪ F'`. The target of `U` is
that of `(j,F')` (as a vertex of `w'`): `t_1` puts `a` to the right and `t_2 = m_a` removes
it. ∎

Summary of the link structure of a datum, used below:

* **(L1)** no pair of `𝒯` is linked to `j` or to a pair of `F'`; pairs in different `T_i` are
  not linked (they are different components of `U ∖ (a,m)`);
* **(L2)** two nodes of `U` with adjacent colours are linked (Lemma 5.3); two nodes of `U`
  have different colours (Lemma 5.1);
* **(L3)** `N` is the only node of `{j} ∪ F'` whose colour is adjacent to `c` (Lemma 5.4,
  for the admissible configuration `(j, F')`, none of whose nodes has the colour `c`).

### 8.3 All products of a datum are the same reduced diagram

For a datum `U = (j, F', 𝒯)` let `σ_U` be the following assignment of final positions to the
strands of `e_I`: mover `j ↦ −j`; `A` (at `−a`) `↦ −m`; `M` (at `+m`) `↦ −a`; for
`p ∈ F' ∪ 𝒯`: `L_p` (at `−a_p`) `↦ −b_p`, `R_p` (at `+b_p`) `↦ +a_p`; all other strands
fixed. The resulting arrangement is the idempotent of the vertex `K'_U` of Lemma 8.4.2.

**Lemma 8.5.** For a datum and every `S ⊆ {1,…,k}`, the concatenation `D(S)` of a reduced
word of `t_1(S)` and a reduced word of `t_2(S)` satisfies (S1) and (S2), and its permutation
is `σ_U`. Hence `β(t_1(S)) · β(t_2(S)) = B`, where `B = B(j, F', 𝒯)` is the element of any
reduced word of `σ_U` from `e_I`.

*Proof.* *Permutation.* In `t_1(S)`: `j ↦ −j`; `A = L_{(a,m)}` goes from `−a` to `−m`;
`M = R_{(a,m)}` from `+m` to `+a`; the pairs of `F' ∪ 𝒯_S` are exchanged. In `t_2(S)` the
mover is the strand at `+a`, i.e. `M`, which goes to `−a`, and the pairs of `𝒯_{S^c}` are
exchanged. Together: `σ_U`.

*Crossings.* Each factor is reduced, so `n_{D(S)} = n_1 + n_2` with `n_1, n_2 ∈ {0,1}` the
inversion indicators of the two factors, given by Lemma 6.1. For the second factor, at the
vertex `J = J(S)`, the mover is `M` (slot `a`), the pairs are `q ∈ 𝒯_{S^c}`, and the
unaffected strands are, with their slots at `J`: the strands unaffected in both factors
("original", slot `y`); the strand `j` (left, slot `j`); `A` (left, slot `m`); and for
`p ∈ F' ∪ 𝒯_S`: `L_p` (left, slot `b_p`), `R_p` (right, slot `a_p`).

*(S1).* We list all pairs with `n_2 = 1` and determine `n_1`.

* `M`, red: `n_1 = 0` (`M = R_{(a,m)}` does not cross the red strand in `t_1`).
* `M` and an original unaffected `y < a`: in `t_1`, `R_{(a,m)}` crosses an unaffected right
  strand only if `a < y < m`, and no left strand; `n_1 = 0`.
* `M` and `j`, if `j < a`: in `t_1` the mover crosses `R_{(a,m)}` iff `m < j`; `n_1 = 0`.
* `M` and `L_p` (`b_p < a`): `R` and `L` strands are never inverted; `n_1 = 0`.
* `M` and `R_p` (`a_p < a`), `p ∈ F' ∪ 𝒯_S`: in `t_1`, `R_{(a,m)}` and `R_p` cross iff
  `a < a_p < b_p < m` (the other nesting is impossible, `m` being the largest slot);
  `n_1 = 0`.
* `M` and `L_q`, `R_q` for `q ∈ 𝒯_{S^c}` with `b_q < a`: in `t_1` these are unaffected strands
  at `−a_q`, `+b_q`, and `b_q < a`; `n_1 = 0` as in the second item.
* `L_q` (`q ∈ 𝒯_{S^c}`) and an original unaffected left `y`, or `R_q` and an original
  unaffected right `y`: both strands are unaffected in `t_1`; `n_1 = 0`.
* `M` and `A`: `A` is unaffected at the slot `m > a`; not crossed (`n_2 = 0`).
* `L_q` and `A`: needs `a_q < m < b_q`; impossible.
* `L_q` and `j`, iff `a_q < j < b_q`: in `t_1` the mover crosses the unaffected strand at
  `−a_q` (as `a_q < j`), so `n_1 = 1`: **a double crossing**. By (L1) `q` is not linked to
  the mover although `a_q < j < b_q`; hence `c(q) ≁ c(j)`, and `c(q) ≠ c(j)` by (L2).
* `L_q` and `L_p` (`p ∈ F' ∪ 𝒯_S`), iff `a_q < b_p < b_q`: in `t_1`, `L_p` crosses the
  unaffected strand at `−a_q` iff `a_p < a_q < b_p`. So a **double crossing** iff
  `a_p < a_q < b_p < b_q`; then `p, q` are interlaced and not linked by (L1), so
  `c(p) ≁ c(q)`, and `c(p) ≠ c(q)` by (L2).
* `R_q` and `R_p` (`p ∈ F' ∪ 𝒯_S`), iff `a_q < a_p < b_q`: in `t_1`, `R_p` crosses the
  unaffected strand at `+b_q` iff `a_p < b_q < b_p`. A **double crossing** iff
  `a_q < a_p < b_q < b_p`; as before the colours are different and not adjacent.
* `L_q, L_{q'}` or `R_q, R_{q'}` nested, `q, q' ∈ 𝒯_{S^c}`: both unaffected in `t_1`;
  `n_1 = 0`.

By Lemma 6.1 for `t_2(S)` this list is complete. So (S1) holds.

*(S2).* Twins among the strands are: `{A, M}`; `{L_p, R_p}` for `p ∈ F' ∪ 𝒯`; two original
unaffected strands; `{j, j̃}` with `j̃` original unaffected (by (L2) the twin of `j` is not a
strand of a pair, and it is not `A` or `M`). `A, M` do not cross: `n_1 = 0` (`L`, `R` of a
pair), `n_2 = 0` (`A` is unaffected at the slot `m > a`). `L_p, R_p` do not cross: in the
factor where `p` is exchanged by Lemma 6.1, in the other one both are unaffected. Two
original unaffected strands do not cross. So the only twins that cross are `j, j̃` with
`j̃ < j` (once, in `t_1`). Let `Y` be a black strand with `c(Y) ∼ c(j)` crossing both.
`Y` is not original unaffected (it would not cross `j̃`). If `Y` is a strand of a pair
`p ∈ F' ∪ 𝒯`, then by (L2) `p` is linked to `j`, so by (L1) `p ∈ F'` and `a_p < j < b_p`;
then `n_1(Y, j) = 0` by Lemma 6.1 (c), and `n_2(Y, j) = 0` since both are unaffected in
`t_2`. If `Y ∈ {A, M}`, then `c ∼ c(j)` and by (L2) `(a,m)` is linked to `j`: `a < j < m`;
then `n_1(Y, j) = 0` by (c), `n_2(A, j) = 0` (both unaffected) and `n_2(M, j) = 0` (it would
need `j < a`). So there is no such `Y`. For the second clause of (S2): the strands crossing
the red strand are `j` and `M`, of different colours.
Lemma 1.2 gives the last statement. ∎

**Lemma 8.6 (the sliding error is the same diagram).** Let `t' = (j, F')` be a frozen term at
`I'` with a bad letter (Lemma 8.3), `β = (i_1, …, i_l)` a reduced word of `t'` (read from
`e_{I'}`, and from `e_I` as `β^R`), `t` the index of its bad letter, and `E_t` the word of
Lemma 2.2. Then `E_t` satisfies (S1), (S2) and has the permutation `σ_U` of the datum
`(j, F', ∅)`. Hence `ψ_{E_t} = B(j, F', ∅)`, and

    ι_R(β(t')) X_{K'} − X_{I'} ι_L(β(t'))  =  r_c t_{c,c(N)} · B(j, F', ∅).

*Proof.* `U = (j, F' ∪ {(a,m)})` is a datum with `𝒯 = ∅` (Lemma 8.4.4).

The strands of `e_{I'}` other than `A` are of four kinds, by Lemma 6.1 and Lemma 8.3: the
strand `S'` of the bad letter; the set `𝒞_1` of the strands crossing `A` in a letter before
`t`; the set `𝒞_2` of those crossing `A` in a letter after `t`; those never crossing `A`,
which are either always to the left of `A` (`𝒜_L`) or always to its right (`𝒜_R`, containing
the red strand). All strands of `𝒞_1 ∪ 𝒞_2 ∪ {S'}` pass `A` from the right to the left.

`E_t` consists of: the letters `i_1 … i_{t−1}`; then `M` crosses all strands standing to the
right of `S'`, i.e. `𝒜_R ∪ 𝒞_2`; then `A` crosses all strands standing to its left, i.e.
`𝒜_L ∪ 𝒞_1`; then the letters `i_{t+1}, …, i_l` raised by 1, in which `M` stands where `A`
stood in `β` after the letter `t` (remark after Lemma 2.1), and `A` is at the far left. So:

* for two strands different from `A, M`: `n_{E_t} = n_β ∈ {0,1}`;
* `n_{E_t}(A, M) = 0`;
* `n_{E_t}(A, Z) = 2` for `Z ∈ 𝒞_1`, `1` for `Z ∈ 𝒜_L`, `0` otherwise;
* `n_{E_t}(M, Z) = 2` for `Z ∈ 𝒞_2`, `1` for `Z ∈ 𝒜_R`, `0` otherwise (in particular `0` for
  `Z = S'`).

*Permutation.* `A` ends at the far left (`−m`); `M` ends where `A` ends in `β`, at `−a`; the
other strands end as in `β`. This is `σ_U`.
*(S1).* The pairs crossing twice are `(A, Z)`, `Z ∈ 𝒞_1`, and `(M, Z)`, `Z ∈ 𝒞_2`. Such a `Z`
is the mover or an `L_p` with `a` inside the node, and `Z ≠ S'`; by (L3) its colour is not
adjacent to `c`, and it is `≠ c`.
*(S2).* Twins crossing in `E_t`: not `A, M`; for the others `n_{E_t} = n_β`, so as in the
proof of Lemma 6.2 only `j, j̃` with `j̃ < j`. Let `Y` be black with `c(Y) ∼ c(j)` crossing
both in `E_t`. If `Y ∉ {A, M}`, then `Y` crosses both in `β`, which is impossible (proof of
Lemma 6.2). If `Y ∈ {A, M}`: `c ∼ c(j)`, so by (L3) `N = j` and `S'` is the mover; but
`n_{E_t}(A, S') = n_{E_t}(M, S') = 0`. Second clause: the strands crossing the red strand are
`j` and `M`, of different colours.
Now Lemma 1.2 gives `ψ_{E_t} = B(j,F',∅)`, and the Sliding Lemma with Lemma 8.3 (exactly one
bad letter, `γ = r_c t_{c,c(N)}`) gives the formula. ∎

### 8.4 Coefficients

Let `Λ` be the exterior algebra on `e_1, …, e_m`, and `ω_V = e_{v_1} ∧ … ∧ e_{v_h}` for
`V = {v_1 < … < v_h}`. For a pair `p` let `s_p` be the derivation of `Λ` with
`s_p(e_{b_p}) = e_{a_p}` and `s_p(e_i) = 0` for `i ≠ b_p`; on a monomial it replaces
`e_{b_p}` by `e_{a_p}` in place (and gives 0 if `e_{b_p}` does not occur). Let `∂_j` be the
right contraction: `∂_j(e_{i_1} ∧ … ∧ e_{i_h}) = (−1)^{h−k} e_{i_1} ∧ … ê_{i_k} … ∧ e_{i_h}`
if `i_k = j`, and `0` if `j` does not occur (well defined on `Λ`). By the definition of `κ`
in `NOTES.md` §2 (`formula.py`, `koszul`):

    ∂_j ( ∏_{p ∈ F} s_p ) ω_I  =  κ(I; j, F) · ω_J .                                        (8.2)

The `s_p` of pairwise disjoint pairs commute, and `∂_j s_p = s_p ∂_j` if `j ∉ {a_p, b_p}`
(both are checked on monomials).

**Lemma 8.7.** For a datum `(j, F', 𝒯)` with `k` components, and `S ⊆ {1, …, k}`:

    coef(t_1(S)) · coef(t_2(S))  =  (−1)^{k − |S|} · C,

with `C` independent of `S`; and for `𝒯 = ∅`:
`coef(t_1) · coef(t_2) = coef'(j, F') · r_c t_{c, c(N)}`, where `coef'` is the coefficient of
the frozen term in `d'`.

*Proof.* *Monomials.* The parent and the connector of a pair of `F'` are the same in
`t_1(S)` as in `(j, F')`. The parent of `(a,m)` is `N`, and its connector is the slot of `N`
inside `(a, m)`: `j`, or `b_p` if `N = p` (`a_p < a < b_p < m`); it contributes
`r_c t_{c, c(N)}`. A pair of `T_i` other than `q_i` has its parent in `T_i`, in `t_1(S)` and
in `t_2(S)`. The parent of `q_i` is `(a,m)` with connector `a` if `i ∈ S`, and the mover `a`
of `t_2(S)` if `i ∉ S`; in both cases it contributes `r_{c(q_i)} t_{c(q_i), c}`. So the
product of the two monomials is independent of `S`.
*Connector signs.* The sign `(−1)^{#\{p : x_p ∈ \text{source vertex}\}}`: for `p ∈ F'` the
connector is a slot `< m` different from `a`, and it lies in `I` iff in `I'`; for `(a,m)` the
connector `j` or `b_p` lies in `I`: factor `−1`; for a pair of `T_i` other than `q_i` the
connector is a slot of a pair of `T_i`, untouched by `t_1(S)`, so it lies in `J(S)` iff in
`I`; for `q_i`: `a ∉ I` if `i ∈ S` (factor `+1`), and `a ∈ J(S)` if `i ∉ S` (factor `−1`).
Total: `(−1)^{k−|S|}` times a sign independent of `S`.
*Koszul signs.* By (8.2) and the commutation rules (`j` is not a slot of `𝒯`),

    κ(t_1(S)) κ(t_2(S)) ω_{K'} = ∂_a s_{𝒯_{S^c}} ∂_j s_{F'} s_{(a,m)} s_{𝒯_S} ω_I
                               = ∂_a ∂_j s_{F'} s_{(a,m)} s_{𝒯} ω_I ,

independent of `S`.
*The case `𝒯 = ∅`.* `ω_I = ω_{I'} ∧ e_m`, so `s_{F'} s_{(a,m)} ω_I = (s_{F'} ω_{I'}) ∧ e_a`
(derivations; `m ∉ I'`, `b_p ≠ m` for `p ∈ F'`), `∂_j(η ∧ e_a) = −(∂_j η) ∧ e_a`, and
`∂_a(ξ ∧ e_a) = ξ`. Hence `κ(t_1) κ(t_2) = −κ'(I'; j, F')`. With the factor `−1` of the
connector of `(a,m)` and the monomial `r_c t_{c,c(N)}` this gives the claim. ∎

### 8.5 End of the proof

**Proposition 8.8.** `E_{I',K'} = 0` if `a ∉ I'`, `a ∉ K'`.

*Proof.* In (8.1), by Lemma 8.3, the Sliding Lemma and Lemma 8.6, the first sum is

    Σ coef'(j, F') · r_c t_{c,c(N)} · B(j, F', ∅)

over the frozen terms `(j, F')` at `I'` with target `K'` for which `N` exists with `a` inside
(the others have no bad letter and contribute 0). By Lemma 8.4.3 the second sum runs over the
data `U = (j, F', 𝒯)` with `K'_U = K'` and the subsets `S`; by Lemmas 8.5 and 8.7 the datum
contributes `C · B(j, F', 𝒯) · Σ_S (−1)^{k−|S|}`, which is `0` if `k ≥ 1`, and
`coef'(j,F') r_c t_{c,c(N)} B(j, F', ∅)` if `𝒯 = ∅`. By Lemma 8.4.4 the data with `𝒯 = ∅` and
target `K'` correspond bijectively to the frozen terms of the first sum. So the two sums of
(8.1) are equal and `E_{I',K'} = 0`. ∎

*Proof of Theorem B.* (8.0) holds by Corollary 8.2 and Proposition 8.8; the Reduction Lemma
4.5 gives `E = 0`, and Proposition 3.3 gives `d² = 0`. ∎

*Checks* (`05_T1_check.py`; for every word: Lemmas 5.1, 5.3, 6.2 for all terms; Lemma 8.1;
Lemma 8.3; Lemma 8.4 (the union is admissible with distinct colours, the families have `2^k`
members); Lemma 8.5 ((S1), (S2) for every product word, and equality with the basis braid of
`σ_U`); Lemma 8.6 ((S1), (S2), the sliding identity and `E_t = β(t_1) β(t_2)`); Lemma 8.7;
Lemma 4.2 for the slot `a`; and finally `E = 0` in all components):

| run | words | families `k = 0 / 1 / 2` | products with `0 / 1 / 2` double crossings |
|---|---|---|---|
| `A 3 all 2 6` (all T1 words of length ≤ 6) | 267 | 1388 / 44 / 0 | 1454 / 22 / 0 |
| `D 4 all 2 6` (all, random red colour) | 2320 | 10410 / 252 / 0 | 10788 / 126 / 0 |
| `A 4 all 5 6` (all) | 2040 | 10128 / 168 / 0 | 10380 / 84 / 0 |
| `D 4 all 7 8 60 1` (sample) | 120 | 4984 / 292 / 11 | 5485 / 122 / 5 |
| `A 5 all 7 9 40 2` (sample) | 120 | 5143 / 180 / 0 | 5387 / 104 / 12 |
| `D 5 all 8 9 30 3` (sample) | 60 | 4854 / 360 / 8 | 5392 / 214 / 0 |
| `D 4 3421342 1342134 2` | 3 | 73 / 16 / 1 | 94 / 12 / 3 |

All assertions pass. (`k = 2` needs a colour with three neighbours; the smallest example is
`D_4`, word `3421342`, `I = {4,5,6,7}`.)

---

## 9. Special cases

**Corollary 9.1 (exactly one repeated colour).** Let `w` have pairwise distinct colours
except that `c` occurs at the two slots `a < b`. Then `d = δ₀ + d_1` with

    d_1 e_I  =  Σ_{j ∈ I : a < j < b, c(j) ∼ c}  −κ(I; j, {(a,b)}) · r_c t_{c,c(j)} · β(I; j, {(a,b)})      (a ∉ I ∋ b),

and `d² = 0`. In the proof of Theorem B for `w_{≤ b}` every datum has `𝒯 = ∅`, `F' = ∅`,
`N = j`, and the only identities used are: Lemma 8.1 for `m_a`; and for `a ∉ I'`, `j ∈ I'`:

    ι_R(m_j) X − X ι_L(m_j) = r_c t_{c,c(j)} · β(I; j, {(a,b)}) · m_a   if a < j, c(j) ∼ c,   and = 0 otherwise.

*Proof.* A pair has colour `c`, so it is `(a,b)`; it must be linked to the mover. The slots
after `b` have new colours (Theorem A). For the step `w' = w_{<b}`, `d' = δ₀` and the frozen
terms are the `m_j`. ∎

**Corollary 9.2.** The tree formula satisfies `d² = 0` for every stack of `sl_n` with `ω_1`
or `ω_2` (and duals), of `so_{2n}` with the vector representation, and for all minuscule
stacks of `D_4` — and for every prefix and every reordering of their words.

*Proof.* The number of occurrences of a colour `i` in the word of a chain `λ → μ` is the
coefficient of `α_i` in `λ − μ`, which is at most its coefficient in `λ − w_0 λ`. For `sl_n`:
`ω_1 − w_0 ω_1 = e_1 − e_n = α_1 + … + α_{n−1}` and
`ω_2 − w_0 ω_2 = (e_1 − e_n) + (e_2 − e_{n−1}) = α_1 + 2(α_2 + … + α_{n−2}) + α_{n−1}`. For
`so_{2n}`: `ω_1 − w_0 ω_1 = 2 e_1 = 2(α_1 + … + α_{n−2}) + α_{n−1} + α_n`. For `D_4` the three
minuscule representations are exchanged by triality. So every colour occurs at most twice,
and this property passes to prefixes and reorderings. ∎

---

## 9A. Theorem C: one colour three times, all other colours once

This is the first case with a colour of multiplicity three. It uses the Reduction Lemma for
the first occurrence and, for the dots that the Reduction Lemma does not remove, the
four-term identity (Proposition 10.2 below, whose proof is independent of this section).

**Theorem C.** Let `w` be a word in which one colour `c` occurs three times and every other
colour at most once. Then the tree formula satisfies `d² = 0`.

*Reduction to one step.* Let `a_1 < a_2 < b` be the slots of colour `c`. The prefix `w_{<b}`
is a T1 word (Theorem B); the letters after `b` have new colours (Theorem A). So it suffices
to treat `w = w'·c` with `m = b` the last slot. Notation: `A_1, A_2, M` are the strands of
`a_1, a_2, m`; all other black strands have pairwise different colours `≠ c`. So

> (∗∗) two black strands of the same colour are two of `A_1, A_2, M`.

*The terms.* A pair has colour `c`, a configuration contains at most one pair (two pairs of
colour `c` are not disjoint), and a pair must be linked to the mover. Hence the terms are:

* in `d'`: `m_j`; and `P_j = (j; (a_1,a_2))` (`a_1 < j < a_2`, `c(j) ∼ c`), from vertices with
  `a_1 ∉`, `a_2 ∈`, to vertices containing `a_1`;
* in `F`: `X`; `G^1_j = (j; (a_1, m))` (`a_1 < j < m`, `c(j) ∼ c`, source `∌ a_1`, target
  `∋ a_1`); `G^2_j = (j; (a_2, m))` (`a_2 < j < m`, `c(j) ∼ c`, source `∌ a_2`, target
  `∋ a_2`).

A mover `a_1` or `a_2` has no pair (a pair has the colour `c`, and a link needs adjacent
colours). The diagrams of `X` and of the `m_j` are moves of a single strand and have a
unique reduced word. Every reduced word of `P_j`, `G^1_j`, `G^2_j` satisfies (S2): by (∗∗) and
Lemma 6.1 two strands of the same colour cross only in `G^1_j`, namely `M` with `A_2` (if
`a_2` is on the right) or `A_1` with `A_2` (if `a_2` is on the left); a black strand of
colour `∼ c` crossing `A_2` is the mover `j` (unaffected strands do not cross each other),
and `j` crosses neither `A_1` nor `M` (Lemma 6.1 (c)); the red strand does not cross `A_2`.
So the elements `β(·)` are well defined (Lemma 1.2).

By Proposition 3.3 and the Reduction Lemma 4.5 with `a = a_1` we have to show
`E_{I',K'} = 0` for `a_1 ∉ K'`. As in (8.1), with `I = I' ∪ m`,

    E_{I',K'} = Σ_{t'} coef'(t') ( ι_R(β(t')) X_{K'} − X_{I'} ι_L(β(t')) )  −  Σ_{(t_1,t_2)} coef(t_1) coef(t_2) β(t_1) β(t_2),        (9A.1)

where `t'` runs through the terms of `d'` from `I'` to `K'` and `(t_1, t_2)` through the
pairs of consecutive terms of `d` (for `w`, with their coefficients for `w`) from `I` to
`K'` in which exactly one of the two is of kind (γ) and none is `X` (for the products
`ι_R(d') F_1` this uses `coef = −coef'`, Lemma 3.2).

*Which contributions exist for `a_1 ∉ K'`.* The terms `P_j` and `G^1_j` start at vertices
without `a_1` and put `a_1` to the right; the only term that removes `a_1` from the right is
`m_{a_1}`; `G^2_j` and the `m_{j'}`, `j' ≠ a_1`, do not move `a_1`.

* First sum: `t'` is `m_j` or `P_j`; `P_j` ends at a vertex containing `a_1`. So `t' = m_j`,
  `K' = I' ∖ j`, and either `j ∈ \{a_1, a_2\}` — **(C1)** — or `j ∉ \{a_1,a_2\}`, `a_1 ∉ I'`
  — part of **(C3)**.
* Products containing `P_j`: after `P_j` the slot `a_1` is on the right, and the other
  factor, of kind (γ), does not remove it. None.
* Products containing `G^1_j`: it must be followed by `m_{a_1}`: `G^1_j · m_{a_1}`, from
  `I' ∌ a_1` to `K' = I' ∖ j` — part of **(C3)**.
* Products of `G^2_j` and `m_{j'}`, in either order:
  `j' = a_1` (then `a_1 ∈ I'`, `a_2 ∉ I'`, `K' = (I' ∖ \{a_1, j\}) ∪ a_2`) — **(C2)**;
  `j' = a_2` (then `a_1 ∉ I'`; the order is `m_{a_2} · G^2_j` if `a_2 ∈ I'` and
  `G^2_j · m_{a_2}` if `a_2 ∉ I'`; `K' = I' ∖ j`) — part of **(C3)**;
  `j' ∉ \{a_1, a_2\}` (then `a_1, a_2 ∉ I'`, `K' = (I' ∖ \{j, j'\}) ∪ a_2`) — **(C4)**; the
  same vertex `K'` is reached by `G^2_{j'}` and `m_j` if `G^2_{j'}` exists.

The three kinds of target vertices are different (`K' = I' ∖ j` contains `a_2` iff `I'`
does; in (C2) `a_1 ∈ I'`; in (C4) `a_1, a_2 ∉ I'` and `a_2 ∈ K'`), so (9A.1) splits into:

* **(C1)** `K' = I' ∖ a_i`: only the term `t' = m_{a_i}` of the first sum;
* **(C2)** `a_1 ∈ I'`, `a_2 ∉ I'`: the two products `m_{a_1} · G^2_j` and `G^2_j · m_{a_1}`;
* **(C3)** `a_1 ∉ I'`, `j ∉ \{a_1,a_2\}`, `K' = I' ∖ j`: `t' = m_j` in the first sum, and the
  products `G^1_j · m_{a_1}` (if `a_1 < j`, `c(j) ∼ c`); `G^2_j · m_{a_2}` (if `a_2 ∉ I'`) or
  `m_{a_2} · G^2_j` (if `a_2 ∈ I'`) (if `a_2 < j`, `c(j) ∼ c`);
* **(C4)** `a_1, a_2 ∉ I'`, `K' = (I' ∖ \{j, j'\}) ∪ a_2`: the products `G^2_j · m_{j'}`,
  `m_{j'} · G^2_j`, and, if also `a_2 < j' < m` and `c(j') ∼ c`, `G^2_{j'} · m_j`,
  `m_j · G^2_{j'}`.

(The script `05_F3_check.py` verifies independently that this list is complete.)

**(C1).** In a reduced word of `m_{a_i}` the mover is the right strand of all its letters;
the only black strand of colour `c` that is a left strand is `A_1`, in its crossing with the
mover `A_2`, whose colour is not adjacent to `c`. No bad letter: the difference vanishes
(Lemma 2.2).

**(C2).** In `m_{a_1} · G^2_j` and in `G^2_j · m_{a_1}` the strands moved are `A_1` (crossing
the red strand and the slots `< a_1`), and `j`, `A_2` (from `−a_2` to `−m`), `M` (from `+m` to
`+a_2`). By Lemma 6.1, `A_1` and `j` cross once (`a_1 < j`), and `A_1` crosses neither `A_2`
nor `M` (their slots stay `> a_1`). So both words are reduced with the same permutation, no
two strands of the same colour cross, (S2) is empty, and the two elements are equal (Lemma
1.2). Coefficients: the monomial is `r_c t_{c,c(j)}` both times, the connector `j` lies in
the source both times, and by (8.2) the Koszul signs are the coefficients in
`∂_j s ∂_{a_1} ω_I` and `∂_{a_1} ∂_j s ω_I = −∂_j s ∂_{a_1} ω_I` (`s = s_{(a_2,m)}`). The two
products cancel.

**(C3).** The reduced word of `m_j` is unique: `j` crosses the right strands of slots `< j`
(from the largest to the smallest), the red strand, and the left strands of slots `< j` (from
the smallest to the largest). Its bad letters are the crossings `χ_i` with `A_i`, for `i`
with `a_i < j`, if `c(j) ∼ c` (the left strand must have colour `c`, and `A_i` is the left
strand of its crossing with `j`); `γ = r_c t_{c,c(j)}`. If `c(j) ≁ c` there is no bad letter
and no product. We show `ψ_{E_{χ_i}} = ` (the corresponding product), in three cases.

*`i = 1`.* In `E_{χ_1}`: `j` crosses what it crosses in `m_j` except `A_1`; `M` crosses all
right strands except `j`, the red strand and the left strands of slots `< a_1`; `A_1` crosses
the left strands of slots `> a_1`; `M` ends at `−a_1`, `A_1` at `−m`. In `G^1_j · m_{a_1}`:
`j` crosses the same strands (Lemma 6.1 (b), (c)); `M` crosses the right slots in `(a_1, m)`
other than `j`, and then, as the mover `a_1`, the right slots `< a_1`, the red strand and the
left slots `< a_1`; `A_1` crosses the left slots in `(a_1, m)`. Both words are reduced with
the same permutation. (S2): the crossings of two strands of colour `c` are `(M, A_2)` if
`a_2 ∈ I'`, and `(A_1, A_2)` if `a_2 ∉ I'`; a black strand of colour `∼ c` crossing `A_2` is
`j`, which crosses neither `M` nor `A_1` in either word; the red strand does not cross
`A_2`. Lemma 1.2 gives equality.

*`i = 2`, `a_2 ∉ I'`.* In `E_{χ_2}`: `j` crosses what it crosses in `m_j` except `A_2`; `M`
crosses all right strands except `j`, the red strand and the left slots `< a_2` (among them
`A_1`); `A_2` crosses the left slots `> a_2`. In `G^2_j · m_{a_2}`: the same crossings (`M`:
right slots in `(a_2,m)` other than `j`, then right slots `< a_2`, red, left slots `< a_2`;
`A_2`: left slots in `(a_2, m)`). Both reduced, same permutation. (S2): the only crossing of
two strands of colour `c` is `(M, A_1)`; a black strand of colour `∼ c` crossing `A_1` is `j`,
which does not cross `M`; the red strand does not cross `A_1`. Equality by Lemma 1.2.

*`i = 2`, `a_2 ∈ I'`.* In `E_{χ_2}`: `j` first crosses the right slots in `(a_2, j)`; then `M`
crosses the right slots `> a_2` other than `j`; `A_2` crosses the right slots `< a_2`, the red
strand and all left strands (among them `A_1`); `j` continues with the right slots `< a_2`,
the red strand, the left slots `< j`. In `m_{a_2} · G^2_j`: `A_2` crosses right slots
`< a_2`, red, left slots `< a_2`, and then, as `L_{(a_2,m)}`, the left slots in `(a_2, m)`;
`M` crosses the right slots in `(a_2, m)` other than `j`; `j` crosses the right slots `< j`
(`a_2` is no longer there), red, the left slots `< j` except `A_2`. Both reduced, same
permutation (`M ↦ +a_2`, `A_2 ↦ −m`). (S2): the only crossing of two strands of colour `c` is
`(A_2, A_1)`; `j` crosses `A_1` but not `A_2`; the red strand crosses `A_2` but not `A_1`.
Equality by Lemma 1.2.

*Coefficients.* For `G^i_j · m_{a_i}` (`a_i ∉ I'`): `coef(t_1) coef(t_2) = coef'(m_j) γ` by
the computation in the proof of Lemma 8.7 (case `𝒯 = ∅`, `F' = ∅`, with `a = a_i`; it uses
only (8.2)). For `m_{a_2} · G^2_j` (`a_2 ∈ I'`): by (8.2) the Koszul sign is the coefficient in
`∂_j s_{(a_2,m)} ∂_{a_2} (ω_{I'} ∧ e_m)`. Now `∂_{a_2}(ω_{I'} ∧ e_m) = −(∂_{a_2} ω_{I'}) ∧ e_m`,
`s_{(a_2,m)}` turns this into `−(∂_{a_2} ω_{I'}) ∧ e_{a_2} = −ω_{I'}` (removing `e_{a_2}` with
the sign of the entries to its right and appending it at the end restores `ω_{I'}`), and
`∂_j` gives `−κ'(I'; j, ∅) ω_{K'}`. With the factor `−1` of the connector `j` (which lies in
the source `I ∖ a_2`): `coef(t_1) coef(t_2) = coef'(m_j) γ`. So in (9A.1) the contributions of
(C3) cancel: `coef'(m_j) γ (ψ_{E_{χ_1}} + ψ_{E_{χ_2}})` against the two products.

**(C4).** If `j'` satisfies `a_2 < j' < m`, `c(j') ∼ c`: the four products are those of
Proposition 10.2 for `(a, b) = (a_2, m)`; (H1) holds because no slot between `a_2` and `m` has
the colour `c`, (H2) because the colours of `j`, `j'` occur once. Their sum with the
coefficients of the tree formula is zero.
Otherwise there are two products. The strands moved are `j, A_2, M` in `G^2_j` and `j'` in
`m_{j'}`. By Lemma 6.1, in both orders: `j, j'` cross once; `A_1` is not crossed by `A_2` or
`M`; `j'` and `M` cross twice in `G^2_j · m_{j'}` iff `a_2 < j'` (`M` passes the right slot
`j'`, then `j'` passes the slot `a_2`), `j'` and `A_2` cross twice in `m_{j'} · G^2_j` iff
`a_2 < j'`, and there are no other double crossings. If `a_2 < j'` then `c(j') ≁ c` (we are
in the case "otherwise") and `c(j') ≠ c`, so (S1) holds. No two strands of the same colour
cross, so (S2) is empty. The permutations agree, so the two elements are equal (Lemma 1.2).
The monomials agree, the connector `j` lies in the source both times, and the Koszul signs
are the coefficients in `∂_{j'} ∂_j s ω_I` and `∂_j s ∂_{j'} ω_I = −∂_{j'} ∂_j s ω_I`. The two
products cancel.

So `E_{I',K'} = 0` whenever `a_1 ∉ K'`, and Theorem C follows from Propositions 4.5 and 3.3. ∎

*Check.* `05_F3_check.py A 3 2 13`, `… D 4 2 134`, `… A 4 2 134` (all words of the three
families, red colour varying): the list (C1)–(C4) is complete; every sliding error word is
reduced, satisfies (S2) and equals the stated product, with the stated coefficient (2496
cases); (C2): 416 cases; (C4): 244 two-term and 82 four-term cases; `d² = 0` in all 13 024
components with `m ∈ I`, `m ∉ K`.

*Remark.* The proof of (C2)–(C4) uses "all other colours once" only through (∗∗) and through
the list of terms. With other colours occurring twice the trees become larger and (R1)-type
arguments are needed; this is not done.

---

## 10. Beyond colour multiplicity two (target (T2)): what carries over, what the data say, what remains

Notation: `w = w'·c` is an arbitrary word without cyclic configurations, the colour `c` occurs
`N` times in `w'`, at the slots `a_1 < … < a_N`. Theorem B is the case "`N ≤ 1` at every step
of the induction", which is the same as "every colour at most twice".

### 10.1 Where "every colour at most twice" is used

| step | uses T1? | what is used |
|---|---|---|
| Lemmas 1.1–1.4, Sliding Lemma 2.1–2.2 | no | relations of the algebra only |
| Cone decomposition 3.1–3.3 | no | (for general words: a compatible choice of the reduced words) |
| Cocycle identity 4.1, Reduction Lemma 4.2–4.6 | no | `d'² = 0`, (F1), first occurrence of a colour |
| Theorem A | no | |
| Lemma 6.1 | no | |
| Lemmas 5.1–5.4 | **yes** | pairs have different colours, the link graph embeds into `Γ` |
| Lemma 6.2 (tree diagrams satisfy (S2)) | **yes** | 5.1, 5.3; false in general (warning after 6.2) |
| section 8, set-up | **yes** (`N = 1`) | the only pair with `m` is `(a, m)`; `a` is in no pair of `w'`; `a` is a first occurrence |
| Lemma 8.1, Corollary 8.2 | only `N = 1` | the proof uses `N = 1`, Lemma 6.1 and the Sliding Lemma, nothing else: **it holds for every word in which `c` occurs once in `w'`** |
| Lemma 8.3 (one bad letter) | **yes** | 5.4 |
| Lemma 8.4 (families) | **yes** | 5.1, 5.2, `Γ` a forest |
| Lemmas 8.5, 8.6 ((S1), (S2)) | **yes** | (L1)–(L3), i.e. 5.1, 5.3, 5.4 |
| Lemma 8.7 (signs) | no | formal properties of `κ` (the exterior algebra calculus), given the family structure |

### 10.2 The general form of the problem

For an arbitrary word `w = w'·c` with `d'² = 0`, the Sliding Lemma turns `E = 0` into

    Σ_{t', χ} coef'(t') γ_χ ψ_{E_χ(t')}  =  Σ_{J'} F_1{}_{;I',J'} ι_L(d'_{J',K'})  −  Σ_{J'} ι_R(d'_{I',J'}) F_1{}_{;J',K'}        (∗)

for all `I', K'` (`t'` the terms of `d'` from `I'` to `K'`, `χ` the bad letters of a reduced
word of `t'`, which are now the crossings of **any** strand of colour `c` standing to the left
of a strand of adjacent colour). By Proposition 4.6, (∗) has to be established directly only
for a set of components from which all others can be reached by the rules (r), (l); by
Proposition 4.5 the components with `a_1 ∉ K'` suffice (`a_1` is a first occurrence; so is
any first occurrence of any other colour).

All statements of this subsection and of 10.3–10.4 about "the data" refer to the basis braids
of `klrw` as diagrams of the tree terms, and are Tier 2: exact computations for the words
listed. In all of them (∗) itself holds (it is equivalent to `d² = 0`).

### 10.3 `N = 1`, the other colours arbitrary

Here Corollary 8.2 holds as proved, so by the Reduction Lemma with `a = a_1`:

> **Remaining identity (R1).** For `a ∉ I'`, `a ∉ K'`: equation (8.1) vanishes, i.e.
> `Σ_{t', χ} coef'(t') γ_χ ψ_{E_χ(t')} = Σ_{(t_1,t_2)} coef(t_1) coef(t_2) β(t_1) β(t_2)`
> (`t_1 ∋ (a,m)`, `t_2` with mover `a`).
> **If (R1) holds for a word `w'·c` with `N = 1` and `d'² = 0`, then `d² = 0` for `w'·c`.**

Data (`05_blocks_general.py A 3 1 80 8 1 a1`, `… D 4 1 60 8 2 a1`: 80 + 60 random non-cyclic
words of length `≤ 8` in which the last colour occurs exactly twice, the other colours
arbitrarily often), restricted to the components of (R1): every term of `d'` has at most one bad
letter (2127 have one, 8130 none); every `E_χ` is a single basis diagram without dots; of the
2949 products `β(t_1) β(t_2)`, 2813 are a single basis diagram without dots and 136 are
**zero**; the basis diagrams cancel in pairs (2456 groups) or in fours (7 groups). No dots occur. Compared with section 8 the only new phenomenon is the
vanishing products; in the examples inspected they are instances of Lemma 1.4 (two strands of
the same colour cross twice: e.g. `A_2`, word `12112`, `I = {3,4,5}`,
`(3; (2,5)) · (2; (1,4))`: the union `(3; (1,4), (2,5))` is blocked, the mover 3 of colour 1
lies inside the pair `(1,4)` of colour 1). So (R1) should be provable by the method of
section 8 once the combinatorics of sections 5–6 is replaced by non-cyclicity; what is
missing is (i) a substitute for Lemma 6.2 (for general words (S2) fails for some tree
diagrams, so the strong form of the Straightening Lemma is not enough), (ii) the family
structure of Lemma 8.4 with blocked unions, for which the partner product does not exist and
the product itself must vanish. **(R1) is not proved.**

In the whole equation (all four blocks) dots do occur for `N = 1`
(2582 product terms with dots among the same 80 words of `A_3`,
`05_blocks_general.py A 3 1 80 8 1`), but all of them lie in components
that follow by Proposition 4.6: `05_closure.py A 3 1 60 8 dots 1`: 358 components with dots,
all deducible from the components without dots.

### 10.4 `N ≥ 2`: the Reduction Lemma does not remove all dots; what it removes

`05_blocks_general.py A 3 2 80 8 3` (80 non-cyclic words, `N = 2`): number of components of
(∗) that are clean / not clean (clean = every `E_χ` and every product is one basis diagram
without dots, cancellation in pairs), by the position of `(a_1, a_2)` in `I'` and in `K'`
(`+` = on the right of the red strand):

| `I'` \ `K'` | `++` | `+−` | `−+` | `−−` |
|---|---|---|---|---|
| `++` | 1006 / 95 | 0 / 109 | 102 / 7 | – |
| `+−` | 411 / 199 | 966 / 135 | 252 / 288 | 102 / 7 |
| `−+` | 553 / 653 | 0 / 1148 | 966 / 135 | 0 / 109 |
| `−−` | – | 553 / 653 | 411 / 199 | 1006 / 95 |

No block is clean. Restricted to the components with `a_1 ∉ K'`
(`05_blocks_general.py A 3 2 80 8 3 a1`): terms of `d'` with 0 / 1 / 2 bad letters:
6597 / 1419 / 810; sliding errors `E_χ` that are 0 / one diagram / several diagrams:
109 / 2894 / 36; products that are 0 / one diagram / several: 463 / 6067 / 735; 704 products
have coefficients with dots; cancellation groups of size 2 / 4 / 6: 4931 / 160 / 28.

**Which dots does the cocycle argument remove?** (`05_closure.py`; a component is *dotted* if
one of its products has a coefficient with dots; "deducible" = reachable by the rules (r),
(l) of Proposition 4.6 from the other components.)

| run | words | dotted components | not deducible from the dot-free ones | not deducible, if the dotted components **not** involving the pair `(a_1, m)` are known |
|---|---|---|---|---|
| `A 3 1 60 8` (`N = 1`) | 60 | 358 | 0 | – |
| `A 3 2 40 7` (`N = 2`) | 40 | 264 | 172 (7 words) | 0 |
| `A 2 2 40 7` (`N = 2`) | 40 | 474 | 308 (10 words) | 0 |
| `D 4 2 25 7` (`N = 2`) | 25 | 110 | 16 (1 word) | 0 |
| `A 2 3 20 8` (`N = 3`) | 20 | 953 | 758 (8 words) | 0 |
| `A 3 3 20 8` (`N = 3`) | 20 | – | – | 0 (of 370) |

(columns 4 and 5 are the modes `dots` and `dots-a1` with the seeds 1, 2, 1, 6, 3, 5).
So, in all samples:

> the Reduction Lemma removes exactly the dotted identities that involve the pair
> `(a_1, m)` of the **first** occurrence of `c`; the dotted identities that only involve the
> pairs `(a_i, m)`, `i ≥ 2`, are not reachable and have to be proved directly.

For `N = 3` it is not enough to know the dotted components of the last pair `(a_3, m)`
(`05_closure.py A 2 3 20 8 dots-aN 3`: 684 of 878 not deducible).

**The smallest dotted identity that is not deducible** is in `A_2`, word `22112` (`c = 2`,
`a_1 = 1`, `a_2 = 2`, slots 3, 4 of colour 1, `m = 5`), `I' = {3,4}`, `K' = {2}`: with
`β_0` the reduced diagram `[−2 → −5, +3 → −3, +4 → −4, +5 → +2]` and `x_{e,i}` the dot on the
`i`-th strand of colour `e` at the source,

    m_3 · (4;(2,5))   =  + r_2 t_{21} ( t_{12} x_{1,1} + t_{21} x_{2,1} ) β_0
    (3;(2,5)) · m_4   =  + r_2 t_{21} ( t_{12} x_{1,2} + t_{21} x_{2,3} ) β_0
    m_4 · (3;(2,5))   =  − r_2 t_{21} ( t_{12} x_{1,2} + t_{21} x_{2,1} ) β_0
    (4;(2,5)) · m_3   =  − r_2 t_{21} ( t_{12} x_{1,1} + t_{21} x_{2,3} ) β_0

(coefficients of the tree formula included; `x_{2,1}` is the dot on `A_2`, `x_{2,3}` on `M`,
`x_{1,1}, x_{1,2}` on the strands 3, 4). The third and the fourth product contain in addition
`∓ r_1 r_2 t_{12} t_{21} β_0'`, where `β_0'` is the diagram in which the strands 3 and 4 do not
cross (`+3 → −4`, `+4 → −3`); these two terms cancel each other. The same four products occur for a T1 word (drop the
slot 1), where they are deduced by the Reduction Lemma with the twin `a`; here the twin
`a_2` is not a first occurrence (the elementary move of `a_2` crosses `A_1`), so this is not
available. In a sample (`A_3`, `N = 2`, 40 words) all 172 non-deducible dotted components
have this shape: exactly four dotted products, two movers `j, j'`, one tree `F ∋ (a_i, m)`
admissible with either mover, the products being `m_j·(j';F)`, `(j;F)·m_{j'}`,
`m_{j'}·(j;F)`, `(j';F)·m_j` (in 8 of them `F` has a second pair;
`05_closure.py A 3 2 40 7 dots 2 shapes`). The four expansions above are printed by
`05_explore.py A 2 22112 v`.

The following proposition proves this four-term identity directly in the simplest situation.
It is valid for arbitrary words, in particular with further slots of colour `c` below `a`
(`N ≥ 2`), and shows the mechanism with exact signs.

**Proposition 10.2 (four-term bigon identity).** Let `w` be an arbitrary word,
`a < j < j' < b` slots with `c(a) = c(b) = c`, `c(j) = e ∼ c`, `c(j') = e' ∼ c`, and `I` a
vertex with `j, j', b ∈ I`, `a ∉ I`; put `K = (I ∖ \{j, j', b\}) ∪ \{a\}`. Assume

* (H1) no slot strictly between `a` and `b` has the colour `c`;
* (H2) no slot `< j` has the colour `e`, and no slot `< j'` has the colour `e'` (so `e ≠ e'`).

Let `A, M` be the strands of `a, b`, and `β_0` the element of a reduced word from `e_I` with
permutation `j ↦ −j`, `j' ↦ −j'`, `A ↦ −b`, `M ↦ +a`. Then all reduced words of the four
terms below have the same element, and

    P_1 := m_j(I) · β(I∖j; j', \{(a,b)\})   =  ( t_{ce} x_A + t_{ec} x_j ) β_0,
    P_4 := β(I; j', \{(a,b)\}) · m_j         =  ( t_{ce} x_M + t_{ec} x_j ) β_0,
    P_2 := β(I; j, \{(a,b)\}) · m_{j'}       =  ( t_{ce'} x_M + t_{e'c} x_{j'} ) β_0,
    P_3 := m_{j'}(I) · β(I∖j'; j, \{(a,b)\}) =  ( t_{ce'} x_A + t_{e'c} x_{j'} ) β_0

(dots at `e_I`). These are the only products of two tree terms from `I` to `K`, and with the
coefficients of the tree formula they add up to zero: `(d²)_{I,K} = 0`.

*Proof.* *Crossings.* The strands moved are `j`, `j'`, `A`, `M`; in each product each of them
moves in exactly one factor. By Lemma 6.1: `j` crosses the red strand and unaffected strands
of slots `< j`, `j'` those of slots `< j'`, `A = L_{(a,b)}` unaffected left strands with slots
in `(a,b)`, `M = R_{(a,b)}` unaffected right strands with slots in `(a,b)`; a mover linked to
`(a,b)` crosses neither `A` nor `M`. By (H1), (H2) none of the four crosses a strand of its
own colour, and unaffected strands do not cross each other. So **no two strands of the same
colour cross** in any of the four product words, (S2) holds trivially, and Lemmas 1.2, 1.3
apply to the factors and to the products. A pair of strands can cross twice only if one of
them moves in the first factor and the other in the second. Going through Lemma 6.1:

* `P_1` (first `j`, then `j', A, M`): `(j, j')` once (in the second factor, `j < j'`);
  `(j, A)` **twice** (`a < j`; then `A` passes the left slot `j ∈ (a,b)`); `(j, M)` never.
* `P_4` (first `j', A, M`, then `j`): `(j', j)` once (first factor); `(M, j)` **twice** (`M`
  passes the right slot `j`; then `j` passes the right slot `a < j`, now occupied by `M`);
  `(A, j)` never.
* `P_2` (first `j, A, M`, then `j'`): `(j, j')` once (second factor); `(M, j')` **twice**;
  `(A, j')` never.
* `P_3` (first `j'`, then `j, A, M`): `(j', j)` once (first factor); `(j', A)` **twice**;
  `(j', M)` never.

In each case the pair crossing twice has adjacent colours `c ∼ e` resp. `c ∼ e'`, and the
permutation is that of `β_0`. Lemma 1.3 gives the four formulas.
*Other products.* A product of two tree terms from `I` to `K` moves `j`, `j'` to the left as
movers, removes `b` from the right and adds `a`; the only ways to do the latter with pairs
are the pair `(a,b)` in one factor, or pairs `(y, b)`, `(a, y)` in the two factors with `y` of
colour `c`, `a < y < b`, excluded by (H1).
*Signs.* By (8.2), with `s = s_{(a,b)}`, and since the connector of `(a,b)` is the mover,
which lies in the source vertex (factor `−1`), the signs of the four products are the
coefficients of `ω_K` in `−∂_{j'} s ∂_j ω_I` (`P_1`), `−∂_j ∂_{j'} s ω_I` (`P_4`),
`−∂_{j'} ∂_j s ω_I` (`P_2`), `−∂_j s ∂_{j'} ω_I` (`P_3`). As `∂_j s = s ∂_j`,
`∂_{j'} s = s ∂_{j'}` and `∂_j ∂_{j'} = −∂_{j'} ∂_j`, the signs are `ε, −ε, ε, −ε` for
`P_1, P_4, P_2, P_3`. The monomials are `r_c t_{ce'}` for `P_1, P_4` and `r_c t_{ce}` for
`P_2, P_3`. Hence

    (d²)_{I,K} = ε r_c ( t_{ce'} ( t_{ce} x_A − t_{ce} x_M ) + t_{ce} ( t_{ce'} x_M − t_{ce'} x_A ) ) β_0 = 0. ∎

*Check.* `05_fourterm.py A 3 40 7 1`, `05_fourterm.py D 4 40 7 2`: 80 random words (arbitrary,
random red colour), 384 cases, 64 of them with a further slot of colour `c` below `a`: the
four products are the only ones, each equals the predicted element, and the sum is zero.

When (H2) fails — e.g. for `22112`, where the two movers are twins — the dot of a mover has
to pass a crossing with a strand of its own colour, which produces an additional term with
one more factor `r` (in the data: `P_3` and `P_4` each get a second basis diagram, in which
the two movers do not cross, with opposite coefficients). This case is **not proved**.

### 10.5 List of what remains for (T2)

Each item is verified by exact computation for the words of the runs quoted (and all are
implied by the verification of `d² = 0` in `PROOFS.md` section 3 and `general_words.py`).

1. **Diagrams.** A statement replacing Lemma 6.2 for general non-cyclic words: the element of
   a tree term does not depend on the reduced word (front 01), or a version of the
   Straightening Lemma with the positional form of the bad triples.
2. **(R1)** of 10.3, for `N = 1` and arbitrary other colours: no dots; single diagrams,
   vanishing products (Lemma 1.4), cancellation in pairs.
3. For `N ≥ 2`, equation (∗) on the components with `a_1 ∉ K'` (or on any set of components
   generating all others by Proposition 4.6). These contain
   * (3a) dot-free identities: sliding errors of terms with several bad letters, some of
     which vanish or have two basis terms; products that vanish or have several basis terms;
   * (3b) the dotted identities attached to the pairs `(a_i, m)`, `i ≥ 2`; in all samples
     they are four-term identities as in Proposition 10.2, in general with twin movers and
     with larger trees. Proposition 10.2 proves the case (H1), (H2), `F = \{(a,b)\}`.
4. Nothing else: by Propositions 3.3 and 4.5, items 1–3 for all non-cyclic words imply C1′.

### 10.6 Attempts that did not work

* *Cancelling all products in pairs.* False: the basis diagrams of `d²` cancel in groups of
  2, 4, 8, with dots (section 0). A general product rule for two tree terms was not found;
  products can have up to 7 basis terms and coefficients that do not factor
  (`A_3`, `132213`, `m_4 · (3; (1,5), (2,6))`).
* *A second reduction with a slot that is not a first occurrence* (e.g. `a_2`): the lowest
  order term `m_{a_2}` crosses a strand of its own colour; `μ μ^* ∈ 𝔪 A`, and right
  multiplication by a crossing of two strands of the same colour has a kernel. Lemma 4.3
  fails.
* *Combining the right-hand rule for one slot with the left-hand rule for another*, or
  using all first occurrences at once: the lowest order of the cocycle identity gives one
  relation between two unknown components (the "Koszul" relations `m_s m_{s'} = m_{s'} m_s`
  produce genuine solutions), so the closure of Proposition 4.6 is strictly smaller than
  everything; `05_closure.py` computes it.
* *Clean components suffice for `N ≥ 2`?* No (10.4).


---

## 11. Status

### Proved completely (human-checkable; inputs: (R1)–(R4) with the signs of the README, (F1) in its left and right module forms, (F3))

| statement | generality | checked by |
|---|---|---|
| Lemma 1.1 (word property), from (F3) | all words of crossings | – (combinatorial) |
| Lemma 1.2 (Straightening), 1.3 (bigons), 1.4 (vanishing) | arbitrary states | `05_straighten.py` |
| Lemmas 2.1, 2.2 (Sliding Lemma, sign `+γ`) | arbitrary words, red strand anywhere | `05_sliding.py` |
| Lemmas 3.1, 3.2, Proposition 3.3 (cone, signs) | arbitrary words (with the convention on reduced words stated there) | `05_cone.py` |
| Lemmas 4.1–4.4, Propositions 4.5, 4.6 (Reduction Lemma, deduction rule) | arbitrary words with `d'² = 0` | `05_reduction.py` (inputs 4.2, 4.4) |
| Theorem A, Corollary 7.1 (new colour; reproves Theorem 5) | arbitrary `w'` | `05_T1_check.py` ("new colour: exact slides") |
| Lemmas 5.1–5.4, 6.2 | T1 words | `05_T1_check.py` |
| Lemmas 8.1–8.7, Proposition 8.8, **Theorem B** (`d² = 0` if every colour occurs at most twice) | T1 words, any `Γ` that is a forest, any red colour | `05_T1_check.py` |
| Corollary 9.1 (one repeated colour), 9.2 (`sl_n` `ω_1, ω_2`; `so_{2n}` vector; `D_4`) | | |
| Proposition 10.2 (four-term bigon identity under (H1), (H2)) | arbitrary words | `05_fourterm.py` |
| **Theorem C** (one colour three times, all others at most once), section 9A | | `05_F3_check.py` |

Inputs taken from elsewhere: the list of relations with signs (README, `alg.py`); Lemma 6.1 =
items 1–6 of the proof of Proposition 3 of `PROOFS.md`; (F1) in both forms (free left and free
right module over the dots), used only in Lemmas 3.1 (injectivity, not needed for Theorem B)
and 4.3; (F3). Theorem B does not use the grading (F2), nor any result of the fronts 01–04, 06.

Referee pass (done after the last change of sections 4 and 8): every case distinction was
re-read against Lemma 6.1; two points were made explicit (the pair `(M, A)` in the list of
Lemma 8.5; Lemma 4.3 now only uses freeness over the dots, with an arbitrary basis). The signs
that enter are: `+γ` in the Sliding Lemma (`05_sliding.py`); the signs of the cone
decomposition (`05_cone.py`); `coef(t_1) coef(t_2) = coef'(t') r_c t_{c,c(N)}` and the
alternation `(−1)^{k−|S|}` (Lemma 8.7; `05_T1_check.py` compares the actual products of the
elements of the formula with the sliding error and within each family); the final identity
`E = 0` in all four blocks (`05_T1_check.py`, "chain map components").

### Runs (all after the last edit of the scripts; every run ended without a failed assertion)

`zsh 05_logs/run_all.sh` (from `proofs/checks`) reproduces all of them; the outputs are in
`proofs/checks/05_logs/*.log`, each ending with `exit code 0`.

| run (`sage -python …`) | result |
|---|---|
| `05_straighten.py 3000 1` | Lemma 1.2: 1371 words; 1.3: 868; 1.4: 381; no failure |
| `05_sliding.py 300 1` | 308 cases, 142 bad letters, 0 failures |
| `05_cone.py A 3 6 40 1`, `05_cone.py D 4 6 30 2` | 70 words, 5062 terms |
| `05_reduction.py A 3 40 6 1`, `05_reduction.py D 4 25 6 2` | 2428 + 2148 cases |
| `05_T1_check.py A 3 all 2 6` | 267 words (all T1 words of length ≤ 6) |
| `05_T1_check.py D 4 all 2 6` | 2320 words (all) |
| `05_T1_check.py A 4 all 5 6` | 2040 words (all) |
| `05_T1_check.py D 4 all 7 8 60 1` | 120 words |
| `05_T1_check.py A 5 all 7 9 40 2` | 120 words |
| `05_T1_check.py D 5 all 8 9 30 3` | 60 words |
| `05_T1_check.py D 4 3421342 1342134 2` | 3 words (`k = 2`) |
| `05_F3_check.py A 3 2 13`, `… D 4 2 134`, `… A 4 2 134` | 132 words (Theorem C) |
| `05_fourterm.py A 3 40 7 1`, `05_fourterm.py D 4 40 7 2` | 384 cases |
| `05_explore_stats.py A 3 3 6 T1` | the numbers of section 0 |
| `05_explore.py A 2 22112 v` | the expansions of 10.4 |
| `05_blocks_general.py A 3 1 80 8 1 [a1]`, `… D 4 1 60 8 2 a1`, `… A 3 2 80 8 3 [a1]` | data of 10.3, 10.4 |
| `05_closure.py …` (11 runs, see `run_all.sh`) | table of 10.4 |

### Verified only by computation (Tier 2)

* (R1) of 10.3 (`N = 1`, other colours arbitrary): structure of the components described
  there (`05_blocks_general.py … a1`).
* The statements of 10.4 on which dotted components are deducible (`05_closure.py`), and the
  shape of the non-deducible ones.
* Equation (∗) for all non-cyclic words of the runs of 10.3–10.4 (equivalent to `d² = 0`
  there).

### Not proved

* Conjecture C1′ for words in which some colour occurs at least three times, beyond the
  family of Theorem C. Precisely: the items 1–3 of 10.5.
* (R1).
* The four-term identity when (H2) fails (twin movers, or a mover crossing a strand of its
  own colour) or for larger trees.

### Failed attempts

See 10.6.

### Statements elsewhere that need a caveat

* README, "Cone structure": correct, with the signs of Lemma 3.2 (`−d'` on `C_R`, `+1` for
  `X_m`).
* For general words a tree diagram can violate (S2) (warning after Lemma 6.2); this does not
  contradict the observed independence of the reduced word, but the strong Straightening
  Lemma 1.2 does not prove it.
