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

What is **not** proved here: Conjecture C1′ for words with a colour occurring three times or
more (target (T2)). Section 10 says exactly which steps use "at most twice", what the data
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

*Proof.* 1. `z P = z μ μ^* ∈ 𝔪^{N+1} A`. By (F1) in the form quoted in the README, `e' A e_0`
is a free right module over `R[x]` (dots at the right end) with basis the `β_σ`; write
`z = Σ β_σ f_σ`. Then `𝔪^{N+1} e' A e_0 = ⊕ β_σ 𝔪^{N+1} R[x]`, so `f_σ P ∈ 𝔪^{N+1} R[x]` for
all `σ`. Grade `R[x]` by the total degree in the variables `r`; `𝔪^{N+1} R[x]` consists of the
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
(more than 990 000 terms).

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
