# 03 — The cyclic condition: combinatorics

*(revised after `R03_referee.md`.)*

Front: the cyclic condition of `PROOFS.md` Cor. 3.1 / Prop. 7, for arbitrary words and for the
minuscule stacks of types A, D, E. Purely combinatorial; all scripts are plain `python3`
(no sage), in `proofs/checks/03_*.py`, shared code in `checks/03_lib.py`.

## 0. Results at a glance

Throughout, `Γ` is any finite forest (every simply-laced Dynkin diagram of finite type is a
tree); "word" means an arbitrary word in the nodes of `Γ` unless "minuscule" is said.

**Proved (complete proofs below):**

| # | statement | where |
|---|---|---|
| 1 | Every colour at most twice ⇒ no block and no cycle in any configuration. | Prop. 2.2 |
| 2 | Inclusion-minimal cyclic configurations are exactly the unblocked configurations whose link graph is a chordless cycle with a pendant path to the mover (a "tadpole"); the mover need **not** lie on the cycle. | Lemma 2.3 |
| 3 | The mover is a chord to `+∞`; replacing the chord of a chordless cycle with the largest right end by a mover at its left end preserves the link graph. Hence every cyclic word has a cyclic configuration whose link graph is a chordless cycle **through the mover**. | Lemmas 2.4, 2.5, Cor. 2.6 |
| 4 | **Cycle shortening.** A chordless cycle of length `L ≥ 6` of an unblocked family can be replaced, using only its own slots, by an unblocked family with a cycle of length `L − 2`. | Lemma 2.7 |
| 5 | **Theorem P (4-cycle / pattern theorem).** A word is cyclic ⇔ it has a cyclic configuration with exactly 3 pairs ⇔ it contains one of three explicit patterns **T**, **B**, **N** (at most 7 letters). | Thm. 2.8 |
| 6 | Every colour at most three times: cyclic ⇔ pattern T or B, and then every minimal cyclic configuration is a 4-cycle through the mover. | Cor. 2.10 |
| 7 | **Blocks matter**, also for minuscule words: ignoring blocks makes strictly more words "cyclic" (e.g. `A_2` word `1212121`, the 3×5 rectangle of `sl_8, ω_3`, the 13-slot `E_6` ideal — the last two carry verified tree complexes). | Prop. 2.11 |
| 8 | Minuscule chain words: exactly `1 − δ_{c,k}` neighbour letters before the first `c`, exactly two between consecutive `c`'s; hence pattern B never starts at the first occurrence. Colour multiplicities are bounded by the coefficients of `λ − w_0λ`. | Lemmas 2.12, 2.14, Cor. 2.13 |
| 9 | **Theorem A.** `sl_n`, `Λ^k ℂ^n`, Young diagram `λ ⊆ k × (n−k)`: the stack is cyclic ⇔ `λ ⊇ (4,4,4,3)` ⇔ `λ_3 ≥ 4` and `λ_4 ≥ 3`. (`(4,4,4,3)` is self-conjugate, so this is already the symmetric statement.) | §3 |
| 10 | Type A: if all diagonals of `λ` have ≤ 3 boxes (e.g. `λ = (4,4,4,3)`), the minimal cyclic configuration is unique (the one of `PROOFS.md` Prop. 8); for `(4,4,4,3)` it is the only cyclic configuration at all. | Prop. 3.3 |
| 11 | `sl_n` with `ω_1, ω_2, ω_3` (and duals) is never cyclic; the full poset of `Λ^k ℂ^n` is cyclic ⇔ `4 ≤ k ≤ n − 4`. | Cor. 3.4 |
| 12 | `D_n` vector representation: never cyclic (all `n`). | Prop. 4.1 |
| 13 | **Theorem D.** `D_n` half-spin, ideal = strict partition `ℓ`: cyclic ⇔ `ℓ_3 ≥ 3` and `ℓ_4 ≥ 1` ⇔ `ℓ ⊇ (5,4,3,1)` (exists iff `n ≥ 6`). | §4 |
| 14 | In types A and D every non-cyclic minuscule stack has all colour multiplicities ≤ 3 (so blocks between two pairs and factors `r_c²` never occur there). | Lemmas 3.2, 4.3, Cor. 6.1 |
| 15 | Unique minimal cyclic ideal / weight `μ_c` in types A, D (all ranks), with the explicit `μ_c` of `NOTES.md` §4 confirmed. | Theorem U (§6) |
| 15a | Models of types A and D: the stacks of `μ` are exactly the linear extensions of one Young diagram / shifted diagram, bijectively with the weights, and the weight order is reverse inclusion (by induction on the chain word; no citation needed). | Lemmas 3.1, 4.2 |
| 16a | `D_n` half-spin: every cyclic ideal gives a non-Deodhar element (all `n`; input: one exact Hecke computation in `D_6`). | Prop. 7.3 |
| 16 | Type A: non-cyclic ⇔ Deodhar (⇔ 321-hexagon-avoiding), via Billey–Warrington Thm. 1. | Prop. 7.1 |

**Proved by exhaustive computer search (no bound on the number of pairs; three independent
methods):**

| # | statement | script |
|---|---|---|
| 17 | **Theorem E.** `E_6` (`ω_1`, `ω_6`; 27 ideals) and `E_7` (`ω_7`; 56 ideals): exactly 3 resp. 22 ideals are cyclic; they are exactly the ideals containing a unique minimal one `I_c` (14 resp. 15 slots), `μ_c = −ω_4 + ω_5` (`E_6, ω_1`), `ω_3 − ω_4` (`E_6, ω_6`), `ω_3 − ω_4 + ω_7` (`E_7`). | `03_typeE.py` |
| 18 | `E_6`, `E_7`: non-cyclic ideals have colour multiplicities ≤ 3. At `I_c` there are exactly two minimal cyclic configurations, one of pattern B and one of pattern N (no T). | `03_typeE.py` |

**Verified by computation only:** Deodhar ⇔ non-cyclic for `D_6 … D_9` half-spin (all
ideals), and its failure in type E (`E_6`: the 13-slot ideal is non-Deodhar but not
cyclic; `E_7`: three such ideals) — `03_deodhar.py`.

**Corrections** (details in §8): the claim in this front's task statement (not in any project
file) that `(4,4,4,3)` is not transposition-symmetric is wrong; `NOTES.md` §4 table, row "`sl_n, ω_k` with
`4 ≤ k ≤ n−3`" should read `4 ≤ k ≤ n−4` (off by one; the second table has it right);
`NOTES.md` §4 "the minimal cyclic configuration always has the same shape" omits the
pattern N (which occurs at the `E_6`, `E_7` thresholds) and the coupling of the two
alternatives; the 13-slot `E_6` ideal is now proved non-cyclic (reading 3 of
`reports/…Demazure.md` is excluded).

The scripts and what each checks:

| script | content | runtime |
|---|---|---|
| `03_lib.py` | definitions, exhaustive search (Lemma 2.3), path search (Cor. 2.6), brute force over all configurations, minuscule chains and heaps | — |
| `03_search_vs_brute.py` | search = brute force on 4000 random words, three modes | 30 s |
| `03_patterns.py` | Theorem 2.8 against search and brute force; Lemmas 2.5, 2.7 step by step (cycles of length 6, 8, 10) | 25 s |
| `03_minuscule.py` | minuscule heaps, ideals ↔ weights, chains = linear extensions, classification | < 10 s per case |
| `03_typeA.py` | model check; all 3432 diagrams in the 7×7 box; minimal configurations | ~1 min |
| `03_typeD.py` | model check; all strict partitions for `D_4 … D_11`; vector rep; brute force up to `D_9` | 5 s |
| `03_typeE.py` | three independent methods on every ideal of `E_6`, `E_7` | 95 s |
| `03_deodhar.py` | Grassmannian hexagon patterns; Deodhar property in the Hecke algebra vs cyclicity | 80 s (+ 17 min with `--big`) |

---

## 1. Definitions

**1.1 Colours.** `Γ` is a finite forest (no loops, no multiple edges, no cycles); its vertices
are *colours*, `c ∼ c'` means joined by an edge (so `c ≁ c`). Facts used about forests: they
are bipartite; there is no closed walk `c_0, c_1, …, c_s = c_0` with `s ≥ 3` and
`c_0, …, c_{s−1}` pairwise distinct (that would be a cycle); two distinct neighbours of a
vertex are not adjacent (that would be a triangle).

**1.2 Words, pairs, interlacing.** A *word* is `w = (c(1), …, c(m))`; its *slots* are
`1, …, m`. A *pair* is `p = (a_p, b_p)` with `a_p < b_p` and `c(a_p) = c(b_p) =: c(p)`; its
*interior* is the open interval `(a_p, b_p)`; put `χ_p(s) = 1` if `a_p < s < b_p`, else `0`.
Two pairs with four distinct slots are *interlaced* if `a_p < a_q < b_p < b_q` or
`a_q < a_p < b_q < b_p`.

> **Lemma 1.1.** For pairs `p, q` with four distinct slots: `p, q` interlaced ⇔ exactly one
> slot of `q` lies in the interior of `p` (`χ_p(a_q) + χ_p(b_q) = 1`) ⇔ exactly one slot of
> `p` lies in the interior of `q`. Not interlaced ⇔ `p, q` nested or disjoint ⇔
> `χ_p(a_q) = χ_p(b_q)`.

*Proof.* Exchanging `p, q` we may assume `a_p < a_q`. The three possible orders are
`a_p < b_p < a_q < b_q` (disjoint: no slot of either in the other), `a_p < a_q < b_q < b_p`
(nested: both slots of `q` in `p`, no slot of `p` in `q`), `a_p < a_q < b_p < b_q`
(interlaced: exactly `a_q` in `p`, exactly `b_p` in `q`). ∎

**1.3 Families and configurations.** A *family* is a set `F` of pairwise disjoint pairs. Its
*link graph* `G(F)` has vertex set `F` and an edge `p — q` (a *link*) iff `p, q` are
interlaced and `c(p) ∼ c(q)`; `p, q` form a *block* iff they are interlaced and
`c(p) = c(q)`. `F` is *unblocked* if it has no block. A *configuration* `(j, F)` is a slot `j`
(the *mover*) and a family `F` none of whose pairs contains the slot `j`. Its link graph
`G(j, F)` has vertex set `{M} ⊔ F` (`M` = the mover), the links of `G(F)`, and an edge
`M — p` iff `a_p < j < b_p` and `c(j) ∼ c(p)`; `(M, p)` is a *block* iff `a_p < j < b_p` and
`c(j) = c(p)`. We write `c(M) := c(j)`. A configuration is *cyclic* if it has no block, `G(j,F)`
is connected, and `#links > |F|`; since `G(j,F)` has `|F| + 1` vertices, the last two
conditions say: `G(j, F)` is connected and contains a cycle. A word is *cyclic* if it has a
cyclic configuration. (This is `PROOFS.md` Cor. 3.1 / Prop. 7 and `NOTES.md` §4. The
conditions `a_p ∉ I`, `b_p ∈ I` of the tree formula play no role: `I = {j} ∪ {b_p : p ∈ F}`
satisfies them.)

Sub-configurations `(j, F')`, `F' ⊆ F`, of an unblocked configuration are unblocked, and
`G(j, F')` is the induced subgraph of `G(j, F)` on `{M} ∪ F'`.

**1.4 Heaps and inheritance.** Links and blocks compare only slots of equal or adjacent
colours, and only through their relative order. Hence (`PROOFS.md` Prop. 7) cyclicity depends
only on the heap of the word. Moreover, **if `u` is a subword of `w`** (a subsequence, not
necessarily a prefix) **and `u` is cyclic, so is `w`**: a configuration of `u` is a
configuration of `w` (on the corresponding slots) with the same links and blocks. So the
non-cyclic words form a class closed under taking subwords — the natural setting for an
induction on the last letter (README, "cone structure").

**1.5 Patterns.** A word `w` *contains*

* **pattern T** if there exist a colour `c`, two distinct neighbours `c' ≠ c''` of `c`, slots
  `x_1 < x_2 < x_3` of colour `c` and, for each `γ ∈ {c', c''}`, a pair `(y_γ, z_γ)` of colour
  `γ` with `x_1 < y_γ < x_2` and `x_3 < z_γ`;
* **pattern B** if there exist a colour `c`, two distinct neighbours `c' ≠ c''` of `c`, slots
  `x_1 < x_2 < x_3` of colour `c` and, for each `γ ∈ {c', c''}`, a pair `(y_γ, z_γ)` of colour
  `γ` with `y_γ < x_1` and `x_2 < z_γ < x_3`;
* **pattern N** if there exist colours `c, d, e` with `d ∼ c` and `e ∼ d` (possibly `e = c`),
  slots `y_1 < y_2 < j < z_2 < z_1` with `c(y_1) = c(y_2) = c(z_2) = c(z_1) = d` and
  `c(j) = c`, and a pair `p = (a, b)` of colour `e` with `j ∉ {a, b}`, one slot in
  `(y_2, z_2)` and the other outside `[y_1, z_1]`, and, if `e = c`, `j ∉ (a, b)`.

(In T and B the slots are automatically pairwise distinct, the colours `c, c', c''` being
distinct; in N the slots of `p` differ from the `y_i, z_i` because `e ≠ d`.)

The associated configurations: **T**: mover `x_3`, pairs `(x_1, x_2)` and the two pairs of
colours `c', c''`; **B**: mover `x_1`, pairs `(x_2, x_3)` and the two pairs; **N**: mover `j`,
pairs `q = (y_1, z_1)`, `q' = (y_2, z_2)`, `p`. In heap language (1.4): T says that the heap has
a chain `x_1 < x_2 < x_3` of colour `c` and, for two distinct neighbours of `c`, elements
`y < z` of that colour with `x_1 < y < x_2` and `x_3 < z`; similarly for B and N.

---

## 2. Arbitrary words

### 2.1 The colour map

> **Lemma 2.1.** For a family `F` or a configuration `(j, F)`, the map `v ↦ c(v)` from the
> vertices of the link graph to `Γ` sends edges to edges. Consequently:
> (a) the link graph is bipartite, all its cycles have even length `≥ 4`;
> (b) for every cycle `(v_1, …, v_L)` of the link graph (indices mod `L`) there is an `i`
> with `c(v_{i−1}) = c(v_{i+1})` — a *turning point* `v_i`.

*Proof.* A link joins adjacent colours by definition. (a) Compose with a 2-colouring of the
forest `Γ`; a simple graph has no cycles of length `< 3`, and bipartite ones none of length 3.
(b) The colours form a closed walk `c(v_1), …, c(v_L)` in one tree component `T` of `Γ`. Fix
a vertex `o` of `T`, and choose `i` with `δ := dist_T(o, c(v_i))` maximal. In a tree,
adjacent vertices have distances to `o` differing by exactly one, and a vertex `≠ o` has
exactly one neighbour closer to `o`. The colours `c(v_{i±1})` are adjacent to `c(v_i)` and
have distance `≤ δ`, so both have distance `δ − 1` (in particular `δ ≥ 1`), so both are the
unique neighbour of `c(v_i)` closer to `o`. ∎

> **Proposition 2.2 (multiplicity ≤ 2).** If every colour occurs at most twice in `w`, no
> configuration has a block and every link graph is a forest. In particular `w` is not
> cyclic.

*Proof.* Two distinct pairs of `F` of the same colour would need four slots of that colour,
and a pair of colour `c(j)` would need three (`j ∉ p`). So `c` is injective on the vertices
of `G(j, F)`, there are no blocks, and by Lemma 2.1 `G(j, F)` is isomorphic to a subgraph of
the forest `Γ`. ∎

This covers `sl_n` with `ω_1, ω_2`, `D_n` with `ω_1`, and all of `D_4` (Lemma 4.0 below).

### 2.2 Minimal cyclic configurations

A *tadpole* is a graph consisting of a cycle `C` (length `≥ 4`) and a path
`P = (v_0, v_1, …, v_l)`, `l ≥ 0`, with `V(C) ∩ V(P) = {v_l}`, and no other edges. A *tree
configuration* is an unblocked configuration whose link graph is a tree.

> **Lemma 2.3 (structure of minimal cyclic configurations).**
> (a) A configuration `(j, F)` is an inclusion-minimal cyclic configuration iff it is
> unblocked and `G(j, F)` is a tadpole with `v_0 = M`. The cycle is then chordless.
> (b) Every cyclic configuration contains an inclusion-minimal one.
> (c) Every inclusion-minimal cyclic configuration is `(j, T ∪ {p})` where `(j, T)` is a tree
> configuration and `p` has exactly two links into `{M} ∪ T`.
> (d) Every tree configuration `(j, T)` arises from `(j, ∅)` by adding one pair at a time,
> each having exactly one link into the configuration built so far and no block with it.
> (e) Hence `w` is cyclic iff there are a tree configuration `(j, T)` and a pair `p`,
> disjoint from `j` and from the pairs of `T`, without block against `M` or `T`, with at least
> two links into `{M} ∪ T`.

*Proof.* We use: (∗) if an unblocked configuration `(j, F)` with connected link graph has a
cyclic proper sub-configuration, then `(j, F ∖ {v})` is cyclic for some `v ∈ F`. Indeed, let `F'' ⊊ F` be
maximal with `(j, F'')` cyclic; as `G(j,F)` is connected, some `v ∈ F ∖ F''` is linked to
`{M} ∪ F''`; `(j, F'' ∪ {v})` is unblocked, connected and contains a cycle, so by
maximality `F'' ∪ {v} = F`.

(a) "⇒": let `(j, F)` be minimal cyclic. Let `C` be a shortest cycle of `G = G(j, F)`; it is
chordless (a chord splits `C` into two shorter cycles). Let `P = (M = v_0, …, v_l)` be a
shortest path from `M` to `V(C)`. The sub-configuration on `V(C) ∪ V(P)` is connected and
contains `C`, so by minimality it is all of `G`. `P` has no chords and no `v_i`, `i ≤ l−2`,
is adjacent to `C` (shortest path). Suppose `l ≥ 1` and `v_{l−1}` is adjacent to a vertex
`x ≠ v_l` of `C`. By (a) of Lemma 2.1, `x` and `v_l` are at even distance `δ ≥ 2` along `C`;
let `A` be a shorter arc of `C` between them, `|V(A)| = δ + 1 ≤ |C|/2 + 1 < |C|`. Then
`V(P) ∪ V(A)` is a proper subset of `V(G)` which induces a connected graph containing the
cycle `v_{l−1}, x, A, v_l, v_{l−1}` — contradicting minimality. So `G` is a tadpole.
"⇐": in a tadpole, deleting `v_i` (`1 ≤ i ≤ l`) disconnects `M` from `C ∖ {v_i}` (for
`i = l`: from `C − v_l`), and deleting a vertex of `C ∖ {v_l}` leaves the tree
`P ∪ (C − v)`. So no `(j, F ∖ {v})` is cyclic, and by (∗) `(j, F)` is minimal.
(b) Finite descent. (c) Take `v ∈ V(C) ∖ {v_l}` (exists, `|C| ≥ 4`); `G − v` is a tree and `v`
is adjacent exactly to its two neighbours on `C`. (d) Order the vertices of the tree
`G(j, T)` by breadth-first search from `M`; every initial segment induces a connected
subgraph of a tree, hence a subtree, and the next vertex has exactly one neighbour in it.
(e) "⇐" is clear; "⇒" by (b), (c). ∎

Part (e) is the exhaustive search `03_lib.Word.search`: it enumerates **all** tree
configurations of every mover by (d) — breadth first, with a seen-set, without any bound on
the number of pairs — and tests every candidate `p` against each. `03_search_vs_brute.py`
compares it with a brute force over all configurations (every mover, every set of disjoint
equal-coloured pairs avoiding it) on 4000 random words of length ≤ 12 in `A_2 … A_5, D_4`:
identical answers.

**The mover need not lie on the cycle of a minimal configuration.** In `λ = (4,4,4,4)`
(type A, §3; boxes `(row, column)`, colour = column − row) the configuration with mover
`(1,3)` and pairs `((1,1),(2,2))`, `((1,2),(3,4))`, `((2,1),(4,3))`, `((3,3),(4,4))` is minimal
cyclic: the four pairs form a 4-cycle, and the mover is linked only to `((1,2),(3,4))`
(`03_typeA.py`: `(4,4,4,4)` has 6 minimal cyclic configurations, 4 of them with the mover at
distance 1 from the cycle; `5^5` has 414, with chordless cycles of length 4, 6, 8 and tails of
length up to 4). But the mover can always be moved onto the cycle:

### 2.3 The mover as a chord; replacement

> **Lemma 2.4 (the mover is a chord to `+∞`).** Let `w⁺ = w · c` be `w` with one slot
> `m + 1` of colour `c` appended. For a configuration `(j, F)` of `w` with `c(j) = c`, the set
> `F⁺ = F ∪ {(j, m+1)}` is a family in `w⁺`, and `M ↦ (j, m+1)`, `p ↦ p` is an isomorphism
> `G(j, F) ≅ G(F⁺)` that also matches blocks.

*Proof.* For `p ∈ F`, `b_p ≤ m < m + 1`, so `p` and `(j, m+1)` are interlaced iff
`a_p < j < b_p` (the other interlacing order `j < a_p < m+1 < b_p` is impossible), and
`c((j, m+1)) = c(j)`. ∎

> **Lemma 2.5 (replacement).** Let `F` be an unblocked family in a word `u` and `C ⊆ F` such
> that the induced subgraph `G(F)[C]` is a chordless cycle. Let `p ∈ C` have the largest right
> slot `b_p` among the pairs of `C`. Then `(a_p, C ∖ {p})` is an unblocked configuration of `u`
> whose link graph is isomorphic to `G(F)[C]` by `M ↦ p`. In particular it is cyclic, and its
> cycle passes through the mover.

*Proof.* `a_p` is not a slot of `C ∖ {p}` (disjointness). For `r ∈ C ∖ {p}` we have
`b_r < b_p`. If `a_r < a_p < b_r`, then `a_r < a_p < b_r < b_p`: `r, p` interlaced.
Conversely, interlaced with `b_r < b_p` means `a_r < a_p < b_r < b_p`. So
`a_r < a_p < b_r` ⇔ `r, p` interlaced, and since `c(a_p) = c(p)`, the mover `a_p` has with
`r` exactly the relation (link / block / none) that `p` has with `r`. Relations among
`C ∖ {p}` are unchanged. `C` had no block, so neither has the new configuration. ∎

> **Corollary 2.6.** A word is cyclic iff it has a cyclic configuration whose link graph is a
> chordless cycle through the mover.

*Proof.* Given a cyclic `(j, F)`, let `C` be a shortest (hence chordless) cycle of
`G(j, F)`. If `M ∈ C`, the sub-configuration on `C` is the required one. Otherwise
`C ⊆ F`, and Lemma 2.5 (with `u = w`) applies. ∎

The path search `03_lib.Word.search(mover_on_cycle=True)` uses this: a cycle through the
mover exists iff some induced path `M — v_1 — … — v_r` plus a pair linked to `M` and to some
`v_i` exists (take a shortest cycle through `M`; it is chordless, since a chord would give a
shorter cycle through `M`). `03_patterns.py` checks Lemma 2.5 on 7549 minimal cyclic
configurations whose cycle avoids the mover (3031 from random words, 4518 from Young diagrams,
`E_7` and `D_7` spin).

### 2.4 Cycle shortening and the pattern theorem

> **Lemma 2.7 (shortening).** Let `F` be an unblocked family in a word `u`, and
> `C = (v_1, …, v_L)` a chordless cycle of `G(F)` with `L ≥ 6`. Let `z = v_i` be a turning
> point (Lemma 2.1(b)), `x = v_{i−1}`, `y = v_{i+1}` (colour `d`), `u' = v_{i−2}`,
> `w' = v_{i+2}`. Put `κ = χ_{u'}(a_y)`, `κ' = χ_{w'}(a_x)`, let `s` be the slot of `x` with
> `χ_{u'}(s) = 1 − κ`, `t` the slot of `y` with `χ_{w'}(t) = 1 − κ'`, and `x* = (min(s,t), max(s,t))`.
> Then `x*` is a pair of colour `d`, the family `F' = (C ∖ {x, y, z}) ∪ {x*}` is unblocked, and
> `G(F')` contains the cycle `(x*, v_{i+2}, v_{i+3}, …, v_{i−2})` of length `L − 2`.

*Proof.* Since `L ≥ 6`, the indices `i−2, …, i+2` are distinct mod `L`, and `u'` and `y`
(indices differing by 3) are not consecutive on `C`, nor are `w'` and `x`.

*`s`, `t` are well defined.* `u'` is linked to `x`, so (Lemma 1.1) exactly one slot of `x` is
inside `u'`; hence exactly one slot `s` of `x` has `χ_{u'}(s) = 1 − κ`. Likewise for `t`.

*`u'`, `w'` are linked to `x*`.* `c(u') ∼ d = c(y)` and `u', y` are not linked (`C` chordless),
so they are not interlaced: `χ_{u'}(a_y) = χ_{u'}(b_y) = κ`. Hence
`χ_{u'}(s) + χ_{u'}(t) = (1 − κ) + κ = 1`: `u'` and `x*` are interlaced (Lemma 1.1; the slots
are distinct), and `c(u') ∼ d`. Symmetrically, `χ_{w'}(a_x) = χ_{w'}(b_x) = κ'` and
`χ_{w'}(s) + χ_{w'}(t) = κ' + (1 − κ') = 1`.

*No block.* `x*` is disjoint from the other pairs of `F'` (its slots belong to `x, y`). A
block of `F'` must involve `x*` and a pair `r ∈ C ∖ {x, y, z}` of colour `d`. Such an `r` is not
interlaced with `x` or `y` (no block in `F`), so `χ_r` is constant on the slots of `x`
(value `ρ_x`) and on the slots of `y` (value `ρ_y`). Also `r` is not a neighbour of `z` on `C`
(those are `x, y`), so `r, z` are not linked, and as `c(r) = d ∼ c(z)` not interlaced: `χ_r` is
constant on the slots of `z`. Suppose `ρ_x = 1`: then `a_r < a_x < b_x < b_r`, the slot of `z`
inside `x` (there is exactly one, `z, x` interlaced) is inside `r`, so both slots of `z` are
inside `r`. If `ρ_y = 0`, then `y` and `r` are disjoint or `y ⊃ r` (both slots of `y` outside
`r`, `y, r` not interlaced); in the first case no slot of `z` is inside `y`, in the second both
are — both contradict that `z, y` are interlaced. So `ρ_x = 1 ⇒ ρ_y = 1`, and symmetrically
`ρ_y = 1 ⇒ ρ_x = 1`. Hence `χ_r(s) + χ_r(t) = ρ_x + ρ_y ≡ 0`, and `r, x*` are not interlaced.

*The cycle.* `v_{i+2}, …, v_{i−2}` are `L − 3` consecutive vertices of `C`, so consecutive
ones are linked; `x*` is linked to both ends; all are distinct. ∎

`03_patterns.py` checks every assertion of this proof (the uniqueness of `s, t`, the
non-interlacing of `u', y`, both links, the absence of blocks, the new cycle) at **every**
turning point of every chordless cycle of length ≥ 6 occurring in the minimal cyclic
configurations of 3000 random words and of `λ = 5^4, 5^5, 6^3 4^3`, `E_7`, `D_7` spin:
4488 surgeries, on cycles of length 6 (3224), 8 (1156), 10 (108).

> **Theorem 2.8 (Theorem P).** For every word `w` over a forest, the following are
> equivalent:
> (i) `w` is cyclic;
> (ii) `w` has a cyclic configuration with exactly three pairs; its link graph is then a
> 4-cycle through the mover;
> (iii) `w` contains pattern T, B or N (1.5).
> The configurations attached to the patterns in 1.5 are cyclic, and every cyclic
> configuration with three pairs is one of them.

*Proof.* (ii) ⇒ (i) is trivial.

(i) ⇒ (ii). Let `(j, F)` be cyclic, `c = c(j)`, `w⁺ = w · c`. By Lemma 2.4, `F_0 := F⁺` is an
unblocked family in `w⁺` whose link graph contains a cycle. Given an unblocked family `F_k` of
`w⁺` with a cycle, let `C_k` be a shortest cycle of `G(F_k)` (chordless; even length `≥ 4`). If
`|C_k| ≥ 6`, Lemma 2.7 gives an unblocked family `F_{k+1}` of `w⁺` with a cycle of length
`|C_k| − 2`, so `|C_{k+1}| ≤ |C_k| − 2`. This stops at a chordless 4-cycle `C` in an unblocked
family of `w⁺`. Lemma 2.5 (with `u = w⁺`) gives the cyclic configuration `(a_p, C ∖ {p})` of
`w⁺`, `p` the pair of `C` with the largest right slot. If a pair of `C` uses the slot `m + 1`,
it is `p` (the largest slot). So the mover `a_p ≤ m` and the three pairs of `C ∖ {p}` are
slots of `w`; their relations are the same in `w`, so this is a cyclic configuration of `w`
with three pairs.

(ii) ⇔ (iii). Let `(j, {q, p, q'})` be cyclic. Its link graph is connected, bipartite, has 4
vertices and at least 4 edges, so it is `K_{2,2}`: a 4-cycle, every vertex of degree 2; name it
`M — q — p — q' — M`. By Lemma 2.1(b) `c(j) = c(p)` or `c(q) = c(q')`.

*Case `c(q) ≠ c(q')`.* Then `c(j) = c(p) =: c`, and `c(q), c(q')` are distinct neighbours of
`c`. No block: `j ∉ (a_p, b_p)`. `q` contains `j` and is interlaced with `p`, so exactly one
slot of `q` is inside `p`. If `b_p < j`: put `x_1, x_2, x_3 = a_p, b_p, j`; the slot of `q`
above `j` is not inside `p`, so the other one is in `(x_1, x_2)`: pattern T. If `j < a_p`: put
`x_1, x_2, x_3 = j, a_p, b_p`; the slot of `q` below `j` is not inside `p`, so the other one is
in `(x_2, x_3)`: pattern B. The case is determined by `j, p`, so `q'` behaves the same way.

*Case `c(q) = c(q') =: d`.* Both contain `j`, have the same colour and are not interlaced,
so they are nested, say `q' ⊂ q`: `y_1 < y_2 < j < z_2 < z_1`; `c(j) ∼ d`. `p` is interlaced with
`q'`: one slot of `p` is in `(y_2, z_2)`, hence in `(y_1, z_1)`; `p` is interlaced with `q`, so
the other slot is not in `(y_1, z_1)`, i.e. outside `[y_1, z_1]`. `c(p) ∼ d`, and if
`c(p) = c(j)` there is no block, i.e. `j ∉ (a_p, b_p)`: pattern N.

*Conversely* the attached configurations are cyclic. **T**: the four pairs/mover use distinct
slots (`c', c'', c` distinct). Mover `x_3` vs `(x_1, x_2)`: equal colour, `x_3 ∉ (x_1,x_2)`: no
relation. Mover vs `(y, z)`: `y < x_2 < x_3 < z`, link. `(x_1, x_2)` vs `(y, z)`:
`x_1 < y < x_2 < z`, interlaced, link. The two pairs of colours `c', c''`: distinct,
non-adjacent colours (1.1): no relation. Four links, no block, connected. **B**: mover `x_1`
vs `(y, z)`: `y < x_1 < z`, link; `(x_2, x_3)` vs `(y, z)`: `y < x_2 < z < x_3`, link; mover vs
`(x_2, x_3)`: `x_1 < x_2`, no relation; rest as for T. **N**: `j` lies inside `q` and `q'`,
`c(j) ∼ d`: two links. `p` has exactly one slot in `q'` and exactly one in `q`: two links. `q, q'`
nested: no relation. Mover vs `p`: if `c(p) = c(j)`, `j ∉ (a_p, b_p)` by hypothesis; otherwise
`c(p), c(j)` are distinct neighbours of `d`, not adjacent. Four links, no block. ∎

*Checks.* `03_patterns.py`: on 3000 random words (types `A_2 … A_5`, `D_4`, `E_6`, lengths
4–16) "cyclic" (exhaustive search) ⇔ "T, B or N" (literal detector written from 1.5), for
1888 of them also against the brute force; every detected pattern is a cyclic
configuration. All three patterns occur (229 T, 233 B, 644 N). The example of the README,
`A_2`, `1221122`, is pattern N (mover 5, `q = (2,7)`, `q' = (3,6)`, `p = (1,4)`).

So the **sharpest condition is exact**: a word is non-cyclic iff it avoids three patterns of
at most seven letters. Useful weaker sufficient conditions follow:

> **Corollary 2.9.** `w` is not cyclic if (a) every colour occurs at most twice; or (b) no
> colour occurs four times and no colour occurring three times has two distinct neighbours
> each occurring at least twice; or (c) (heap form) for every colour `c` and every chain
> `x_1 < x_2 < x_3` of colour `c` in the heap, at most one neighbour colour of `c` has an
> element above `x_3`, and at most one has an element below `x_1`, and no colour occurs four
> times.

*Proof.* T and B need three slots of `c` and two slots of each of two distinct neighbours;
N needs four slots of `d`. (c): T needs two neighbour colours with elements above `x_3`, B
two with elements below `x_1`. ∎

> **Corollary 2.10 (multiplicity ≤ 3).** If every colour occurs at most three times, `w` is
> cyclic iff it contains pattern T or B. Moreover every inclusion-minimal cyclic configuration
> is then a 4-cycle through the mover of type T or B.

*Proof.* N needs four slots of `d`. For the second claim let `(j, F)` be minimal cyclic with
cycle `C`. Every colour carries at most one pair of `F`, and a pair of colour `c(j)` uses the
other two slots of that colour; so the vertices of `G(j, F)` have pairwise distinct colours,
except possibly `M` and one pair `p_0` with `c(p_0) = c(j) =: c`. A turning point of `C` has two
`C`-neighbours of equal colour, so `M, p_0 ∈ C`. The two arcs of `C` from `M` to `p_0` are walks
`c, d_1, …, d_s, c` in `Γ` (`s ≥ 1`) with `d_1, …, d_s` distinct and `≠ c`; for `s ≥ 2` this
would be a cycle in `Γ`, so `s = 1` on both arcs and `|C| = 4`. As `M ∈ C`, the tadpole has no
tail: `G(j, F) = C`, a cyclic configuration with three pairs, of type T or B by Theorem 2.8. ∎

(The "pattern S" of `01_foundations.md` Theorem B.6(b) — mover `c'`, pairs `q ⊂ p` with tree
path `q — x — M — p` and a further slot of colour `c` inside `q` — is, in the multiplicity ≤ 3
setting, a special occurrence of T or B: the three slots of colour `c'` are `a_q, b_q, j`, and
`x`, `p` supply the two neighbour colours. Theorem 2.8 subsumes it.)

### 2.5 Blocks are relevant

Call a configuration *cyclic ignoring blocks* if `G(j, F)` is connected and contains a cycle,
blocks allowed (they are neither links nor forbidden).

> **Proposition 2.11.** "Has a configuration that is cyclic ignoring blocks" is strictly
> weaker than "cyclic", also for minuscule words:
> (a) `A_2`, `w = 1212121`: mover 4, pairs `(1,5)` (colour 1), `(2,6)` (colour 2), `(3,7)`
> (colour 1) is connected with 4 links on 3 pairs, with two blocks (`(1,5), (3,7)` interlaced;
> the mover inside `(2,6)`); `w` is not cyclic.
> (b) `sl_8`, `ω_3`, full poset (3 × 5 rectangle, 15 slots, word `321432543654765`): cyclic
> ignoring blocks (mover slot 8 of colour 4 inside the colour-4 pair `(4,12)`, with pairs
> `(1,9)`, `(7,15)` of colours 3, 5), but not cyclic (Theorem A).
> (c) `E_6`, `ω_1`: the ideals of 11, 12, 13 slots are cyclic ignoring blocks but not cyclic;
> `D_6` half-spin: four ideals (10, 11, 12, 12 slots); `sl_{10}`, `ω_5`: 54 versus 42 ideals.

*Proof.* (a) The configuration: by inspection. Non-cyclicity: by Theorem 2.8 it suffices to
exclude T, B and N. T and B need two distinct neighbours of a colour, and in `A_2` every colour
has only one. For N (colour 1 occurs four times, at 1, 3, 5, 7; colour 2 three times) with `d = 1`, `y_1 < y_2 < j < z_2 < z_1` forces `y_1, y_2, z_2, z_1 = 1, 3, 5, 7`,
`j ∈ {4}`, `e = 2 = c(j)`, and `p ⊂ {2, 4, 6} ∖ {j}` must have a slot in `(3, 5)` — only `4 = j`:
impossible; `d = 2` needs four slots of colour 2. (b), (c): `03_minuscule.py --blocks` and the
check in §8 (script `03_minuscule.py`, function `classify(..., allow_blocks=True)`). ∎

Two of these words carry tree complexes that were verified: the 13-slot `E_6` stack
(`d² = 0` on the full complex, `NOTES.md` §4) and `A_7 ω_3` (random vertices, `NOTES.md` §5).
So the no-block condition in the definition of "cyclic" is not only needed for the theorems
of §§3–4 but is also the one consistent with the Tier 2 evidence for conjecture C1.

### 2.6 Minuscule words

A *minuscule chain word* (for `λ = ω_k`) is a word with `⟨μ_{t−1}, α_{c(t)}^∨⟩ = 1` for all
`t`, where `μ_0 = ω_k`, `μ_t = μ_{t−1} − α_{c(t)}` (simply-laced, Cartan matrix of `Γ`). The
stacks of `NOTES.md` §1 are such words.

> **Lemma 2.12 (two-neighbour property).** In a minuscule chain word, for every colour `c`:
> (a) exactly `1 − δ_{c,k}` letters before the first occurrence of `c` are neighbours of `c`;
> (b) exactly two letters strictly between two consecutive occurrences of `c` are neighbours
> of `c` (counted with multiplicity).

*Proof.* `⟨μ_t, α_c^∨⟩ = δ_{c,k} − 2E_c(t) + N_c(t)` with `E_c(t)`, `N_c(t)` the numbers of
`s ≤ t` with `c(s) = c`, resp. `c(s) ∼ c` (`⟨α_c, α_c^∨⟩ = 2`, `⟨α_d, α_c^∨⟩ = −1` for
`d ∼ c`, `0` otherwise). At the first occurrence `t_1`: `1 = δ_{c,k} + N_c(t_1 − 1)`. For
consecutive occurrences `t < t'`: `⟨μ_{t'−1}, α_c^∨⟩ − ⟨μ_{t−1}, α_c^∨⟩ = 1 − 1 = 0`, and it
equals `−2 + #{t < s < t' : c(s) ∼ c}`. ∎

> **Corollary 2.13.** In a minuscule chain word, pattern B never has `x_1` = the first
> occurrence of `c`. If every colour occurs at most three times, the word is cyclic iff some
> colour `c` occurs exactly three times, at `x_1 < x_2 < x_3`, such that the two neighbour
> letters between `x_1` and `x_2` have distinct colours `c' ≠ c''` and both `c'` and `c''`
> occur after `x_3`.

*Proof.* B needs two neighbour letters (`y` of colour `c'`, `y` of colour `c''`) before `x_1`;
before the first occurrence there is at most one. With multiplicities `≤ 3`, T or B needs `c`
to occur exactly three times, so `x_1` is its first occurrence and B is excluded; in T, `x_1`
and `x_2` are consecutive occurrences, so the letters of colours `c'`, `c''` in `(x_1, x_2)`
are the two neighbour letters there (Lemma 2.12(b)). Conversely the condition is T. ∎

> **Lemma 2.14 (multiplicity bound).** In a minuscule chain word from `λ` to `μ`, colour `c`
> occurs exactly `[λ − μ : α_c]` times, and `[λ − μ : α_c] ≤ [λ − w_0 λ : α_c]`.

*Proof.* The first claim is the definition. A chain can be prolonged as long as some
`⟨μ, α_i^∨⟩ = 1`; the weights stay in the finite orbit `Wλ` (`μ − α_i = s_i μ`), and the
height of `λ − μ` grows, so the prolongation stops at `ν` with no pairing equal to `1`. As `λ`
is minuscule, all pairings of `ν` lie in `{−1, 0, 1}`, so `ν` is antidominant, `ν = w_0 λ`.
Prolonging only increases the multiplicities. (Standard facts: weights of the orbit of a
minuscule weight pair with every coroot to `−1, 0, 1`; the orbit has a unique antidominant
element `w_0 λ`.) ∎

---

## 3. Type A

### 3.1 The model

`sl_n` (`Γ = A_{n−1}`, colours `1, …, n−1`), `λ = ω_k`. Let `R = [1,k] × [1,n−k]` (boxes
`(r, s)` = (row, column)), colour `c(r, s) = k + s − r`, ordered componentwise:
`(r,s) ≤ (r',s')` iff `r ≤ r'` and `s ≤ s'`. A Young diagram `λ ⊆ R` (an order ideal) has the
*row-major word*: its boxes row by row, left to right.

> **Lemma 3.1 (model).** For `e ∈ ℤ` let `D_e = {(r, r+e)} ∩ R` (the boxes of colour `k+e`,
> listed componentwise increasing: `D_e(i) = (i, i+e)` for `e ≥ 0`, `D_e(i) = (i−e, i)` for
> `e < 0`), and for a Young diagram `λ ⊆ R` let `a_e(λ) = |λ ∩ D_e|`.
> (a) For every linear extension of `λ`, the heap of its colour word is `λ` with the
> componentwise order.
> (b) The row-major word of `λ` is a minuscule chain word for `ω_k`.
> (c) `λ ∩ D_e` is the initial segment `D_e(1), …, D_e(a_e(λ))`. Hence `λ` is determined by the
> numbers `a_e(λ)`, and `λ ⊇ λ'` iff `a_e(λ) ≥ a_e(λ')` for all `e`.
> (d) Every chain word `ω_k → μ` is the colour word of a linear extension of the Young diagram
> `λ(μ)` with `a_e(λ(μ)) = [ω_k − μ : α_{k+e}]` (the number of letters `k+e`), and every linear
> extension of a Young diagram is a chain word. Consequently the stacks of `μ` are exactly the
> linear extensions of `λ(μ)`; `μ ↦ λ(μ)` is a bijection from the weights of `Λ^k ℂ^n` onto
> the Young diagrams in `R`; `μ ≤ ν` (`ν − μ ∈ ℕ`-span of simple roots) iff `λ(μ) ⊇ λ(ν)`; and
> for `μ = e_S`, `S = {s_1 < … < s_k}`, `λ(μ)_i = s_{k+1−i} − (k+1−i)`.

*Proof.* (a) A linear extension lists `λ` compatibly with the componentwise order. The covering
relations `(r,s) ⋖ (r,s+1)`, `(r,s) ⋖ (r+1,s)` join adjacent colours, so componentwise ⊆ heap
order. Conversely two boxes of equal or adjacent colours are componentwise comparable: equal
colours `(r,s)`, `(r+t, s+t)`; colours `c` and `c+1`, boxes `(r,s)` and `(r', s−r+1+r')`: if
`r ≤ r'` the first is below the second, if `r' ≤ r − 1` the second is below the first. So every
generating relation of the heap (equal or adjacent colours, in word order) is a componentwise
relation, and so is its transitive closure.
(b) By (a), in any linear extension the letters before a box `x = (r,s)` of equal or adjacent
colour are exactly the boxes below `x` of those colours (they are comparable with `x` and
precede it). Count them: same colour: `min(r,s) − 1`; colour `c+1`: boxes `(r', s−r+1+r')` with
`max(1, r−s) ≤ r' ≤ r−1`, i.e. `min(r−1, s)`; colour `c−1`: boxes `(r', s−r−1+r')` with
`max(1, r−s+2) ≤ r' ≤ r`, i.e. `min(r, s−1)` (boxes with colour `0` or `n` would lie outside
`R`, so no non-existent colour is counted). The pairing is
`δ_{r,s} + min(r−1,s) + min(r,s−1) − 2(min(r,s) − 1)`, which is `1` in each of the cases
`r = s`, `r < s`, `r > s`. This holds for every linear extension, in particular for the
row-major one.
(c) If `D_e(i) ∈ λ` and `i ≥ 2`, then `D_e(i−1) ≤ D_e(i)` lies in `λ`.
(d) *Claim.* Let `λ ⊆ R` be a Young diagram, `μ_λ = ω_k − Σ_e a_e α_{k+e}` (`a_e = a_e(λ)`),
`c = k + d` a colour, and `y = D_d(a_d + 1)` the next box of `D_d`. Then
`⟨μ_λ, α_c^∨⟩ = δ_{d,0} + a_{d−1} + a_{d+1} − 2a_d ∈ {−1, 0, 1}`, and it equals `1` iff `y ∈ R`
and `λ ∪ {y}` is a Young diagram (`y` *addable*). (Diagonals `D_e` that miss `R` have `a_e = 0`;
they are exactly those of the non-existent colours `k + e ∉ [1, n−1]`.)
*Proof of the claim*, `d ≥ 1`. The left neighbour of `D_d(i)` is `D_{d−1}(i)`, its upper
neighbour `D_{d+1}(i−1)`. Since `λ` is a Young diagram: `D_d(a_d) ∈ λ` gives
`D_{d−1}(a_d) ∈ λ` and (for `a_d ≥ 2`) `D_{d+1}(a_d − 1) ∈ λ`; `D_{d+1}(a_{d+1}) ∈ λ` gives its
left neighbour `D_d(a_{d+1}) ∈ λ`; `D_{d−1}(a_{d−1}) ∈ λ` gives (for `a_{d−1} ≥ 2`) its upper
neighbour `D_d(a_{d−1} − 1) ∈ λ`. With (c): `ε_− := a_{d−1} − a_d ∈ {0,1}` and
`ε_+ := a_d − a_{d+1} ∈ {0,1}`, and the pairing is `ε_− − ε_+`. It is `1` iff `ε_− = 1` and
`ε_+ = 0`, i.e. iff the left neighbour `D_{d−1}(a_d + 1)` of `y` is in `λ` and either `a_d = 0`
or the upper neighbour `D_{d+1}(a_d)` of `y` is in `λ`. These are the lower covers of `y`; if they
are in `λ ⊆ R`, so is the row of `y`, and its column is that of the upper neighbour or, for
`a_d = 0`, is `1 + d ≤ n − k` (as `c ≤ n − 1`); so this is exactly "`y` addable". `d = 0`: the
neighbours of `D_0(i) = (i,i)` are `D_{−1}(i−1)` (left) and `D_1(i−1)` (upper), and the same
reasoning gives `a_{±1} ∈ {a_0 − 1, a_0}`, pairing `1 − (a_0 − a_1) − (a_0 − a_{−1})`, equal to `1`
iff `a_1 = a_{−1} = a_0`, iff `y = (a_0+1, a_0+1)` is addable. `d ≤ −1`: transposition
`(r,s) ↦ (s,r)` maps `λ` to `λ'` in `R' = [1,n−k] × [1,k]`, `D_e` to `D_{−e}`, preserves
addability and the pairing formula; apply the case `−d ≥ 1` to `λ'` (for `Λ^{n−k}`).
*Induction.* The empty word is the colour word of the empty diagram. If `u` is the colour word of
a linear extension of `λ`, the weight reached by `u` is `μ_λ` (the letters `k+e` are the boxes of
`D_e`); if `u·c` is a chain word, the claim gives an addable `y` of colour `c`; `y` is maximal
in `λ ∪ {y}` (a box above `y` in `λ` would put `y` in `λ`), so `u·c` is the colour word of a linear
extension of `λ ∪ {y}`. Conversely every linear extension of a Young diagram is a chain word, by
the proof of (b). *Consequences.* By (c), `λ(μ)` is determined by `μ`, and distinct diagrams give
distinct weights; every diagram is reached (b). The weights of `Λ^k ℂ^n` are the `e_S`, `C(n,k)` of
them, as many as Young diagrams in `R`, and the formula for `λ(e_S)` exhibits the bijection:
adding the box `(r, λ_r + 1)` of colour `c = k + λ_r + 1 − r` subtracts `α_c = ε_c − ε_{c+1}`,
i.e. replaces `c` by `c + 1` in `S`; starting from `S = {1, …, k}`, the element `s_{k+1−r}` is
`k + 1 − r + λ_r` throughout. Weight order: `ν − μ ∈ ℕ`-span iff
`a_e(λ(μ)) ≥ a_e(λ(ν))` for all `e`, iff `λ(μ) ⊇ λ(ν)` by (c). Finally, no box is addable to `R`,
so by the claim the row-major word of `R` cannot be prolonged: it ends at the lowest weight. ∎

This makes the type A model self-contained. (For the general minuscule theory see
R. A. Proctor, *Bruhat lattices, plane partition generating functions, and minuscule
representations*, European J. Combin. 5 (1984) 331–350, and J. R. Stembridge, *Minuscule
elements of Weyl groups*, J. Algebra 235 (2001) 722–743, where `λ`-minuscule elements are shown
to be fully commutative; I have not checked theorem numbers, and nothing here depends on them.)
*Checks:* `03_typeA.py` (`model_check`: row-major words of all rectangles up to 6×6 are chain
words to the lowest weight with componentwise heap; `addable_check`: the claim for all 10 164
pairs (diagram in the 6×6 box, diagonal); weight order = reverse inclusion for `sl_n`,
`n ≤ 10`); `03_minuscule.setup` (for every weight, #chain words = #linear extensions of its
ideal); `03_deodhar.py` (the formula for `λ(e_S)`, `n ≤ 9`); and the referee's `R03_typeA.py`
(`A_11 ω_6`, all 924 weights, one commutation class of chain words each).

Since cyclicity depends only on the heap (1.4), **"the stack of `μ` is cyclic" means "the
row-major word of `λ(μ)` is cyclic"**, and everything below is about Young diagrams.

### 3.2 The criterion

> **Lemma 3.2.** If `λ ⊉ (4,4,4,3)`, every colour occurs at most three times in `λ`. If some
> colour occurs four times, `λ ⊇ (4,4,4,4)`.

*Proof.* The boxes of colour `k + d` are the diagonal `D_d = {(r, r+d)}`; distinct boxes of
`D_d` lie in distinct rows and in distinct columns. `λ ⊉ (4,4,4,3)` means `λ_3 ≤ 3` or
`λ_4 ≤ 2`. If `λ_3 ≤ 3`: a box `(r, r+d)` with `d ≥ 1` has `r ≤ 2` (for `r ≥ 3`,
`r + d ≤ λ_r ≤ 3` is impossible), so `|D_d| ≤ 2`; a box with `d ≤ 0` lies in column
`r + d ≤ 3` (for `r ≥ 3` because `λ_r ≤ 3`, for `r ≤ 2` because `d ≤ 0`), so `|D_d| ≤ 3`. If
`λ_4 ≤ 2`: symmetrically, `d ≤ −1` gives columns `≤ 2`, so `|D_d| ≤ 2`; `d ≥ 0` gives rows
`≤ 3`, so `|D_d| ≤ 3`. Second claim: four boxes on `D_d` give `(4, 4+d) ∈ λ` (`d ≥ 0`) or
`(4−d, 4) ∈ λ` (`d < 0`), hence `(4,4) ∈ λ`. ∎

> **Theorem A.** For a Young diagram `λ ⊆ k × (n−k)`, the stack (any linear extension of
> `λ`, colours `k + s − r`) is cyclic if and only if `λ ⊇ (4,4,4,3)`, i.e. iff `λ_3 ≥ 4` and
> `λ_4 ≥ 3`, i.e. iff the boxes `(3,4)` and `(4,3)` belong to `λ`. For `μ = e_S`,
> `S = {s_1 < … < s_k}`: if `k ≥ 4`, iff `s_{k−2} ≥ k+2` and `s_{k−3} ≥ k`; if `k ≤ 3`, never.

*Proof.* "⇐" is `PROOFS.md` Prop. 8: the configuration given there is pattern T with
`c = k`, `x_1, x_2, x_3 = (1,1), (2,2), (3,3)`, pairs `((2,1), (4,3))` and `((1,2), (3,4))`
(Theorem 2.8 re-proves that it is cyclic).

"⇒". Suppose `λ ⊉ (4,4,4,3)`. By Lemma 3.2 every colour occurs at most three times, and by
Lemma 3.1(b) the row-major word is a minuscule chain word. If it were cyclic, Corollary 2.13
would give a diagonal `D_d` with exactly three boxes `x_1 < x_2 < x_3` such that two distinct
neighbours of `k + d` — necessarily `k + d ± 1`, the diagonals `D_{d±1}` — have boxes above
`x_3`. If `d ≥ 0`: `x_3 = (3, 3+d)`; a box `(r, r+d+1)` of `D_{d+1}` is above `x_3` iff `r ≥ 3`,
so `(3, 4+d) ∈ λ`, whence `(3,4) ∈ λ`; a box `(r, r+d−1)` of `D_{d−1}` is above `x_3` iff
`r ≥ 3` and `r + d − 1 ≥ 3 + d`, i.e. `r ≥ 4`, so `(4, 3+d) ∈ λ`, whence `(4,3) ∈ λ`. If
`d < 0`: `x_3 = (3−d, 3)`; a box of `D_{d+1}` above it is `(r, r+d+1)` with `r ≥ 3−d`, so
`(3−d, 4) ∈ λ` and, as `3 − d ≥ 4`, `(4,4) ∈ λ`; a box of `D_{d−1}` above it is `(r, r+d−1)`
with `r ≥ 4−d`, so `(4−d, 3) ∈ λ` and `(4,3) ∈ λ`. In both cases `λ ⊇ (4,4,4,3)`:
contradiction. The formula in `S`: for `k ≥ 4`, `λ_3 = s_{k−2} − (k−2)` and
`λ_4 = s_{k−3} − (k−3)`; for `k ≤ 3`, `λ` has at most three rows, so `λ_4 = 0`. ∎

*Checks.* `03_typeA.py 7`: all 3432 partitions in the 7×7 box (2116 cyclic, 1316 not): the
exhaustive search agrees with `λ ⊇ (4,4,4,3)`; Lemma 3.2 holds; the Prop. 8 configuration is
cyclic in every `λ ⊇ (4,4,4,3)`. `03_minuscule.py` (all ideals of `A_7 ω_4`, `A_8 ω_4`,
`A_9 ω_4`, `A_9 ω_5`): 2, 9, 25, 42 cyclic ideals, one minimal.

**On the symmetry `k ↔ n − k`.** The task statement of this front says `(4,4,4,3)` is not symmetric under
transposition. It is: its rows are `4,4,4,3` and its columns are `4,4,4,3` (column 4 has the
boxes of rows 1–3 only). Transposition `(r, s) ↦ (s, r)` maps the stack of `λ` for `Λ^k` to
the stack of `λ'` for `Λ^{n−k}` with colours `c ↦ n − c` (box `(s, r)` of `λ'` has colour
`(n−k) + r − s = n − c(r,s)`), an automorphism of `A_{n−1}`; so `λ` is cyclic iff `λ'` is,
consistent with `λ ⊇ (4,4,4,3) ⇔ λ' ⊇ (4,4,4,3)`. There is nothing to correct in the
criterion; the exhaustive computation over the 6×6 and 7×7 boxes confirms it as stated.

### 3.3 Minimal cyclic configurations in type A

> **Proposition 3.3.** (a) If `λ ⊇ (4,4,4,3)` and every diagonal of `λ` has at most three
> boxes (e.g. `λ = (4,4,4,3), (5,4,4,3), (4,4,4,3,1), (5,5,4,3)`), then `λ` has exactly one
> inclusion-minimal cyclic configuration: mover `(3,3)`, pairs `((1,1),(2,2))`,
> `((2,1),(4,3))`, `((1,2),(3,4))`. Every cyclic configuration of such `λ` contains it.
> (b) For general `λ` the inclusion-minimal cyclic configurations are the unblocked tadpoles
> of Lemma 2.3, and every cyclic `λ` contains a configuration of pattern T (the one in (a)).

*Proof.* (a) By Corollary 2.10 and Corollary 2.13 every minimal cyclic configuration is a
4-cycle of pattern T with `x_1 < x_2 < x_3` the three boxes of a diagonal `D_d` and with boxes of
`D_{d±1}` above `x_3`. The proof of Theorem A shows that `d ≥ 1` forces `(4, 3+d) ∈ λ` and
`d ≤ −1` forces `(3−d, 4) ∈ λ`; both put `(4,4)` into `λ`, so `D_0` would have four boxes. So
`d = 0`, `x_i = (i, i)`. The pair of colour `k − 1` has a box in `(x_1, x_2)` — the only box of
`D_{−1}` between `(1,1)` and `(2,2)` is `(2,1)` — and a box of `D_{−1}` above `(3,3)`, i.e.
`(r, r−1)` with `r ≥ 4`; as `|D_{−1}| ≤ 3`, this is `(4,3)`. Symmetrically `((1,2),(3,4))`.
(b) Lemma 2.3 and Theorem A. ∎

*Addendum to (a) (suggested by the referee).* For `λ = (4,4,4,3)` this configuration is the
**only** cyclic configuration: any cyclic configuration contains it, and the boxes it leaves free
carry only two further pairs, `((1,3),(2,4))` (colour `k+2`) and `((3,1),(4,2))` (colour `k−2`);
the first has both slots inside `((1,2),(3,4))` (they lie between `(1,2)` and `(3,4)` in the heap),
the second both inside `((2,1),(4,3))`, so neither is linked to anything and adding them
disconnects. For `(5,4,4,3)`, `(4,4,4,3,1)`, `(5,5,4,3)` the same holds by brute force over all
configurations (`03_typeA.py`, function `all_cyclic_brute`; also the referee's `R03_typeA.py`).

Beyond (a) there is no short list: `03_typeA.py` finds, as (number of minimal cyclic
configurations; cycle lengths; distances of the mover from the cycle):
`(4,4,4,4)`: 6; 4; 0–1. `(5,5,5,5)`: 32; 4, 6; 0–3. `5^5`: 414; 4, 6, 8; 0–4. Example of a
chordless 6-cycle through the mover in `(5,5,5,5)`: mover `(2,2)`, pairs `((1,2),(2,3))`,
`((1,3),(3,5))`, `((3,4),(4,5))`, `((3,3),(4,4))`, `((2,1),(4,3))` (colours along the cycle
`k, k+1, k+2, k+1, k, k−1`). All three patterns occur in type A: among the 924 partitions in
the 6×6 box, 362 are cyclic; all contain T, 262 also B, 162 also N.

### 3.4 Consequences (task 3)

> **Corollary 3.4.** (a) `sl_n` with `ω_1, ω_2, ω_3, ω_{n−3}, ω_{n−2}, ω_{n−1}`: no stack is
> cyclic, for every `n`. (b) The full minuscule poset `k × (n−k)` of `Λ^k ℂ^n` is cyclic iff
> `4 ≤ k ≤ n − 4` (so `n ≥ 8`; first case `sl_8, ω_4`). (c) The cyclic stacks of `Λ^k ℂ^n` are
> those of the weights `μ ≤ μ_c`, `μ_c = e_S`, `S = {1,…,k−4} ∪ {k, k+2, k+3, k+4}`
> (`λ(μ_c) = (4,4,4,3)`), whenever `4 ≤ k ≤ n−4`; there are none otherwise.

*Proof.* (a) `λ ⊆ k × (n−k)` has at most `k ≤ 3` rows, so `λ_4 = 0`, or at most `n − k ≤ 3`
columns, so `λ_3 ≤ 3`. (b) `(4,4,4,3) ⊆ k × (n−k)` iff `k ≥ 4` and `n − k ≥ 4`. (c) Theorem A
and Lemma 3.1(d) (stacks of `μ` = linear extensions of `λ(μ)`, the formula for `λ(e_S)`, and
weight order = reverse inclusion). ∎

(For `ω_1, ω_2` also Prop. 2.2 with Lemma 2.14 applies: `[ω_k + ω_{n−k} : α_c] = min(c, k, n−k, n−c) ≤ 2`.)

---

## 4. Type D

`Γ = D_n` (`n ≥ 4`), Bourbaki labels: `1 — 2 — … — (n−2)`, `(n−2) — (n−1)`, `(n−2) — n`.

### 4.1 Vector representation

> **Proposition 4.1.** For `D_n`, `ω_1`, every colour occurs at most twice in every stack;
> hence no stack is cyclic (all `n ≥ 4`).

*Proof.* `ω_1 = ε_1 = α_1 + … + α_{n−2} + ½(α_{n−1} + α_n)` (`α_i = ε_i − ε_{i+1}`, `i < n`,
`α_n = ε_{n−1} + ε_n`). A chain step `μ ↦ μ − α_i = s_i μ` keeps `μ` in `{±ε_1, …, ±ε_n}` (the
`s_i` act by signed permutations), so every stack ends at some `μ = ±ε_i`, and colour `c` occurs
`[ε_1 − μ : α_c]` times. Since `ε_1 + μ` is `0`, `2ε_1` or a positive root, it lies in the
`ℕ`-span of the simple roots, so `[ε_1 − μ : α_c] ≤ [2ε_1 : α_c] ∈ {2, 1}`. Prop. 2.2. (This is
Lemma 2.14 made explicit, without the general facts on minuscule orbits.) ∎

(*Lemma 4.0, for `D_4`:* the three minuscule representations of `D_4` are permuted by
triality, and for `ω_1` the bound is `(2,2,1,1)`; so all of `D_4` is covered by Prop. 2.2.)
`03_typeD.py` checks the chain `1, 2, …, n−2, n−1, n, n−2, …, 1` and the absence of cyclic
ideals for `n ≤ 11`.

### 4.2 Half-spin representations: the model

By the diagram automorphism exchanging `n−1, n` it suffices to treat `ω_n`. Let
`P_n = {(u, v) : 1 ≤ u < v ≤ n}`, ordered componentwise, with colours
`c(u, v) = n − (v − u)` if `v − u ≥ 2`, and `c(u, u+1) = n` for `u` odd, `n − 1` for `u` even.
*Row `u`* is `(u, u+1), …, (u, n)`; its colours are `n` or `n−1`, then `n−2, n−3, …`, pairwise
distinct. The order ideals of `P_n` are the sets `I_ℓ = {(u, v) : v − u ≤ ℓ_u}` for the strict
partitions `ℓ = (ℓ_1 > ℓ_2 > … > ℓ_t > 0)` with `ℓ_u ≤ n − u` (a down-set meets row `u` in an
initial segment of length `ℓ_u`, and `(u, u + ℓ_u) ∈ I` forces `(u−1, u + ℓ_u) ∈ I`, i.e.
`ℓ_{u−1} ≥ ℓ_u + 1`); there are `2^{n−1}` of them.

For `1 ≤ r ≤ n−1` let `T_r = {(u, u+r)}`, listed as `T_r(i) = (i, i+r)`, `1 ≤ i ≤ n − r`
(colour `n − r` for `r ≥ 2`; for `r = 1` colour `n` at odd `i`, `n − 1` at even `i`), and for an
ideal `I` let `b_r(I) = |I ∩ T_r|`.

> **Lemma 4.2 (model).** (a) For every linear extension of an ideal `I` of `P_n`, the heap of
> its colour word is `I` with the componentwise order. (b) The row-major word of `I_ℓ` is a
> minuscule chain word for `ω_n`. (c) `I ∩ T_r` is the initial segment of length `b_r(I)`; the
> colour content of `I` is: `b_r(I)` letters `n − r` (`r ≥ 2`), `⌈b_1/2⌉` letters `n` and
> `⌊b_1/2⌋` letters `n − 1`. Hence `I` is determined by its colour content, and `I ⊇ I'` iff the
> content of `I` is componentwise `≥` that of `I'`. (d) Every chain word `ω_n → μ` is the colour
> word of a linear extension of the ideal `I(μ)` with content `ω_n − μ`, and every linear
> extension of an ideal is a chain word. Consequently the stacks of `μ` are exactly the linear
> extensions of `I(μ)`; `μ ↦ I(μ)` is a bijection from the `2^{n−1}` weights of the half-spin
> representation onto the strict partitions; and `μ ≤ ν` iff `I(μ) ⊇ I(ν)`.

*Proof.* (a) Covering relations `(u,v) ⋖ (u,v+1)` and `(u,v) ⋖ (u+1,v)` (`u + 1 < v`) join
adjacent colours: `n−r` and `n−r−1` (`r ≥ 2`) or `n/n−1` and `n−2` (`r = 1`); resp. `n−r`, `n−r+1`
(`r ≥ 3`) or `n−2` and `n/n−1` (`r = 2`). Equal or adjacent colours are comparable: equal
colours lie on one `T_r`; for colours `n−r`, `n−r−1` (`r ≥ 2`): `(u, u+r) ≤ (u', u'+r+1)` iff
`u ≤ u'`, and `(u', u'+r+1) ≤ (u, u+r)` iff `u' ≤ u − 1`; for `n−2` and `n` or `n−1`:
`(u, u+2) ≤ (u', u'+1)` iff `u ≤ u' − 1`, and `(u', u'+1) ≤ (u, u+2)` iff `u' ≤ u`. Conclude as in
Lemma 3.1(a).
(b) As in Lemma 3.1(b), in any linear extension the pairing at `x` is `δ_{c(x), n} + N − 2E`
with `N`, `E` the numbers of boxes below `x` of adjacent resp. equal colour.
`x = (u, u+r)`, `r ≥ 3`: `E = u − 1`, `N = (u − 1) + u` (colour `n−r−1`: `(u', u'+r+1)`,
`u' ≤ u−1`; colour `n−r+1`: `(u', u'+r−1)`, `u' ≤ u`); pairing `1`. `r = 2`: `E = u − 1`,
`N = (u−1) + u` (colour `n−3`: `u' ≤ u−1`; colours `n−1`, `n`: `(u', u'+1)`, `u' ≤ u`); pairing `1`.
`r = 1`: `E = ⌊(u−1)/2⌋` (same parity), `N = u − 1` (colour `n−2`: `(u', u'+2)`, `u' ≤ u−1`),
`δ = 1` iff `u` odd; pairing `1 + (u−1) − (u−1) = 1` (`u` odd) or `(u−1) − (u−2) = 1` (`u` even).
(c) If `T_r(i) ∈ I` and `i ≥ 2`, then `T_r(i−1) ≤ T_r(i)` is in `I`. The content is read off; it
determines all `b_r` (`b_1 = ⌈b_1/2⌉ + ⌊b_1/2⌋`), hence `I`. Inclusion: `I ⊇ I'` iff
`b_r(I) ≥ b_r(I')` for all `r`; for `r ≥ 2` this is the content of colour `n − r`, and for `r = 1`,
`b_1 ≥ b'_1` iff `⌈b_1/2⌉ ≥ ⌈b'_1/2⌉` and `⌊b_1/2⌋ ≥ ⌊b'_1/2⌋` (both are non-decreasing in `b_1`,
and their sum is `b_1`).
(d) *Claim.* Let `I` be an ideal, `b_r = b_r(I)` (`b_n := 0`), `μ_I = ω_n − content(I)`, `c` a
colour. (i) If `c = n − r`, `r ≥ 2`: `⟨μ_I, α_c^∨⟩ = b_{r−1} + b_{r+1} − 2b_r ∈ {−1, 0, 1}`, and it
is `1` iff `y = T_r(b_r + 1)` is *addable* (`y ∈ P_n`, `I ∪ {y}` an ideal). (ii) If `c ∈ {n, n−1}`:
the pairing lies in `{−1, 0, 1}` and is `1` iff `y = T_1(b_1 + 1)` is addable and has colour `c`.
*Proof.* The ideal property gives `b_{r+1} ∈ {b_r − 1, b_r}` for `r ≥ 1`: `T_{r+1}(i)` has left
neighbour `T_r(i)`, so `b_r ≥ b_{r+1}`; `T_r(i)`, `i ≥ 2`, has upper neighbour `T_{r+1}(i−1)`, so
`b_{r+1} ≥ b_r − 1`. (i) The neighbours of `n − r` are `n − r ± 1`, except that for `r = 2` the
colours `n−1`, `n` (all of `T_1`, `b_1` boxes) replace `n − r + 1`, and for `r = n − 1` there is
no `n − r − 1` (`T_n = ∅`). So the pairing is `(b_{r−1} − b_r) − (b_r − b_{r+1})`, a difference of
two numbers in `{0, 1}`, equal to `1` iff `b_{r−1} = b_r + 1` and `b_{r+1} = b_r`, i.e. iff the
left neighbour `T_{r−1}(b_r + 1)` of `y` and (if `b_r ≥ 1`) its upper neighbour `T_{r+1}(b_r)` lie
in `I`; these are the lower covers of `y`, and then `y ∈ P_n` (its row is that of the left
neighbour; its column `b_r + 1 + r` is that of the upper neighbour, or `1 + r ≤ n` if `b_r = 0`).
(ii) The only neighbour of `n` and of `n − 1` is `n − 2`. For `c = n` the pairing is
`1 − 2⌈b_1/2⌉ + b_2`: if `b_1` is even, `1 − (b_1 − b_2) ∈ {0, 1}`, equal to `1` iff `b_2 = b_1`; if
`b_1` is odd, `−(b_1 − b_2) − 0 ∈ {−1, 0}`. For `c = n − 1` it is `−2⌊b_1/2⌋ + b_2`: if `b_1` is
odd, `1 − (b_1 − b_2) ∈ {0, 1}`, `= 1` iff `b_2 = b_1`; if even, `−(b_1 − b_2) ∈ {−1, 0}`. The box
`y = (b_1 + 1, b_1 + 2)` has colour `n` iff `b_1` is even, and its only lower cover is the upper
neighbour `T_2(b_1)` (if `b_1 ≥ 1`), which lies in `I` iff `b_2 = b_1`; and then `y ∈ P_n`.
*Induction and consequences* exactly as in Lemma 3.1(d), with (c) in place of 3.1(c): a chain
word `u·c`, `u` a linear extension of `I`, gives an addable `y` of colour `c`, maximal in
`I ∪ {y}`; distinct ideals have distinct contents, hence distinct weights; every ideal is reached
(b); there are `2^{n−1}` strict partitions with parts `≤ n − 1` (subsets of `{1, …, n−1}`; then
`ℓ_u ≤ n − u` holds automatically) and `2^{n−1}` weights of the half-spin representation
(`½(±1, …, ±1)` with an even number of minus signs, all of multiplicity one), so all weights are
reached; weight order: `ν − μ ∈ ℕ`-span iff content`(I(μ)) ≥` content`(I(ν))` iff
`I(μ) ⊇ I(ν)` by (c). No box is addable to `P_n`, so the row-major word of `P_n` ends at the
lowest weight. ∎

(The referee's `R03_referee.md` §5.3 gives the equivalent sign-vector description: `I_ℓ` has its
minus signs at the positions `{n − ℓ_u}`, plus `n` if `ℓ` has an odd number of parts.)
*Checks* for `4 ≤ n ≤ 11`: `03_typeD.py` (`model_check`: the row-major word of `P_n` is a chain to
the lowest weight with componentwise heap, `2^{n−1}` ideals; `addable_check`: the claim for every
ideal and every colour; weight order = reverse inclusion for `n ≤ 9`), `03_minuscule.setup`
(chain words = linear extensions), and the referee's `R03_typeD.py` (`D_4 … D_9`).

### 4.3 The criterion

> **Lemma 4.3.** If `ℓ_4 = 0` or `ℓ_3 ≤ 2`, every colour occurs at most three times in `I_ℓ`.

*Proof.* Each colour occurs at most once per row. If `ℓ_4 = 0` there are at most three rows.
Otherwise strictness and `ℓ_3 ≤ 2` force `ℓ_3 = 2`, `ℓ_4 = 1`, `ℓ_5 = 0`: rows 3, 4 are
`(3,4), (3,5)` (colours `n`, `n−2`) and `(4,5)` (colour `n−1`), and colour `n` occurs in rows 1, 3,
colour `n−1` in rows 2, 4, colour `n−2` at most in rows 1, 2, 3, all other colours only in rows
1, 2. ∎

> **Theorem D.** Let `n ≥ 4` and `ℓ` a strict partition with `ℓ_u ≤ n − u`. The stack of
> `I_ℓ` (half-spin representation of `D_n`) is cyclic iff `ℓ_3 ≥ 3` and `ℓ_4 ≥ 1`, i.e. iff
> `ℓ ⊇ (5,4,3,1)`, i.e. iff `(3,6), (4,5) ∈ I_ℓ`. Such `ℓ` exist iff `n ≥ 6`.

*Proof.* (`ℓ_3 ≥ 3` and `ℓ_4 ≥ 1` imply `ℓ_2 ≥ 4`, `ℓ_1 ≥ 5` by strictness, and
`I_{(5,4,3,1)}` is the ideal generated by `(3,6)` and `(4,5)`; `(3,6)` needs `n ≥ 6`.)

"⇐": pattern T with `c = n − 2`: `x_1, x_2, x_3 = (1,3), (2,4), (3,5)` (colour `n−2`); colour
`n−3`: `y = (1,4)`, `z = (3,6)`; colour `n−1`: `y = (2,3)`, `z = (4,5)`. Indeed
`(1,3) < (1,4) < (2,4)`, `(1,3) < (2,3) < (2,4)`, `(3,5) < (3,6)`, `(3,5) < (4,5)` componentwise, and
`n−3`, `n−1` are distinct neighbours of `n−2`. By Lemma 4.2(a) the order of these slots in the
word is the componentwise order; Theorem 2.8 gives the cyclic configuration: mover `(3,5)`,
pairs `((1,3),(2,4))`, `((1,4),(3,6))`, `((2,3),(4,5))`.

"⇒": suppose `ℓ_4 = 0` or `ℓ_3 ≤ 2`. By Lemma 4.3 and Lemma 4.2(b), Corollary 2.13 applies: a
cyclic `I_ℓ` would have a colour `c` with exactly three occurrences `x_1 < x_2 < x_3`, the two
neighbour letters between `x_1, x_2` of distinct colours `c', c''`, and both occurring after
`x_3`. Colours `n`, `n−1` occur at most twice (rows `u` odd resp. even, `u ≤ 4`). So
`c = n − r`, `r ≥ 2`, occurring in rows 1, 2, 3 (rows `≥ 4` have length `≤ 1`):
`x_u = (u, u+r)`. The boxes of colours adjacent to `c` lying between `x_1` and `x_2` — in the
heap, equivalently in any linear extension, since such boxes are comparable with `x_1` and
`x_2` (Lemma 4.2(a)) — are those strictly between them in the componentwise order, namely
`(1, 2+r)` (colour `n−r−1`) and `(2, 1+r)` (colour `n−r+1` if `r ≥ 3`, `n−1` if `r = 2`); in
the row-major word further letters (the rest of row 1, the start of row 2) lie between them,
of non-adjacent colours. If `r ≥ 3`: a box of colour
`n−r+1` above `x_3 = (3, 3+r)` is `(u, u+r−1)` with `u ≥ 4`; row 4 has length `≤ 1 < r − 1`.
Impossible. If `r = 2`: colour `n−1` above `(3,5)` is `(u, u+1)` with `u ≥ 4`, so `(4,5) ∈ I_ℓ`,
`ℓ_4 ≥ 1`; colour `n−3` above `(3,5)` is `(u, u+3)` with `u ≥ 3`, so `(3,6) ∈ I_ℓ`, `ℓ_3 ≥ 3`.
Contradiction. ∎

> **Corollary 4.4.** (a) The full half-spin poset of `D_n` is cyclic iff `n ≥ 6`; `D_4`, `D_5`
> have no cyclic stack. (b) For `n ≥ 6` the cyclic weights are exactly `μ ≤ μ_c` with
> `I_{μ_c} = I_{(5,4,3,1)}` (13 slots), `μ_c = ½(+,…,+,−,−,−,+,−,+)` (in `ε`-coordinates; the
> `D_6` pattern `½(−,−,−,+,−,+)` on the last six coordinates), `ω_n − μ_c = α_{n−5} + 2α_{n−4} +
> 3α_{n−3} + 3α_{n−2} + 2α_{n−1} + 2α_n`; for `D_6`, `μ_c = −ω_3 + ω_4 − ω_5` = lowest weight
> `+ α_4 + α_6`. (c) Every non-cyclic half-spin stack has colour multiplicities `≤ 3`.

*Proof.* (a) `ℓ = (n−1, …, 1)` has `ℓ_3 = n − 3`, `ℓ_4 = n − 4`. (b) Theorem D; the root content
is read off the 13 boxes (rows `(1,2),…,(1,6)`; `(2,3),…,(2,6)`; `(3,4),(3,5),(3,6)`; `(4,5)`:
colours `n` twice, `n−1` twice, `n−2`, `n−3` three times, `n−4` twice, `n−5` once). With
`α_i = ε_i − ε_{i+1}` (`i < n`), `α_n = ε_{n−1} + ε_n`, this content is
`ε_{n−5} + ε_{n−4} + ε_{n−3} + ε_{n−1}` (coefficients `1, 2−1, 3−2, 3−3, 2−3+2, −2+2` at
`ε_{n−5}, …, ε_n`), so `μ_c = ½(1, …, 1) − (ε_{n−5} + ε_{n−4} + ε_{n−3} + ε_{n−1}) =
½(+,…,+,−,−,−,+,−,+)` (also checked by `03_typeD.py`, `6 ≤ n ≤ 11`). The full staircase minus
`I_{(5,4,3,1)}` in `D_6` is `{(4,6), (5,6)}`, of colours `4`, `6`. (c) Lemma 4.3. ∎

*Checks.* `03_typeD.py 11`: for `4 ≤ n ≤ 11`, all `2^{n−1}` strict partitions: exhaustive search
⇔ `ℓ_3 ≥ 3 and ℓ_4 ≥ 1`; for the non-cyclic ones multiplicities `≤ 3`, no pattern T/B/N, and
(for `n ≤ 9`) the brute force over all configurations finds nothing; the explicit
configuration is cyclic for the cyclic ones. Counts of cyclic ideals: `D_6 … D_{11}`:
3, 16, 54, 148, 361, 820 (`03_minuscule.py` up to `D_{12}`: 1780, one minimal ideal, 13 slots,
in every case).

---

## 5. Types E₆ and E₇ (computer-assisted)

Bourbaki labels (`E_6`: `1 — 3 — 4 — 5 — 6`, `2 — 4`; `E_7`: add `6 — 7`). The poset `P` is the
heap of the chain `ω_k →` lowest weight chosen greedily (smallest available node;
`03_lib.minuscule_full_word`): `E_6, ω_1`: `1342543165423456`; `E_6, ω_6`:
`6542314354265431`; `E_7, ω_7`: `765423143542654317654234567`.

> **Theorem E.** (a) `E_6`, `ω_1`: of the 27 order ideals (= weights) exactly 3 are cyclic
> (14, 15, 16 slots); they are the ideals containing `I_c`, 14 slots, word `13425431654234`,
> `μ_c = −ω_4 + ω_5` = lowest weight `+ α_5 + α_6`, `ω_1 − μ_c = (2,2,3,4,2,1)`.
> (b) `E_6`, `ω_6`: the image of (a) under the diagram automorphism; `μ_c = ω_3 − ω_4`,
> `ω_6 − μ_c = (1,2,2,4,3,2)`, word `65423143542654`.
> (c) `E_7`, `ω_7`: of the 56 ideals exactly 22 are cyclic; they are the ideals containing
> `I_c`, 15 slots, word `765423143542654`, `μ_c = ω_3 − ω_4 + ω_7`,
> `ω_7 − μ_c = (1,2,2,4,3,2,1)`.
> (d) In all three cases every non-cyclic ideal has colour multiplicities `≤ 3`; the 13-slot
> ideal of `E_6` (the Billey–Jones pattern, word `1342543165423`) is **not** cyclic; `I_c` has
> exactly two inclusion-minimal cyclic configurations, both 4-cycles through the mover, one of
> pattern B and one of pattern N, and no configuration of pattern T:
>
> | case | pattern B | pattern N |
> |---|---|---|
> | `E_6 ω_1` | mover 6 (c4); `(11,14)` c4, `(2,13)` c3, `(4,12)` c2 | mover 10 (c5); `(3,14)` ⊃ `(6,11)` c4, `p = (2,7)` c3 |
> | `E_6 ω_6` | mover 7 (c4); `(10,14)` c4, `(2,13)` c5, `(4,11)` c2 | mover 8 (c3); `(3,14)` ⊃ `(7,10)` c4, `p = (2,9)` c5 |
> | `E_7 ω_7` | mover 8 (c4); `(11,15)` c4, `(3,14)` c5, `(5,12)` c2 | mover 9 (c3); `(4,15)` ⊃ `(8,11)` c4, `p = (3,10)` c5 |
>
> (slots of the word of `I_c`; in `E_6 ω_1` colour 4 sits at slots 3, 6, 11, 14, and the
> pattern-B mover is its second occurrence, consistent with Corollary 2.13.)
> (e) The order on weights is reverse inclusion of ideals, so "cyclic iff `μ ≤ μ_c`".

*Proof (computer).* `python3 proofs/checks/03_typeE.py` (95 s). For every ideal, cyclicity is
decided by three independent methods, which agree on all 27 + 27 + 56 ideals:

* **M1** — the exhaustive search of Lemma 2.3(e) (`Word.search`): all tree configurations of
  every mover, no bound on the number of pairs.
* **M2** — the literal pattern detector for T, B, N (`03_patterns.patterns`), valid by
  Theorem 2.8.
* **M3** — brute force over **all** configurations: every mover `j` and every set of pairwise
  disjoint equal-coloured pairs avoiding `j` (`Word.brute`), tested against the definition
  1.3 directly. For a cyclic ideal it stops at the first cyclic configuration; for the
  non-cyclic ideals it examines every configuration (2 632 for `E_6`, 14 505 for `E_7` in
  total).

Then it checks that the cyclic ideals are exactly those containing the unique minimal one,
lists all cyclic configurations of `I_c` with three pairs and all minimal ones
(`Word.minimal_cyclic`, which by Lemma 2.3(c) contains every minimal configuration), and
compares the weight order with inclusion. ∎

**What the computer proof assumes.** (1) The definitions 1.3 as implemented in
`03_lib.Word.pair_rel`, `mover_rel`, `graph`, `is_cyclic_config` (about 40 lines). My pattern
detector agrees with them on 3000 random words (`03_patterns.py`), but it is by the same author
from the same reading, so this is a consistency check only. Genuinely independent checks are
the referee's: `checks/R03_core.py` (definitions written from `PROOFS.md` Cor. 3.1, brute
force, connected-growth search, pattern detector), `R03_typeE.py` (Theorem E re-derived, the
B/N table slot for slot), and `R03_compare_author.py` (agreement with `03_lib` / `03_patterns`
on 3000 random words, 462 cyclic, 0 disagreements); see `R03_referee.md` §5. M3 uses nothing
else. (2) For M1: Lemma 2.3; for M2: Theorem 2.8 — both are
not needed for the conclusion, since M3 alone decides every ideal. (3) The identification of
stacks with ideals: `P` is the heap of one chain to the lowest weight, its ideals correspond
to the weights, and every chain word `ω_k → μ` is a linear extension of `I_μ`.
`03_minuscule.setup` checks all of this for the cases at hand: the 27 resp. 56 ideals give 27
resp. 56 distinct weights, every ideal's subword is a chain, and for every weight the number
of chain words to it equals the number of linear extensions of its ideal (linear extensions
of `I_μ` are chain words to `μ` — the pairing at a letter only depends on the equal- and
adjacent-coloured letters below it — so the two sets coincide). It does not assume the
general theorems of Proctor and Stembridge. (4) Python's integer arithmetic. There is no
bound on the number of pairs anywhere.

**On the bounded search of `cyclefree.find_cycle`.** Theorem 2.8 shows a posteriori that the
bound was harmless: every cyclic word has a cyclic configuration with **three** pairs, so
`find_cycle(st, max_pairs=3)` is already exhaustive for the yes/no question (it grows connected
configurations, and the 3-pair configuration `M — q — p — q'` is connected after adding `q`,
then `p`, then `q'`). The earlier computations with `max_pairs = 4` (and 6 near the threshold)
are therefore correct as they stand; `NOTES.md` §4's "[verified … cycles with up to 4 pairs]"
can be upgraded to "proved".

---

## 6. The unique minimal cyclic weight (task 6)

> **Theorem U.** Let `(Γ, ω_k)` be a minuscule representation of a simply-laced simple Lie
> algebra. Either no stack is cyclic — exactly for `sl_n` with `min(k, n−k) ≤ 3`, `D_n` with
> `ω_1`, and the half-spin representations of `D_4`, `D_5` — or there is a unique minimal
> cyclic order ideal `I_c`, and the stack of `μ` is cyclic iff `I_μ ⊇ I_c`, i.e. iff
> `μ ≤ μ_c` (`μ_c − μ` a non-negative combination of simple roots):
>
> | representation | `I_c` | `μ_c` | `λ − μ_c` | slots |
> |---|---|---|---|---|
> | `sl_n`, `Λ^k`, `4 ≤ k ≤ n−4` | `λ = (4,4,4,3)` | `e_S`, `S = {1..k−4} ∪ {k,k+2,k+3,k+4}`; `sl_8`: lowest `+ α_4` | `(1,2,3,3,3,2,1)` on nodes `k−3 … k+3` | 15 |
> | `D_n` half-spin `ω_n`, `n ≥ 6` | `ℓ = (5,4,3,1)` | `½(+,…,+,−,−,−,+,−,+)`; `D_6`: `−ω_3+ω_4−ω_5` | `(1,2,3,3,2,2)` on nodes `n−5 … n` | 13 |
> | `E_6`, `ω_1` | Theorem E(a) | `−ω_4 + ω_5` = lowest `+ α_5 + α_6` | `(2,2,3,4,2,1)` | 14 |
> | `E_6`, `ω_6` | Theorem E(b) | `ω_3 − ω_4` | `(1,2,2,4,3,2)` | 14 |
> | `E_7`, `ω_7` | Theorem E(c) | `ω_3 − ω_4 + ω_7` | `(1,2,2,4,3,2,1)` | 15 |
>
> (`D_n` with `ω_{n−1}`: apply the automorphism `n−1 ↔ n`.) This is a theorem for types A and
> D (all ranks) and a computer-assisted theorem for `E_6`, `E_7`. All entries agree with
> `NOTES.md` §4.

*Proof.* Type A: Theorem A and Corollary 3.4. Type D: Prop. 4.1, Theorem D, Corollary 4.4.
Type E: Theorem E. The translation "ideal containment ⇔ weight order" (and "stacks of `μ` =
linear extensions of `I_μ`") is Lemma 3.1(d) for type A and Lemma 4.2(d) for the half-spin
representations, proved in the models; for the vector representation of `D_n` nothing is needed
(no cyclic stack); for `E_6`, `E_7` it is checked by `03_typeE.py` / `03_minuscule.setup`.
(`E_8` has no minuscule representation.) ∎

Equivalently (not used, quoted) `w ≥ w_c` in the Bruhat order of minimal coset representatives
(for minuscule `λ` the Bruhat order on `W^J` is the inclusion order of ideals of `P`, Proctor 1984),
i.e. the Schubert variety of `μ` contains that of `μ_c`.

> **Corollary 6.1 (non-cyclic minuscule stacks are thin).** In every non-cyclic minuscule
> stack (types A, D: proved, Lemmas 3.2, 4.3 and Prop. 4.1; type E: computed, Theorem E(d))
> every colour occurs at most three times. Consequently, in the cycle-free minuscule range,
> admissibility condition 2 of `NOTES.md` §2 (no interlaced pairs of equal colour) is
> vacuous, no coefficient of the tree formula contains `r_c²`, and every minimal cyclic
> configuration of a stack with multiplicities `≤ 3` is a 4-cycle through the mover
> (Corollary 2.10).

This proves the remark in `NOTES.md` §2 ("Condition 2 and repeated factors such as `r_c²` need
a colour occurring four times; they do not occur in the cycle-free stacks") for types A, D,
and confirms it for type E.

*Uniformity.* The threshold configuration is pattern T in types A and D and patterns B and N
(no T) in type E; so a single uniform heap pattern does not describe the threshold, but
Theorem 2.8 does, in all types at once: **a minuscule stack is cyclic iff its heap contains T,
B or N.** A case-free proof of Theorem U (why the cyclic ideals form a principal up-set) is
not known to me.

---

## 7. Relation with Deodhar elements (task 7)

**Literature (quoted from the source).** S. Billey, G. Warrington, *Kazhdan–Lusztig
polynomials for 321-hexagon-avoiding permutations*, J. Algebraic Combin. 13 (2001) 111–136,
arXiv:math/0005052. **Theorem 1:** "Let `a = s_{i_1} ⋯ s_{i_r}` be a reduced expression for
`w ∈ S_n`. The following are equivalent: 1. `w` is 321-hexagon-avoiding. 2. … `P_{x,w} =
Σ q^{d(σ)}` … over all masks on `a` whose product is `x`. 3. The Poincaré polynomial for the full
intersection cohomology group of `X_w` is … `(1+q)^{l(w)}`. 4. The Kazhdan–Lusztig basis element
`C'_w` satisfies `C'_w = C'_{s_{i_1}} ⋯ C'_{s_{i_r}}`. 5. The Bott–Samelson resolution of `X_w`
is small. 6. `IH^*(X_w) ≅ H^*(Y)`, where `Y` is the Bott–Samelson resolution of `X_w`." —
"Say that `w` is hexagon-avoiding if it avoids each of the patterns in
`{[4,6,7,1,8,2,3,5], [4,6,7,8,1,2,3,5], [5,6,7,1,8,2,3,4], [5,6,7,8,1,2,3,4]}`" (their (7)),
and 321-avoiding permutations are the fully commutative ones. Elements satisfying 4 are called
*Deodhar* (Billey–Jones) or *tight* (Lusztig, Fan–Green); by Theorem 1 the property does not
depend on the reduced expression. (Billey–Jones, arXiv:math/0612043, Thm. 2.3 and the list
of minimal non-Deodhar patterns, Thm. 5.12 and Fig. 1, are quoted from
`research_notes/…/homogeneous_reps_minuscule_heaps.md`; not re-read by me.)

> **Proposition 7.1 (type A: non-cyclic ⇔ Deodhar).** Let `λ ⊆ k × (n−k)`, `S = λ`'s set
> (`λ_i = s_{k+1−i} − (k+1−i)`), and `w ∈ S_n` the minuscule element of `e_S`, i.e. the
> Grassmannian permutation `w(1) < … < w(k)`, `w(k+1) < … < w(n)`, `w({1,…,k}) = S`. The
> following are equivalent: (i) `w` is Deodhar; (ii) `w` is 321-hexagon-avoiding;
> (iii) `λ ⊉ (4,4,4,3)`; (iv) the stack of `e_S` is not cyclic.

*Proof.* (i) ⇔ (ii): Billey–Warrington Thm. 1 (1 ⇔ 4). (iii) ⇔ (iv): Theorem A.
(ii) ⇔ (iii). `w` is the concatenation of two increasing sequences, so every subsequence has
at most one descent; hence `w` avoids `321` and the two hexagon patterns with two descents
(`46718235`, `56718234`). The patterns `46781235`, `56781234` have one descent, between
positions 4 and 5, so an occurrence at positions `i_1 < … < i_8` has `i_4 ≤ k < i_5`: it is a
choice of `a_1 < … < a_4` in `S` and `b_1 < … < b_4` in `T = [n] ∖ S`, with
`b_4 < a_1` (pattern `56781234`) resp. `b_3 < a_1 < b_4 < a_2` (pattern `46781235`); every such
choice is an occurrence. For `s_i ∈ S`, `#{t ∈ T : t < s_i} = s_i − i = λ_{k+1−i}`. Hence
`56781234` occurs iff some `s_i` with `i ≤ k − 3` has `s_i − i ≥ 4`, i.e. iff `λ_4 ≥ 4`; and
`46781235` occurs iff there are `i < i' ≤ k − 2` with `s_i − i ≥ 3` and
`(s_{i'} − i') − (s_i − i) ≥ 1` (a `T`-element between `s_i` and `s_{i'}`), i.e. iff there are rows
`3 ≤ r' < r` with `λ_r ≥ 3` and `λ_{r'} > λ_r`. If one of these holds then `λ_4 ≥ 3` and
`λ_3 ≥ 4`; conversely, if `λ_3 ≥ 4` and `λ_4 ≥ 3`, then either `λ_4 ≥ 4` or `(r', r) = (3, 4)`
works. So `w` contains a hexagon pattern iff `λ ⊇ (4,4,4,3)`. (Whether the stack word spells `w`
or `w^{−1}` is immaterial: both properties are invariant under inversion — the pattern set is
closed under inversion, `46781235^{−1} = 56718234`, the others are involutions.) ∎

*Checks.* `03_deodhar.py` Part 1: all 8166 Grassmannian permutations with `n ≤ 12` (321-avoidance,
hexagon patterns ⇔ `λ ⊇ (4,4,4,3)`; the ideal of `e_S` has the colour content of `λ(S)` for
`n ≤ 9`). Part 2 tests the Deodhar property directly in the Hecke algebra (Soergel's
normalisation; `C'_{s_1} ⋯ C'_{s_l}` is the KL basis element iff all its coefficients of `H_y`,
`y ≠ w`, lie in `vℤ[v]`): `A_7 ω_4`, `A_8 ω_4`, `A_8 ω_3`, `A_9 ω_5` — non-Deodhar ⇔ cyclic on
every ideal (2, 9, 0, 42 ideals).

> **Lemma 7.2 (non-Deodhar is inherited).** Let `W` be a Weyl group, `a`, `a'` reduced words
> of `u`, `u'` with `ℓ(uu') = ℓ(u) + ℓ(u')`, and `w = uu'`. Write
> `Π(a) = C'_{s_1} ⋯ C'_{s_l}`. If `Π(a) ≠ C'_u` (or `Π(a') ≠ C'_{u'}`), then `Π(aa') ≠ C'_w`. If
> `u` is fully commutative, `Π(a)` does not depend on the reduced word `a` (commutation moves
> only), so "Deodhar" is a property of `u`. If `W_J ⊆ W` is a standard parabolic subgroup and
> `a` is a word in `J`, the statement `Π(a) = C'_u` means the same in `H(W_J)` and in `H(W)`.

*Proof.* Soergel's normalisation: `H_s² = 1 + (v^{−1} − v) H_s`, `C'_s = H_s + v`,
`C'_y = H_y + Σ_{x<y} h_{x,y} H_x` with `h_{x,y} ∈ vℤ[v]` having the coefficients of the KL
polynomial `P_{x,y}`. The only input is **non-negativity of the coefficients of KL
polynomials** (Kazhdan–Lusztig, *Schubert varieties and Poincaré duality*, Proc. Symp. Pure
Math. 36 (1980), for Weyl groups; Elias–Williamson, *The Hodge theory of Soergel bimodules*,
Ann. of Math. 180 (2014), for all Coxeter groups), together with the multiplication rule
(Kazhdan–Lusztig, Invent. Math. 53 (1979), (2.3.a,b)):
`C'_y C'_s = (v + v^{−1}) C'_y` if `ys < y`, and `C'_y C'_s = C'_{ys} + Σ_{z<y, zs<z} μ(z,y) C'_z`
if `ys > y`, where `μ(z,y) ≥ 0` is the coefficient of `v` in `h_{z,y}`. Multiplying by one
`C'_s` at a time, `C'_y Π(b)` and `Π(b)` are `ℕ[v^{±1}]`-combinations of KL basis elements, for
every `y` and every word `b`.

Write `Π(a) = Σ_y m_y C'_y`, `m_y ∈ ℕ[v^{±1}]`. As `a` is reduced, the `H`-expansion of `Π(a)` is
`H_u` plus terms `H_x` with `ℓ(x) < ℓ(u)`, so `m_u = 1` and `m_y ≠ 0` only for `ℓ(y) < ℓ(u)`.
Then `Π(aa') = Σ_y m_y C'_y Π(a')`. In `C'_u Π(a')` the `H`-expansion is `H_w` plus terms of
length `< ℓ(w)` (multiplying by `H_s` or `v` raises the length by at most one, and only the
product `H_u H_{s'_1} ⋯ H_{s'_r} = H_w` reaches `ℓ(w)`), so by unitriangularity
`C'_u Π(a') = C'_w + Σ_{z ≠ w} n_z C'_z` with `n_z ∈ ℕ[v^{±1}]`. Hence
`Π(aa') − C'_w = Σ_{z≠w} n_z C'_z + Σ_{y ≠ u} m_y C'_y Π(a')`, a sum of `ℕ[v^{±1}]`-combinations
of the basis, which vanishes only if every summand does. But `C'_y Π(a') ≠ 0`: at `v = 1` the
Hecke algebra becomes `ℤW`, `C'_y ↦ y + Σ_{x<y} h_{x,y}(1) x` and `C'_s ↦ 1 + s`, elements with
non-negative coefficients, and a product of such non-zero elements is non-zero. So
`Π(aa') = C'_w` forces `m_y = 0` for `y ≠ u`, i.e. `Π(a) = C'_u`. The case of `a'` is the same
with left multiplication (apply the anti-involution `H_x ↦ H_{x^{−1}}`). Fully commutative `u`:
`C'_s C'_t = C'_t C'_s` for commuting `s, t`, and all reduced words of `u` are related by
commutations. Parabolic: `H(W_J) ⊆ H(W)` is a subalgebra containing the `C'_x`, `x ∈ W_J`
(KL polynomials of `W_J` are those of `W`). ∎

> **Proposition 7.3 (type D, one direction).** For the half-spin representation of `D_n`,
> `n ≥ 6`, every `I_ℓ ⊇ I_{(5,4,3,1)}` (i.e. every cyclic ideal) gives a non-Deodhar element.
> This relies on Lemma 7.2 and on the exact Hecke-algebra computation that the 13-slot
> element of `D_6` is not Deodhar (`03_deodhar.py`, Tier 2).

*Proof.* Precisely: for every chain word `a = (c(1), …, c(m))` of `I_ℓ`, with
`w_ℓ := s_{c(1)} ⋯ s_{c(m)}`, we show `Π(a) ≠ C'_{w_ℓ}`. The chain words of
`I_ℓ` are its linear extensions (Lemma 4.2(d)); any two linear extensions of a finite poset are
related by exchanging adjacent incomparable letters, which here have distinct non-adjacent
colours, so commute; hence `Π(a)` does not depend on the chain word and we may take one that
begins with a chain word `a_0` of `I_{(5,4,3,1)}` (an ideal of `I_ℓ`). Chain words are reduced: with
`x = s_{c(t−1)} ⋯ s_{c(1)}` and `s = s_{c(t)}`, `⟨xλ, α_s^∨⟩ = 1 > 0` and `λ` dominant give
`x^{−1}α_s > 0`, so `ℓ(sx) = ℓ(x) + 1`; hence the reversed word, and so `a`, is reduced. `I_{(5,4,3,1)}` uses only
colours `n−5, …, n`, which span a `D_6` subdiagram, and `a_0` is the `D_6` word relabelled
(the colour rule of 4.2 depends only on `n − (v−u)` and the parity of `u`); by the computation
`Π(a_0) ≠ C'` in `D_6`, hence in `D_n` (parabolic part of Lemma 7.2). Lemma 7.2 gives
`Π(a) ≠ C'_{w_ℓ}`. (That minuscule elements are fully commutative [Stembridge 2001], so that
this is a property of the element `w_ℓ`, is not needed for the statement as given.) ∎

**Computed only.** Non-Deodhar ⇔ cyclic on every ideal of `D_6, D_7, D_8, D_9` half-spin (3, 16,
54, 148 ideals), and for `D_6, D_7` vector (none). The missing direction for all `n` (the
ideals with `ℓ_3 ≤ 2` or `ℓ_4 = 0` give Deodhar elements) would follow from Billey–Jones's
classification of minimal non-Deodhar patterns in type D, which I have not checked.

**Type E: the two notions differ (computed).** `E_6, ω_1` (and `ω_6`): 4 non-Deodhar ideals
(13, 14, 15, 16 slots) versus 3 cyclic ones; the 13-slot ideal is non-Deodhar (it is
Billey–Jones's second `E_6` pattern, per the research notes) but **not** cyclic (Theorem E,
three methods). `E_7`: 25 non-Deodhar versus 22 cyclic ideals; the three extra ones have 14,
15 and 16 slots. Since the tree formula is a verified differential on the full complex of the
13-slot `E_6` stack (`NOTES.md` §4), the cycle condition, not the Deodhar condition, is the
one that governs the tree formula there.

---

## 8. Corrections and updates to `NOTES.md`, `PROOFS.md` and the task statement

1. **Task statement of this front, task 2** (the coordinator's prompt; the claim is not in
   `README_FOR_PROVERS.md` nor in any project `.md` file): "`(4,4,4,3)` is not symmetric under
   transposition" is **wrong**: its
   conjugate is `(4,4,4,3)`. The criterion `λ ⊇ (4,4,4,3)` is already invariant under
   `k ↔ n−k`, `λ ↔ λ'`, colours reversed (§3.2); exhaustive computation over the 6×6 and 7×7
   boxes confirms it unchanged.
2. **`NOTES.md` §4, first table**, row "`sl_n`, `ω_k` with `4 ≤ k ≤ n−3`": with
   `sl_n = A_{n−1}` the correct range is **`4 ≤ k ≤ n−4`** (e.g. `sl_8, ω_5` is a 5×3 rectangle and
   not cyclic); the second table ("`4 ≤ k ≤ n−4`") is right. (If `n` there meant the rank, the
   row is right but inconsistent with "`sl_8`".)
3. **`NOTES.md` §4, "The minimal cyclic configuration always has the same shape …"**: true at
   the thresholds of types A and D (pattern T, Prop. 3.3, Theorem D), but (a) the two
   parenthetical alternatives are coupled — either both pairs are "between `x_1, x_2` and above
   `x_3`" (T, mover `x_3`) or both "below `x_1` and between `x_2, x_3`" (B, mover `x_1`); (b) there
   is a third shape, **N** (two nested pairs of one colour `d` around the mover, and a pair
   interlacing both), which occurs at the thresholds of `E_6` and `E_7` (each `I_c` has exactly
   one B and one N configuration and no T) and in arbitrary words (e.g. the README's
   `1221122`); (c) in larger ideals minimal cyclic configurations are tadpoles with chordless
   cycles of length 4, 6, 8, … and the mover off the cycle (§3.3). The correct general
   statement is Theorem 2.8.
4. **`NOTES.md` §4 / `PROOFS.md` Tier 2 line "A stack is cyclic iff `μ ≤ μ_c` … cycles with ≤ 4
   pairs (≤ 6 near the threshold)"** and **`PROOFS.md` C3**: now Tier 1 for types A and D (all
   ranks, Theorems A, D, Prop. 4.1) and an exhaustive (unbounded) computer proof for
   `E_6, E_7` (Theorem E). By Theorem 2.8, searches with `max_pairs ≥ 3` were exhaustive all
   along.
5. **`NOTES.md` §4 first table**, "`sl_n, ω_1, ω_2, ω_3` … checked `n ≤ 11`", "`so_{2n}` vector,
   checked `n ≤ 8`", "`D_4`, `D_5` spinors": now proved for all `n` (Cor. 3.4, Prop. 4.1,
   Cor. 4.4).
6. **`reports/KLR resolutions trees bigons Demazure.md`, reading 3** ("the cycle search …
   misses an `E_6` configuration"): excluded — the 13-slot `E_6` ideal is not cyclic (three
   methods, including the brute force over all its configurations). The type-E mismatch with
   Deodhar is real.
7. **`PROOFS.md` Prop. 7** can be strengthened: cyclicity is inherited by all super-words (a
   subword of a non-cyclic word is non-cyclic), not only by extensions of a prefix (1.4).
8. **`NOTES.md` §2**, "Condition 2 and repeated factors such as `r_c²` need a colour occurring
   four times; they do not occur in the cycle-free stacks": proved for types A, D (Cor. 6.1),
   computed for E.
9. No error found in `PROOFS.md` Prop. 8 or Cor. 3.1. Conjecture C1′ (README) can be restated
   as: *the tree formula satisfies `d² = 0` for every word avoiding the patterns T, B, N*; the
   hypothesis is closed under subwords, as an induction on the last letter needs.

---

## 9. Status

**Proved completely** (only the definitions of §1, elementary facts about forests, and the
Cartan data of types A, D):
Lemma 1.1; Lemma 2.1; Prop. 2.2; Lemma 2.3 (structure of minimal cyclic configurations, and
exhaustiveness of the tree search); Lemmas 2.4, 2.5, Cor. 2.6 (mover on the cycle); Lemma 2.7
(shortening); **Theorem 2.8** (cyclic ⇔ three-pair configuration ⇔ pattern T, B or N); Cors. 2.9,
2.10; Prop. 2.11(a); Lemma 2.12, Cor. 2.13 (minuscule chain words; uses only the chain
condition); **Lemma 3.1** (type A model: heap of every linear extension, chain property,
stacks of `μ` = linear extensions of `λ(μ)`, bijection with the weights, weight order =
reverse inclusion — by induction on the chain word, no citation needed); Lemma 3.2,
**Theorem A**, Prop. 3.3(a) and its addendum for `(4,4,4,3)`, **Cor. 3.4**; Prop. 4.1;
**Lemma 4.2** (the same for the half-spin model); Lemma 4.3, **Theorem D**, **Cor. 4.4**;
**Theorem U** for types A and D; Cor. 6.1 for types A, D.

**Proved assuming standard results:** Lemma 2.14 (the general multiplicity bound; uses that
weights in the orbit of a minuscule weight pair with every coroot to `−1, 0, 1`, and that the
orbit has a unique antidominant element; it is no longer used in types A, D). Prop. 7.1 assumes
Billey–Warrington Thm. 1 (quoted verbatim). Lemma 7.2 assumes the non-negativity of the
coefficients of KL polynomials and the KL multiplication rule (Kazhdan–Lusztig 1979, 1980).
The references Proctor 1984 and Stembridge 2001 are kept for orientation only; no argument
depends on them (§3.1).

**Proved by exact finite computation (Tier 2, pure Python, no bound on configuration
size):** Theorem E and Theorem E(d) (three independent methods; assumptions listed in §5);
Prop. 2.11(b)(c); Prop. 7.3's input (the 13-slot `D_6` element is not Deodhar); Prop. 7.1's
Hecke-algebra cross-check.

**Verified by computation only:** Deodhar ⇔ non-cyclic for `D_6 … D_9` half-spin (all
ideals; one direction proved for all `n`, Prop. 7.3) and for `A_7 … A_9` cases (proved in
general by Prop. 7.1); the type-E mismatch with Deodhar; the census of minimal cyclic
configurations in type A beyond Prop. 3.3(a), and the uniqueness of the cyclic configuration
in `(5,4,4,3)`, `(4,4,4,3,1)`, `(5,5,4,3)` (Prop. 3.3 addendum). All statements of this file
were re-derived by the referee's independent code (`R03_referee.md` §5).

**Not done / open.** (1) A case-free proof of Theorem U (why the cyclic ideals of a minuscule
poset form a principal up-set; types A, D are proved case by case, E by computer). (2) The
converse of Prop. 7.3 for all `n` (needs the Billey–Jones type-D pattern list). (3) A
description of all minimal cyclic configurations of large Young diagrams: there is no finite
list (chordless cycles of length up to 8 already in `5^5`), but by Theorem 2.8 none is needed
for deciding cyclicity.

**Failed attempts.** A first shortening move (re-pairing the two equal-coloured neighbours
`x, y` of a turning point into the other non-interlaced pairing, keeping or dropping `z`) fails
on 16 of 442 six-cycles tested (scratch computation); the single chord `x*` of Lemma 2.7 works
at every turning point, and its block-freeness has the proof given there.
