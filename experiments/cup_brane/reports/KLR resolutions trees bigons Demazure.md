# Hexagon avoidance predicts where tree formulas break

No paper in the KLR, KLRW, weighted-KLR or Coulomb-branch literature contains the structure the cup-brane project has found. That structure is a closed, tree-indexed differential for cube resolutions of standard-type modules over a KLRW algebra deformed by r_i and t_ij, plus spanning-tree-with-bigon corrections when the link graph has a cycle. The cube shape itself is classical: Brundan–Kleshchev–McNamara (BKM) build cube resolutions of root modules, and Webster builds subset-cube resolutions of standard modules (KIHRT Prop. 7.6). Both papers say in so many words that they could not, or would not, write the differential once colours repeat, and that is exactly the regime the project's tree formula covers. The threshold has a sharp match in Kazhdan–Lusztig combinatorics. Billey–Warrington's 321-hexagon-avoidance is the same as Deodhar's condition that the KL basis element C′_w factors as the product of the C′_s along a reduced word, which is also the same as the Bott–Samelson resolution being small. For a Grassmannian permutation it fails exactly when λ ⊇ (4,4,4,3). Billey–Jones's minimal non-Deodhar patterns live in A7, D6 and E6, which are precisely the project's three seeds. A check run for these notes confirms the match on every type A and D minuscule case tested. In type E there is a one-step mismatch: Deodhar fails at the 13-element E6 ideal, while the project's first cycle appears at 14. So either the link graph is a different invariant, or the unsolved 13-slot E6 stack hides terms nobody has looked for. The local mechanism is established algebra. Rouquier's (i,j,i) braid relation fails by the divided difference of Q_ij, and for Q_ij = t_ij u + t_ji v with ψ = r_i∂ that difference is r_i t_ij, the weight of one tree edge. The cyclic correction is, as a formal identity, a directed Kirchhoff (first Symanzik) polynomial, and the Demazure form ∏(r_p ∂_p) ∏ Q is the same expansion that proves the matrix-tree theorem. No source connects either polynomial to KLR algebras. Aganagic–LePage–Rapčák's algorithm covers every minuscule colour in principle, but its only worked examples are sl₂ and gl(1|1), where these cycles cannot occur. As far as these sources show, the project's formulas are new. The literature points to three routes to a proof: reading the cube as Deodhar masks, solving a Maurer–Cartan equation with Rouquier's defect as curvature, and proving a matrix-tree identity for link graphs with several cycles.

*Evidence labels.* **Established** means proved or stated in the cited source. **Project computation** means output of this repository's cup-brane code, documented in [NOTES.md](../NOTES.md), [GLUING.md](../GLUING.md) and [HANDOVER.md](../HANDOVER.md). **Notes computation** means a check run while preparing the research notes, with scripts in [research_notes/…/code](../research_notes/KLR%20resolutions%20trees%20bigons%20Demazure/code/deodhar_minuscule.py). Those scripts' `sys.path` lines point to a scratchpad folder, so they need adjusting before re-running. The main Deodhar table was re-run and reproduced while writing this report. **Speculative** marks a connection that no source proves.

## Closed-form cube differentials stop exactly where colours repeat

**Established: for finite-type KLR algebras the homological theory is proved, but only abstractly.** Kato proves, in types ADE over characteristic zero, that the algebras have finite global dimension and that standard and proper standard modules are Ext-orthogonal. He defines standard modules as quotients of projectives and never resolves them ([Kato, arXiv:1203.5254](https://arxiv.org/abs/1203.5254), Thms A–B). McNamara proves that **R(ν) has global dimension |ν|** (Thm 4.7) and calls his construction of cuspidal modules "non-constructive" ([McNamara, arXiv:1207.5860](https://arxiv.org/abs/1207.5860)). BKM extend Ext-vanishing to all finite types over any field (Thm 3.12). They also bound pd Δ(λ) ≤ ht α − l, where l is the number of parts of λ (Cor. 4.11) ([BKM, arXiv:1210.6900](https://arxiv.org/abs/1210.6900)). Affine cellularity yields standard modules, reduced standard modules and finite global dimension, but neither paper contains a resolution ([Kleshchev–Loubert–Miemietz, arXiv:1210.6542](https://arxiv.org/abs/1210.6542); [Kleshchev–Loubert, arXiv:1310.4467](https://arxiv.org/abs/1310.4467)).

**Established: the published closed forms have no correction terms.** The richest is BKM Thm 4.12. For a multiplicity-free root α, the root (cuspidal) module has a cube resolution indexed by {0,1}^{n−1}, and each entry is a single signed permutation element (−1)^{σ1+…+σr} τ_w. The proof "relies on the fact that the braid relations hold exactly in H_α under the assumption that α is multiplicity-free." For general roots BKM build the same cube by iterated mapping cones (§4.5, eq. 4.17), then concede: **"Unfortunately we have not been able to find a satisfactory description of such elements τσ,ρ, except in the following special case"** ([BKM, arXiv:1210.6900](https://arxiv.org/abs/1210.6900)).

Kleshchev's survey resolves every irreducible module of a type-A cuspidal block by a cube over T ⊆ [1,d]. Its edges are signed crossings, except for one edge given by the central sum of dots z_ρ. Induction products then resolve proper standard and costandard modules, and Prop. 6.14 shows that bigons collapse to dots there ([Kleshchev, arXiv:1401.6151](https://arxiv.org/abs/1401.6151), §6.7–6.8, (6.30)–(6.33)). It is the only cube with a dotted edge found in this literature.

Buursma–Kleshchev–Steinberg (BKS) resolve Δ(α^m) in type A on a grid {0..m}^{b−a+1} with divided-power idempotents. They realise it as a direct summand of an induced BKM resolution, which keeps every equal-colour correction hidden inside thick strands ([BKS, arXiv:1906.11376](https://arxiv.org/abs/1906.11376), Thm A). Their companion paper carries a warning. The Ext algebra between standard modules is a Boolean lattice for roots (Thm A), but for θ = α1+2α2+α3, the first type-A weight with a repeated colour, **the A∞ Ext category is non-formal** (Example C) ([BKS, arXiv:1906.11380](https://arxiv.org/abs/1906.11380)).

Two more explicit complexes come from the tensor-product side. One is Webster's three-term minimal resolution of the sl₂ cup module, whose maps are the unique red–black crossings ([Webster, arXiv:1312.7357](https://arxiv.org/abs/1312.7357), Prop. 4.4). The other is Kang–Kashiwara's two-term resolution, Lemma 4.17, eq. (4.22). The composite of its map P with the near-inverse Q is **a^Λ(t_i) ∏_a Q_{i,ν_a}(t_i, x_a), a product of bigon polynomials** ([Kang–Kashiwara, arXiv:1102.4677](https://arxiv.org/abs/1102.4677)). That is the published prototype of "pulling strands across and back gives a product of Q's".

**Established: tensor product and weighted KLR algebras give existence and shape, not formulas.** Webster's T^λ is standardly stratified (Cor. 5.22). Every standard module has a finite projective resolution (Cor. 5.24), proved by induction rather than written out. When every λ_i is minuscule the category is highest weight, with finite global dimension (Prop. 8.3) ([Webster, KIHRT, arXiv:1309.3796](https://arxiv.org/abs/1309.3796)).

Prop. 7.6 is the structural twin of the project's cube. For two red strands, the standard S^{κ_j}_i is resolved by ⊕_{|T|=k} P^{κ_T}_{i_T}(χ_T) over T ⊂ [j+1,n], where the strands in T are moved left of the second red strand in reversed order. Webster then stops: **"The same principle can be used for any value of ℓ … but carefully writing this down is a bit more subtle and difficult than the ℓ = 2 case, so we will not do so here."** Prop. 7.8 resolves the cup simple by modules filtered by standards, with length 2ρ^∨(λ). Prop. 9.14 gives Koszulity for sl_n with fundamental weights, and Webster expects T^λ "will never be Koszul" otherwise ([KIHRT](https://arxiv.org/abs/1309.3796)).

Mackaay–Webster show that an algebra acts on the projective resolution of its module only in the A∞ sense (§2.3). They also show that a module deforms iff its class s_M ∈ Ext²(M,M) vanishes, and that s_M is the chain map ∂̃²/h obtained by lifting the differential to the deformed algebra (Prop. 2.22, Lemma 2.23) ([Mackaay–Webster, arXiv:1502.06011](https://arxiv.org/abs/1502.06011)).

The weighted KLR paper supplies the right polynomial tools but no resolutions. It has a Demazure-operator polynomial representation (Prop. 2.7) and a canonical flat deformation Q̃ of the Q-polynomials (§2.7, Prop. 2.23). It also shows that the deformed black–red–black triple point is a divided difference (3.4a) ([Webster, arXiv:1209.2463](https://arxiv.org/abs/1209.2463)).

The remaining explicit constructions miss the target in different ways:
- **BGG resolutions** (Bowman–Norton–Simental, Bowman–Hazi–Norton) are explicit alternating sums of one-dimensional homomorphisms, but they resolve by standard modules, not projectives ([BNS, arXiv:1803.08736](https://arxiv.org/abs/1803.08736), Thms A–B; [BHN, arXiv:2004.13082](https://arxiv.org/abs/2004.13082)).
- **Brundan–Stroppel** construct linear projective resolutions of cell modules inductively, with Kazhdan–Lusztig multiplicities ([Brundan–Stroppel, arXiv:0806.3472](https://arxiv.org/abs/0806.3472), Thm 5.3).
- **Naisse–Vaz** give a formal dg model of the cyclotomic quotient. It uses the Cautis–Lauda scalars t_ij and r_i, with **r_i required to be a unit** (Def. 3.1, Thm 4.4) ([Naisse–Vaz, arXiv:1710.06293](https://arxiv.org/abs/1710.06293)).
- **Lacabanne–Naisse–Vaz** add an explicit three-term cofibrant replacement of the cup/cap bimodule (Prop. 5.1) ([LNV, arXiv:2005.06257](https://arxiv.org/abs/2005.06257)).
- **Khovanov–Lauda–Sussan–Yonezawa and Khovanov–Sussan** deform only the red strands, a direction orthogonal to (r_i, t_ij) ([KLSY, arXiv:1802.05358](https://arxiv.org/abs/1802.05358); [Khovanov–Sussan, arXiv:1605.02678](https://arxiv.org/abs/1605.02678)).

**Established: the parameters are gauge only while invertible.** Cautis–Lauda note that a unit r_i rescales to 1 and that t_ij can be set to 1 on a tree ([Cautis–Lauda, arXiv:1111.1431](https://arxiv.org/abs/1111.1431), §2). McNamara–Tingley show that every isomorphism fixing the idempotents is a torus rescaling. On a cycle the invariant is the product of the t_ij over the product of the t_ji (Lemma 2.14). Setting t_ij = 0 gives "pseudo-KLR" algebras with strictly more irreducible modules (Def. 2.15, Thm 2.16) ([McNamara–Tingley, arXiv:1512.04458](https://arxiv.org/abs/1512.04458)). Lauda shows that the ratios t_ji/t_ij form a 1-cocycle on the Dynkin graph, trivial on trees ([Lauda, arXiv:1812.07654](https://arxiv.org/abs/1812.07654), §3.3). With inverted parameters, r and t are pure bookkeeping. A formula over ℤ[r,t] that also holds on the degenerate fibres is not covered by any rescaling argument in the literature.

| Construction | Index set | Entries | Dots / bigons | Repeated-colour corrections |
|---|---|---|---|---|
| BKM Thm 4.12 ([1210.6900](https://arxiv.org/abs/1210.6900)) | cube {0,1}^{n−1} | single ±τ_w | none | none (braid relations hold exactly) |
| BKM §4.5, general roots | same cube | not determined | unknown | "not … satisfactory description" |
| Kleshchev §6.7 ([1401.6151](https://arxiv.org/abs/1401.6151)) | cube T ⊆ [1,d] | ±ψ plus one central z_ρ | central sum of dots | none |
| BKS Thm A ([1906.11376](https://arxiv.org/abs/1906.11376)) | grid {0..m}^{b−a+1} | ±ψ between divided powers | inside idempotents | absorbed by thick calculus |
| Webster Prop. 7.6 ([1309.3796](https://arxiv.org/abs/1309.3796)) | subsets T ⊂ [j+1,n] | not written | not written | "will not do so here" |
| Kang–Kashiwara (4.22) ([1102.4677](https://arxiv.org/abs/1102.4677)) | two terms | a^Λ(x_1)τ_1⋯τ_n | composite Q∘P = ∏ Q | intertwiner-like g_a |
| Naisse–Vaz / LNV ([1710.06293](https://arxiv.org/abs/1710.06293)) | Koszul-type, odd generators | closed-form d_N | h_n, ε_r in dots | r_i a unit |
| **Project: tree formula** ([NOTES.md §2](../NOTES.md)) | cube I ⊆ slots | ±∏ r_c t_{cc′}, dot-free | none | mover plus exchange pairs forming a tree |
| **Project: cyclic terms** ([NOTES.md §6](../NOTES.md)) | same cube | spanning trees × bigon; hand-overs | one dot per cycle so far | partially understood |

**Project computation.** In each stack the slots form an order ideal of a minuscule poset. A cube vertex is the set I of slots whose strands lie right of the red strand, with idempotent [slots not in I, decreasing] W_k [slots in I, increasing]. When no cyclic configuration exists, the full differential is a sum over a mover j together with disjoint equal-colour exchange pairs whose link graph is a tree rooted at the mover. Each such term has coefficient ε ∏_p r_{c(p)} t_{c(p), c(x_p)}, with no dots and no u ([NOTES.md §2](../NOTES.md)).

The evidence:
- An independent re-implementation written from the text alone reproduces **all 40,084 stored terms**.
- The formula matches the order-by-order solver term by term up to 10 strands.
- d² = 0 holds on random vertices of stacks with up to 21 strands.
- The complete ansatz has **kernel 0** in every case with at most 6 strands, so the correction is unique ([NOTES.md §5](../NOTES.md)).

**Speculative positioning.** The project's idempotent ordering matches Webster's Prop. 7.6: crossed strands sit on the left, in reversed order. That makes the tree formula read naturally as the explicit differential Webster declined to write, specialised to words coming from minuscule chains. After the deformation, it is also the first closed description of BKM-type cube entries in a repeated-colour regime. Two steps remain unproved. The first is identifying the project's U-brane stacks with Webster's standard modules. The second is checking that BKM's root-module cubes fall under the same mechanism.

## Deodhar's hexagon reproduces (4,4,4,3) in types A and D but runs one step early in type E

**Established: homogeneous representations supply the dictionary but no threshold.** Kleshchev–Ram classify the irreducible KLR modules concentrated in one degree, which are the same as the calibrated ones, on which all dots act by zero (Cor. 3.2). Each is a module S(C) whose basis is a connected component of the admissible-transposition graph, with ψ_r acting by permutation (Thm 3.4). Such components are skew shapes. **Skew shapes with d boxes correspond bijectively to fully commutative elements of length d**, with standard tableaux matching reduced words (Prop. 3.7). Dominant minuscule elements obey Stembridge's condition: between consecutive occurrences of a generator there are "exactly two terms (possibly equal to each other) that do not commute with" it (Prop. 3.8). The Peterson–Proctor hook formula gives their dimensions (Thm 3.10) ([Kleshchev–Ram, arXiv:0809.0557](https://arxiv.org/abs/0809.0557)).

Park proves that for minuscule Λ the cyclotomic quotient R^{Λ}(wΛ) is isomorphic to End(S(b)), hence a simple algebra (Thm 2.9) ([Park, arXiv:1909.00313](https://arxiv.org/abs/1909.00313)). Feinberg–Lee classify and enumerate the homogeneous modules in types D and A without naming any threshold ([Feinberg–Lee, arXiv:1401.0845](https://arxiv.org/abs/1401.0845); [arXiv:1505.04075](https://arxiv.org/abs/1505.04075)).

In the project's language a slot is a bead, a colour is a runner, and an order ideal is a straight shape. The project's rule of "exactly two adjacent-colour slots between consecutive equal colours" is Stembridge's condition. Homogeneous modules have y = 0, and the cyclotomic algebra is semisimple. So every dot in the cyclic corrections must come from the deformation, and from working with projectives over the non-cyclotomic algebra.

**Established: Deodhar's theorem and the hexagon.** Billey–Jones state, after Deodhar, that for Coxeter groups with nonnegative KL polynomials the following conditions on w are equivalent (Thm 2.3):
- every proper mask on a reduced word has fewer zero-defects than plain zeros;
- **C′_w = C′_{w1}⋯C′_{wk}**;
- P_{x,w} is the generating function of defects over masks;
- **the Bott–Samelson resolution of X_w is small**;
- the intersection-cohomology Poincaré polynomial is (1+q)^{ℓ(w)}.

Their §6 treats masks explicitly as 0/1 labellings of heap entries. In types A and B, w is Deodhar iff it avoids short braids and hexagons as embedded factors (Thm 5.9, from Billey–Warrington). The type-A hexagon is u = [46718235] ∈ S₈, whose heap has rows 3, 4, 4, 3. That is **the 14-element skew heap (4,4,4,3)/(1)** (Ex. 5.8) ([Billey–Jones, arXiv:math/0612043](https://arxiv.org/abs/math/0612043); [Billey–Warrington, arXiv:math/0005052](https://arxiv.org/abs/math/0005052)).

Billey–Jones classify the minimal non-Deodhar patterns in every finite Weyl group (Thm 5.12, Fig. 1). They live on **I₂(m), A7, B7/C7, D6, D7, D8, E6 and E7**, with four E6 patterns of lengths 14, 13, 13 and 13. For Deodhar elements the μ-coefficients are always 0 or 1 ([Jones, J. Algebraic Combin.](https://link.springer.com/content/pdf/10.1007/s10801-008-0131-6.pdf)). Perrin's small resolutions of minuscule Schubert varieties concern a different property, smooth relative minimal models, so they do not supply this threshold ([Perrin, arXiv:math/0601117](https://arxiv.org/abs/math/0601117)).

**Inference: a type-A proof sketch.** Minuscule elements are fully commutative, so they never contain short braids, and Deodhar reduces to hexagon avoidance. A translate of the hexagon heap (4,4,4,3)/(1) sits inside an order ideal of the Grassmannian heap exactly when the ideal contains (4,4,4,3). That reproduces the project's verified statement, checked for all subsets in sl₈ through sl₁₀: e_S is cyclic iff the Young diagram of S contains (4,4,4,3) ([NOTES.md §4](../NOTES.md)). It also explains why ω₁, ω₂ and ω₃ never cycle, since a hexagon needs four rows and four columns.

**Notes computation: the comparison on every minuscule element tested.** The script checks criterion (5) on every Λ-minuscule element ([deodhar_minuscule.py](../research_notes/KLR%20resolutions%20trees%20bigons%20Demazure/code/deodhar_minuscule.py), re-run for this report). The table compares its output with the project's cyclic thresholds.

| (g, ω) | heap size \|P\| | non-Deodhar weights (heap sizes) | project's cyclic weights | verdict |
|---|---|---|---|---|
| A7, ω4 | 16 | 2 (15, 16) | μ_c = lowest + α4: 15, 16 | identical |
| A8 ω4 / A9 ω4 / A9 ω5 | 20 / 24 / 25 | 9 / 25 / 42, smallest 15 | λ ⊇ (4,4,4,3) | identical; counts equal the number of shapes ⊇ (4,4,4,3) |
| A8 ω3; D4; D5 spinor; D6, D7 vector | 18; 6; 10; 10, 12 | none | none (cycle-free) | identical |
| D6, half-spin | 15 | 3 (13, 14, 15) | μ_c = lowest + α6 + α4: 13–15 | identical |
| D7, D8 half-spin | 21, 28 | 16, 54; smallest 13 | 13 | identical under the literal encoding |
| **E6, 27** | 16 | **4 (13, 14, 15, 16)** | **3 (14, 15, 16)** | **Deodhar fails one weight earlier** |
| **E7, 56** | 27 | **25, smallest 14** | 22 under the literal encoding; smallest 15, equal to the project's μ_c | **3 extra Deodhar weights** |

The type-E mismatch is robust:
- **The 13-element ideal is a Billey–Jones pattern.** The 13-element E6 minuscule element (weight (0,−1,−1,1,0,0) in Bourbaki labels, root content λ − μ = (2,2,3,3,2,1)) is literally Billey–Jones's second E6 pattern s5s1s2s3s0s1s2s5s4s3s2s1s0. The label map is BJ 0,1,2,3,4,5 → Bourbaki 1,3,4,5,6,2 ([e6_bj_compare.py](../research_notes/KLR%20resolutions%20trees%20bigons%20Demazure/code/e6_bj_compare.py)).
- **The project's threshold sits one step below it.** The project's μ_c = −ω4 + ω5 has λ − μ_c = (2,2,3,4,2,1). By arithmetic on these two weights, **μ_c = μ₁₃ − α4**: the project's threshold is the Deodhar threshold plus one more slot of the trivalent colour.
- **The local pattern differs.** The minimal violating mask at size 13 zeroes all three trivalent-colour slots. The neighbours of that colour-4 chain split as {2, 5} between the first and second chain element, {3, 5} between the second and third, and {2, 3} above the third. Only colour 2 spans the whole chain, so the two-colour straddling pattern of the project's minimal cyclic configuration is absent even though the element is non-Deodhar ([e6_chain.py](../research_notes/KLR%20resolutions%20trees%20bigons%20Demazure/code/e6_chain.py)).
- **Types A and D match locally too.** In A7 the neighbours are {3.1, 5.1} between the first two colour-4 slots and {3.3, 5.3} above the third. Those are exactly the project's two straddling pairs, and D6 is analogous.
- **The encoding matters only slightly.** A literal encoding of the verbal cyclic pattern places the E6 threshold at 15 if c′ ≠ c″ is required, and at 14 if c′ = c″ is allowed. The project's enumeration gives 14.

At each threshold the product of the C′_s decomposes as C′_w plus extra elements C′_y, each with multiplicity 1 ([bad_terms.py](../research_notes/KLR%20resolutions%20trees%20bigons%20Demazure/code/bad_terms.py)):

| Case | Number of extra C′_y | Lengths ℓ(y) |
|---|---|---|
| A7 (4,4,4,3) | one | 7 |
| D6, 13-element ideal | two | 7, 5 |
| E6, 13-element ideal | two | 7, 5 |
| E6, 14-element ideal | three | 8, 8, 6 |

Projecting to the parabolic sign module instead gives the parabolic KL element for every minuscule w tested. So the threshold lives in the full Hecke algebra, i.e. in the Bott–Samelson resolution of X_w ⊂ G/B, and not in the minuscule module ([parabolic_check.py](../research_notes/KLR%20resolutions%20trees%20bigons%20Demazure/code/parabolic_check.py)).

**Speculative: the cube is a categorified Deodhar expansion.** The stack word is a reduced word of the minuscule element w, so the number of slots m equals ℓ(w). The project's 2^m cube vertices and Deodhar's 2^{ℓ(w)} masks are therefore **indexed by the same subsets of the same heap**. A Deodhar defect is a local condition at a heap entry, decided by the equal-colour and adjacent-colour entries below it. That is the same kind of data that decides whether an exchange pair is linked. Deodhar's property says the naive sum over masks already is the KL element, which is the natural analogue of "the tree formula is complete". On this reading the extra C′_y are the shadows of the project's cyclic terms. Through Soergel's categorification and the Elias–Williamson theorem (a step these notes did not re-verify against sources), they would be the extra indecomposable summands of the Bott–Samelson bimodule.

The E6 mismatch allows three readings:
1. The link graph is coarser than Deodhar's invariant, and the A/D agreement is a coincidence of heaps whose separating neighbours come in pairs.
2. The 13-slot E6 stack carries extra terms not attached to cycles, for instance of hand-over type.
3. The cycle search, which used up to four pairs and re-tested with up to six near the threshold, misses an E6 configuration ([NOTES.md §4](../NOTES.md)).

Solver evidence for whole stacks stops at 10 strands, and the local pieces have 14–16 strands ([NOTES.md §5–6](../NOTES.md)). So the 13-slot E6 stack has never been checked directly, and checking it decides among the three readings.

## Aganagic's programme covers every minuscule colour yet solves only sl₂

**Established: the brane dictionary and the algorithm.** In the Coulomb-branch picture:

| Brane | Geometry | Algebraic counterpart |
|---|---|---|
| T-brane | thimble | projective module |
| I-brane | product of intervals | simple module |
| E-brane | product of figure-eights | cup |
| U-shaped Lagrangian | cone T_{i+1} → T_i | standard module |

([ALR, arXiv:2305.13480](https://arxiv.org/abs/2305.13480); [Tong, arXiv:2511.06431](https://arxiv.org/abs/2511.06431))

Aganagic–LePage–Rapčák (ALR) state their algorithm for any simply-laced ^Lg with minuscule colours (§1.2, §2.1). Break a product Lagrangian into a rectangular grid of T-branes, then correct its differential order by order, δ_ħ = Σ ħ^k δ_k, by solving a linear inhomogeneous equation {δ_0, δ_k}_0 = D_k at each order (eqs 3.17–3.18). They concede that "the deformation problem may not have any solutions, and if it does, the solution may not be unique". They then argue existence, and **uniqueness up to gauge** because Hom spaces between T-branes are at most one-dimensional in each degree (§3.4.5). Truncation of the series follows from a broken symmetry (§3.4.3) ([ALR](https://arxiv.org/abs/2305.13480)). This is the deformation problem the project's solver solves. The project finds no gauge freedom at all inside a stack in any case checked ([NOTES.md §3](../NOTES.md)).

**Established: only sl₂ is explicit.** The single cup is resolved as T_2{−1}[2] ⊕ (T_1 ⊕ T_3{−1})[1] ⊕ T_2, which ALR call "the standard Koszul resolution of the structure sheaf of a ℙ¹" (§5.6.1). Aganagic gives explicit matrices for the same complex on the A_{m−1} surface ([Aganagic II, arXiv:2105.06039](https://arxiv.org/abs/2105.06039), §6.5). Braided cups need dotted corrections, with order u^ℓ meaning ℓ dots (§5.5.3, §5.6.2). In every worked example **first-order ħ-corrections without dots suffice** (§5.6.3–5.6.5), and taking Hom with the simple cap I_𝒰 sets every dot to zero (§5.5.7) ([ALR](https://arxiv.org/abs/2305.13480)).

LePage–Shende prove that the ADLSZ embedding sends Webster's cup ∪_W^n, a simple KLRW module, to Aganagic's cup and intertwines the braid actions. As a result **Aganagic's sl₂ invariant is Khovanov homology over ℤ** (Thm 1.1). They also prove that the U-Lagrangian is isomorphic to the step brane J (Lemma 10.3) and that the cup simple is a cone υ₋ → υ₊ (Prop. 10.4). They warn that naive braiding yields "a complex of size 3^n … (not all of whose maps we are entirely certain we can rigorously determine)" ([LePage–Shende, arXiv:2505.00327](https://arxiv.org/abs/2505.00327)).

**Established: for general g there are statements, not computations.**
- **B-side cup.** It is 𝒪 of a product of minuscule Grassmannians G/P_i. The minuscule restriction is needed because otherwise "𝒳 develops singularities due to monopole bubbling" ([Aganagic I, arXiv:2004.14518](https://arxiv.org/abs/2004.14518), §1.1, §6.2).
- **A-side interval brane.** Its coordinates are ordered "in the order in which we subtract the corresponding simple roots from μ_i to get −μ_j". That order "depends crucially on the fact μ_i and μ_j are highest weights of minuscule representations" ([Aganagic II](https://arxiv.org/abs/2105.06039), §8.2). It is exactly the project's slot order along a minuscule chain.
- **Webster's cups and caps.** In Webster's cylindrical setting the cup is exact, and the cap is bounded when the colour is minuscule (Lemma 7.6), a result that "holds in effectively no non-minuscule cases" (Remark 7.7). The invariant equals KIHRT's, and in type A_n it equals Khovanov–Rozansky sl_{n+1} homology (Thm 7.12) ([Webster, CSQCB II, arXiv:2211.02099](https://arxiv.org/abs/2211.02099)).
- **Fukaya side.** ADLSZ embed cylindrical KLRW categories into Fukaya categories of multiplicative Coulomb branches whenever these are smooth, which is "known when Γ is of ADE type" (Thm 1.7). Generation remains Conj. 1.8 ([ADLSZ, arXiv:2406.04258](https://arxiv.org/abs/2406.04258)).

Beyond sl₂, no published source contains an explicit thimble resolution of a cup: not for sl_n with n ≥ 3 in any Λ^k, not for a D_n spinor, and not for an E_n minuscule. None gives a closed formula for the corrections δ_k or describes code. Shende–Zhou mention an ongoing project with Aganagic and LePage but report no cup results ([Shende–Zhou, arXiv:2604.01208](https://arxiv.org/abs/2604.01208)).

**Established: stable envelopes appear only as supports.** Aganagic describes the core of 𝒳 as "the union of supports of all stable envelopes", and obtains the integrand insertions as limits of elliptic stable envelopes ([Aganagic II](https://arxiv.org/abs/2105.06039), §3.1, §8.2). Tong shows that the two Lagrangian realisations of a standard module, the U-shaped T-brane and the step I-brane, are related by Hamiltonian flow for (Γ, d) = (•, 1) ([Tong](https://arxiv.org/abs/2511.06431)). Danilenko studies Maulik–Okounkov stable envelopes on slices of the affine Grassmannian, with a recursion for PSL₂ (read in abstract only) ([Danilenko, arXiv:2210.09967](https://arxiv.org/abs/2210.09967)). **No paper states that a stable envelope is a KLRW standard module.** The project's usage fits this picture but goes beyond it: S_μ is the product (stack around p₁) ⊠ (stack around p₂), with no cross terms, verified on 15 envelopes ([NOTES.md §2](../NOTES.md)).

**Established: the parameter dictionary.** The ADLSZ category is defined over ℂ[ħ, η] (Fig. 1):

| Relation | ADLSZ value |
|---|---|
| dot passing a crossing | ħ |
| bigon of two linked black strands | η(dot) − η(dot) |
| red–black bigon | η·dot |
| braid relation with a neighbour or red strand | ηħ |

([ADLSZ](https://arxiv.org/abs/2406.04258))

LePage–Shende add that "in general, one can introduce independent u, ħ parameters and gradings for each node of Γ" (§2.3). ALR note that both parameters "may be set to 1 by rescaling … if we keep them generic, they are useful as book-keeping parameters" (§5.4.10) ([LePage–Shende](https://arxiv.org/abs/2505.00327); [ALR](https://arxiv.org/abs/2305.13480)). The project's algebra is exactly this per-node refinement, with t_ij per oriented edge ([HANDOVER.md §3.1](../HANDOVER.md)):

| Project | ADLSZ / ALR |
|---|---|
| r_c | ħ of colour c |
| t_{cc′} | η, refined per oriented edge |
| u_c | the framing (red–black) parameter |
| defect r_i t_ij | ηħ |

**Inference: what the sl₂ examples could not see.** ALR's empirical rule that ħ-corrections need no dots is exactly the cycle-free regime. sl₂ has one colour, so there are no adjacent-colour links and no cycles. The project's dotted ħ-corrections at A7 ω4, D6 and E6 are the first places where that rule must fail.

The project also finds that u never enters a stack and appears only in the gluing maps between envelopes ([GLUING.md §1](../GLUING.md)). Combined with ALR §5.5.7, the dotted Kirchhoff terms therefore drop out of link homology computed against the simple cap, while the dot-free hand-over terms survive. **For computing invariants, the hand-over terms are the bottleneck**, not the dotted cycle terms.

## Rouquier's braid defect makes the tree edges; the matrix-tree theorem makes the cycles

**Established: "Demazure of Q" is Rouquier's braid relation.** Among Rouquier's defining relations for H_n(Q), when ν_a = ν_{a+2}:

τ_{a+1}τ_aτ_{a+1} − τ_aτ_{a+1}τ_a = (x_{a+2} − x_a)^{−1} [Q_{ν_a ν_{a+1}}(x_{a+2}, x_{a+1}) − Q_{ν_a ν_{a+1}}(x_a, x_{a+1})].

He describes the algebras as a flat family **"filtered with associated graded algebra a wreath product of a polynomial algebra by a nil Hecke algebra"** ([Rouquier, arXiv:0812.5023](https://arxiv.org/abs/0812.5023), §3.2.1 and introduction). In Khovanov–Lauda's polynomial representation, equal-colour crossings act by divided differences ([Khovanov–Lauda, arXiv:0803.4121](https://arxiv.org/abs/0803.4121), §2.3, Prop. 2.3).

For Q_ij = t_ij u + t_ji v, the divided difference in the outer variables is the constant t_ij. Rescale the equal-colour crossing so that xψ − ψx = r_i, and the defect becomes **r_i t_ij times the smoothed diagram**. That is exactly the relation implemented in the project's code, and the label exchange the project derives for each tree edge ([HANDOVER.md §3.1](../HANDOVER.md); [NOTES.md §3](../NOTES.md)). At r_i = 0 equal-colour crossings commute with dots. So the undeformed differential δ₀ lives over Rouquier's associated graded algebra, and the true differential is its filtered lift.

Kang–Kashiwara–Kim's intertwiners, φ_a = (x_{a+1} − x_a)τ_a − 1 for equal labels, satisfy **φ_a² = Q_{ν_aν_{a+1}}(x_a, x_{a+1}) + δ and the braid relations on the nose**. The composite R_{N,M}R_{M,N} is a central product of Q-factors (Lemma 1.3.1, Prop. 1.4.4) ([KKK, arXiv:1304.0323](https://arxiv.org/abs/1304.0323)). The "−1" in φ_a isolates an identity-diagram term, which is the algebraic shape of a hand-over.

**Established: perturbation theory yields paths, trees or loops, depending on the structure.**
- **Paths.** The basic perturbation lemma gives b₁ = b + pAi with A = (1 − δh)^{−1}δ, a sum over linear chains of insertions ([Crainic, arXiv:math/0403266](https://arxiv.org/abs/math/0403266), Thm 2.4).
- **Trees.** They appear when a multilinear structure is transferred, as planar trees with vertices μ_k and internal edges h ([Markl, arXiv:math/0401007](https://arxiv.org/abs/math/0401007), Prop. 6; [Kontsevich–Soibelman, arXiv:math/0011041](https://arxiv.org/abs/math/0011041), §6.4, Thm 3).
- **Loops.** They need a trace or contraction:
  - modular operads replace "sums over trees" by "sums over general Feynman graphs" ([Chuang–Lazarev, arXiv:0802.3507](https://arxiv.org/abs/0802.3507));
  - quantum L∞ minimal models are Feynman expansions ([Doubek–Jurčo–Pulmann, arXiv:1712.02696](https://arxiv.org/abs/1712.02696));
  - wheeled PROPs encode traces through directed cycles ([Markl–Merkulov–Shadrin, arXiv:math/0610683](https://arxiv.org/abs/math/0610683)).
- **Hypercubes.** Their structure maps are sums over increasing chains, i.e. paths again ([Zemke, arXiv:2109.11520](https://arxiv.org/abs/2109.11520), eq. 2.6).
- **Eisenbud–Shamash.** Their higher homotopies correct a lifted resolution by multi-indices, not trees ([Sobieska, arXiv:2212.06999](https://arxiv.org/abs/2212.06999)).

No source gives a tree formula for the corrected differential of a complex of projectives over a deformed algebra.

**Established: obstruction theory frames existence and uniqueness.** Lowen proves that the obstruction to deforming M ∈ D^b(A) along c ∈ HH²(A) is χ(c)_M ∈ Ext²(M,M) ([Lowen, arXiv:0707.2602](https://arxiv.org/abs/0707.2602)). Mackaay–Webster's ∂̃²/h is this class computed on a projective resolution ([Mackaay–Webster](https://arxiv.org/abs/1502.06011), Lemma 2.23). Rickard's lifting theorem, seen only through secondary sources, makes tilting complexes lift uniquely over complete local deformations ([notes on Schaps](https://homepages.abdn.ac.uk/d.j.benson/pages/papers/s/schaps/Tiltings.pdf); [arXiv:2303.04893](https://arxiv.org/abs/2303.04893)). Braverman–Gaitsgory derive PBW for nonhomogeneous Koszul algebras from a deformation argument ([Braverman–Gaitsgory, arXiv:hep-th/9411113](https://arxiv.org/abs/hep-th/9411113)).

**Speculative but precise: the Maurer–Cartan picture.** Lift the cube termwise. Then κ = δ₀² lies in (r t)·End², and each leaf of κ is one Rouquier defect. The correction x must solve κ + [δ₀, x] + x² = 0. Given a contracting homotopy h, the recursion x = −h(κ + x²) generates rooted trees whose leaves are κ-insertions.

This explains why the project sees branching trees rather than the path sums of the perturbation lemma: corrections compose with each other. It also recasts the project's uniqueness result. Kernel 0 is Ext¹ vanishing in the relevant multidegree, and solvability at every order is vanishing of the Ext² obstruction. The 181 of 750 orders on local pieces where the solver found a kernel ([NOTES.md §6](../NOTES.md)) mark where that vanishing fails locally.

**Established identity: the single-cycle rule is a Kirchhoff polynomial.** Write c_e for the tree-edge weight r t and Q_e for the bigon polynomial. Then

Σ_T ∏_{e∈T} c_e ∏_{e∉T} Q_e = (∏_e Q_e)·det L_root(c/Q) = (∏_e c_e)·U_G(Q/c),

where U_G is the first Symanzik polynomial and L_root the reduced Laplacian at the mover ([Bogner–Weinzierl, arXiv:1002.3458](https://arxiv.org/abs/1002.3458)). When G has a single cycle, its spanning trees are the graphs G − e for e on the cycle. So **the project's rule Σ_{e on cycle} sign(G−e)·mono(G−e)·(t_{xy}[dot x] + t_{yx}[dot y]) is this polynomial with one non-tree edge** ([NOTES.md §6](../NOTES.md)).

The project's edge weights t_{c(p),c(parent)} depend on orientation towards the root, so the relevant form is the directed (arborescence) one. Forman's theorem gives that form: cycle-rooted spanning forests, each cycle weighted by (1 − w) (Kenyon Thm 6). Its Dirichlet version at a boundary set S has components that are either trees containing exactly one vertex of S or cycle-rooted trees avoiding S (Thm 7) ([Kenyon, arXiv:1001.4028](https://arxiv.org/abs/1001.4028)). Whether the project's sign(G−e) equals a directed-cofactor sign is unchecked.

**Speculative but precise: the Demazure form is the matrix-tree expansion.** The project observes that its dotted part equals a sum over orientations in which every pair has an outgoing link, which suggests the formula ∏_p (r_p ∂_p) ∏_links Q_link ([NOTES.md §6](../NOTES.md)). Expanding it with the twisted Leibniz rule ∂_p(fg) = ∂_p(f)g + s_p(f)∂_p(g) goes as follows:
- every non-root vertex chooses one incident linear Q to differentiate;
- an edge chosen twice dies;
- the choices form functional digraphs, i.e. trees rooted at the mover, or unicycles.

That is the expansion behind combinatorial proofs of the matrix-tree theorem, where the unicycles cancel. If the reflections s_p act like a nontrivial holonomy, cycle-rooted terms survive, as in Forman–Kenyon. Those survivors are the natural candidates for the hand-over terms. The project's finding that in-between cycle nodes come with opposite tree signs, so that only an r-correction remains, looks like the beginning of that cancellation.

The nearest published statement of the form "Demazure of a product over edges is a tree count" is divided symmetrisation. Petrov's Thm 2 evaluates DS_G on trees, but no formula exists for graphs with cycles ([Petrov, arXiv:1512.07136](https://arxiv.org/abs/1512.07136)). Since ∂_{w₀}(g) = Sym(g/Δ), DS_G(f) = ±∂_{w₀}(f ∏_{non-edges}(x_i − x_j)).

## A7, D6 and E6 are the smallest Coxeter graphs that break Deodhar's factorisation

**Project computation.** Each representation has a **unique minimal cyclic weight μ_c**: a stack is cyclic iff μ ≤ μ_c. Every larger case contains one of three seeds on a sub-diagram ([NOTES.md §4](../NOTES.md)).

| Seed | dim V | heap size = dim G/P | project μ_c (strands; λ − μ_c) | first non-Deodhar ideal | Billey–Jones minimal graph | E7 symmetric space (speculative) |
|---|---|---|---|---|---|---|
| A7, Λ⁴ℂ⁸ | 70 | 16, Gr(4,8) | lowest + α4 (15; 1,2,3,3,3,2,1) | 15 | A7: hexagon, 14-element heap | EV = E7/(SU(8)/±1), dim 70 |
| D6, half-spin | 32 | 15, OG(6,12) | lowest + α6 + α4 (13; 1,2,3,3,2,2) | 13 | D6: one pattern | EVI = E7/(SO(12)·SU(2)), dim 64 = 2·32 |
| E6, 27 | 27 | 16, OP² | lowest + α6 + α5 (14; 2,2,3,4,2,1) | **13** | E6: patterns of length 13, 13, 13, 14 | EVII = E7/(E6·SO(2)), dim 54 = 27 + 27 |

The minimal cyclic configuration always has the same shape: three consecutive slots of one colour, carrying the mover and one pair, plus pairs in two adjacent colours, one straddling low and one high. Interlaced same-colour pairs and factors r_c² need a colour that occurs four times.

The notes observe that the full minuscule posets containing the cyclic pattern are exactly those in which **some colour occurs at least four times**, i.e. the maximal coefficient of ω − w₀ω is at least 4. These are:
- Gr(k,n) with min(k, n−k) ≥ 4;
- D_n half-spin with n ≥ 6;
- E6 and E7.

The D_n vector representation (maximal coefficient 2) and the D5 spinor (3) fall short. This characterises the cyclic families, not the threshold ideals: the (4,4,4,3) ideal contains colour 4 only three times.

**Established: the seeds are Billey–Jones's minimal non-Deodhar types.** The simply-laced graphs in Billey–Jones's Fig. 1 are A7, D6, D7, D8, E6 and E7. **The rank-minimal ones in each family, A7, D6 and E6, are exactly the project's seeds** ([Billey–Jones, arXiv:math/0612043](https://arxiv.org/abs/math/0612043), Thm 5.12). D7 and D8 carry further minimal patterns, and type D even admits an infinite antichain of non-Deodhar elements (Remark 5.10, Ex. 5.11). The notes computation and the project's D_n spinor threshold, 13 for every n ≥ 6, together show these extra patterns create no new minuscule seeds. Of all the explanations considered, this is the only one tied to a theorem about heaps.

**Speculative: an E7 coincidence.** By dimension count, the three seed representations are the isotropy representations of E7's three compact symmetric spaces: EV (dim 70, rank 7), EVI (dim 64, quaternion-Kähler) and EVII (dim 54, Hermitian) ([Wikipedia: Symmetric space](https://en.wikipedia.org/wiki/Symmetric_space); [Wikipedia: E7](https://en.wikipedia.org/wiki/E7_(mathematics))). The arithmetic works: 133 = 63 + 70 = (3 + 66) + 64 = (78 + 1) + 54. But no source links E7 symmetric pairs to Deodhar patterns or to cycles in minuscule heaps.

Freudenthal's magic square does not explain the triple either. Landsberg–Manivel place OP² among the Severi varieties in row 2 and the spinor variety S₁₂ in the Legendrian row 3, and **Gr(4,8) is absent from the square** ([Landsberg–Manivel, arXiv:math/9908039](https://arxiv.org/abs/math/9908039)).

The defensible position is the Coxeter-theoretic one. A7, D6 and E6 are the smallest simply-laced graphs whose fully commutative heaps can host a non-tight factorisation. Λ⁴ℂ⁸, the 32 and the 27 are the first minuscule representations whose heaps are large enough to contain one. The E7 pattern stays a coincidence until a mechanism appears.

## Eight papers to read first, ranked

| Rank | Paper | Where to look | Why first |
|---|---|---|---|
| 1 | Billey–Jones, "Embedded factor patterns for Deodhar elements in Kazhdan–Lusztig theory", [arXiv:math/0612043](https://arxiv.org/abs/math/0612043) | Thm 2.3, Def. 2.1, Thm 5.12 with Fig. 1, §6 (Ex. 6.4) | The only published condition matching both the threshold and the seed triple. Its mask/defect calculus on heaps is what a cube-as-masks test needs, and its E6 words locate the type-E discrepancy. |
| 2 | Webster, "Knot invariants and higher representation theory", [arXiv:1309.3796](https://arxiv.org/abs/1309.3796) | Cor. 5.22–5.24, Props 7.6, 7.8, 8.3, 9.14 | Defines the standard modules and the subset-cube resolution whose differential the project writes. "We will not do so here" makes the project's formula a direct completion, and type-A Koszulity gives a check at a specialisation. |
| 3 | Aganagic–LePage–Rapčák, [arXiv:2305.13480](https://arxiv.org/abs/2305.13480) | §2.1, §3.3–3.4 (eqs 3.17–3.18, §3.4.3–3.4.5), §5.5–5.6 | The same order-by-order deformation problem for all minuscule ^Lg, with gauge uniqueness and dot bookkeeping. The project's A7, D6 and E6 data are the first examples beyond its sl₂ scope. |
| 4 | Brundan–Kleshchev–McNamara, [arXiv:1210.6900](https://arxiv.org/abs/1210.6900) | Thm 4.10, §4.5 (eq. 4.17), Thm 4.12, the A3 example | The prototype KLR cube. Its admitted gap for roots that are not multiplicity-free is the repeated-colour regime the project's mechanism handles. |
| 5 | Rouquier, "2-Kac-Moody algebras", [arXiv:0812.5023](https://arxiv.org/abs/0812.5023), with Kang–Kashiwara–Kim, [arXiv:1304.0323](https://arxiv.org/abs/1304.0323) | §3.2.1–3.2.2; KKK Lemma 1.3.1, Prop. 1.4.4 | The braid relation as the divided difference of Q gives r t per edge, and the filtered/associated-graded statement gives the Rees picture. Intertwiners satisfy the braid relations exactly, a basis in which the tree sums may collapse and the hand-overs separate. |
| 6 | Kenyon, "Spanning forests and the vector bundle Laplacian", [arXiv:1001.4028](https://arxiv.org/abs/1001.4028), with Bogner–Weinzierl, [arXiv:1002.3458](https://arxiv.org/abs/1002.3458) | Kenyon Thms 1, 5–7; B–W on Symanzik polynomials and the all-minors matrix-tree theorem | Turns the single-cycle rule into a determinant. Directed and Dirichlet cycle-rooted expansions are the template for hand-overs and for predictions with several cycles. |
| 7 | Billey–Warrington, "Kazhdan–Lusztig polynomials for 321-hexagon-avoiding permutations", [arXiv:math/0005052](https://arxiv.org/abs/math/0005052) | main theorem, equivalences, the hexagon | The type-A theorem behind the (4,4,4,3) match, with an explicit KL formula and the small-Bott–Samelson criterion. |
| 8 | Mackaay–Webster, [arXiv:1502.06011](https://arxiv.org/abs/1502.06011), with Lowen, [arXiv:0707.2602](https://arxiv.org/abs/0707.2602) | M–W §2.3, Prop. 2.22, Lemma 2.23; Lowen's main theorem | The obstruction formalism for lifting projective resolutions to deformed KLRW algebras: the framework for proving existence and uniqueness of the corrected differential. |

## Six conjectures the literature makes testable

| # | Conjecture | Literature hook | Concrete test |
|---|---|---|---|
| 1 | **The cube is a Deodhar-mask complex.** Vertices I ⊆ slots are masks on the heap of w (m = ℓ(w)). The tree formula is complete iff w is Deodhar, and the cyclic terms track the extra C′_y. | Billey–Jones Thm 2.3, §6; Billey–Warrington | Run the full-ansatz solver on the unsolved 13-slot E6 stack, which is the Billey–Jones pattern. Compare the minimal violating masks in A7 and D6 (A7 zeros {1.1, 2.2, 3.3, 4.2, 4.3, 5.3, 6.2, 7.1}) with the minimal cyclic configurations. Match the 1, 2, 2, 3 extra C′_y against counts of new term families. |
| 2 | **Matrix-tree identity.** On a configuration with link graph G, the dotted part is Σ_T sign_T ∏_{e∈T} r t ∏_{e∉T} Q_e, a directed reduced-Laplacian determinant at the mover. This predicts b₁(G) dots for b₁ independent cycles. | Bogner–Weinzierl; Kenyon Thm 6; Abdesselam, [arXiv:math/0306396](https://arxiv.org/abs/math/0306396) | Solve two-cycle pieces in D7, E7 or larger A_n ω_k, and check that sign(G−e) is a cofactor sign. |
| 3 | **Demazure form.** d_cyc = ∏_p r_p ∂_p ∏_links Q_link in the polynomial representation, with the hand-over terms as cycle-rooted survivors of the s_p-twists. | Rouquier §3.2.1; Kenyon Thm 7; Petrov Thm 2 | Implement the operator and compare it with the 336 dotted and 188 dot-free extra terms in [CATALOGUE_cyclic.txt](../CATALOGUE_cyclic.txt). |
| 4 | **Maurer–Cartan / curvature.** d = δ₀ + x solves κ + [δ₀, x] + x² = 0, where κ = δ₀² is built from Rouquier defects. The tree sum is the recursion x = −h(κ + x²) for an explicit homotopy h, and uniqueness is Ext¹ vanishing in the relevant multidegree. | Lowen; Mackaay–Webster Lemma 2.23; Markl Prop. 6; Kontsevich–Soibelman §6.4 | Build h on End(P₀) for the undeformed Koszul cube over the associated graded algebra. Check that the recursion reproduces ε ∏ r t, including the project's sign κ(−1)^{#x_p∈I}. |
| 5 | **The method closes BKM's gap.** The mover-plus-exchange mechanism yields BKM's τ_{σ,ρ} for roots that are not multiplicity-free. | BKM §4.5, Thm 4.12 | Compute the root-module cube for the highest root of D4 (central multiplicity 2) with the solver, and compare it with the tree-formula prediction. |
| 6 | **Hand-overs control invariants.** Hom into the simple cap kills dots, so only tree and hand-over terms reach link homology for Λ⁴ℂ⁸, the 32 and the 27. | ALR §5.5.7; LePage–Shende Thm 1.1 | Compute Hom(glued I-brane, simple) on the A7 ω4 pieces with and without the dotted cyclic terms. |

## Conclusion

The project's structure sits where three literatures meet without citing each other. KLR resolution theory, whose closed forms stop at multiplicity-free words, has no explanation for it. Kazhdan–Lusztig combinatorics already owns the threshold, and graph-polynomial theory already owns the shape of the cycle corrections.

The most consequential observation is that the cube's 2^{ℓ(w)} vertices and Deodhar's 2^{ℓ(w)} masks are indexed by the same subsets of the same heap. If the tree regime is the Deodhar regime, the tree formula is a categorified Bott–Samelson expansion and the cyclic terms are its Kazhdan–Lusztig corrections. That reading would also explain the seed triple without appealing to E7. The 13-slot E6 stack is the decisive experiment: it is a Billey–Jones minimal pattern, the project has never solved it, and either outcome teaches something. Extra terms there would promote the A/D coincidence to a conjecture covering all types. Their absence would prove the link graph is a genuinely different invariant.

On the algebra side, the content lives in the parameters left uninverted. Over a Laurent ring, r and t rescale to 1, so it is the multigrading that makes the correction unique and its combinatorics visible. The Kirchhoff and Demazure forms then make one sharp, testable prediction: b₁ dots for b₁ independent cycles. And because dots vanish against the simple cap, the gap between the current formulas and actual link-homology computations for Λ⁴ℂ⁸, the half-spin 32 and the 27 is the poorly understood dot-free hand-over terms, not the elegant dotted Kirchhoff terms.
