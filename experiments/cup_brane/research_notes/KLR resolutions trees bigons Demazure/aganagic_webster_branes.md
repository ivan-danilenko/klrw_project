# Explicit resolutions of cup/cap branes, I-branes, T-branes and stable envelopes in the Coulomb branch / KLRW approach to knot categorification

Method note. The primary sources were read from their arXiv LaTeX sources (latest versions, downloaded 2026-11-25). Section, theorem and equation numbers below were computed from those sources: comment blocks were excluded and `\newtheorem` counters were followed. Journal versions may be numbered differently. "Stated" means the paper says it. Anything marked **Inference** is my own reading.

Notation across papers:
- ALR = Aganagic–LePage–Rapčák, arXiv:2305.13480.
- A-I = Aganagic, Part I, arXiv:2004.14518.
- A-II = Aganagic, Part II, arXiv:2105.06039.
- ADLSZ = Aganagic–Danilenko–Li–Shende–Zhou, arXiv:2406.04258.
- LS = LePage–Shende, arXiv:2505.00327.
- KIHRT = Webster, "Knot invariants and higher representation theory", arXiv:1309.3796.
- CSQCB I/II = Webster, arXiv:1905.04623 and arXiv:2211.02099.

Dictionary of the brane names:
- **T-branes** are thimbles, i.e. products of real lines on the cylinder 𝒜. They correspond to projective modules.
- **I-branes** are dual thimbles, i.e. products of intervals between punctures. They correspond to simple modules. The cap I_𝒰 is one of them.
- **E-branes** are products of "figure-eight" curves around pairs of punctures. These are the cup branes. In sl_2 an E-brane is h^*h_* of an I-brane.
- **U-shaped Lagrangians** are cones T_{i+1} → T_i. They correspond to standard modules.

## Q1. What explicit cup/cap resolutions exist for sl_2 (and sl_n fundamentals)? How do they compare with Webster's cup/cap bimodules and Khovanov's arc algebra?

### Takeaway
For sl_2 everything is explicit, at three levels.
1. ALR §5.6.1 resolves a single cup (figure-eight brane E_𝒰) by four T-branes, T_2{-1}[2] ⊕ (T_1 ⊕ T_3{-1})[1] ⊕ T_2. ALR call this "the standard Koszul resolution of the structure sheaf of a ℙ¹". It has the same 1–2–1 shape as Webster's resolution of the cup simple module, θ{-2} → θ_+{-1} ⊕ θ_-{-1} → θ.
2. LePage–Shende (2025) prove that Webster's cup ∪_W^n, a *simple* KLRW module, goes to Aganagic's cup ∪_A^n = E_Π(n) under the ADLSZ embedding. They also prove the full braid-group intertwining, so Aganagic's invariant equals Khovanov homology over ℤ.
3. For several cups, or braided cups, ALR build the resolution as a product grid ("toric grid") of one-dimensional resolutions. They then correct it order by order: first in u (terms with dots), then in ħ.

For sl_n with n ≥ 3 and fundamental weights, no explicit A-brane resolution of a cup is written down in this literature. The comparisons that exist are algebraic: Webster's invariant equals Sussan, Mazorchuk–Stroppel and Khovanov–Rozansky homology.

### Cited Findings
**ALR, the sl_2 theory (§5)**
- ALR §2.1.9 states that, to keep the length contained, the paper specializes to su_2 and gl_{1|1} with links coloured by their 2-dimensional defining representations. Y = Sym^d(𝒜). The cap I_𝒰 is the product of d intervals (red segments). The braided cup ℬE_𝒰 is obtained "by replacing the blue segments with products of d figure eights in the su_2 case, or ovals for gl_{1|1}" — [ALR](https://arxiv.org/abs/2305.13480).
- Link homology is defined as Hom_{𝒟_Y}(ℬE_𝒰, I_𝒰). Because I_𝒰 is a simple of the thimble algebra A, a projective resolution (ℬE(T), δ) of ℬE_𝒰 by T-branes yields the Floer complex "for free": apply hom(−, I_𝒰) and use the duality hom(T_𝒞, I_𝒞′) = δ_{𝒞,𝒞′}ℂ (§2.2.4, §2.3.1–2.3.3, §3.5) — [ALR](https://arxiv.org/abs/2305.13480).
- **Single sl_2 cup (ALR §5.6.1).** Y = 𝒜 with four punctures, mirror to the resolved A_3 surface. Isotoping E_{𝒰_2} and breaking it at y = 0 and y = ∞ gives E_{𝒰_2}(T) = T_2{−1}[2] ⊕ (T_1 ⊕ T_3{−1})[1] ⊕ T_2. The maps are the four intersection points at y = ∞ and y = 0. The differential squares to zero both downstairs in A and upstairs in 𝒜, and "The complex is the standard Koszul resolution of the structure sheaf of a ℙ¹ in 𝒳". ALR also note that only three of the four cup branes are structure sheaves of vanishing ℙ¹'s; the fourth is described in A-II — [ALR](https://arxiv.org/abs/2305.13480).
- **Same complex in A-II, with explicit matrices.** A-II §6.5 (the A_{m−1}-surface example), eq. (cupm), writes E_i ≅ T_i{−1,0} → T_{i+1}{−1,0} ⊕ T_{i−1} → T_i. The first map is (a_{i+1,i}, −b_{i−1,i})^T and the second is (b_{i,i+1}, a_{i,i−1}). It is the mirror of the resolution of N_i = f^*f_*S_i (S_i = 𝒪 of the 2-cycle) by projectives P_i. Each arrow is a cone, i.e. a connected sum of thimbles over an intersection point at one of the two ends of 𝒜 — [A-II](https://arxiv.org/abs/2105.06039).
- **One-strand braided cup with a dot correction (ALR §5.6.2).** For a braided one-dimensional ℬE_𝒰 the direct-sum brane is T_4{−3}[2] ⊕ (T_6{−3} ⊕ T_4{−2} ⊕ T_1)[1] ⊕ (T_5{−2} ⊕ T_2).
  - The geometric differential δ_{u=0} is already a resolution downstairs.
  - Upstairs, after filling in the puncture divisor D_a, exactly one map with a dot is allowed on degree grounds. One finds δ² = (u+x)(…), hence x = −u.
  - "The correction of order u^ℓ corresponds to introducing maps with ℓ dots." Passing back downstairs kills the dots — [ALR](https://arxiv.org/abs/2305.13480).
- **Two-dimensional example (ALR §5.6.3).** This is a d = 2 brane: the figure-eight together with an outer brane.
  - The product grid has terms T_{34}{−1}[3] ⊕ (T_{44}{−1}, T_{23}, T_{24})[2] ⊕ (T_{24}, T_{34}, T_{22})[1] ⊕ T_{23}.
  - δ_{u,0} squares to zero in 𝒜_{u,0} but not in 𝒜_{u,ħ}. "In fact, the first order deformation δ = δ_{u,0} + ħδ_{u,1} turns out to suffice."
  - The solution sets x_1 = uħ and x_2 = 0. The resulting Floer differential is ℂ →^{uħ} ℂ, which counts a disk through one puncture and the diagonal — [ALR](https://arxiv.org/abs/2305.13480).
- **Reduced unknot (ALR §5.6.4).** The product of the two one-dimensional resolutions is "a grid on a torus of size 6×4".
  - The u ≠ 0 differential is the product of the one-dimensional u-corrected differentials, but "only a subset of one-dimensional u ≠ 0 corrections lift to the product".
  - For the ħ step, they restrict δ_{u,1} to the simplest terms (no crossings, no dots): "19 such possible entries". Solving the first-order equation gives a one-parameter family of solutions.
  - Final Floer complex: 0 → ℂ{−1} →^{uħ} ℂ{−1} →^0 ℂ{−2} → 0 — [ALR](https://arxiv.org/abs/2305.13480).
- **Hopf link (ALR §5.6.5).** Dotted u-corrections appear. It "proves sufficient to turn on only the first order in ħ deformation … and further restrict the possible terms in δ_{u,1} to maps without dots". The equations give, for example, x_4 = −x_8 = −uħ. The trefoil is sketched in §5.6.6 — [ALR](https://arxiv.org/abs/2305.13480).
- **gl_{1|1} analogues (ALR §4.5).**
  - A single cup is the two-term complex T_0[1] ⊕ T_2.
  - Corrections there are in ħ only; for gl_{1|1}, ħ deforms the differential of the dg algebra, not its product.
  - The example in §4.5.4 needs one map with coefficient −ħ (an empty rectangle).
  - The 8_19 example (§4.5.6) "requires a second-order correction in ħ to δ_0".
  - The algebra is a cylinder version of the Lipshitz–Ozsváth–Thurston algebra, with m_ℓ = 0 for ℓ > 2 (§4.3.11–4.3.12) — [ALR](https://arxiv.org/abs/2305.13480).
- **LePage's String-Math write-up, arXiv:2307.07034 (§4.2).** It restates the sl_2 algorithm.
  - Figure-eight E_2 resolution; for the Hopf-link brane ℬE_1, "This differential squares to zero when u = 0 but not when u ≠ 0 … we must add dotted correction terms", fixed to x_1 = −u and x_2 = u.
  - For a d = 2 product brane, "There are only two possible corrections allowed by degrees", fixed to x_1 = 0 and x_2 = −uħ.
  - Abstract: "This algorithm can be extended to all Lie algebras" — [LePage 2023](https://arxiv.org/abs/2307.07034).

**LePage–Shende: identification with Webster's cups and with Khovanov homology**
- Theorem 1.1: for Γ = • (sl_2), the embedding Perf 𝒜^cyl(Γ, d, a) ↪ Fuk(𝓜^×(Γ, d), 𝒲_a) of ADLSZ "intertwines the braid group representations ρ_W and ρ_A, and carries ∪_W^n ↦ ∪_A^n". Hence Kh(β̄) ≅ Hom(∪_A^n, ρ_A(β)∪_A^n). This holds over ℤ, whereas Seidel–Smith is "only known to recover Khovanov homology over ℚ" — [LS](https://arxiv.org/abs/2505.00327).
- Webster's cup ∪_W^n is the simple KLRW module S_{Π(n)}. Lemma 10.2: 𝔸(S_{Π(n)}) ≅ E_{Π(n)}, the product of figure-eights. The proof computes Hom(T_{Π(n)}, E_{Π(n)}) ≅ ℤ by taking "the tensor product of n copies of the complex for a single component" (§10.1) — [LS](https://arxiv.org/abs/2505.00327).
- Lemma 10.1: a figure-eight "can be resolved as a cone on some morphism between two downward U objects" — [LS](https://arxiv.org/abs/2505.00327).
- §10.2 ("Aganagic's caps") gives a sketch: "It has since been understood that ∪_A^n ≅ ∩_A^n".
  - Lemma 10.3: the U Lagrangian ≅ the step Lagrangian J.
  - Proposition 10.4: I ≅ 𝔸(S).
  - Corollary 10.5: E ≅ I.
  - Footnote: the authors do not check that I_Π is a bona fide object of the wrapped category — [LS](https://arxiv.org/abs/2505.00327).
- **Webster's resolution of the sl_2 cup simple**, quoted by LS in §3 from Webster's "Tensor product algebras, Grassmannians and Khovanov homology" (Contemp. Math. 680, 2016, §4.3): θ{−2} → θ_+{−1} ⊕ θ_−{−1} → θ → S_θ. The differentials are right multiplication by red–black crossings, and the braces are J-degree shifts — [LS](https://arxiv.org/abs/2505.00327).
- LS Proposition 10.4 adds: the resolution of S_θ "is the cone over a map in homological degree zero from υ_− to υ_+". Here υ_+ := {θ_+{−1} → θ} and υ_− are the two U-objects, i.e. standard-type cones — [LS](https://arxiv.org/abs/2505.00327).
- **Braiding.** LS, footnote in §1: braiding generating objects naively gives "a complex of size 3^n on the ρ_A side (not all of whose maps we are entirely certain we can rigorously determine) whereas, on the ρ_W side, the expected complex has n+1 terms".
  - They therefore use a cleverer resolution, Λ_n := (υ_+ × θ_+^{n−1})^{⊕n} → θ_+^n.
  - Prop. 8.1 shows θ^n ≅ Λ_n. Prop. 4.1 shows 𝔹_{τ(π)}(Λ_n) = Λ′_n — [LS](https://arxiv.org/abs/2505.00327).

**Webster's cup/cap functors and the comparison with category O / Khovanov–Rozansky**
- KIHRT §7.1 sets up the coevaluation (cup) and evaluation (cap) for a pair (λ, λ*) with λ* = −w_0λ.
  - Definition 7.1 builds the sequence 𝐢^λ_𝐰 = (i_1^{(λ^{i_1})}, i_2^{((w_1λ)^{i_2})}, …) from a reduced word for w_0 (with divided powers). It has 2ρ^∨(λ) black strands.
  - Lemma 7.3: the standard and projective module S^{(0,2ρ^∨)}_{𝐢_λ} has a unique simple quotient 𝔏 (written L_λ in the source).
  - Definition 7.4: 𝕂 = RHom(L̇_λ, −)(2⟨λ,ρ⟩)[−2ρ^∨(λ)] and 𝔼 = − ⊗^L L̇_{λ*}.
  - Proposition 7.5: these categorify coevaluation and evaluation — [KIHRT](https://arxiv.org/abs/1309.3796).
- KIHRT Proposition 9.21, under the standing assumption of §9.4 that the weights are fundamental: Webster's invariant matches Sussan's and Mazorchuk–Stroppel's functors (via a Koszul/Ringel duality functor Ξ) for the defining representation of sl_n. "In particular, it coincides with Khovanov homology when g = sl_2 and Khovanov–Rozansky homology when g = sl_3."
  - Prop. 9.20 compares the cup/cap functors with Sussan's and Mazorchuk–Stroppel's.
  - Webster says he believes it agrees with Khovanov–Rozansky for all n with the defining representation, conjectured in Mazorchuk–Stroppel; that statement is in the text, not a theorem — [KIHRT](https://arxiv.org/abs/1309.3796).
- CSQCB II Theorem 7.12: the Coulomb-branch/cylindrical-KLRW invariant equals KIHRT's invariant. "If Γ is of type A_1, then Φ(K) coincides with Khovanov homology", and in type A_n it coincides with Khovanov–Rozansky sl_{n+1} homology. Webster points to Mackaay–Webster, Th. A, for the type-A comparisons — [CSQCB II](https://arxiv.org/abs/2211.02099).
- A-II §7.7 deduces the same over ℂ from equivariant HMS plus "theorems F and 7.12 of [W2]", with i = M + 2J + i_0 and j = 2J + j_0 — [A-II](https://arxiv.org/abs/2105.06039).
- Relation to Khovanov's own complexes: A-II §7.9 uses grid-diagram presentations (after Droz) to match a subset of Khovanov generators with intersection points of ℬE_𝒰 ∩ I_𝒰. For the trefoil this gives 16 intersection points and a grading table — [A-II](https://arxiv.org/abs/2105.06039).
- Arc algebra on the Seidel–Smith side: Mak–Smith (arXiv:1907.07624, JEMS) realize Stroppel's extended arc algebra in the Fukaya–Seidel category of a Lefschetz fibration on a two-Jordan-block type-A nilpotent slice. They also give a symplectic model of two-block parabolic category O and a cylindrical model for Hilbert schemes — [Mak–Smith](https://arxiv.org/abs/1907.07624). ADLSZ thank Mak, Tian and Yuan for explaining [Mak–Smith] and [Honda–Tian–Yuan] — [ADLSZ](https://arxiv.org/abs/2406.04258).

### Inferences
- **Inference.** The ALR single-cup complex (T_2 at both ends, T_1 ⊕ T_3 in the middle) and Webster's θ{−2} → θ_+ ⊕ θ_− → θ are the same complex under θ ↔ T_2 and θ_± ↔ T_3, T_1. LS prove E ≅ 𝔸(S_θ), which supports this, but neither paper writes the matrix dictionary down. The shape is a 2-term Koszul complex: a square with one vertex in degree 2, two in degree 1 and one in degree 0.
- **Inference.** For n disjoint, unbraided cups the resolution should be the n-fold tensor product of these squares, i.e. a 2n-dimensional Koszul cube. LS use this tensor-product structure for Hom(T, E). ALR's grid construction gives it geometrically. ALR's examples need ħ-corrections only when factor branes interleave (nested or braided). Disjoint, unbraided cups have no reason to interact, but I found no explicit statement that their product needs no correction.
- **Inference.** The pattern in all reported sl_2 examples is: u-corrections (dots) come from one-dimensional factors, and first-order ħ-corrections without dots suffice. This is an empirical observation in ALR's examples, not a theorem.

### Gaps
- I found no explicit A-brane (thimble) resolution of a cup for sl_n, n ≥ 3, in ALR, ADLSZ, LS or LePage 2023.
- I could not access LePage's PhD thesis ("Homological Link Invariants from Lagrangian Floer Theory", Berkeley physics, supervised by Aganagic; listed on her CV per search results), so I cannot say whether it contains sl_3 or ADE computations. The eScholarship PDF I fetched turned out to be a different Berkeley thesis.
- I did not re-verify in this session the arc-algebra side: Khovanov's H^n, Brundan–Stroppel's explicit linear projective resolutions of cell modules, and the Chen–Khovanov subquotients. No primary text was read for them.

## Q2. What is known for minuscule representations of general simply-laced g: existence, shape, closed formulas, ħ/dot corrections, algorithms, conjectures?

### Takeaway
The framework is formulated for all simply-laced g with minuscule colours. The cup is 𝒪 of a product of minuscule Grassmannians G/P_i on the B-side, and an interval-type brane ordered by root subtraction on the A-side. ALR state an algorithm valid for any ^Lg: build the product grid of one-dimensional resolutions, then deform order by order in u and ħ by solving linear equations. They argue existence and uniqueness up to gauge, and that the series truncate.

No closed formula and no worked example beyond su_2 and gl_{1|1} is published on the A-side.

On the algebraic side the existence of a finite resolution of the cup simple 𝔏 holds exactly in the minuscule case (finite global dimension). Its shape is explicit only for ℓ = 2 red strands, which is the cup case itself:
- Webster gives a projective resolution of each standard module as a cube indexed by subsets T (KIHRT Prop. 7.6).
- He also gives a resolution of 𝔏 by standardly filtered modules of length 2ρ^∨(λ) (KIHRT Prop. 7.8).
- He says the general-ℓ version is "more subtle and difficult … we will not do so here."

### Cited Findings
- **Scope of ALR (§2.1).** "Pick ^Lg, which is either a simply laced Lie algebra, or the Lie superalgebra gl(m|n). Pick also … pairs [of marked points], labeled by a minuscule representation V_i of ^Lg, and its conjugate V_i^*." The theory "extends to non-simply laced Lie algebras ^Lg, and to sp(m|2n), with some minor modifications. It is possible it exists more generally yet." §1.2: "We give an algorithm for computing link homologies that applies for any ^Lg" — [ALR](https://arxiv.org/abs/2305.13480).
- **The algorithm (ALR §3).**
  - Product Lagrangians L = L_1 × … × L_d are broken at the ends of 𝒜 into T-branes, giving a d-dimensional rectangular grid with differential δ_O (§3.3.1–3.3.3). Relative gradings are read off from the edge maps (§3.3.2).
  - δ_O is a differential on Y_O = Y ∖ D_O, with D_O ⊃ diagonals. Filling the divisor back in deforms δ_O.
  - The expansion is δ_ħ = Σ_k ħ^k δ_k (eq. 3.17). At each order a linear inhomogeneous equation {δ_0, δ_k}_0 = D_k(δ_0, …, δ_{k−1}) is solved (eq. 3.18, §3.4.4).
  - "In general, the deformation problem may not have any solutions, and if it does, the solution may not be unique." Existence holds because both complexes resolve the same product Lagrangian avoiding the diagonal. Uniqueness up to gauge e^b δ e^{−b} follows from equivariance, since hom_{Y_0}(T_𝒞, T_𝒞′) is at most one-dimensional in each degree (§3.4.5).
  - Truncation of the series follows from a broken symmetry under which ħ has degree 1 (§3.4.3).
  - For su_2, D_O = Δ ∪ (puncture divisors), and filling the punctures in is the easier step (§3.4.6) — [ALR](https://arxiv.org/abs/2305.13480).
- **Two-stage deformation (ALR §5.5.3).** For su_2 the deformation goes δ_{0,0} → δ_{u,0} = Σ u^ℓ δ_{ℓ,0} → δ_{u,ħ} = Σ ħ^k δ_{u,k}. "The terms in δ_{ℓ,0} are corrections involving ℓ dots, since at ħ = 0 the theory has a symmetry that scales the dots and u oppositely. There are at most finitely many terms". The upstairs complex has more information; "taking the hom with the I_𝒰-branes … all the dots in δ get set to zero anyhow … because the I_𝒰 branes are simples" (§5.5.7) — [ALR](https://arxiv.org/abs/2305.13480).
- **Upstairs vs downstairs (ALR §5.3.7–5.4).**
  - Upstairs, "a cup and a cap are the same object": 𝒰 = 𝒪_U on 𝒳, with U = (ℙ¹)^d for su_2.
  - Downstairs, I_𝒰 maps to 𝒰 under h_*, and ℬE_𝒰 = h^*ℬ𝒰. By adjunction, Hom_{𝒟_Y}(ℬE_𝒰, I_𝒰) = Hom_{𝒟_𝒳}(ℬ𝒰, 𝒰).
  - T-branes carry a local system of modules over ℬ = ℂ[z_1, …, z_d]/(symmetric functions), of rank d!. The downstairs algebra is A = 𝒜/ℐ (§5.4.14–5.4.16).
  - The proof that the right-hand side is a link invariant "for any simply-laced Lie algebra ^Lg" is credited to A-I — [ALR](https://arxiv.org/abs/2305.13480).
- **B-side cup for general minuscule colours (A-I §6.2).** For 2m vertex operators coloured (V_i, V_i^*), 𝒳 = Gr^{μ⃗_{2m}}_0. The vanishing cycle is U_i = Gr^{μ_i} = G/P_i, "the minuscule Grassmannian" (Appendix A.4). The cup object is 𝒰 = 𝒪_U with U = G/P_1 × … × G/P_m, and locally 𝒳 ∼ T^*U.
  - §6.4: the cup and cap functors are Fourier–Mukai transforms with kernel 𝒪_{C_i}, following Cautis–Kamnitzer, with 𝒞_i(ℱ) = i_*f^*ℱ.
  - §1.1: the restriction to minuscule colours is needed because "Otherwise, 𝒳 develops singularities due to monopole bubbling" — [A-I](https://arxiv.org/abs/2004.14518).
- **A-side shape of the general minuscule interval brane (A-II §8.2, "Fusion and vanishing cycles").** For 𝒳 = Gr^{(μ_i, μ_j)}_{μ_k} = T^*F_k, the special brane is L_k : a_i ≤ y_1, …, y_{D_k} ≤ a_j, a product of straight intervals, with D_k = d_i + d_j − d_k and d_i = ⟨μ_i, ρ⟩.
  - "We restrict t's so that y's corresponding to roots that are linked remain ordered on L_k. The order in which y_i's appear … is the order in which we subtract the corresponding simple roots from μ_i to get −μ_j, or vice versa. The fact that this is well defined depends crucially on the fact μ_i and μ_j are highest weights of minuscule representations."
  - Braiding acts as ℬL_k = L_k[D_k]{C^g_k} (eq. CDS) — [A-II](https://arxiv.org/abs/2105.06039).
- **ICM survey (arXiv:2207.14104).** "In [ALR], we give an explicit algorithm for computing homological link invariants for … ^Lg and links colored by its minuscule representations" (§1.4). Cups and caps coloured by minuscule representations come from structure sheaves of (products of) minuscule Grassmannians (§4.3, §5.4). A subsection "Projective resolutions from geometry" (§6.3) restates ALR. The ICM numbering was computed from the source without a check for comment blocks. — [Aganagic ICM 2022](https://arxiv.org/abs/2207.14104).
- **Webster: finiteness exactly for minuscule weights.**
  - KIHRT Prop. 8.3: "If all λ_i are minuscule, then the cohomology of Φ(T)(𝕂) is finite-dimensional." In that case the preorder on standards is a partial order, End(S) ≅ 𝕂 for indecomposable standards, T^λ-mod is highest weight, and so it has finite homological dimension.
  - §8.1: "The only case where the invariant is known to be finite dimensional is when the representations λ are minuscule" — [KIHRT](https://arxiv.org/abs/1309.3796).
- **Shape of resolutions for two red strands (KIHRT §7.1).**
  - Prop. 7.6: "The standard S_𝐢^{κ_j} has a projective resolution of the form ⋯ → ⊕_{|T|=k} P^{κ_T}_{𝐢_T}(χ_T) → ⋯ → P^{κ_j}_𝐢 → S^{κ_j}_𝐢." Here T ⊂ [j+1, n], and 𝐢_T moves the strands of T left of the second red strand in reversed order, with χ_T = Σ_{k∈T}⟨α_{i_k}, −λ_2 + Σ_{j<m<k} α_{i_m}⟩. The proof is an iterated cone built from the exact sequences 0 → S_{𝐢″}^{κ_{j+1}}(…) → 𝔉_{i_n} S_{𝐢′}^{κ_j} → S_𝐢^{κ_j} → 0.
  - Directly after: "The same principle can be used for any value of ℓ to construct an explicit description of a projective resolution for any standard, but carefully writing this down is a bit more subtle and difficult than the ℓ = 2 case, so we will not do so here."
  - Cor. 7.7: Ext^i(standard, 𝔏) is nonzero only for i = 2ρ^∨(λ).
  - Prop. 7.8: 𝔏 has a resolution ⋯ → M_1 → M_0 → 𝔏 → 0 with M_{2ρ^∨−j} in the subcategory generated by the standards S^{κ_j}_𝐢, M_j = 0 for j > 2ρ^∨(λ), and M_{2ρ^∨(λ)} ≅ (standard)(−2⟨λ,ρ⟩) — [KIHRT](https://arxiv.org/abs/1309.3796).
- **Koszulity.** KIHRT §9 (around Prop. "sln-Koszul") states that for g = sl_n with fundamental weights the algebras T^λ_μ are Koszul (via Brundan–Kleshchev/Hu–Mathas and BGS uniqueness of Koszul gradings). A-II §5.4 argues heuristically that Koszulity would give each I_𝒞 a linear projective resolution graded by path length — [KIHRT](https://arxiv.org/abs/1309.3796); [A-II](https://arxiv.org/abs/2105.06039).
- **Conjectures stated.**
  - ADLSZ Conj. 1.8: the 𝒯_θ generate Fuk(𝓜^×(Γ, d), 𝒲). Remark 1.9: thimble generation does not apply because "our 𝒲 has no critical points whatsoever" — [ADLSZ](https://arxiv.org/abs/2406.04258).
  - CSQCB II Conj. 7.13 (annular invariants vs Hochschild homology vs Queffelec–Rose) — [CSQCB II](https://arxiv.org/abs/2211.02099).
  - LS footnote: the annular braid group "presumably recovers the annular Khovanov homology" — [LS](https://arxiv.org/abs/2505.00327).

### Inferences
- **Inference.** For minuscule λ the divided powers in Webster's 𝐢^λ_{𝐰_0} are all 0 or 1. The cup then has 2ρ^∨(λ) = dim G/P_λ single black strands, ordered by a reduced word of the minimal coset representative of w_0 in W/W_λ. This should match Aganagic's ordering of the y's in L_k "in the order we subtract simple roots from μ_i to get −μ_j", and his D_0 = dim G/P_i. No paper writes this identification down.
- **Inference.** Combining KIHRT Props. 7.6 and 7.8 gives a projective resolution of the cup simple 𝔏 as an iterated cone. The levels are indexed by the number j of black strands crossed over the second red strand (a filtration by "weights" of the right factor). Each standard at level j is resolved by a subset cube. This is the "iterated cones over the weight poset" shape. Webster proves existence and the shape of the filtration but writes no explicit differentials beyond sl_2.
- **Inference.** For general minuscule g, ALR's algorithm would start from a product grid in a product of Sym^{d_a}(𝒜), one factor per Dynkin node. Interactions between different nodes enter through the bigon-type relations (η, or u per node, in the ADLSZ and LS conventions). ALR do not discuss how many orders of correction to expect.

### Gaps
- I found no published explicit thimble resolution for an E_n, D_n spinor or sl_n Λ^k cup, and no closed formula for the corrections δ_k.
- I found no computer implementation or code release described in ALR, LePage 2023 or LS. The texts contain no mention of a program.
- Shende–Zhou (arXiv:2604.01208, 2026) mention "an ongoing joint project with Aganagic and LePage" that applies symmetric-power sectorial covers to these knot-homology settings. No results on cup resolutions are available yet — [Shende–Zhou](https://arxiv.org/abs/2604.01208).

## Q3. How are stable envelopes related to standard modules or to products of U-shaped Lagrangians? Is there an identification, and are projective resolutions of standards discussed?

### Takeaway
The explicit identification in this literature is between standard modules of the KLRW algebra and U-shaped Lagrangians, i.e. cones T_{i+1} → T_i, equivalently "step" I-branes. It is proved for sl_2 with d = 1 by LePage–Shende (Lemma 10.3) and given a Hamiltonian-flow proof by Tong (2025).

Stable envelopes appear in A-II and the ICM survey in only two roles:
1. The core X of 𝒳 is "the union of supports of all stable envelopes".
2. The integrand insertions Φ_σ are limits of elliptic stable envelopes.

I found no statement in ALR, ADLSZ, LS or CSQCB I/II that identifies stable-envelope objects with KLRW standard modules. Projective resolutions of standards are written down by Webster (KIHRT Prop. 7.6, two red strands), by LS (2-term cones υ_±), and by Guicardi for Verma-type KLRW (2026).

### Cited Findings
- A-II §3.1 ("The core"): "The singular holomorphic Lagrangian X is the union of supports of all stable envelopes [MO, ese]. Equivalently, X is the union of all attracting sets of Λ-torus actions on 𝒳, where we let Λ vary over all chambers" — [A-II](https://arxiv.org/abs/2105.06039). The ICM survey repeats this in §5.2 ("The core of the monopole moduli space") — [ICM](https://arxiv.org/abs/2207.14104).
- A-II §8.2: the insertions Φ_σ in the integral formula for conformal blocks "can also be obtained, as explained in [AFO], as limits of elliptic stable envelopes [ese] of the Nakajima quiver variety which is the Higgs branch". §7.2 has a similar statement — [A-II](https://arxiv.org/abs/2105.06039).
- **Tong (arXiv:2511.06431, Nov 2025), "Hamiltonian flow between standard module Lagrangians".** "In Aganagic's Fukaya category of the Coulomb branch … each standard module is known to admit two Lagrangian realizations: the 'U'-shaped T-brane and the step I-brane."
  - The U brane is "the symplectic connected sum (equivalently, the mapping cone) of the morphism T_{i+1} → T_i".
  - The paper shows the step I-brane is the infinite-time limit of a Hamiltonian evolution of U, "serving as a generalized thimble". This is done for (Γ, d) = (•, 1).
  - The algebraic isomorphism U ≅ J is attributed to LS Lemma 10.3 — [Tong](https://arxiv.org/abs/2511.06431).
- LS Lemma 10.3 ("The U Lagrangian is isomorphic to the Lagrangian J") is proved by explicit disk counts showing p·q = id and q·p = id. Prop. 10.4: the resolution of the simple S_θ is a cone υ_− → υ_+, and surgery at the corresponding intersection point gives I — [LS](https://arxiv.org/abs/2505.00327).
- Webster's standard modules (KIHRT §5) are quotients of projectives by "standardly violating" diagrams, i.e. diagrams where a black strand crosses a red strand in the forbidden direction. KIHRT Thm. B: T^λ is standardly stratified, and the stratification categorifies the decomposition of V_λ into tensor products of weight spaces. KIHRT Prop. 7.6 resolves standards for ℓ = 2 by subset cubes — [KIHRT](https://arxiv.org/abs/1309.3796).
- Webster's resolution of the cup simple by standards (KIHRT Prop. 7.8) has length 2ρ^∨(λ). For sl_2 its length-1 case is 0 → S_− → S_+ → 𝔏, which matches LS's "cone over υ_− → υ_+" — [KIHRT](https://arxiv.org/abs/1309.3796); [LS](https://arxiv.org/abs/2505.00327).
- Guicardi (arXiv:2609.10941, 2026), "On Diagrammatic Categorification of Verma Modules I: Braiding", works with dg-enhanced KLRW algebras for tensor products of sl_2 Verma modules.
  - It has a section on standard modules (§3) and explicit projective resolutions of braiding bimodules. Cup and cap bimodules for finite-dimensional strands are defined following Naisse–Vaz and Webster, with their projective resolutions (§5.1).
  - §7.1 proposes cups and caps for Verma-coloured strands — [Guicardi](https://arxiv.org/abs/2609.10941).
- Danilenko (arXiv:2210.09967, Selecta Math. 2024) studies cohomological Maulik–Okounkov stable envelopes for slices of the affine Grassmannian, the B-side spaces 𝒳 here. He gives a recursive formula for G = PSL_2 and the first-order correction in general (per search abstract; not read in full) — [Danilenko](https://arxiv.org/abs/2210.09967). A-I cites [Danilenko] for the theorem that the KZ equation is the quantum differential equation of these slices (A-I §3.2) — [A-I](https://arxiv.org/abs/2004.14518).

### Inferences
- **Inference.** On the B-side the attracting sets (supports of stable envelopes) for a chamber of the Λ-torus are the natural candidates for the objects that become standard modules under 𝒟_𝒳 ≅ cylindrical KLRW-mod. On the A-side these would be U-shaped/step branes. Tong's U ↔ J ↔ standard and A-II's "core = union of supports of stable envelopes" are consistent with this. Nobody states the identification "K-class of standard = stable envelope" as a theorem in these papers.
- **Inference.** For a product of U-shaped Lagrangians, ALR's grid algorithm would give the projective resolution as a product of 2-term complexes (a cube), corrected in u and ħ. This matches the subset-cube shape of KIHRT Prop. 7.6.

### Gaps
- I found no primary statement identifying the elliptic or K-theoretic stable envelope with a specific KLRW standard module, or with an exceptional collection, in this literature. A search for a Bezrukavnikov–Okounkov-type statement returned nothing verifiable. The only hint is in CSQCB II, in source text that Webster excised (marked `\excise`, so not in the compiled paper), which mentions "Bezrukavnikov and Okounkov's conjecture relating wall-crossing functors to quantum cohomology".
- Projective resolutions of standards for ℓ ≥ 3 red strands are not written down (KIHRT, after Prop. 7.6).

## Q4. Does Webster's CSQCB II (or the tensor product algebra papers) give explicit cup/cap complexes for general minuscule weights? Which deformations appear?

### Takeaway
CSQCB II §7.2 defines cups via an exact Morita-equivalence bimodule B_∪, so the cup needs no resolution. The cap is the derived tensor with B_∩. That functor is well defined, i.e. B_∩ has a finite resolution, precisely when the colour is minuscule (Lemma 7.6). The finite resolution is imported from the two-red-strand planar case (KIHRT §7–8) by cylindricalization, not written explicitly.

The deformations in play are:
- flavours β on edges (Webster's "cylindrical flavored KLRW", comparable to weighted KLR);
- a quantization parameter h, whose classical limit h = 0 is the cylindrical KLRW algebra;
- on the Fukaya side, the parameters η/u and ħ (ADLSZ, ALR, LS), which can be rescaled to 1 but are useful as gradings.

### Cited Findings
- **Cups and caps in CSQCB II §7.2.**
  - The wall where two adjacent red strands labelled j, j^* collide gives a filtration with deepest piece 𝒞_z, where z = ρ^∨(ϖ_j + ϖ_{j*}) = 2ρ^∨(ϖ_j).
  - Lemma 7.4: R^{(z)} := R^𝐣/R^𝐣 e_z R^𝐣 is Morita equivalent to R^{𝐣′} with v′ = v − z. Informally the two red strands collide and "pinch off" z_i black strands of label i, where ϖ_j + ϖ_{j*} = Σ z_i α_i.
  - Example 7.5 (type A_n): z_i = i for i ≤ j, z_i = j for j ≤ i ≤ j^*, and z_i = n+1−i for i ≥ j^*. For sl_2, one black strand is pinched off.
  - The bimodules B_∪ and B_∩ are "obtained from the quantum trace and coevaluation bimodules defined in [Webmerged, §7.3] by the cylindricalization operation".
  - ∪_ℓ = B_∪ ⊗ − is exact. The left adjoint ∪_ℓ^* (derived tensor with B_∩) has "no guarantee that this functor will preserve being a bounded complex" — [CSQCB II](https://arxiv.org/abs/2211.02099).
- **Lemma 7.6.** ∪_ℓ^* is well defined on D^b(fd-mod) if j is minuscule. Proof: for 𝐣 = (j, j^*), T̃^𝐣 has finite global dimension (proof of KIHRT Prop. 8.3), so the simple L_0 has a finite projective resolution. Adding strands on both sides and cylindricalizing (an exact operation) resolves R^𝐣/R^𝐣 e_z R^𝐣.
  - Remark 7.7: "This result holds in effectively no non-minuscule cases. As in [Webmerged], we could consider a category of complexes which are bounded above or below … for simplicity, we don't work through the details."
  - Def. 7.8: ∩_ℓ = ∪_ℓ^*[−m](−m). This is also ∪_ℓ^![m](m) by KIHRT Thm. 8.11. Cups and caps at other positions are obtained by conjugating with twists 𝔹_σ — [CSQCB II](https://arxiv.org/abs/2211.02099).
- **Theorems and conjecture.** Thm. 7.11: an annular tangle invariant Φ(T) on D^b(R^𝐣 fd-mod) with Φ(∪_k) = ∪_k, labels in minuscule fundamentals. Thm. 7.12: it agrees with KIHRT and with Khovanov/KR. Conj. 7.13: annular comparisons. The introduction states that this "agrees with [Aganagic, Th. 5*]" and that "there is no need for the asterisk" — [CSQCB II](https://arxiv.org/abs/2211.02099).
- **Which minuscule weights.** CSQCB II §7.2 lists them: "all fundamental weights in type A, the vector and both spin representations in type D, two representations for E_6 and one for E_7". BFN resolutions exist "for type D and E quivers with w_i only non-zero on nodes with minuscule fundamental representations" (§5) — [CSQCB II](https://arxiv.org/abs/2211.02099).
- **Flavours and weighted KLR (CSQCB II §5–6).** The algebra is a "cylindrical flavored KLRW algebra … subject to local relations like those of the weighted KLR algebra [WebwKLR]". A flavoring is β_e ∈ 𝔸 on each edge of Γ^𝐰. §6.2 ("Change of flavor") includes the comparison 𝔸 = ℝ/ℤ vs 𝔸′ = 𝔽_p. §6.3 compares with unflavored cylindrical KLRW, "where each corporeal strand coincides with all of its ghosts". The relation to weighted KLR via a Morita bimodule is cited to Kamnitzer–Webster–Weekes–Yacobi, §7.4 — [CSQCB II](https://arxiv.org/abs/2211.02099).
- **Quantization (CSQCB II §8.2–8.3).** Quantum cylindrical KLRW algebras carry a parameter h. Lemma 8.x (label `lem:h=0`): setting h = 0 recovers the cylindrical KLRW algebra of Def. "cKLRW". CSQCB I constructs Bezrukavnikov–Kaledin-type tilting generators on BFN-resolved Coulomb branches, with "wall-crossing functors described by natural bimodules" (abstract) — [CSQCB II](https://arxiv.org/abs/2211.02099); [CSQCB I](https://arxiv.org/abs/1905.04623).
- **ADLSZ parameters.** The KLRW category is defined over ℂ[ħ, η]. In ADLSZ Fig. 1:
  - the red–black bigon equals η·(dot);
  - the bigon of linked black strands equals η(dot) − η(dot);
  - the dot-pass-crossing relations equal ħ;
  - the braid relations with neighbour or red strand equal ηħ.
  - §4 uses the degenerate cases ħ = η = 0 (basis of "weighted taut strand diagrams") and ħ = 0. Relations are derived from disk counts at the matter divisor (η; §7.2) and the root divisor (ħ; §8.3). The main theorem is proved in §8.4.
  - Main Thm. 1.7 embeds the cylindrical KLRW category into Fuk(𝓜^×(Γ, d), 𝒲) when 𝓜^× is smooth ("known when Γ is of ADE type").
  - The ℤ-grading comes from deleting {r_{1,…,1} = 0}: same-label crossings have degree −1 and dots +1 (§1.3) — [ADLSZ](https://arxiv.org/abs/2406.04258).
- **ALR and LS conventions.** In ALR the u-deformation fills the puncture divisor D_a and corrects relation #5. The ħ-deformation fills the diagonal and corrects #4, #6, #7. "Both parameters u and ħ may be set to 1 by rescaling the algebra generators; if we keep them generic, they are useful as book-keeping parameters" (§5.4.10) — [ALR](https://arxiv.org/abs/2305.13480).
  - LS §2.3: u has rescaling weight (2,0,−1) and ħ has (0,1,1) with respect to (red–black crossing, black–black crossing, dot). The J-grading (1,−2,2) leaves both invariant "and eventually gives rise to the q's". "In general, one can introduce independent u, ħ parameters and gradings for each node of Γ."
  - LS Remark 2.5: "The comparison between a Hom space and Khovanov homology happens after setting u = ħ = 1 … if one sets these parameters to zero, the resulting Hom space does not give a knot invariant" — [LS](https://arxiv.org/abs/2505.00327).
- LePage 2023 §4.1: "With u = 1, these relations are the same as those for the KLRW algebra". ħ = 0 corresponds to Y_0 = Y ∖ Δ, and u = ħ = 0 to Y_0 ∖ D_a — [LePage 2023](https://arxiv.org/abs/2307.07034).

### Inferences
- **Inference.** In CSQCB II the cup is exact, and the work of resolving is on the cap side. In ALR the projective resolution is of the cup E_𝒰, with the cap I_𝒰 a simple. In KIHRT the coevaluation is RHom(𝔏, −) with 𝔏 simple. These are three placements of the same "resolve a simple" problem. LS show I ≅ E for sl_2, so both A-side cups are resolutions of that simple.
- **Inference.** ADLSZ's η plays the role of ALR's u. This rests on: the red–black bigon equals η·dot in ADLSZ and u·dot in ALR's relation #5 (filling the puncture/matter divisor), and in both papers ħ goes with the root/diagonal divisor. The dictionary is implicit, not stated.

### Gaps
- I did not verify the word "redotted" in any of these papers. It does not occur in CSQCB II. The weighted/flavored terminology is what CSQCB II uses.
- No explicit matrices for B_∩ resolutions for E_6, E_7 or spin weights are given in CSQCB II. Existence only is shown, via finite global dimension.

## Q5. Which representations make the construction harder (non-fundamental weights, Λ^k with k ≥ 4, spinors of D_n, E_6, E_7)? Do dots appear in the differentials?

### Takeaway
The literature locates the difficulty at non-minuscule colours, not at particular minuscule ones.
- On the B-side, monopole bubbling makes 𝒳 singular.
- On the algebraic side, standards lose End = 𝕂 and global dimension can be infinite. An example is the sl_2 unknot coloured by the 3-dimensional representation, whose invariant is infinite-dimensional (KIHRT §8.2).
- Cap functors stop being bounded (CSQCB II Remark 7.7).

I found no statement that Λ^k for large k, the D_n spinors, or the E_6/E_7 minuscules are harder within the minuscule class, and no computations for them.

Dots do appear in the differentials: they are the u-corrections in ALR, with order u^ℓ meaning ℓ dots. They disappear in the downstairs complex and after pairing with the simple cap.

### Cited Findings
- A-I §1.1: "One limitation of our approach is that we restrict to knots colored by minuscule representations of ^Lg. Otherwise, 𝒳 develops singularities due to monopole bubbling." Appendix A.1: unless μ is minuscule, Gr^μ is not compact, and the source of non-compactness is monopole bubbling. A-I §3.1: "A minuscule representation is also a fundamental representation" — [A-I](https://arxiv.org/abs/2004.14518).
- KIHRT §8.2 ("The unknot for g = sl_2"): with label 2, 𝔏 has the standard resolution 0 → S^{(0,0)}_{1^{(2)}}(−2) → S^{(0,1)}_{1,1}/(y_1+y_2)(−1) → S^{(0,2)}_{1^{(2)}} → L_2 → 0.
  - The middle standard has End = 𝕂[y_1, y_2]/(y_1², y_2²). It produces an infinite periodic resolution ⋯ →^{y_1+y_2} A(−4) →^{y_1−y_2} A(−2) →^{y_1+y_2} A.
  - The unknot invariant comes out as φ(U) = q^{−2}t² + 1 + q²t^{−2} + (q^{−2} − q^{−2}t)/(1 − t²q^{−4}), which is not a Laurent polynomial — [KIHRT](https://arxiv.org/abs/1309.3796).
- CSQCB II Remark 7.7 (cap finiteness "holds in effectively no non-minuscule cases") and its list of minuscule fundamentals (all of type A, vector and both spinors in D, two for E_6, one for E_7) — [CSQCB II](https://arxiv.org/abs/2211.02099).
- ADLSZ arXiv source, commented-out text that is not in the compiled paper, after Thm. 1.7: "In general, the B-side space becomes dangerously terrible if there is a red strand colored by a non-miniscule highest weight μ. It is a union of spaces of different dimensions, labeled by weights ν … not all fundamental representations are minuscule … An example … Γ = D_4, and the fundamental weight dual to … the central node". This is unpublished authorial commentary — [ADLSZ source](https://arxiv.org/abs/2406.04258).
- A-II §8.2: the root-ordering of the interval brane L_k "depends crucially on the fact μ_i and μ_j are highest weights of minuscule representations". A-II §2.5: since V_i, V_j are minuscule, tensor-product multiplicities are 0 or 1 — [A-II](https://arxiv.org/abs/2105.06039).
- **Dots in differentials.**
  - ALR §5.5.3: "The terms in δ_{ℓ,0} are corrections involving ℓ dots".
  - ALR §5.6.2: the one-strand braided cup needs a single dotted term with coefficient −u.
  - ALR §5.6.5: Hopf-link "dotted corrections". The ħ-corrections could be taken without dots in the examples.
  - ALR §5.5.7: dots drop out after Hom with I_𝒰 — [ALR](https://arxiv.org/abs/2305.13480).
  - LePage 2023: the Hopf-link brane resolution needs "dotted correction terms" with x_1 = −u and x_2 = u — [LePage 2023](https://arxiv.org/abs/2307.07034).
  - LS Lemma 2.2 uses the u-grading to exclude dots from degree-matched morphisms. A single red–black crossing is the unique morphism in its degree "Adding dots will increase the u-degree" — [LS](https://arxiv.org/abs/2505.00327).
  - Webster's sl_2 standard resolution for the label-2 unknot involves the dot combinations y_1 ± y_2 (above) — [KIHRT](https://arxiv.org/abs/1309.3796).

### Inferences
- **Inference.** Within minuscule colours, the expected growth in difficulty is combinatorial. The number of black strands in a cup is 2ρ^∨(λ) = dim G/P_λ: for example 16 for the E_6 minuscule, 27 for E_7, and k(n−k) for Λ^k in sl_n. The number of T-brane summands and possible correction terms in ALR's grid grows accordingly. More Dynkin nodes also mean more bigon-type interactions (η per edge). The papers do not discuss this.
- **Inference.** The u/dot corrections are needed upstairs (on 𝒴, i.e. for the cylindrical KLRW with ℬ_∞-local systems), but they are killed downstairs and in the final Hom with the simple cap. For computing link homology itself, only the dot-free part of δ matters, even though dotted terms can be needed for δ² = 0 upstairs.

### Gaps
- I found no source discussing Λ^k with k ≥ 4, or D_n spinors and E_6/E_7 minuscules, as specifically harder cases. There are no computations for them on either side.
- I found no source on whether ħ-corrections beyond first order are ever needed in the su_2 theory. gl_{1|1} 8_19 needs second order (ALR §4.5.6); the su_2 examples reported need only first order.
