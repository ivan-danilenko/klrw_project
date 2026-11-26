# Explicit projective resolutions of standard-type modules over KLR, cyclotomic KLR, Webster tensor product (KLRW) and weighted KLR algebras, and their deformations

Method note: I read the full arXiv texts (PDF to text) of every paper cited below with an arXiv number, except where a line is marked "(abstract/search only)". Theorem numbers refer to the arXiv version named. Published numbering may differ, especially for Webster's Memoir. "Proves" means the statement is in the source. "Inference" means it is my reading of relevance to the user's deformed-KLRW cube resolution (parameters r_i in x psi - psi x = r_i, and t_ij in Q_ij = t_ij u + t_ji v).

## Q1. What do BKM, Kato, McNamara, Kleshchev–Loubert–Miemietz, Kleshchev (surveys, cuspidal/standard theory) and their successors prove about projective resolutions of (proper) standard modules? Are the differentials explicit?

### Takeaway
In finite type the homological theory (affine quasi-heredity, Ext-orthogonality, finite global dimension equal to the height) is proved abstractly by Kato (ADE, char 0), McNamara and Brundan–Kleshchev–McNamara (BKM, all finite types, any field). Closed-form differentials exist only in a few cases:
- BKM Thm 4.12: root modules of multiplicity-free roots. This is a Koszul-like cube on {0,1}^{n-1} whose entries are single signed permutation elements (no dots, no correction terms).
- Kleshchev's survey §6.7: all simples of type-A cuspidal blocks. This is a cube over subsets T ⊆ [1,d], with crossings plus one central sum-of-dots edge.
- Buursma–Kleshchev–Steinberg (BKS) Thm A: Δ(α^m) in type A. This is a grid {0..m}^{b-a+1} with divided-power idempotents.

BKM say explicitly that they could not describe the differential entries outside the multiplicity-free case.

### Cited Findings
- **BKM, "Homological properties of finite type Khovanov–Lauda–Rouquier algebras", arXiv:1210.6900 (Duke Math. J. 163 (2014) 1353–1404).** Any finite type, any field. The abstract states they "construct some Koszul-like projective resolutions of standard modules corresponding to multiplicity-free positive roots" — [arXiv:1210.6900](https://arxiv.org/abs/1210.6900)
  - Thm 3.12 (all finite types, any characteristic): Ext^d(Δ(λ), V) = 0 for d ≥ 1 when V has composition factors L(μ) with μ ⊁ λ. Also Dim Ext^d(Δ(λ), ∇̄(μ)) = δ_{d0} δ_{λμ}. This extends Kato's Thm 4.12 — [arXiv:1210.6900](https://arxiv.org/abs/1210.6900)
  - Lemmas 3.7–3.9: the root module Δ(α) has an endomorphism x with τ x_2 = x_1 τ + 1, and NH_m ≅ End(Δ(α)^{∘m})^op. Divided powers Δ(α^m) are cut out by a primitive nilHecke idempotent, and Δ(λ) ≅ Δ(γ_1^{m_1})∘…∘Δ(γ_s^{m_s}) — [arXiv:1210.6900](https://arxiv.org/abs/1210.6900)
  - Thm 4.10: for a minimal pair (β, γ) of α there is a short exact sequence 0 → q^{−β·γ}Δ(β)∘Δ(γ) →φ Δ(γ)∘Δ(β) → [p_{β,γ}+1]Δ(α) → 0. The map φ is explicit (Lemma 4.9): v1⊗v2 ↦ τ_w 1_{γ,β} ⊗ (v2⊗v1), with w the block-swap permutation — [arXiv:1210.6900](https://arxiv.org/abs/1210.6900)
  - Cor 4.11: pd Δ(λ) ≤ n − l (n = ht α, l = number of parts) — [arXiv:1210.6900](https://arxiv.org/abs/1210.6900)
  - §4.5: recursive mapping-cone construction P_d(α) = P_d(γ,β) ⊕ q^{−β·γ}P_{d−1}(β,γ). Explicitly, P_d(α) = ⊕_{σ∈{0,1}^{n−1}, |σ|=d} q^{d_{α,σ}} H_α 1_{i_{α,σ}} (eq. 4.17), a cube. The differential is right multiplication by a matrix (τ_{σ,ρ}) with τ_{σ,ρ} = 0 unless σ, ρ differ in exactly one entry. BKM write: "Unfortunately we have not been able to find a satisfactory description of such elements τσ,ρ, except in the following special case." — [arXiv:1210.6900](https://arxiv.org/abs/1210.6900)
  - Thm 4.12: for α multiplicity-free (all roots in type A), τ_{σ,ρ} = (−1)^{σ1+…+σr} 1 τ_w when σ, ρ differ in the r-th entry, with w the unique permutation such that 1_{i_{α,σ}} τ_w = τ_w 1_{i_{α,ρ}}. The proof "relies on the fact that the braid relations hold exactly in H_α under the assumption that α is multiplicity-free." — [arXiv:1210.6900](https://arxiv.org/abs/1210.6900)
  - Worked example, highest root of A3: 0 → q²H 1_{1321} → qH 1_{1213} ⊕ qH 1_{1312} → H 1_{1123} → Δ(α) → 0, with matrices (−τ1τ2, τ2) and (τ1; τ1τ2) — [arXiv:1210.6900](https://arxiv.org/abs/1210.6900)
  - Conventions: BKM use sign choices ε_{i,j} = −ε_{j,i} in τ_k² (a special choice of Q_ij) and dot-crossing coefficient exactly 1 (§2.2). There is no r_i parameter and no general t_ij — [arXiv:1210.6900](https://arxiv.org/abs/1210.6900)
- **Kato, "PBW bases and KLR algebras", arXiv:1203.5254 (Duke Math. J. 163 (2014) 619–663).** Types ADE, geometric (char 0). Thm A (= Cor 2.9): R_β has finite global dimension. Thm B (= Thm 4.11 and corollaries): Ext-orthogonality between standard modules E_b and duals of proper standard modules Ē_b, with ext^•(E_b, E_b') = 0 for b <_i b'. The standard modules are defined as quotients of P_b and are not given explicit resolutions. The text gives no explicit resolution with differentials. — [arXiv:1203.5254](https://arxiv.org/abs/1203.5254)
- **McNamara, "Finite dimensional representations of KLR algebras I: finite type", arXiv:1207.5860.** Classifies simples via cuspidals and standards for any convex order. Thm 4.7: "the KLR algebra R(ν) has global dimension |ν|", proved by Ext-vanishing and induction, with no explicit resolution. Prop 4.6 shows Ext^i(S_α, S_α) = 0 for i ≥ 2 for cuspidals. The author calls the construction of cuspidals "non-constructive". — [arXiv:1207.5860](https://arxiv.org/abs/1207.5860)
- **Kleshchev–Loubert–Miemietz, "Affine cellularity of KLR algebras in type A", arXiv:1210.6542, and Kleshchev–Loubert, arXiv:1310.4467 (IMRN 2015, all finite types).**
  - R_α is graded affine cellular, with affine cell ideals generated by idempotents. This gives finite global dimension through Koenig–Xi Thm 4.4 and "a theory of standard and reduced standard modules" — [arXiv:1210.6542](https://arxiv.org/abs/1210.6542); [arXiv:1310.4467](https://arxiv.org/abs/1310.4467)
  - Neither text contains the word "resolution", so neither has explicit resolutions — [arXiv:1210.6542](https://arxiv.org/abs/1210.6542)
- **Kleshchev survey, "Representation theory and cohomology of KLR algebras", arXiv:1401.6151.**
  - §5.2, Thm 5.13 restates BKM Thm 4.12 — [arXiv:1401.6151](https://arxiv.org/abs/1401.6151)
  - §6 is an independent elementary treatment of type A over an arbitrary commutative ring O — [arXiv:1401.6151](https://arxiv.org/abs/1401.6151)
    - Prop 6.3: R_α ≅ R′_α ⊗ O[X], where X ↦ z_α is a central element equal to a sum of dots y_r over the positions carrying the first label.
    - Thms 6.24–6.25: a type-A cuspidal block R′_ρ is Morita equivalent to B_n = B^{⊗n}, where B is the path algebra of the quiver with two vertices and arrows both ways, with b of degree 1.
    - (6.8): B_n has a linear minimal "cube" resolution, with terms indexed by subsets {r1<…<rm} ⊆ [1,n] and differential Σ_k (∏_{l<k} σ_{r_l}) b_{r_k}.
  - §6.7, (6.30)–(6.32): explicit resolutions of every irreducible L_λ of R_ρ (λ a skew shape with one box of each content). The terms are P_T = q^{s(T)} E(i_{sp_T λ}) for T ⊆ [1,d]. The edge maps are ∂_{T,t} = σ(T,t)·ψ(sp_T λ, sp_{T∖t} λ) for t < d and σ(T,d)·z_ρ for t = d, where σ(T,t) = (−1)^{#{s∈T, s<t}} and sp_r are "row splitting" operators. The cuspidal module L_ρ is the case λ = (d) — [arXiv:1401.6151](https://arxiv.org/abs/1401.6151)
  - Prop 6.14(i): products ψ(λ, sp_r λ) ψ(sp_r λ, λ) equal a single dot or minus a sum of dots, i.e. bigons become dots — [arXiv:1401.6151](https://arxiv.org/abs/1401.6151)
  - §6.8, (6.33): induction products give resolutions of proper standard and costandard modules — [arXiv:1401.6151](https://arxiv.org/abs/1401.6151)
  - Thm 6.35 (pd L ≤ d), Cor 6.36 (gd R_α = d) and Cor 6.38 (Ext(Δ̄(π), ∇̄(σ)) = 0 for π ≠ σ) follow from these resolutions — [arXiv:1401.6151](https://arxiv.org/abs/1401.6151)
- **Buursma–Kleshchev–Steinberg, "Resolutions of standard modules over KLR algebras of type A", arXiv:1906.11376.** Over any commutative ring.
  - Thm A gives an explicit projective resolution of the semicuspidal standard module Δ(α^m), α = α_a+…+α_{b+1}. Terms P_λ = q^{s_λ} R_θ e_λ are indexed by compositions λ ∈ {0,…,m}^{b−a+1} (a grid, not a cube). The e_λ are concatenations of divided-power idempotents 1_{i^{(m)}} = ψ_{w0} y_{0,m}. The differential is d_n^{μ,λ} = sgn_{λ;i} e_μ ψ_{λ;i} e_λ when μ = λ + e_i, and 0 otherwise — [arXiv:1906.11376](https://arxiv.org/abs/1906.11376)
  - All standard modules in type A follow by induction product, since Δ(π) ≅ ∘_s Δ(β_s^{m_s}) — [arXiv:1906.11376](https://arxiv.org/abs/1906.11376)
  - It is described as the "thick calculus generalization" of BKM Thm 4.12. The proof realises it as a direct summand, via chain maps f, g with g∘f = id, of the m-th induced power of the BKM resolution — [arXiv:1906.11376](https://arxiv.org/abs/1906.11376)
- **Buursma–Kleshchev–Steinberg, "Some extension algebras for standard modules over KLR algebras of type A", arXiv:1906.11380.**
  - Thm A: for θ a positive root, Ext^m(Δ(ρ), Δ(σ)) ≅ q^{−m} X_ρ if D_ρ ⊆_m D_σ and 0 otherwise. Here D_ρ ⊆ [1,l−1] and X_ρ is a quotient polynomial ring, so the Ext category is a Boolean lattice of subsets.
  - Thm B: the case of type A2, computed with symmetric polynomials and Demazure operators ∂_w.
  - Example C: for θ = α1+2α2+α3 the A∞-category Ext*(Δ,Δ) is non-formal — [arXiv:1906.11380](https://arxiv.org/abs/1906.11380)
- **Kleshchev–Steinberg, "Homomorphisms between standard modules over finite type KLR algebras", arXiv:1505.04222 (Compos. Math. 153 (2017) 621–646).** Concerns Hom spaces, not resolutions (abstract/search and BKS bibliography only) — [arXiv:1505.04222](https://arxiv.org/abs/1505.04222)

### Inferences
- In all closed-form finite-type resolutions above, every matrix entry is a single permutation element ψ_w, possibly sandwiched by divided-power idempotents, or in the cuspidal-block case a central sum of dots. No sum of correction terms appears. This is because multiplicity-free words have no two equal colours, so nilHecke dot-slide corrections and i–j–i triple-point corrections never fire (BKM's own explanation). The user's r_c- and t_{cc′}-weighted correction terms live exactly in the regime where BKM say they "have not been able to find a satisfactory description of τσ,ρ", i.e. words with repeated colours.
- BKS avoid repeated-colour corrections by working with thick strands (divided-power idempotents) rather than writing them out. A formula with explicit r-terms for repeated colours would be new relative to BKM and BKS, as far as these sources show.
- BKM and Kleshchev use a fixed choice of signs. Transporting their resolutions to general t_ij uses the rescaling isomorphisms in Q7 (valid for invertible parameters).

### Gaps
- No source found gives closed-form differentials for root modules of non-multiplicity-free roots (types D, E, B, C, F, G), nor for imaginary or semicuspidal strata in affine type. A search for 2020–2023 follow-ups to BKS turned up nothing.
- Kleshchev's §6 type-A cuspidal-block resolutions appear in the survey. I could not determine whether they are also published in a research article.
- I did not read Kleshchev, "Affine highest weight categories and affine quasihereditary algebras" (Proc. LMS 110 (2015), cited in BKS) or Kleshchev–Muth on affine types for resolution statements.

## Q2. Webster's tensor product algebras T^λ and weighted KLR algebras: what is known about standard modules, standard stratification, projective resolutions, and explicit complexes for cups/caps?

### Takeaway
- T^λ is standardly stratified, and quasi-hereditary (highest weight) when all λ_i are minuscule. Every standard module and every simple has a projective resolution (abstract existence).
- The only explicit resolution in the Memoir is for two red strands (ℓ = 2). It is a cube over subsets T of black strands moved left of the second red strand, built by iterated cones. Webster declines to write out the differential for ℓ > 2.
- For sl_2 fundamental weights there are explicit 3-term minimal resolutions of the cup module L_1 (Webster, Grassmannians paper) and an A∞-structure on resolutions of cup bimodules (Mackaay–Webster).
- The weighted KLR paper supplies the canonical deformation of the Q-polynomials and a Demazure-operator polynomial representation, but no resolutions.

### Cited Findings
- **Webster, "Knot invariants and higher representation theory", arXiv:1309.3796 v3 (Mem. Amer. Math. Soc. 250 (2017), no. 1191, as cited in KLSY's bibliography).** Numbering below is from arXiv v3 — [arXiv:1309.3796](https://arxiv.org/abs/1309.3796)
  - Standard stratification and finiteness:
    - Cor 5.22: T^λ is standardly stratified, with standard modules the standard quotients of indecomposable projectives and preorder given by dominance on root functions. Cor 5.23 gives the same for T̃^λ.
    - Cor 5.24: "Every standard module has a finite length projective resolution." It is proved by induction on the preorder, gluing resolutions of kernels (Lemma 5.29). This is not explicit.
    - Thm 5.28: every simple has a projective resolution in C↑(T^λ). The remark after it says T^λ is Morita equivalent to a positively graded algebra in characteristic 0 with symmetric Cartan matrix and "polynomials Q_ij carefully chosen", and that outside these cases there may be no such Morita equivalence.
    - Intro (§1 outline) and proof of Prop 8.3: when all λ_i are minuscule, T^λ-mod is highest weight and of finite global dimension.
  - Cups and caps:
    - Def 7.4: the coevaluation functor is K^{λ,λ*}_∅ = RHom(L̇_λ, −)(2⟨λ,ρ⟩)[−2ρ^∨(λ)], and the evaluation is − ⊗^L L_{λ*}.
    - Prop 7.5: these categorify coevaluation and evaluation.
    - Prop 7.6 (ℓ = 2): the standard S^{κ_j}_i has a resolution ⊕_{|T|=k} P^{κ_T}_{i_T}(χ_T), with T ⊂ [j+1,n]. Here i_T lists i_1…i_j, then T in reversed order, then [j+1,n]∖T. The shift is χ_T = Σ_{k∈T} ⟨α_{i_k}, −λ_2 + Σ_{j<m<k} α_{i_m}⟩.
    - The resolution of Prop 7.6 is built from the exact sequence (7.25), 0 → S^{κ_{j+1}}_{i″}(shift) → F_{i_n} S^{κ_j}_{i′} → S^{κ_j}_i → 0 (from Prop 5.5), by lifting the left map to resolutions and taking the cone. Webster writes: "The same principle can be used for any value of ℓ … but carefully writing this down is a bit more subtle and difficult than the ℓ = 2 case, so we will not do so here."
    - Cor 7.7: Ext^i(M_λ, L_λ) ≠ 0 only for i = 2ρ^∨(λ).
    - Prop 7.8: L_λ has a resolution by standard-filtered modules M_j of length 2ρ^∨(λ).
  - Non-minuscule behaviour:
    - §8.2 (sl_2, label 2): L_2 has a 3-term standard resolution. A = k[y1,y2]/(y1², y2²) and A/(y1+y2) has an infinite periodic minimal resolution with maps alternating y1+y2 and y1−y2. This is why non-minuscule unknots give infinite-dimensional invariants.
    - §4: T^{2ω}_1 ≅ k[y]/(y²), and its trivial module has an infinite minimal resolution.
  - Prop 9.14: for g = sl_n and λ a list of fundamental weights, T^λ_μ is Koszul (via Hu–Mathas [HM15, Thm C] and the Morita equivalence Thm 5.34, or [Webc, Thm B]). Webster adds: "If λ is not a list of fundamental weights, then we expect that T^λ will never be Koszul."
- **Webster, "Tensor product algebras, Grassmannians and Khovanov homology", arXiv:1312.7357 (sl_2, fundamental weights).**
  - Cor 2.8: T^ℓ is cellular, and its cell modules S_λ are the standard modules of a quasi-hereditary structure. Thm 3.7: T^ℓ is Koszul dual to Stroppel's generalized arc algebras — [arXiv:1312.7357](https://arxiv.org/abs/1312.7357)
  - Prop 4.4: in T²_0 (5-dimensional), the minimal projective resolutions of the simples are P(0,1){−1} →ψ P(0,0) → L_0 and P(0,1){−2} →ψ P(0,0){−1} →φ P(0,1) → L_1. Here ψ, φ are the unique red–black crossing basis vectors and L_1 is the "cup" module — [arXiv:1312.7357](https://arxiv.org/abs/1312.7357)
  - RHom(L_1, P) = [e(0,1)P →φ e(0,0)P{1} →ψ e(0,1)P{2}]. Prop 4.5: cup and cap are biadjoint up to shift. Prop 4.6: Ext•(L_1, L_1) ≅ H*(S²) — [arXiv:1312.7357](https://arxiv.org/abs/1312.7357)
  - Def 4.7: the cup bimodule K_i is spanned by "+i-Stendhal diagrams" modulo relations (4.16a,b). Its projective resolution as a left module is given only "schematically" (3 terms). The right T^ℓ-action on that complex exists only in the A∞ sense, citing [MW §2.3] — [arXiv:1312.7357](https://arxiv.org/abs/1312.7357)
- **Mackaay–Webster, "Categorified skew Howe duality and comparison of knot homologies", arXiv:1502.06011 (Adv. Math. 330 (2018) 876–945).**
  - §2.3 (eq. 2.7): an algebra acting on a module acts on its projective resolution in the A∞ sense. The higher maps α_t are chosen as null-homotopies — [arXiv:1502.06011](https://arxiv.org/abs/1502.06011)
  - The deformation of T̃^p over k[z_1,…,z_ℓ] modifies relation (2.4) by z_k on the k-th red strand. It is a special case of the "canonical deformation" of reduced weighted KLR algebras, and the diagram basis persists — [arXiv:1502.06011](https://arxiv.org/abs/1502.06011)
  - Prop 2.22 (Gerstenhaber–Schack): a module deforms iff its class s_M ∈ Ext²(M,M) vanishes. Lemma 2.23: s_M equals the chain map ∂̃²/h obtained by lifting the differential of a projective resolution to the deformed algebra — [arXiv:1502.06011](https://arxiv.org/abs/1502.06011)
- **Webster, "Weighted Khovanov–Lauda–Rouquier algebras", arXiv:1209.2463 v4.** The text contains no projective resolutions.
  - Prop 2.7: polynomial representation in which same-label crossings act by Demazure operators ∂_{j,j+1} = (s_j − 1)/(y_{j+1} − y_j), and strand–ghost crossings multiply by Q_e(y_k, y_j) — [arXiv:1209.2463](https://arxiv.org/abs/1209.2463)
  - §2.6: steadied quotients — [arXiv:1209.2463](https://arxiv.org/abs/1209.2463)
  - §2.7: canonical deformation Q̃_e(u,v) = Q_e(u,v) + Σ z_{e,a,b} u^a v^b. Prop 2.23: free (hence flat) over S = k[z] — [arXiv:1209.2463](https://arxiv.org/abs/1209.2463)
  - For T^λ realised through the Crawley-Boevey quiver: ρ^∨(λ) parameters z_{i,j,k} with p_{i,j}(u) = ∏_k (u − z_{i,j,k}). Relation (3.4a) deforms the black–red–black triple point so that it equals (p_{i,j}(y_{r+1}) − p_{i,j}(y_r))/(y_{r+1} − y_r), a divided difference — [arXiv:1209.2463](https://arxiv.org/abs/1209.2463)
  - Remark 4.14: the deformation is equivariant Borel–Moore homology for G_m^{E(Γ)} — [arXiv:1209.2463](https://arxiv.org/abs/1209.2463)
- **Kang–Kashiwara, "Categorification of highest weight modules via KLR algebras", arXiv:1102.4677 (Invent. Math. 190 (2012) 699–742).** This is the key explicit two-term resolution behind cyclotomic and red-strand constructions.
  - Thm 4.7: exact sequence 0 → q^{(α_i|2Λ−β)} F_i M → F̄_i M → F^Λ_i M → 0 — [arXiv:1102.4677](https://arxiv.org/abs/1102.4677)
  - Lemma 4.17, (4.22): 0 → K_1 →P K_0 →π F^Λ → 0 is a projective resolution of F^Λ as a right R^Λ(β)[t_i]-module. P is right multiplication by a^Λ(x_1)τ_1⋯τ_n — [arXiv:1102.4677](https://arxiv.org/abs/1102.4677)
  - A near-inverse Q is built from the intertwiner-like elements g_a of (4.14): g_a = τ_a for unequal labels, and g_a = (x_{a+1}−x_a) − (x_{a+1}−x_a)²τ_a for equal labels — [arXiv:1102.4677](https://arxiv.org/abs/1102.4677)
  - Q∘P acts as a^Λ(t_i) ∏_{a: ν_a≠i} Q_{i,ν_a}(t_i, x_a), a monic product of bigon polynomials — [arXiv:1102.4677](https://arxiv.org/abs/1102.4677)

### Inferences
- Webster's Prop 7.6 is, as far as these sources show, the closest published analogue of a cube-shaped resolution of a standard module over a KLRW algebra (subsets T of black strands pulled across a red strand). Its differential is defined only up to the choices in lifting and coning, with no closed form.
- A closed formula for one red strand of minuscule weight in a deformed algebra would fill exactly the gap Webster leaves ("will not do so here"), at least for ℓ = 1 or 2 (my inference).
- Kang–Kashiwara's Q∘P = ∏ Q_{i,ν_a} is the prototype of "composite of strand-pulling maps = product of bigon polynomials". The user's spanning-tree sums of bigon polynomials look like a multi-strand, cyclic generalisation of this. That is an analogy only; it is not proved in any source.
- The deformation literature (Mackaay–Webster, Webster's weighted KLR §2.7, KLSY) deforms the Q-polynomials by lower-order or equivariant terms (red-strand variables z). It does not deform the dot-slide relation by r_i. The user's r_i-deformation is a different direction, discussed in Q7.

### Gaps
- No source found writes the differential of Webster's Prop 7.6 explicitly, or treats ℓ > 2.
- I did not verify whether the Memoir (published) numbering matches arXiv v3.
- I did not read Losev–Webster, "On uniqueness of tensor product categorifications" (arXiv:1303.4617) or Brundan–Losev–Webster (arXiv:1310.0349) for further statements on standard modules.

## Q3. Khovanov–Lauda–Sussan–Yonezawa (deformed Webster algebras) and Khovanov–Sussan (redotted Webster algebra): which deformation parameters, and which explicit resolutions or complexes?

### Takeaway
Both papers deform only the red strands: red strands carry dots (equivariant variables or elementary symmetric functions). Black strands obey the undeformed nilHecke relations, since only sl_2 is treated, so there is a single black colour and no t_ij. Their explicit complexes are Rickard/Rouquier-type braid complexes of bimodules, not projective resolutions of standard modules.

### Cited Findings
- **KLSY, arXiv:1802.05358 (Transformation Groups 27 (2022) 919–982).** The algebra W(s,n) has red strands labelled by symmetric powers s_i of the sl_2 vector representation, n black strands, and red dots E(d)_j identified with elementary symmetric functions e_d (zero for d > s_i) — [arXiv:1802.05358](https://arxiv.org/abs/1802.05358)
  - Black dot–crossing relations (4.17)–(4.18) have coefficient exactly δ_{i_j,i_{j+1}} (no r-parameter) — [arXiv:1802.05358](https://arxiv.org/abs/1802.05358)
  - Black–red bigon (4.19): ψ_j² = Σ_{d=0}^{s} (−1)^d E(d) x^{s−d}, a product ∏(x − red roots) written with elementary symmetric functions — [arXiv:1802.05358](https://arxiv.org/abs/1802.05358)
  - Triple point with a red middle strand (4.20): Σ_{d1+d2+d3=s−1} (−1)^{d3} x_j^{d1} E(d3) x_{j+2}^{d2}, i.e. a divided difference of the bigon polynomial — [arXiv:1802.05358](https://arxiv.org/abs/1802.05358)
  - Remark 4.1 gives the cyclotomic quotient. §4.4: Webster's T ≅ W(s,n)/I(s,n), where I is the ideal generated by the red dots — [arXiv:1802.05358](https://arxiv.org/abs/1802.05358)
  - Main theorems: Thm 9.1 gives a 2-functor U(gl_m) → Bim. Thm 9.3: Rickard complexes give a braid group action on the homotopy category, which is strong for s = (1^m) via Elias–Krasner — [arXiv:1802.05358](https://arxiv.org/abs/1802.05358)
  - The intro says that "closely related deformations were studied for fundamental representations by Mackaay and Webster" — [arXiv:1802.05358](https://arxiv.org/abs/1802.05358)
- **Khovanov–Sussan, "The Soergel category and the redotted Webster algebra", arXiv:1605.02678 (J. Pure Appl. Algebra 222 (2018) 1957–2000).** W(n,k) has n red (fundamental) strands, k black strands and red dots y. Works over an infinite field of characteristic ≠ 2 — [arXiv:1605.02678](https://arxiv.org/abs/1605.02678)
  - Relations: (8)–(9) dot-slide with coefficient δ; (10) black–red bigon equal to ± black dot ± red dot; (11) triple point with red middle equal to −1; (12) cyclotomic condition — [arXiv:1605.02678](https://arxiv.org/abs/1605.02678)
  - §6 Theorem: W(n,1)-gpmod is equivalent to singular Soergel bimodules ∅R^{1̂} — [arXiv:1605.02678](https://arxiv.org/abs/1605.02678)
  - Thm 2: W(n,1)/m ≅ A^!_n — [arXiv:1605.02678](https://arxiv.org/abs/1605.02678)
  - Prop 13: a length-2 bimodule projective resolution of A^!_n, used to compute Hochschild cohomology (Props 12, 14, 15) — [arXiv:1605.02678](https://arxiv.org/abs/1605.02678)
  - §8: braid group action through complexes of bimodules — [arXiv:1605.02678](https://arxiv.org/abs/1605.02678)
- **Yonezawa, "KLR subalgebras and redotted Webster algebras", arXiv:2203.15964.** Defines KLR subalgebras generalising the redotted algebras of KLSY. Their quotients are isomorphic to Webster's tensor product algebras in general type (abstract) — [arXiv:2203.15964](https://arxiv.org/abs/2203.15964)

### Inferences
- These deformations are the equivariant direction (red-strand variables). They are orthogonal to the user's (r_i, t_ij), which deform black–black relations.
- None of these papers gives projective resolutions of standard or cup modules, so they are relevant mainly as precedent for working in a deformed KLRW algebra and for homotopy-category (finite complex) technology.

### Gaps
- Not checked: whether Yonezawa's general-type KLR subalgebras carry black–black deformation parameters. I read only the abstract and intro.

## Q4. Naisse–Vaz and successors: do the dg-enhancements give Koszul-type resolutions of cyclotomic quotients with explicit differentials? Are there deformation parameters?

### Takeaway
Yes, in the sense of formal dg-models. The b-KLR algebra with odd "floating dots" and the closed-form differential d_N is a formal dg-algebra whose homology is the cyclotomic quotient. It is a Koszul-type model of the cyclotomic ideal. Its parameters are exactly the Cautis–Lauda scalars t_ij, s^{tv}_ij and r_i, with r_i required to be a unit. The same construction for KLRW (nail generator) appears in Lacabanne–Naisse–Vaz, which also contains an explicit 3-term cofibrant replacement of the cup/cap bimodule.

### Cited Findings
- **Naisse–Vaz, "2-Verma modules", arXiv:1710.06293 (v3, 2021).**
  - Def 3.1 (KLR with Cautis–Lauda scalars): t_ij ∈ k^×, s^{tv}_ij ∈ k, r_i ∈ k^×, with t_ii = 1 and t_ij = t_ji when d_ij = 0. The ii dot-slide relation carries "+ r_i". The triple-point relation carries r_i Σ s^{tv}_ij Σ_{u+ℓ=t−1} (…) — [arXiv:1710.06293](https://arxiv.org/abs/1710.06293)
  - Def 3.3 (b-KLR algebra): regions carry floating dots, which are odd and anticommute. Relations (9)–(11) express how floating dots pass strands, using s^{tv}_ij — [arXiv:1710.06293](https://arxiv.org/abs/1710.06293)
  - Def 4.2: d_N is extended by the Leibniz rule. On a floating dot it is given in closed form by sums of complete homogeneous polynomials h in the dots times elementary symmetric ε_r(x_K) — [arXiv:1710.06293](https://arxiv.org/abs/1710.06293)
  - Thm 4.4: "(R_b(m), d_N) is formal with homology H ≅ R^N_p(m)". For p = g this is the usual cyclotomic KLR algebra — [arXiv:1710.06293](https://arxiv.org/abs/1710.06293)
  - Prop 4.3: acyclicity when the weight is out of range. Cor 5.2 and Thm 5.17: the sl_2-commutator as a short exact sequence or cone. The authors say this reinterprets Kang–Kashiwara's proof — [arXiv:1710.06293](https://arxiv.org/abs/1710.06293)
  - Earlier sl_2 version: Naisse–Vaz arXiv:1603.01555 (A_k, d_n) is formal with homology the cyclotomic nilHecke algebra (as recalled in the 1710.06293 intro) — [arXiv:1710.06293](https://arxiv.org/abs/1710.06293)
- **Cautis–Lauda, "Implicit structure in 2-representations of quantum groups", arXiv:1111.1431.** Introduces the scalars t_ij, s^{pq}_ij and r_i ∈ k^× (§2). Remark (1): "It is always possible to rescale the ii-crossing by r_i so that r_i = 1". Remark (3): if the simply-laced graph is a tree, one can rescale so that t_ij = t_ji = 1 — [arXiv:1111.1431](https://arxiv.org/abs/1111.1431)
- **Lacabanne–Naisse–Vaz, "Tensor product categorifications, Verma modules and the blob 2-category", arXiv:2005.06257 (Quantum Topology 12 (2021) 705–812, per search).**
  - T^{λ,N}_b adds a blue λ-strand and a "nail" (a tight floating dot) of homological degree 1 to the affine KLRW algebra. d_N sends the nail to a diagram with N dots on the leftmost strand — [arXiv:2005.06257](https://arxiv.org/abs/2005.06257)
  - Thm 3.13: (T^{λ,N}_b, d_N) is formal with homology the KLRW algebra T^{(N,N)}_b (an extra red strand N on the left) — [arXiv:2005.06257](https://arxiv.org/abs/2005.06257)
  - §5.2 (Prop 5.1): explicit cofibrant replacement of the cap/cup bimodule B_i as a left module, a 3-term complex following Webster 1312.7357 §4.3 — [arXiv:2005.06257](https://arxiv.org/abs/2005.06257)
  - Exactness of that complex is proved from the basis theorem (Thm 3.11). A diagram in which a strand crosses a red strand on the left, also written as a different diagram with the same strand crossing a red strand on the right, can be rewritten as the straight strand with a dot — [arXiv:2005.06257](https://arxiv.org/abs/2005.06257)
- **Dupont–Naisse, arXiv:2103.14760.** dg-KLRW algebras for tensor products of Verma and integrable sl_2-modules. Bases are obtained by rewriting modulo braid-like isotopy (Thms 4.6, 5.6). Thm 6.5 is a quasi-isomorphism of cones — [arXiv:2103.14760](https://arxiv.org/abs/2103.14760)

### Inferences
- The Naisse–Vaz/LNV dg-algebras are the "Koszul resolution of the cyclotomic or red-strand relation" in algebra form: one odd generator per strand class, with differential equal to the cyclotomic polynomial. Applying them to a module gives Koszul-type complexes, and cube-shaped ones when several odd generators are present.
- The scalars r_i and t_ij there are units. The user's deformation, which presumably lets r_i, t_ij degenerate (formal parameters), is a genuinely different family. At unit values Cautis–Lauda Remark (1) normalises r_i = 1, and trees normalise t_ij.

### Gaps
- No source found writes these dg-models as explicit projective resolutions of specific standard or cup modules with tree-indexed terms.
- Published venue of 1710.06293 not verified.

## Q5. Bowman–Norton–Simental and Bowman–Hazi–Norton: explicit BGG resolutions over (cyclotomic) quiver Hecke algebras and their differentials

### Takeaway
These are BGG resolutions by standard (cell or Specht) modules, not by projectives. The differentials are explicit: alternating sums of one-dimensional "one-column homomorphisms" (or "simple reflection homomorphisms") with explicitly computed signs on commuting diamonds. They resolve unitary, or homogeneous, simples.

### Cited Findings
- **Bowman–Norton–Simental, arXiv:1803.08736 (v3, 2018; published as "On BGG resolutions of unitary modules for quiver Hecke and Cherednik algebras", Selecta Math. 2022, per search).** Characteristic-free.
  - Thm A and Thm B: complexes C•(λ) = ⊕_{ν} Δ(ν)⟨ℓ(ν)⟩, resp. ⊕ S_n(ν)⟨ℓ(ν)⟩, "with differential given by an alternating sum over all 'one-column homomorphisms'". They are exact except in degree 0, where H_0 = L(λ), resp. D_n(λ) — [arXiv:1803.08736](https://arxiv.org/abs/1803.08736)
  - Thm 4.3: Hom(Δ(α), Δ(β)) = k·φ^α_β is one-dimensional, with φ^α_β given explicitly on a tableau basis element — [arXiv:1803.08736](https://arxiv.org/abs/1803.08736)
  - Thm 4.16: diamond relations φ∘φ = ε·φ∘φ with explicit signs ε. Cor 4.10: compositions of two one-column homomorphisms vanish in the degenerate case — [arXiv:1803.08736](https://arxiv.org/abs/1803.08736)
  - Thm 5.2: exactness for cyclotomic quiver Hecke R_n(κ) with e > hℓ, κ h-admissible and λ in the fundamental alcove — [arXiv:1803.08736](https://arxiv.org/abs/1803.08736)
  - Thm 5.3: explicit action on the simple, which is homogeneous with 1-dimensional weight spaces — [arXiv:1803.08736](https://arxiv.org/abs/1803.08736)
  - Applications: Betti numbers and regularity of subspace arrangements — [arXiv:1803.08736](https://arxiv.org/abs/1803.08736)
- **Bowman–Norton–Simental, arXiv:2107.01190.** Classifies unitary (= calibrated, for the important parameters) modules of cyclotomic Hecke algebras and "cohomologically construct[s] these modules via BGG resolutions" (abstract) — [arXiv:2107.01190](https://arxiv.org/abs/2107.01190)
- **Bowman–Hazi–Norton, "The modular Weyl–Kac character formula", arXiv:2004.13082.** For anti-spherical Hecke categories, not KLR directly.
  - Thm A: the unique homogeneous simple L(1_{P\W}) is one-dimensional — [arXiv:2004.13082](https://arxiv.org/abs/2004.13082)
  - Thm B: BGG resolution C•(1_{P\W}) = ⊕_w Δ(w)⟨ℓ(w)⟩, with differential an alternating sum of "simple reflection homomorphisms", for char p > 2 — [arXiv:2004.13082](https://arxiv.org/abs/2004.13082)
  - The paper notes that in type A, diagrammatic Hecke categories relate to truncations of quiver Hecke algebras — [arXiv:2004.13082](https://arxiv.org/abs/2004.13082)
- **Brundan–Stroppel, "Highest weight categories arising from Khovanov's diagram algebra II: Koszulity", arXiv:0806.3472 (Transformation Groups 15 (2010)).** This is the arc-algebra side of sl_2 KLRW.
  - Thm 5.3: linear projective resolution of every cell module V(λ), with P_n(λ) = ⊕_μ p^{(n)}_{λ,μ} P(μ)⟨n⟩ (coefficients of Kazhdan–Lusztig polynomials). The maps are built inductively using projective (cap) functors G_t, with no closed form — [arXiv:0806.3472](https://arxiv.org/abs/0806.3472)
  - Thms 5.6 and 5.13: K_Λ is Koszul — [arXiv:0806.3472](https://arxiv.org/abs/0806.3472)
  - Thm 7.3: BGG-type resolutions of Kostant modules by cell modules — [arXiv:0806.3472](https://arxiv.org/abs/0806.3472)

### Inferences
- Signed sums over a cube or alcove poset with one-dimensional edge maps and explicit diamond signs are structurally like the user's cube. But the terms are standard modules, and no dots, bigon sums or tree sums appear in the differentials. These papers give no precedent for tree-indexed correction terms.

### Gaps
- Not checked: whether any BNS or BHN differential, rewritten in a projective resolution (by resolving each Δ(ν)), acquires correction terms.

## Q6. Explicit minimal resolutions for nilHecke, rank 2/3 KLR algebras, and Koszulity results (type A)

### Takeaway
The explicit small examples in the literature are:
- the nilHecke algebra, which is a matrix algebra over its centre, the symmetric polynomials, so resolutions reduce to Koszul complexes over Sym;
- the A3 highest-root example of BKM;
- the type-A2 Ext computation of BKS, which uses Demazure operators;
- Webster's sl_2 examples.

Koszulity is known, non-constructively, for cyclotomic quiver Schur algebras and for T^λ with fundamental weights in type A, and for T^ℓ of sl_2. Webster expects T^λ never to be Koszul for non-fundamental λ.

### Cited Findings
- The nilHecke algebra "is a matrix algebra over its center", see [Ro2 §2] and [KLM §4]. Kleshchev uses the idempotent e_m = y_2 y_3² ⋯ y_m^{m−1} ψ_{w0} — [arXiv:1401.6151](https://arxiv.org/abs/1401.6151)
- KLSY note that the center Z(NH_n) is the ring of symmetric polynomials — [arXiv:1802.05358](https://arxiv.org/abs/1802.05358)
- BKS use divided-power idempotents 1_{i^{(d)}} = ψ_{w0} y_{0,d}. Lemma 2.5: ψ_{w0} f ψ_{w0} = 0 for deg f < d(d−1)/2, and ψ_{w0} y_{0,d} ψ_{w0} = ψ_{w0} — [arXiv:1906.11376](https://arxiv.org/abs/1906.11376)
- Rank 2 and 3 explicit matrices:
  - A3 highest root: BKM display the 3-term resolution with explicit 1×2 and 2×1 matrices of crossings (see Q1) — [arXiv:1210.6900](https://arxiv.org/abs/1210.6900)
  - A2: BKS compute E_θ for θ = c1α1 + c2α2 (Thm B) using Λ_k, Schur bases and Demazure operators ∂_w — [arXiv:1906.11380](https://arxiv.org/abs/1906.11380)
  - A3: BKS show that θ = α1 + 2α2 + α3 has a non-formal A∞ Ext-category (Example C) — [arXiv:1906.11380](https://arxiv.org/abs/1906.11380)
  - BKS Example 3.5 gives the explicit 2-step resolution of Δ((α1+α2)²) — [arXiv:1906.11376](https://arxiv.org/abs/1906.11376)
- Koszulity results:
  - Maksimau, arXiv:1307.6013: Thm 1.1 (= Thm 2.67) says the graded basic cyclotomic quiver Schur algebra is isomorphic to the Koszul-graded endomorphism algebra of parabolic category O at negative level, so the category is Koszul. It relies on SVV [26, Thm 6.4] — [arXiv:1307.6013](https://arxiv.org/abs/1307.6013)
  - Hu–Mathas, arXiv:1110.1699: for e = 0 the cyclotomic quiver Schur algebras are Koszul over fields of characteristic zero (abstract). Thm C gives graded Schur functors compatible with parabolic category O — [arXiv:1110.1699](https://arxiv.org/abs/1110.1699)
  - Shan–Varagnolo–Vasserot, arXiv:1107.0146: parabolic/singular Koszul duality for affine category O, applied to q-Schur and cyclotomic rational DAHA — [arXiv:1107.0146](https://arxiv.org/abs/1107.0146)
  - Webster, arXiv:1306.0074: Thm C says T^s_μ (weighted KLR or diagrammatic Cherednik) is standard Koszul with Koszul dual of the same type under rank–level duality. §6 depends on the categorical dimension conjecture of RSVV/Losev — [arXiv:1306.0074](https://arxiv.org/abs/1306.0074)
  - Webster, Prop 9.14 of 1309.3796: sl_n with fundamental λ gives Koszul T^λ_μ, and other λ are expected never to be Koszul — [arXiv:1309.3796](https://arxiv.org/abs/1309.3796)
  - Webster, Thm 3.7 of 1312.7357: T^ℓ is Koszul dual to generalized arc algebras — [arXiv:1312.7357](https://arxiv.org/abs/1312.7357)
- Mackaay–Webster construct projective resolutions of simples via the Koszul resolution of [BGS96] (§3, text near line 2023 of the arXiv PDF) — [arXiv:1502.06011](https://arxiv.org/abs/1502.06011)

### Inferences
- The Koszulity results give linear minimal resolutions abstractly, through the Koszul dual. None writes the linear differentials in KLR diagrams, except Brundan–Stroppel's inductive construction on the arc-algebra side and the small explicit cases above.
- For the user's minuscule (fundamental-weight) type-A setting, Koszulity of the undeformed T^λ is known. This suggests the specialisation r = t = 0 or 1 of the user's cube could be compared with a linear Koszul resolution (inference).

### Gaps
- No paper found with explicit minimal projective resolutions of all simples of KLR algebras in rank 2 (A2, B2, G2) or A3 written as matrices of diagrams, beyond the examples listed.

## Q7. Are there known formulas where a differential or homotopy is a sum over trees or spanning trees, or where r·(smoothed crossing) corrections are organised combinatorially?

### Takeaway
I found no KLR, KLRW or weighted-KLR paper whose resolution differential or homotopy is a sum over trees or spanning trees, or whose correction terms are organised as r·(label-exchanging smoothings). The closest rigorous precedents for tree-indexed products of t_ij are the parameter-rescaling results:
- McNamara–Tingley: the torus action z_ij on KLR algebras; for trees all Q are equivalent; on a cycle the invariant is the product of t_ij over t_ji.
- Lauda: the v_ij = t_ji/t_ij form a 1-cocycle on the Dynkin graph; the explicit isomorphism for trees uses products along root-to-vertex paths.

Generic tree sums arise from the homological perturbation lemma (general background; see the adjacent notes file).

### Cited Findings
- **McNamara–Tingley, "Face functors for KLR algebras", arXiv:1512.04458.**
  - Proof of Lemma 2.14: (k^*)^{I×I} acts by ψ_j e_i ↦ z_{i_j i_{j+1}} ψ_j e_i and y_j e_i ↦ z_{i_j i_j}^{−1} y_j e_i, so Q′_ij(u,v) = z_ij z_ji Q_ij(z_ii u, z_jj v). "All possible isomorphisms of KLR algebras that fix the idempotents e_i arise in this way", and the equal-colour crossing must be scaled by z_ii^{−1} because of the dot-slide relation — [arXiv:1512.04458](https://arxiv.org/abs/1512.04458)
  - Lemma 2.14(1): for a tree, all choices of Q_ij are related. Lemma 2.14(2): for I = Z/nZ with Q_{i,i+1} = s_i u + t_i v, the algebra is of geometric type iff s_1⋯s_n / t_1⋯t_n = (−1)^n — [arXiv:1512.04458](https://arxiv.org/abs/1512.04458)
  - Def 2.15: "pseudo-KLR" algebras allow t_ij = 0. Thm 2.16: then R((1−a_ij)i + j) has at least 2 − a_ij irreducibles, more than in the KLR case, because Q_ij becomes divisible by v — [arXiv:1512.04458](https://arxiv.org/abs/1512.04458)
- **Lauda, "Parameters in categorified quantum groups", arXiv:1812.07654 (Algebr. Represent. Theory, DOI 10.1007/s10468-019-09890-8).**
  - §3.3: "The algebras R_Q are governed by the products v_ij = t_ij^{−1} t_ji … a k^×-valued 1-cocycle on the graph Γ". Cohomologous choices give isomorphic (non-canonically) algebras, and for trees all choices are cohomologous — [arXiv:1812.07654](https://arxiv.org/abs/1812.07654)
  - §3.3.1: the explicit isomorphism for trees fixes a root vertex, orients edges away from it and uses parameters D_i built as products over the edges of the unique path P_i from the root to i (eq. 3.16) — [arXiv:1812.07654](https://arxiv.org/abs/1812.07654)
  - §3.2: rescalings of an i-dot and of the ii-crossing must be inverse to each other — [arXiv:1812.07654](https://arxiv.org/abs/1812.07654)
- **Cautis–Lauda.** r_i can be rescaled to 1 when it is a unit, and t_ij can be set to 1 on trees (Remarks (1) and (3), §2) — [arXiv:1111.1431](https://arxiv.org/abs/1111.1431)
- **Webster.** T^λ is Morita equivalent to a positively graded algebra only for "carefully chosen" Q_ij (char 0, symmetric type), and may fail otherwise (remark after Thm 5.28) — [arXiv:1309.3796](https://arxiv.org/abs/1309.3796)
- **Perturbation lemma background.** Homotopy-transfer formulas are commonly written as sums over rooted trees (Kontsevich–Soibelman style), e.g. in Vallette's "Algebra+Homotopy=Operad" (search snippet only; not read in full) — [arXiv:1202.3245](https://arxiv.org/abs/1202.3245)
- **Mackaay–Webster.** The obstruction ∂̃²/h of Lemma 2.23 is the first-order term of exactly such a perturbation when a resolution is lifted to a deformed KLRW algebra — [arXiv:1502.06011](https://arxiv.org/abs/1502.06011)
- **Search result.** A targeted search ("spanning trees" with quiver Hecke, KLR or Webster algebra) returned no relevant sources — (web search, no hit)

### Structure checklist (what appears in each explicit construction)
| Construction | Cube / subsets | Trees | Q_ij bigons | Dots | Demazure ops | Equal-colour corrections |
|---|---|---|---|---|---|---|
| BKM Thm 4.12 (root modules, multiplicity-free) [1210.6900](https://arxiv.org/abs/1210.6900) | yes, {0,1}^{n−1} | no | no | no | no | none (braid relations exact) |
| BKM §4.5 general roots [1210.6900](https://arxiv.org/abs/1210.6900) | yes (cube terms) | no | unknown | unknown | unknown | entries "not found" |
| Kleshchev survey §6.7 (type A cuspidal blocks) [1401.6151](https://arxiv.org/abs/1401.6151) | yes, T ⊆ [1,d] | no | only implicitly (ψψ = dots) | yes, central z_ρ = Σ y_r | no | no |
| BKS Thm A (Δ(α^m), type A) [1906.11376](https://arxiv.org/abs/1906.11376) | grid {0..m}^{b−a+1} | no | no | only inside divided-power idempotents | no (absorbed in idempotents) | avoided via thick calculus |
| BKS Ext (A2) [1906.11380](https://arxiv.org/abs/1906.11380) | Boolean lattice for roots | no | no | yes | yes (∂_w) | n/a |
| Webster Prop 7.6 (ℓ = 2 standards) [1309.3796](https://arxiv.org/abs/1309.3796) | yes, T ⊂ [j+1,n] | no | not specified | not specified | not specified | differential not written |
| Webster Prop 4.4 (sl_2 cup) [1312.7357](https://arxiv.org/abs/1312.7357) | 3-term | no | no | no | no | no |
| Kang–Kashiwara (4.22) [1102.4677](https://arxiv.org/abs/1102.4677) | 2-term | no | yes (Q∘P = a^Λ ∏ Q_{i,ν_a}) | yes | implicit | g_a uses (x−x′) and (x−x′)²τ for equal labels |
| Naisse–Vaz d_N [1710.06293](https://arxiv.org/abs/1710.06293); LNV [2005.06257](https://arxiv.org/abs/2005.06257) | Koszul-type (odd generators) | no | yes (s^{tv}_ij, t_ij in floating-dot relations) | yes (h_n, ε_r) | no | r_i ∈ k^× in relations |
| BNS / BHN BGG [1803.08736](https://arxiv.org/abs/1803.08736), [2004.13082](https://arxiv.org/abs/2004.13082) | alcove/Bruhat poset | no | no | no | no | no (by standard modules) |
| Weighted KLR (3.4a) deformation [1209.2463](https://arxiv.org/abs/1209.2463) | n/a | no | deformed Q̃ | yes | yes (relation is a divided difference) | n/a |

### Inferences
- The products ∏ r_c t_{cc′} indexed by trees in the user's formula are consistent with the rescaling picture above. On any spanning tree of the "colour/strand graph" the t's are gauge (Lauda's cocycle is trivial on trees). Cycles carry the gauge-invariant holonomy ∏ t_ij/t_ji (McNamara–Tingley Lemma 2.14(2)).
- It is therefore natural (inference, unproved) that new terms appear exactly when the graph has a cycle. A sum over spanning trees of bigon polynomials is reminiscent of Kirchhoff/matrix-tree determinants of a weighted Laplacian with edge weights Q_{cc′}. No source found states such a formula for KLR algebras.
- r_i is gauge only while invertible (Cautis–Lauda, Lauda §3.2), and t_ij = 0 changes the representation theory (McNamara–Tingley Thm 2.16). So a resolution valid uniformly over Z[r, t], including the degenerate fibres, is not covered by any rescaling argument in the literature.
- The tree-sum structure is plausibly the homological-perturbation expansion around the degenerate algebra (r = t = 0), where the cube is a Koszul complex. The Mackaay–Webster Lemma 2.23 obstruction is the first-order term of such an expansion (inference).

### Gaps
- No source found for: differentials or homotopies as sums over trees or spanning trees in KLR/KLRW resolutions; correction terms of the form r·(smoothed crossing exchanging labels of equal-coloured strands); or matrix-tree-type formulas with bigon weights Q_ij = t_ij u + t_ji v.
- I did not read in full Rouquier, "Quiver Hecke algebras and 2-Lie algebras" (arXiv:1112.3619), which discusses dependence on Q. It may contain further invariants of the parameters.
- McNamara, "Representations of KLR algebras III" (arXiv:1407.7304) is the reference [M] for "geometric type" choices of Q. Not read.
