# Homogeneous KLR representations, minuscule heaps, and the (4,4,4,3) threshold

Scope: Kleshchev–Ram homogeneous representations and their follow-ups; the combinatorics of minuscule and fully commutative heaps; and any known property whose threshold matches the "link graph has a cycle" threshold (type A: λ ⊇ (4,4,4,3); minimal cases (A7, ω4), (D6, half-spin), (E6, 27)).

**Headline finding.** Billey–Warrington's "321-hexagon-avoiding" condition matches the threshold exactly in type A, and Billey–Jones's "Deodhar element" condition does the same in all type D cases tested. Both conditions say that the Kazhdan–Lusztig basis element C'_w equals the monomial C'_{s1}…C'_{sl}. Equivalently, Deodhar's defect formula over masks (subsets of the heap) computes every KL polynomial P_{x,w} with no recursion. In type E the match fails. The Deodhar condition fails one step earlier in E6 (at the 13-element ideal, three steps above the lowest weight), and at 3 extra weights in E7. The literature results are verified below. The matching against the user's minuscule data is my own computation, labelled as such.

Own-computation scripts (scratchpad, may be cleared):
- `/private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code/deodhar_minuscule.py`
- `pattern_vs_deodhar.py`, `variants.py`, `e7_run.py`, `bad_terms.py`, `violating_masks.py`, `e6_chain.py`, `e6_bj_compare.py`, `parabolic_check.py`, all in the same folder.

How the computation works: in the Hecke algebra with H_s² = 1 + (v⁻¹ − v)H_s, take the product (H_{s1}+v)…(H_{sl}+v) along a reduced word of each Λ-minuscule w. Elements x are encoded by x⁻¹ρ in fundamental-weight coordinates. The product equals the KL basis element \underline H_w if and only if every coefficient of H_y with y ≠ w lies in vZ[v], because the product is bar-invariant.

---

## Q1. What exactly do Kleshchev–Ram prove (arXiv:0809.0557)?

### Takeaway
For a simply laced Γ, Kleshchev–Ram classify the irreducible graded R_α-modules that are concentrated in one degree. These are the same as the irreducible modules on which all y_r act by 0 (Cor. 3.2). They are the modules S(C), one for each connected component C of the "admissible transposition" graph on words that satisfies condition (3.1). Each S(C) has basis {v_i : i ∈ C}, with y_r = 0 and ψ_r v_i = v_{s_r i} or 0. Such components are the same thing as "skew shapes", i.e. heaps of fully commutative elements of length d (Prop. 3.7). Heaps of dominant minuscule elements ("straight shapes") have dimension given by the Peterson–Proctor hook formula (Thm. 3.10). The paper says nothing about cyclotomic quotients.

### Cited Findings
- Bibliographic data: A. Kleshchev, A. Ram, "Homogeneous representations of Khovanov–Lauda algebras", arXiv:0809.0557. v1 is dated 3 Sep 2008 and v2 23 Sep 2008; the arXiv page lists no journal reference — [arXiv abs](https://arxiv.org/abs/0809.0557). The journal reference J. Eur. Math. Soc. 12 (2010) 1293–1306 is from memory and was not verified.
- The abstract says the paper constructs "irreducible graded representations of simply laced Khovanov-Lauda algebras which are concentrated in one degree". The combinatorics of skew shapes and standard tableaux was "developed previously by Peterson, Proctor and Stembridge". The Peterson–Proctor hook formula gives the dimensions for straight shapes — [arXiv PDF](https://arxiv.org/pdf/0809.0557).
- The introduction says that homogeneous representations "turn out to be similar to seminormal representations of affine Hecke algebras. In type A this can be explained using [BK]" (Brundan–Kleshchev, "Blocks of cyclotomic Hecke algebras and Khovanov–Lauda algebras") — [arXiv PDF](https://arxiv.org/pdf/0809.0557).
- §2.1 (setting):
  - Γ is a graph without loops or multiple edges; cycles are allowed and I may be infinite. The Cartan matrix has a_ij = −1 for neighbours.
  - Relations (2.2)–(2.11) hold. In particular, ψ_r² e(i) = 0 if i_r = i_{r+1}, = e(i) if a = 0, and = ±(y_r − y_{r+1})e(i) for neighbours, depending on orientation.
  - The braid deformation (2.11) is ±e(i) when i_{r+2} = i_r is a neighbour of i_{r+1}.
  - Grading: deg y_r = 2 and deg ψ_r e(i) = −a_{i_r i_{r+1}}.
  - Source: [arXiv PDF](https://arxiv.org/pdf/0809.0557).
- §2.4: s_r is an *admissible transposition* for i if a_{i_r i_{r+1}} = 0. The *weight graph* G_α joins i and s_r i. **Prop. 2.3:** i and j lie in the same component if and only if their {a,b}-subsequences agree for every pair of neighbours a, b — [arXiv PDF](https://arxiv.org/pdf/0809.0557).
- §2.5 (configurations): configurations are beads on a "Γ-abacus" dropped under gravity. The paper notes they "are essentially the same as heaps defined by Viennot [V], see also Stembridge [S1, S2]".
  - Standard λ-tableaux are linear extensions of the configuration.
  - **Prop. 2.4:** components of G_α correspond to configurations, and the weights in a component correspond bijectively to standard tableaux.
  - Source: [arXiv PDF](https://arxiv.org/pdf/0809.0557).
- §3.1, **Prop. 3.1:** let M be irreducible and calibrated (all y_r act as 0), and let i be a weight of M. Then:
  - (i) there is no r with i_r = i_{r+1};
  - (ii) there is no r with i_r, i_{r+1} neighbours and i_{r+2} = i_r;
  - (iii) dim M_i = 1;
  - (iv) the weights of M form one connected component of G_α.
  - **Cor. 3.2:** an irreducible M is homogeneous if and only if it is calibrated.
  - Source: [arXiv PDF](https://arxiv.org/pdf/0809.0557).
- §3.2: a component C is *homogeneous* if every i ∈ C satisfies condition **(3.1)**: "if i_r = i_s for some r < s then there exist t, u with r < t < u < s such that a_{i_r i_t} = a_{i_r i_u} = −1".
  - **Lemma 3.3:** (3.1) holds for all i ∈ C if it holds for one i ∈ C. It is equivalent to Prop. 3.1 (i) and (ii) holding for all i ∈ C, i.e. no "aa" and no "aba" anywhere in the component.
  - Source: [arXiv PDF](https://arxiv.org/pdf/0809.0557).
- **Theorem 3.4:** let C be a homogeneous component and S(C) the space with basis {v_i : i ∈ C}, where
  - e(j)v_i = δ_{ij} v_i,
  - y_r v_i = 0,
  - ψ_r v_i = v_{s_r i} if s_r i ∈ C, and 0 otherwise.
  - Then S(C) is a homogeneous irreducible R_α-module, S(C) ≇ S(C′) for C ≠ C′, and every homogeneous irreducible module is isomorphic to some S(C).
  - The proof uses injectivity of the character map ([KL1, Thm 3.17] = their Thm 2.2).
  - Source: [arXiv PDF](https://arxiv.org/pdf/0809.0557).
- **Definition 3.5 (skew shape):** a configuration is a skew shape "if whenever B1 and B2 are two beads of λ on the same runner then there are at least two beads on different neighboring runners separating B1 from B2". In type A∞ these are the usual skew diagrams in Russian notation. **Theorem 3.6** restates Thm 3.4 with basis v_T indexed by standard tableaux (ψ_r v_T = v_{s_r T} if s_r T is standard) — [arXiv PDF](https://arxiv.org/pdf/0809.0557).
- §3.4: the character is ch S(λ) = Σ_{T∈T(λ)} e^{i^T} (eq. 3.2). Restriction to R_β ⊗ R_γ gives ⊕_{μ⊆λ} S(μ) ⊗ S(λ/μ) (eq. 3.3), generalising the skew Schur formula — [arXiv PDF](https://arxiv.org/pdf/0809.0557).
- §3.5, **Prop. 3.7:** set w_λ := r_{i_d}…r_{i_1} for i = i^T. Then:
  - w_λ is independent of T;
  - (i) the expression is reduced;
  - (ii) λ ↦ w_λ is a bijection between skew shapes with d boxes and fully commutative elements of length d;
  - (iii) standard tableaux correspond to reduced words of w_λ.
  - Source: [arXiv PDF](https://arxiv.org/pdf/0809.0557).
- **Prop. 3.8** (= Stembridge, Minuscule elements, Prop. 2.5): a reduced word w = r_{i1}…r_{id} is dominant minuscule if and only if:
  - (i) between every pair of consecutive occurrences of r_i there are "exactly two terms (possibly equal to each other) that do not commute with r_i";
  - (ii) "the last occurrence of each generator r_i is followed by at most one generator that does not commute with r_i".
  - Dominant minuscule elements are fully commutative [S2, Prop. 2.1].
  - Straight shape := a skew shape λ with w_λ dominant minuscule. Lemma 3.9 gives the bead version.
  - Source: [arXiv PDF](https://arxiv.org/pdf/0809.0557).
- **Theorem 3.10 (Peterson–Proctor hook formula):** for a straight shape with d beads, dim S(λ) = #T(λ) = d! / ∏_{β∈Φ(w_λ)} ht(β), where Φ(w) = {β > 0 : w⁻¹β < 0}. The proof is credited to Nakada [N2] — [arXiv PDF](https://arxiv.org/pdf/0809.0557).
- KR's reference list (verified from the PDF) — [arXiv PDF](https://arxiv.org/pdf/0809.0557):
  - Stembridge, "On the fully commutative elements of Coxeter groups", J. Algebraic Combin. 5 (1996) 353–385;
  - Stembridge, "Minuscule elements of Weyl groups", J. Algebra 235 (2001) 722–;
  - Proctor, "Minuscule elements of Weyl groups, the numbers game, and d-complete posets", J. Algebra 213 (1999) 272–303;
  - Proctor, "Dynkin diagram classification of λ-minuscule Bruhat lattices and of d-complete posets", J. Algebraic Combin. 9 (1999) 61–94;
  - C.K. Fan, J. Algebraic Combin. 5 (1996) 175–189;
  - Nakada, Osaka J. Math. 45 (2008).

### Inferences
- In the user's language: an order ideal of a minuscule poset is a straight shape, a "slot" is a bead, the colour is the runner, and the module is S(λ) with dim S(λ) equal to the number of linear extensions. The user's "exactly two adjacent-colour elements between consecutive equal colours" is condition (i) of Prop. 3.8.
- The fully-commutative condition (3.1) only requires "at least two". The wording "different neighboring runners" in Def. 3.5 is ambiguous. Stembridge's "possibly equal" (Prop. 3.8) and the D4 vector heap 1,2,{3,4},2,1 show that the two separating elements may share a colour.
- The calibrated/seminormal form (y = 0, ψ acting by permutation) means every KR homogeneous module is "combinatorial": a cube complex built from such modules has no y-terms. Any "extra term" in the user's complexes must come from the deformation or from projective, non-homogeneous objects.

### Gaps
- The JEMS journal reference and its page numbers were not verified on a publisher page.
- KR v2 contains no statement about cyclotomic quotients R^Λ; this was checked by a full read of the arXiv v2 text. Cyclotomic results are in Park (Q2).

---

## Q2. Follow-ups: Feinberg–Lee, cyclotomic/minuscule, Specht and Lyndon-word papers

### Takeaway
The verified follow-ups classify and enumerate fully commutative elements: Feinberg–Lee in type D ("packets", Catalan's triangle) and in type A (Dyck paths). Park proves that for minuscule Λ the cyclotomic quotient R^{Λ}(wΛ) is a simple algebra, isomorphic to End of the KR homogeneous module S(b), with an explicit basis indexed by pairs of reduced words. Casbi uses KR's "strongly homogeneous" modules (the dominant minuscule ones) in the Baumann–Kamnitzer–Knutson / MV-cycle setting. None of these papers identifies a threshold like (4,4,4,3).

### Cited Findings
- G. Feinberg, K.-H. Lee, "Fully commutative elements of type D and homogeneous representations of KLR-algebras", arXiv:1401.0845. v1 is dated 4 Jan 2014 and v2 29 Jul 2015, 19 pp. The paper decomposes the FC elements of type D_n into natural subsets and, as an application, classifies and enumerates the homogeneous representations of type D_n KLR algebras — [arXiv abs](https://arxiv.org/abs/1401.0845).
  - Section outline (from the PDF): 1.2 Homogeneous representations; 1.3 FC elements; 2.1 Canonical reduced words; 2.2 Packets; 2.3 Catalan's Triangle — [arXiv PDF](https://arxiv.org/pdf/1401.0845).
  - Published in J. Combinatorics (2015), per a search snippet; not confirmed on the arXiv page — [UConn copy](https://www2.math.uconn.edu/~khlee/Papers/fully-D.pdf).
- G. Feinberg, K.-H. Lee, "Homogeneous representations of Type A KLR-algebras and Dyck paths", arXiv:1505.04075 (May 2015). It refines the classification using Dyck paths, enumerates the homogeneous representations, and gives a dimension formula. The abstract relates them to KR's cuspidal representations. Per the arXiv comment, the type A results were "separated and expanded" from arXiv:1401.0845 — [arXiv abs](https://arxiv.org/abs/1505.04075). (The "1309.xxxx" number in the brief appears to be wrong; the type A paper is 1505.04075.)
- E. Park, "Cyclotomic quiver Hecke algebras corresponding to minuscule representations", arXiv:1909.00313, v3 11 Jun 2020, to appear in J. Korean Math. Soc. — [arXiv abs](https://arxiv.org/abs/1909.00313). From the paper — [arXiv PDF](https://arxiv.org/pdf/1909.00313):
  - **Prop. 2.4** (citing [1, Cor. 4.7] and [14, Lemma 1.11]): for Λ ∈ P⁺ and ξ = wΛ, R^Λ(ξ) is simple.
  - **Prop. 2.5:** every irreducible R-module is absolutely irreducible.
  - **Theorem 2.9:** for X ∈ {A_n, D_n, E6, E7}, i a minuscule node and b ∈ B(Λ_i) of weight ξ:
    - (1) S(b), the KR module on the reduced words of the minuscule element w(b), is a homogeneous simple R^{Λ_i}(ξ)-module;
    - (2) R^{Λ_i}(ξ) has basis {c_{ν,μ} : ν, μ ∈ P(b)}, which makes it isomorphic to End(S(b)).
  - **Remark 2.10:** in type A the basis is indexed by pairs of standard tableaux.
  - **Prop. 2.11:** for type C_n, R^{Λ_n}(ξ) is trivial (simple modules are 1-dimensional).
  - **Theorem 3.4:** type B_n, using shifted Young tableaux.
- E. Casbi, "Equivariant multiplicities of simply-laced type flag minors", arXiv:2005.07051, to appear in Represent. Theory. It studies the Baumann–Kamnitzer–Knutson morphism D̄, which takes special values on dual canonical basis elements corresponding to "Kleshchev–Ram strongly homogeneous modules". It uses the Kang–Kashiwara–Kim–Oh monoidal categorification — [arXiv abs](https://arxiv.org/abs/2005.07051).

### Inferences
- If the user's deformed algebra specialises to the cyclotomic quotient R^Λ(wΛ) at a minuscule weight, the undeformed algebra is a full matrix algebra over k (Park, Thm 2.9 / Prop. 2.4). It is therefore semisimple, and every complex of projectives over it splits. Any non-trivial homological behaviour, such as the user's "extra terms", must come from the deformation parameters or from working over the non-cyclotomic R_α.

### Gaps
- None of the following were located or verified in this pass. The recollections are from memory and are flagged as such.
  - "Kim–Lee": not found.
  - "Loubert": not found.
  - Brundan–Kleshchev–Wang, "Graded Specht modules": from memory arXiv:0901.0218, J. reine angew. Math. 655 (2011).
  - Muth's skew Specht modules: from memory, "Graded skew Specht modules and cuspidal modules for KLR algebras of affine type A".
  - Kleshchev–Ram, "Representations of KLR algebras and combinatorics of Lyndon words": from memory arXiv:0909.1984, Math. Ann. 349 (2011).
- The precise theorems of Feinberg–Lee (packet decomposition, enumeration formulas) could not be extracted because PDF text extraction failed.
- Search results also surfaced two possibly relevant papers that I did not read: "Heaps, crystals, and preprojective algebra modules" (arXiv:2202.02490), and "Classifications of Γ-colored minuscule posets and Γ-colored d-complete posets" (arXiv:2012.15787 / 2008.06745) — [arXiv 2202.02490](https://arxiv.org/pdf/2202.02490), [arXiv 2012.15787](https://www.arxiv.org/pdf/2012.15787).

---

## Q3. Combinatorics: is there a known property with threshold "λ ⊇ (4,4,4,3)" or minimal cases Gr(4,8), OG(6,12), E6/P1?

### Takeaway
**Yes, in type A, and in type D on every case computed.** The property is that w is a *Deodhar element*: C'_w = C'_{s1}…C'_{sl}, equivalently the Bott–Samelson resolution of X_w is small, equivalently P_{x,w} is the defect generating function over masks.
- Type A (Billey–Warrington): Deodhar means 321-hexagon-avoiding. The hexagon is the 14-element skew heap (4,4,4,3)/(1), a 4×4 square with two opposite corners removed. For a Grassmannian permutation, avoiding it means exactly λ ⊉ (4,4,4,3).
- General finite Weyl groups (Billey–Jones): the minimal non-Deodhar patterns live in Coxeter graphs A7, B7/C7, D6, D7, D8, E6, E7. The simply laced minimal types A7, D6, E6 are exactly the user's minimal cases.
- Own computation: on minuscule weights, the Deodhar property agrees with the user's cycle-free property in all A and D cases tested. In E6 it fails one step earlier (13-element ideal); in E7 it fails at 3 extra weights.

No other candidate checked (Gorenstein, lci, factorial, singular locus, small resolutions, jeu de taquin) has a verified matching threshold.

### Cited Findings
**Deodhar framework (as restated by Billey–Jones):**
- S. Billey, B. Jones, "Embedded factor patterns for Deodhar elements in Kazhdan–Lusztig theory", arXiv:math/0612043, Ann. Comb. 11 (2007) 285–333 — [arXiv abs](https://www.arxiv.org/abs/math/0612043); [Springer](https://link.springer.com/doi/10.1007/s00026-007-0321-2).
- Masks and defects:
  - A mask σ on a reduced word w = w_1…w_k is a 0/1 word, and w^σ is the corresponding subexpression.
  - Position j is a *defect* if ℓ(w^{σ[j−1]} w_j) < ℓ(w^{σ[j−1]}) (eq. 2.4); the value of σ_j does not matter.
  - **Def. 2.1:** a zero-defect has σ_j = 0 and is a defect; a plain-zero has σ_j = 0 and is not a defect. σ is Deodhar if #zero-defects < #plain-zeros (eq. 2.5). w is Deodhar if every proper mask is Deodhar.
  - Source: [arXiv PDF](https://arxiv.org/pdf/math/0612043).
- **Theorem 2.3 [Deodhar 1990].** For Coxeter groups with nonnegative KL polynomials, the following are equivalent:
  - (1) the expression w is Deodhar;
  - (2) the element w is Deodhar;
  - (3) C'_w = q^{−ℓ(w)/2} Σ_σ q^{d(σ)} T_{w^σ};
  - (4) P_{x,w} = Σ_{σ: w^σ = x} q^{d(σ)};
  - (5) C'_w = C'_{w1}⋯C'_{wk};
  - (6) the Bott–Samelson resolution of X_w is small;
  - (7) Σ_i dim IH^{2i}(X_w) q^i = (1+q)^{ℓ(w)}.
  - **Remark 2.4:** (1)–(6) are implicit in Deodhar; (4)⇔(7) is proved in Billey–Warrington. Lusztig and Fan–Green call the elements satisfying (5) "tight".
  - Source: [arXiv PDF](https://arxiv.org/pdf/math/0612043).
- §6: for short-braid-avoiding elements, masks are treated as "assignments of 0's and 1's to the entries in the heap". **Example 6.4:** in type A, an entry is a defect if and only if its strings cross an odd number of times below it — [arXiv PDF](https://arxiv.org/pdf/math/0612043).

**Type A hexagon (Billey–Warrington):**
- **Example 5.8:** the hexagon is u = s3s2s1s5s4s3s2s6s5s4s3s7s6s5 = [46718235] ∈ S8. Its heap has rows of sizes 3, 4, 4, 3 (14 elements) on the 7 columns of A7. **Theorem 5.9 [BW01]:** in types A and B, w is Deodhar if and only if w avoids short braids and hexagons as embedded factors — [arXiv PDF](https://arxiv.org/pdf/math/0612043).
- S. Billey, G. Warrington, "Kazhdan–Lusztig polynomials for 321-hexagon-avoiding permutations", arXiv:math/0005052, J. Algebraic Combin. 13 (2001) 111–136 — [arXiv abs](https://arxiv.org/abs/math/0005052); [Springer](https://link.springer.com/article/10.1023/A:1011279130416). What it gives:
  - a combinatorial formula for P_{x,w} when w is 321-hexagon-avoiding;
  - equivalent conditions, including that the Bott–Samelson resolution is small;
  - the statement that Deodhar's algorithm becomes trivial for these w;
  - a method for computing the singular locus of X_w.
- The four classical patterns, recalled from memory: 46718235, 46781235, 56718234, 56781234. Billey–Jones confirm that "avoiding the hexagon embedded factor pattern can be characterized by avoiding 4 classical permutation patterns" on FC elements, and they confirm [46718235] — [arXiv PDF](https://arxiv.org/pdf/math/0612043).
- By direct check, 46781235 is the Grassmannian permutation of λ = (4,4,4,3) and 56781234 that of λ = (4,4,4,4) (λ_i = w(k+1−i) − (k+1−i)). The other two have two descents.
- Enumeration of 321-hexagon-avoiding permutations by a 7-term linear recurrence: Stankova-Frenkel and West, arXiv:math/0106073 — [arXiv](https://arxiv.org/pdf/math/0106073).

**Billey–Jones classification, all finite Weyl groups:**
- **Theorem 5.12:** w is Deodhar ⇔ w⁻¹ is Deodhar ⇔ w avoids the embedded factor patterns of Figure 1 and, in type D, the 1-line "diamond" pattern FLHEX₀ = [16785234] (with bars). Figure 1 ("Minimal non-Deodhar patterns") lists:
  - I2(m): short braids;
  - A7: HEX;
  - B7/C7: BHEX;
  - D6: one pattern;
  - D7: HEX₂, HEX₃a, HEX₃b;
  - D8: the diamond;
  - E6: four patterns (words of lengths 14, 13, 13, 13);
  - E7: five patterns.
  - Source: [arXiv PDF](https://arxiv.org/pdf/math/0612043).
- The E6 words, in BJ labels (chain 0–1–2–3–4 with 5 attached to 2), are:
  - s5s0s1s2s5s3s4s2s3s1s2s5s0s1
  - s5s1s2s3s0s1s2s5s4s3s2s1s0
  - s1s2s5s3s4s2s3s1s2s5s0s1s2
  - s2s5s1s2s3s0s1s2s5s4s3s2s1
  - Source: [arXiv PDF](https://arxiv.org/pdf/math/0612043). (The D-type words in the extracted text are garbled and I could not reliably transcribe them.)
- **Def. 5.5 / Cor. 5.6:** non-Deodhar elements are closed upward under embedded-factor containment (factor containment after a Coxeter embedding of the support). **Remark 5.10 / Ex. 5.11:** type D heaps can contain "alcoves"/"holes", and there is an infinite antichain of non-Deodhar type D elements — [arXiv PDF](https://arxiv.org/pdf/math/0612043).
- B. Jones (J. Algebraic Combin. 2009): for Deodhar elements, the leading coefficients μ(x,w) are always 0 or 1 — [Springer PDF](https://link.springer.com/content/pdf/10.1007/s10801-008-0131-6.pdf).

**Small resolutions of minuscule Schubert varieties:**
- N. Perrin, "Small resolutions of minuscule Schubert varieties", arXiv:math/0601117, Compositio Math. 143 (2007) 1255–1312. He describes all relative minimal models of a minuscule Schubert variety via quiver (heap) combinatorics and shows they are small. X has a small resolution if and only if it has a smooth relative minimal model, and a combinatorial criterion is given — [arXiv abs](https://arxiv.org/abs/math/0601117). (This is a different property from smallness of the *Bott–Samelson* resolution.)

**Own computation (not literature):** Deodhar test (criterion (5)) on every Λ-minuscule element, compared with a literal encoding of the user's verbal cyclic pattern. Script: [deodhar_minuscule.py](file:///private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code/deodhar_minuscule.py) and companions.

| (g, ω) | \|P\| | heap sizes of the non-Deodhar weights | cyclic pattern, literal encoding |
|---|---|---|---|
| A7, ω4 | 16 | {15, 16} | identical |
| A8, ω4 | 20 | 9 weights, smallest 15 | identical |
| A9, ω4 | 24 | 25 weights | identical |
| A9, ω5 | 25 | 42 weights | identical |
| A8, ω3 | 18 | none | none |
| D4, ω1 / ω4 | 6 | none | none |
| D5, ω5 | 10 | none | none |
| D6, D7 vector | 10, 12 | none | none |
| D6, ω6 | 15 | {13, 14, 15} | identical |
| D7, ω7 | 21 | 16 weights, smallest 13 | identical |
| D8, ω8 | 28 | 54 weights, smallest 13 | identical |
| E6, ω1 | 16 | {13, 14, 15, 16} | {15, 16} with c′ ≠ c″; {14, 15, 16} if c′ = c″ is allowed |
| E7, ω7 | 27 | 25 weights, smallest 14 | 22 weights, smallest 15 |

- The type A counts 9, 25 and 42 equal the number of λ in the box containing (4,4,4,3), counted by hand.
- The 13-element E6 minuscule element (weight (0,−1,−1,1,0,0) in fundamental-weight coordinates, Bourbaki labels) is *equal* to BJ's second E6 pattern s5s1s2s3s0s1s2s5s4s3s2s1s0 under the label map BJ 0,1,2,3,4,5 → Bourbaki 1,3,4,5,6,2. All four BJ E6 words also test non-Deodhar in my code, which is consistent with BJ — [e6_bj_compare.py](file:///private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code/e6_bj_compare.py).
- Own computation: extra summands at the threshold. The product decomposes as C'_w + Σ C'_y, all multiplicities 1 (each offending coefficient has lowest term exactly v⁰) — [bad_terms.py](file:///private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code/bad_terms.py).
  - A7, (4,4,4,3): one extra y, with ℓ(y) = 7 and ℓ(w) = 15;
  - D6, 13-element: two, with ℓ = 7, 5;
  - E6, 13-element: two, with ℓ = 7, 5;
  - E6, 14-element: three, with ℓ = 8, 8, 6.
- Own computation: minimal Deodhar-violating masks, heap read bottom-up, labels "colour.occurrence" — [violating_masks.py](file:///private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code/violating_masks.py).
  - A7 (4,4,4,3): a unique mask with 8 zeros {1.1, 2.2, 3.3, 4.2, 4.3, 5.3, 6.2, 7.1}; zero-defects {2.2, 3.3, 5.3, 6.2}.
  - D6 13-element: a unique mask with zeros {1.1, 2.2, 3.3, 4.2, 4.3, 5.2}; zero-defects {2.2, 3.3, 5.2}.
  - E6 13-element: zeros {2.2, 3.2, 3.3, 4.1, 4.2, 4.3}; zero-defects {2.2, 3.2, 3.3}.
  - Here x₂, x₃ are the 2nd and 3rd colour-4 elements, and in A and D the elements 3.3/5.3 (resp. 3.3/5.2) are the "above x₃" members of the user's pairs.
- Own computation: neighbours of the colour-4 chain x₁ < x₂ < x₃, by gap ("gap g" = above g chain elements) — [e6_chain.py](file:///private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code/e6_chain.py).
  - A7 (4,4,4,3): gap1 {3.1, 5.1}, gap3 {3.3, 5.3}. Two colour pairs, matching the user's pattern.
  - D6 13-element: gap1 {3.1, 5.1}, gap3 {3.3, 5.2}. Matches.
  - E6 13-element: gap1 {2.1, 5.1}, gap2 {3.2, 5.2}, gap3 {2.2, 3.3}. Only colour 2 spans gap1→gap3, so the user's two-colour pattern is absent, yet the element is non-Deodhar.
- Own computation, side result: in the parabolic "sign" module (H_s ↦ −v for s ∈ W_Λ), the projected product is the parabolic KL basis element for *every* minuscule w tested (A7, D5, D6, E6, A8), so it shows no threshold. The "trivial" module projection fails already at length 2 because of a normalisation artefact. Hence the threshold lives in the full Hecke algebra (Bott–Samelson for X_w ⊂ G/B), not in the minuscule module — [parabolic_check.py](file:///private/tmp/claude-501/-Users-marco-Documents--research-klrw-project/74b900cf-f14a-404c-8ada-376dc11468cc/scratchpad/code/parabolic_check.py).

### Inferences
- **Structural parallel.** The user's cube of projectives indexed by subsets of the heap parallels Deodhar's 2^{ℓ(w)} masks, which BJ §6 explicitly treats as 0/1 labellings of heap entries. A defect is a local condition at a heap element determined by equal-colour and adjacent-colour elements below it.
  - Deodhar's condition (zero-defects < plain-zeros for every mask) is the regime in which the "naive" mask/subword sum is already the answer, with no recursive corrections.
  - This plausibly corresponds to the user's tree regime, and the corrections C'_y plausibly correspond to the user's "true extra terms".
  - In types A and D the two thresholds coincide on all data computed. This is a strong coincidence, but no published theorem connects KLR cube complexes to Deodhar masks.
- **Categorical form (inference from standard facts, not verified here).** By Soergel's categorification theorem together with Elias–Williamson's proof of Soergel's conjecture (char 0), criterion (5) says that the Bott–Samelson Soergel bimodule BS(w) is indecomposable (≅ B_w). The computed extra C'_y are the extra indecomposable summands of BS(w). For FC w these are Temperley–Lieb-type "monomial" elements (Fan; Fan–Green).
- **Type A proof sketch (inference).** w_λ is 321-avoiding with heap λ. An embedded hexagon factor is a convex subheap that is a translate of (4,4,4,3)/(1), which is symmetric under the reversal embedding. An order ideal λ contains such a translate if and only if λ ⊇ (4,4,4,3). This is consistent with the user's claims that ω₁, ω₂, ω₃ are cycle-free for all n (a hexagon needs 4 rows and 4 columns) and that Gr(4,8) is minimal.
- **Fully tight types (inference).** From BJ Figure 1, every FC element of A_n (n ≤ 6), of D_n (n ≤ 5), and hence of D4 and D5, is Deodhar. This matches the user's cycle-free D4 and D5 spinor cases.
- **E-type discrepancy.** In E6 the Deodhar condition fails at the 13-element ideal (three steps above lowest), where the user reports a tree. The violating mask there zeroes three consecutive trivalent-colour elements. Either the link graph misses the E6 pattern, or it is a different invariant.
  - My literal encoding of the user's verbal pattern (c′ ≠ c″) gives the E6 threshold at 15 elements, not the stated 14. Allowing c′ = c″ gives 14. The verbal description may therefore not match the code exactly in E6.
- **A reformulation (own observation, inference).** The full minuscule posets containing the cyclic pattern are exactly those in which some colour occurs ≥ 4 times, i.e. the maximal coefficient of ω − w₀ω is ≥ 4:
  - Gr(k,n): min(k, n−k) ≥ 4;
  - D_n half-spin: n − 2 ≥ 4;
  - E6: 4; E7: 6;
  - excluded: D_n vector (2) and D5 spinor (3).
  - This describes which families are cyclic, not the threshold ideals: (4,4,4,3) has colour 4 only 3 times.

### Gaps
- No source found showing that any of the following has threshold (4,4,4,3) or minimal cases Gr(4,8)/OG(6,12)/E6/P1, and I did not fetch sources on them:
  - Gorenstein, lci or factorial loci of minuscule Schubert varieties;
  - Lakshmibai–Weyman multiplicities;
  - Brion–Polo generic singularities;
  - Perrin's Gorenstein locus;
  - Thomas–Yong / Buch–Samuel jeu de taquin;
  - Koszulness or syzygy thresholds.
  - From memory and unverified: the Gorenstein criterion for Grassmannian Schubert varieties (all inner corners on one antidiagonal, Svanes / Woo–Yong) has a different shape, and all Grassmannian Schubert varieties have small resolutions (Zelevinsky), so small-resolution *existence* is not the threshold.
- Perrin's explicit criterion for which minuscule Schubert varieties admit small resolutions was not extracted, so I cannot say whether it singles out the user's cases in types D and E.
- BJ's D6/D7/D8 pattern words were garbled in text extraction. The D-type agreement rests on my computation, not on matching their words.
- Proctor's 15 slant-irreducible classes: verified only that Class 1 is shapes, Class 2 shifted shapes, Class 3 a three-parameter family, and Class 15 a single poset tied to E7. Classes outside 1–3 correspond to elements of "general type E" Kac–Moody Weyl groups. Every connected d-complete poset is a slant sum of slant-irreducibles in 15 families — [Proctor's page](https://proctor.math.unc.edu/Classif.html); [JACO 9](https://emis.univie.ac.at/journals/JACO/Volume9_1/j48700v65753066w.html); [EJC paper](https://www.combinatorics.org/ojs/index.php/eljc/article/download/v28i1p15/pdf). The full list and the double-tailed diamond's place in it were not retrieved. From memory, unverified: the double-tailed diamond d_n(1) is the D_n vector minuscule poset.
- Other minuscule-poset sources:
  - N.J. Wildberger, "Minuscule posets from neighbourly graph sequences" (preprint 2001; MaRDI lists a 2003 publication; the European J. Combin. reference is unverified). It shows that if a maximal neighbourly X-heap is two-neighbourly then X is A_n, D_n (n ≥ 4), E6 or E7 — [preprint](https://viennot.org/wa_files/wildberger1.pdf); [MaRDI](https://portal.mardi4nfdi.de/wiki/Item:Q1405002).
  - R.M. Green, *Combinatorics of Minuscule Representations*, Cambridge Tracts in Math. 199 (CUP 2013). Chapters include Heaps over graphs; Minuscule representations; Full heaps over affine Dynkin diagrams; Combinatorics of Weyl groups; The 28 bitangents — [Lehmanns](https://lehmanns.de/shop/naturwissenschaften/24400321-9781107026247-combinatorics-of-minuscule-representations); [CUP](https://www.cambridge.org/core/books/combinatorics-of-minuscule-representations/minuscule-representations/0935170626E0B33BDDECFA1EC53AF665). Section numbers were not verified, and an MAA review describes the chapter structure inconsistently.

---

## Q4. Literature relating Λ⁴C⁸ (A7), the 32 of D6 and the 27 of E6

### Takeaway
Established: the three minimal cases are exactly the three symmetric spaces of E7.
- EV = E7/(SU(8)/±1): dimension 70, rank 7.
- EVI = E7/(SO(12)·SU(2)): dimension 64, rank 4, quaternion-Kähler.
- EVII = E7/(E6·SO(2)): dimension 54, rank 3, Hermitian.

Their isotropy representations are Λ⁴C⁸, C²⊗S⁺₃₂ and 27⊕27̄. This last statement is standard but was not verified on a page; the dimensions match.

In Landsberg–Manivel's geometric magic square, the Cayley plane OP² = E6/P1 sits in row 2 (Severi varieties) and the spinor variety S₁₂ ⊂ P³¹ (D6 half-spin) in row 3 (Legendrian varieties). Gr(4,8) is not in the square.

Independently, and verified: A7, D6, E6 are the simply laced Coxeter graphs that carry Billey–Jones's minimal non-Deodhar patterns (Q3). Of the two observations, this is the only one tied to a theorem about heaps.

### Cited Findings
- Symmetric-space classification table — [Wikipedia: Symmetric space](https://en.wikipedia.org/wiki/Symmetric_space):
  - EV: G/K = E7/(SU(8)/{±I}), dim 70, rank 7;
  - EVI: E7/(SO(12)·SU(2)), dim 64, rank 4;
  - EVII: E7/(E6·SO(2)), dim 54, rank 3.
  - EVI is listed as quaternion-Kähler and EVII as Hermitian.
- E7 real forms: EV = E7(7), split, maximal compact SU(8)/{±1}; EVI = E7(−5), maximal compact SU(2)·SO(12)/centre; EVII = E7(−25), maximal compact SO(2)·E6/centre — [Wikipedia: E7](https://en.wikipedia.org/wiki/E7_(mathematics)). The same page did *not* state the branching rules (133 = 63+70, etc.).
- J.M. Landsberg, L. Manivel, "The projective geometry of Freudenthal's magic square", arXiv:math/9908039 — [ar5iv](https://ar5iv.labs.arxiv.org/html/math/9908039). Rows of the geometric square:
  - row 2 (Severi): v₂(P²), P²×P², G(2,6), OP²;
  - row 3: Gω(3,6), G(3,6), S₁₂, E7/P7. The third-row varieties are Legendrian for an Sp₆(A)-invariant symplectic form.
  - row 4: adjoint varieties.
- Landsberg–Manivel, "Legendrian varieties", arXiv:math/0407279. This is a general study of Legendrian manifolds; the abstract does not name these examples — [arXiv abs](https://arxiv.org/abs/math/0407279).

### Inferences
- Isotropy representations, by dimension count: (e7, sl8) gives p = Λ⁴C⁸ (dim 70 = C(8,4)); (e7, sl2⊕so12) gives C²⊗32 (64); (e7, e6⊕C) gives 27⊕27̄ (54).
- In Kac's description of involutions, these are the inner involutions of E7 obtained from the extended diagram Ẽ7 by deleting one node of mark 2, giving A7 or A1×D6, or the two mark-1 nodes, giving E6×T1. This is from memory, not verified.
- Whether this E7 coincidence explains the threshold is **speculative**. No source links E7 symmetric pairs to Deodhar patterns or to minuscule-heap cycles.
- Prehomogeneity is *not* the common property: Spin12×GL1 on 32 and E6×GL1 on 27 are prehomogeneous, whereas GL8 on Λ⁴C⁸ is a θ-representation with a 7-dimensional invariant quotient. This is from memory (Sato–Kimura; Vinberg θ-groups) and unverified.
- The magic square places the 27 and the 32 together, in rows 2 and 3, but not Λ⁴C⁸. So it does not explain the triple either.

### Gaps
- No primary-source statement was found of the isotropy representations (Helgason's tables, or Kac's *Infinite-dimensional Lie algebras* Ch. 8), of the Kac coordinates, or of orbit classifications of SL8 on Λ⁴C⁸ (Antonyan) or Spin12 on 32 (Igusa).
- No paper found ties Gr(4,8), OG(6,12) and E6/P1 together through Schubert-variety singularities.
