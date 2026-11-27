# R06 — Referee report on `06_main_intertwiner.md` (sections 1–6 and 9; 7–8 spot-checked)

## RESUME NOTES (open)

* **Finished (read line by line, verdicts below):** §1 (Lemma 1.1), §2 (Lemma 2.1, Lemma 2.3, all
  five cases recomputed by hand), §3 (Props 3.2, 3.3, 3.5), §4 (Props 4.2, 4.3, 4.4), §5 (Props
  5.1, 5.2, Remark 5.3), §6 (Lemma 6.1, Def. 6.2, Lemmas 6.3–6.8, Cor. 6.9, proof of Thm 6.2
  in §6.6: the three classes T0/T1/T2, the identities (P0), (P1), (P2), completeness).
  §7–8 and §9 read once.
* **Doing now:** writing my own numerical checks `proofs/checks/R06_*.py` (independent
  construction of `Φ_j`, `d_Φ`, `G` from the text of Definition 6.2, test of
  `d_Φ G = G d_tree` etc.).
* **Remaining:** (1) the numerical checks listed in the brief (all words of length ≤ 6 in
  `A_3`, `D_4` with multiplicity ≤ 2, every red colour; random 7–9 strand words in `A_5`,
  `D_5`, `E_6`; perturbations; multiplicity-3 failure); (2) re-run 2–3 experiments of §7–8;
  (3) §9 detailed verdict (Prop. 9.2, 9.3, Cor. 9.4); (4) final judgement.
* **Scripts / logs:** none yet (will be `proofs/checks/R06_*.py`, logs `proofs/checks/R06_logs/`).

---

## Verdicts so far

| statement | verdict | remarks |
|---|---|---|
| Lemma 1.1 (`ψ f = (s f)ψ + r ∂f`) | correct and complete | base cases agree with (S); Leibniz rule for `∂` checked. |
| Lemma 2.1 | correct and complete | = Lemma D.1 of `01`. |
| Lemma 2.3 (exact braid relations for `g`) | correct and complete | the five colour patterns are exhaustive for 3 strands with ≤ 1 red; every case recomputed (dot transport through `ψ_2ψ_1`, `ψ_1ψ_2`; in case 4 the bigons `ψ_1² − ψ_2² = t_{cc'}(x_1 − x_3)` resp. `u_k(x_1−x_3)` cancel the cubic defect; case 5 expansion of `φ_2φ_1`, `ψ_1φ_2φ_1`, `φ_2φ_1ψ_2` line by line). Independent of `01` Thm D.3 (which uses F1); this proof is F1-free. |
| Prop. 3.2 | correct and complete | commutation moves: `g_i` involves only positions `i,i+1`; braid moves: Lemma 2.3. |
| Prop. 3.3 | correct and complete | |
| Def. 3.4 / Prop. 3.5 (product rule) | correct and complete | induction checked, both cases `{a,b} ∈ / ∉ inv(s,s′)`; `q_{ab}` symmetric so well defined on unordered pairs. |
| Prop. 4.2 (naturality) | correct and complete | the three cases for `ψ_i` on `(X,Y,S)` are exhaustive (X,Y not same black colour → Lemma 2.3; same colour ≠ c(m) → exact (B) on `(c,c,T)`; all three equal → second part of Lemma 2.3). Red strand and dots included. |
| Prop. 4.3 (`d_Φ² = 0`) | correct and complete | `{i,j}` is inverted in neither path; product rule gives `𝒟(std I → std(I∖{i,j}))` for both; signs opposite. |
| Prop. 4.4 | correct and complete | `ζ_m` is a well-defined positional polynomial because it is symmetric in the other colour-`c(m)` dots. |
| Prop. 5.1 | correct and complete | induction along `≺` (well founded); left multiplication by `ζ_I` at the source, Theorem E.1(a) — correct side. |
| Prop. 5.2 | correct and complete | the chain-map condition of `Γ` on `C_R → C_L` is exactly (★) with the cone sign `−d′⊗1`. |
| Remark 5.3 | correct as a remark, one imprecision | see problem 1 below. |
| Lemma 6.1 (Λ1)–(Λ4) | correct and complete | (Λ3) exponent `#{i>a} + #{a<i<b} ≡ 1 + #{i>b}` checked. |
| Def. 6.2 / triangularity | correct | `I_F ≺ I` for `F ≠ ∅` since `a_p < b_p`. |
| Lemma 6.3 | correct and complete | colour map is injective on `{ρ} ⊔ 𝔉`; (ii) via uniqueness of paths in a tree. |
| Lemma 6.4 | correct and complete | (ii): only same-colour crossing in a tree term is mover/twin with twin `< j`; (iii) uses Lemma 6.3(ii) with `ρ = j`, `𝔉 = F` (connected by admissibility). |
| Lemma 6.5 | correct and complete | `z_p` slides to the crossing through `ψ`'s of different colours; `D_2′` is a relabelling of `D_2` hence reduced; doubly inverted pairs of `D_1, D_2′` are pairs `{a,y}`/`{j,y}` with `y` crossing both, `q = 1` by 6.4(iii); `W` cannot occur. |
| Lemma 6.6 (a)–(f) | correct and complete | all 6 tables recomputed from the (start, middle, end) positions; nested pairs give no doubly inverted pair. |
| Lemma 6.7 (tree lemma) | correct and complete | |
| Lemma 6.8 | correct and complete | `ε_q` agrees with `formula.py` (connector in the source vertex ⇒ −1). |
| Cor. 6.9 | correct and complete | every link between two vertices of `T` is an edge of `T` (component of a forest). |
| Thm 6.2, proof §6.6 | **correct and complete (by hand)**; numerical checks pending | classes T0, T1, T2 exhaustive and disjoint (see detailed notes); coefficients and signs of (P0), (P1), (P2) recomputed; types (α)–(δ) handled. |

## Problems found so far

1. **(cosmetic / repairable) Remark 5.3, triangularity of a factorised `G`.** The components
   `h : I∪{k} → J` of a one-step map need not satisfy `J ≺ I∪{k}` for the order `≺` of §5
   (sum of the slots): e.g. `I∪{k} = {1,5}`, `J = {3,4}`. So the factorised
   `G = Γ^{(1)}⋯Γ^{(m)}` is triangular for the *colexicographic* order (compare the largest
   element of the symmetric difference), not necessarily for `≺`. Prop. 5.1 holds verbatim
   for any well-founded order, so nothing breaks; but Prop. 5.1 should be stated for an
   arbitrary well-founded order (or colex) if it is to be applied to factorised `G`
   (§7–8, Prop. 9.1). Irrelevant for Theorem 6.2/Cor. 6.3 (there `I_F ≺ I` holds).

(More to come: §9 verdicts, remaining checks.)

## Independent checks so far (own code: `checks/R06_lib.py`; nothing imported from `06_lib.py`)

`R06_lib.py` builds, from the text only: `g_i` (φ for equal black colours, ψ otherwise),
`𝒟(s → s′)` along a (random or fixed) reduced word, `Φ_j(I)`, `d_Φ`, `G` of Definition 6.2
literally (sum over all `F ⊆ 𝔓(I)`, `ε(I,F)`, `∏(−r_p)`, `∏ z_p`, `∏ Q(x_{b_p}, x_{a_q})`,
brackets `[σ_F]` computed as `𝒟`), and the tree differential from `formula.Stack` +
`sample_verify.element` (klrw basis braids).

| script / log | what | result |
|---|---|---|
| `R06_basic.py` → `R06_logs/basic.log` | conventions (`ψ²` on `(1,2)` = `t_{12}x_1 + t_{21}x_2`; `D('t_1_2')` = edge variable), (S), Lemma 1.1 on a cubic `f`, Lemma 2.1, Lemma 2.3 on **every** labelled 3+W-strand state of 13 colour words in `A_3`, `D_4` with all relevant red colours (1 960 identities, including `ψ_iφφ = φφψ_{i+1}`), negative control (ψ-braid on `(1,2,1)` not exact), Prop. 3.2/3.3/3.5 on 48 random triples of labelled states (4 words, `A_2,A_3,D_4`, red included, `q_{aW} = u_k x_a` included) | 43/43 ok |
| `R06_natural.py` → `R06_logs/natural.log` | Prop. 4.2 and 4.4 on 150 random elements (sums of 3 random words of length 5 with random dots, random integer coefficients) on 10 words (`A_2, A_3, D_4`; red equal to the moving colour, adjacent, far; blocks containing equal neighbours and the red strand) | naturality 0/150 failures; `ζ_m` commutes 0/300 failures; negative control (ψ instead of φ in `Φ`): 138/150 differ |
| `R06_main.py all A 3 6` → `all_A3_*.log` | **all** 270 words of length ≤ 6 over `A_3` with multiplicities ≤ 2, **every** red colour (810 cases): `d_Φ² = 0`, `d_tree² = 0`, `d_Φ G = G d_tree` in every component | 0 failures |
| `R06_main.py word … text noQ LL RR plusr nosign` | power of the test: `1212` (A_2 red 1) and `213213` (A_3 red 2) with `G` perturbed (drop `Q`; `Q(x_{a_p},x_{a_q})` or `Q(x_{b_p},x_{b_q})` instead of the mixed bigon; `+r` instead of `−r`; drop `ε(I,F)`) | text: 0/40, 0/300 failing components; perturbed: 4, 4, 4, 22, 14 of 40 and 42, 42, 42, 170, 117 of 300 fail |
| multiplicity 3, `G` of Def. 6.2 literally (all same-colour pairs) | `111, 1121, 1211, 12121, 11211` (A_2 red 1), `123131` (A_3 red 1), `21322` (A_3 red 2), `342312453` (A_5 red 3) | `d_Φ G ≠ G d` in 5/14, 12/42, 12/42, 33/126, 61/138, 60/322, 28/112, 840/5040 components, while `d_Φ² = 0` and `d_tree² = 0` hold: the formula is indeed special to multiplicity ≤ 2 |
| `R06_main.py word A 5 3 342312451` | 9 strands, four colours twice | 0/4625 failing |
| `R06_classes.py` | **structure of the proof of §6.6**: every term of `d_Φ G` and `G d` computed in the algebra, right terms split into main/resolved part (`z_p` removed from `γ`, then Lemma 6.5), split compared with the actual term, each piece assigned to `T0/T1/T2` by the rules of §6.6, **each class checked to vanish separately**; also Lemma 6.5 at every twin term and "`[σ_F]` = klrw basis braid" | `1212`: 105 terms, 60 classes; `213213`: 808 terms, 450 classes: all classes 0, no split / Lemma 6.5 / basis-braid errors |

Running: `all D 4 6` (all 2 324 words, every red colour, 9 296 cases), random 7–9 strand
words in `A_5`, `D_5`, `E_6`, class-by-class checks on a batch.
