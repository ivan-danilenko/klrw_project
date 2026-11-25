# Closed form for the cup (I-) brane resolution — hand-over

**Status.** This file was written by a session that could only *read* the repository
(SageMath was not installed there). Nothing below has been run. Statements are marked as

- **[code]** — read off the source, with file references;
- **[hand]** — a pen-and-paper computation, not machine-checked;
- **[guess]** — a working hypothesis.

A session picking this up should start at "Start here" at the bottom.

---

## 1. Goal

For a simply-laced Lie algebra 𝔤 and a minuscule fundamental weight λ, the cup brane
(I-brane) is an object of D^b of the deformed KLRW algebra with two red strands (λ and λ*).
It is built from *stable envelopes* S_μ, one for each weight μ of V_λ: U-shaped branes for a
chain λ → μ nested around the first puncture, and for a chain μ → w₀λ nested around the
second. The S_μ are glued by iterated cones, each time along a chain map that is conjectured
to be the unique non-zero one.

Each S_μ is a cube of T-branes whose leading differential δ₀ is geometric and squares to zero
only modulo the deformation parameters. The true differential is currently obtained by
solving integer linear systems order by order in those parameters.

**Minimum deliverable.** A general expression for the fully corrected differential of every
S_μ — or reliable evidence that no such expression exists.

**Stretch.** A closed form for the gluing maps, hence for the whole I-brane.

**Data to collect.** sl_n for n = 3, 4, 5, 6 (all minuscule ω_k) and D₄, D₅, D₆ (vector and
spinor weights where affordable).

## 2. Decisions already made by the project owner

| Topic | Decision |
|---|---|
| Parameters | ħ = vertex parameters `r_i`; u = framing parameters `u_i`; edge parameters `t_ij`. All kept generic, with the full multigrading that `standard_ebranes` switches on (`dot_scaling`, `edge_scaling`, `vertex_scaling`). Specialisations follow by substitution. |
| u versus ħ | **Both u- and ħ-corrections are needed in general.** "No pure-u order" (section 6) is only a statement about the first obstruction in δ₀². Never assume u-independence; the formula must cover all mixed orders. |
| Scope | Expect S_μ = (stack around p₁) ⊠ (stack around p₂). The target is a formula for **one stack of nested U-branes around one red strand**, for words coming from chains in a minuscule weight poset, uniform over all reduced words (not only the arbitrary chain `standard_ebranes` picks). |
| Normal form | The differential is only defined up to gauge δ ↦ gδg⁻¹. Aim for the gauge with the fewest terms and smallest integer coefficients. |
| Branch | All work on `claude/youthful-ride-a6dudi`. `main` is never touched. |
| Where | Computations run on the owner's machine (existing Sage, compiled extensions, Gurobi licence). |

## 3. Conventions extracted from the code

### 3.1 Algebra, parameters, relations [code]

`KLRWAlgebra` (`klrw/klrw_algebra.py`) is free over a polynomial (or Laurent) ring
`KLRWDotsAlgebra` (`klrw/dot_algebra.py`) with basis the braids. Variables, in generator
order:

| Variable | Default name | Meaning |
|---|---|---|
| vertex parameter | `r_<i>` | ħ for colour i |
| edge parameter | `t_<i>_<j>` | one per oriented Dynkin edge |
| framing parameter | `u_<i>` | edge V_i → W_i |
| dot | `x_<i>_<k>` | dot on the k-th strand of colour i (1-based, counted in the **left** state) |

Edge parameters for non-adjacent pairs, and for W → V, are the constant 1
(`dot_algebra.py`, `assign_default_values`). Names come from
`FramedDynkinDiagram_with_dimensions.names` (`klrw/framed_dynkin.py`).

Relations, as implemented in `RightDotAction` and `_right_action_of_s_iter_`
(`klrw_algebra.py`), simply-laced case:

| Relation | Formula |
|---|---|
| same colour, double crossing | ψ² = 0 |
| dot through a same-colour crossing | x·ψ − ψ·x' = ± r_i (crossing smoothed) |
| bigon, adjacent colours i, j | t_ij·x_i + t_ji·x_j |
| bigon, non-adjacent colours | 1 |
| bigon, black i with red i | u_i·x |
| bigon, black i with red j ≠ i | 1 |
| triple point (i, j, i), j adjacent to i | braid relation fails by − r_i·t_ij × (smoothed diagram) |
| triple point (V_i, W_i, V_i) | braid relation fails by − r_i·u_i × (smoothed diagram) |
| all other triple points | braid relation holds exactly |

"Geometric part" (`geometric_part`) = set every `r`, `t`, `u` to zero.

### 3.2 Grading [code]

`QuiverGradingGroup` (`klrw/gradings.py`) is free on: q (ordinary equivariant degree),
c(v) per vertex (`vertex_scaling`), e(a,b) per unoriented edge of the framed quiver
(`edge_scaling`), d(v) per vertex (`dot_scaling`).

| Generator | Degree |
|---|---|
| crossing, same colour v | −2q + c(v) |
| crossing, adjacent colours a, b (also V_i with W_i) | q + e(a,b) |
| crossing, unrelated colours | 0 |
| dot on colour v | 2q + d(v) |
| r_v | c(v) + d(v) |
| t_{a→b}, u_a (tail a) | 2e(a,b) − d(a) |

With all three scalings on, the parameter monomial of a matrix entry is essentially fixed by
its multidegree. This is why the data should be collected in the fully graded algebra.

### 3.3 States, braids, products [code]

- A state is a tuple of `NodeInFramedQuiver` (V = moving, W = framing).
- A braid stores its **right** state and a reduced word in "lexmin" normal form; `word[0]`
  is next to the left state. `KLRW.element_from_braid(word, left_state=..., ...)` accepts an
  arbitrary word.
- `a * b` glues a's right end to b's left end (path concatenation, read left to right).
  Mismatched states give zero, or `ValueError` if `warnings=True`.
- Coefficients sit on the left of the braid; dot indices refer to the left state.
- Differentials are stored over `KLRW.opposite`; use `entry.value` to get the KLRW element.
- `KLRW.one()` raises; use `KLRW._one_()`.

### 3.4 The stable-envelope cube [code]

`stable_envelope(KLRW, left_framing, right_framing, left_sequence=..., right_sequence=...)`
in `klrw/stable_envelopes.py`. With n = len(left) + len(right), vertices are ε ∈ {0,1}ⁿ,
homological degree |ε|, differential of degree −1. The state at ε is

```
[left strands with ε=0, in REVERSED order]  W_left
[left strands with ε=1, in order]  [right strands with ε=0, in order]  W_right
[right strands with ε=1, in REVERSED order]
```

So the first left strand is the innermost U around the left puncture, and the last right
strand is the innermost U around the right puncture. δ₀ flips a single 1 → 0 (the strand
moves from the right of its puncture to the left, crossing the red strand and every strand
nested inside it) with the minimal braid and Koszul sign (−1)^(number of 1's to the right).
Equivariant shifts of the projectives are fixed by requiring δ₀ to be homogeneous.

`standard_ebranes(Phi, lambd)` (`klrw/standard_ebranes.py`) builds, for every weight μ, one
arbitrary maximal chain λ → μ (left word) and μ → w₀λ (right word), calls `stable_envelope`,
base-changes to ℚ, and glues along `pos_dual.linear_extension()` using the unique class in
`stab_mu[shift].hom_set(e_brane).homology()`.

### 3.5 The correction algorithm [code]

`corrected_diffirential_csc` in `klrw/perfect_complex_corrections.py` (name misspelled in
the source):

1. Find the smallest exponent tuple occurring in δ² (`min_exp_in_square`). The comparison is
   lexicographic by variable position — `r`'s first, then `t`/`u`, then dots — so monomials
   with low powers of ħ are treated first.
2. Drop the dot part; this parameter monomial is the current order.
3. Ansatz: for **every** pair of projectives in adjacent homological degrees, every basis
   element of the correct multidegree whose parameter part is exactly that monomial (any
   number of dots the degree allows).
4. Solve the linearised system {δ_geom, δ_k} = −(δ²)_k for integer unknowns (`solve_system`,
   Gurobi, no objective).
5. Add the correction, recompute δ², repeat (at most 10 iterations).

### 3.6 Reading a complex out [code, untested recipe]

```python
D = KLRW.base()
names = D.variable_names()

def entry_to_tuples(opposite_entry):
    out = []
    for braid, coeff in opposite_entry.value:
        mono = {}
        for exp, c in coeff.iterator_exp_coeff():
            key = tuple((names[p], int(k)) for p, k in exp.sparse_iter())
            mono[key] = int(c)
        out.append((tuple(braid.left_state().as_tuple()),
                    tuple(braid.right_state().as_tuple()),
                    tuple(braid.word()), mono))
    return out

for hom_deg, mat in C.differential:          # mat: C_h -> C_{h-1}, rows = codomain
    for (i, j), entry in mat.dict(copy=False).items():
        ...
C.projectives()                               # {hom_deg: tuple of (state, equivariant_degree)}
```

`KLRWPerfectComplex` asserts δ² = 0 on construction (`free_complex.py`,
`Differential.__init__`), so building a complex from a conjectured differential is already
an exact check.

## 4. Caveats and known problems [code]

- **The solver output is not canonical.** One arbitrary integer solution per order. Any
  pattern-matching needs a gauge choice first.
- `solve_system` casts Gurobi's float solution with `dtype="intc"`, which truncates instead
  of rounding (0.9999999 → 0). Round before casting in any wrapper.
- An unreachable right-hand side raises `AssertionError("Inconsistent system: ...")` in the
  Cython extension instead of reporting infeasibility.
- `stable_envelope` calls `PerfectComplex` with the default `parallel_processes=8` and
  `verbose=False`; system sizes are therefore not printed.
- Gurobi is used in three places: `perfect_complex_corrections.solve_system`,
  `homotopy.solve_system` (both "integer x with Ax = b"; the second returns `None` when
  infeasible), and a small grading ILP `dot_algebra._solve_ilp_on_gradings_` that sits under
  every unrestricted `graded_component.basis()` call, hence under all Hom/Ext computations.
- Ext groups use exact Sage linear algebra, over a field only (ℤ is `NotImplementedError`).
- Gluing maps live over the Laurent ring: the stored B₂ output in `swipe_move.ipynb`
  contains `r_2^-1`. `homotopy._homotopy_iterations_` refuses invertible parameters.
- Broken or stale: `klrw/combinatorial_ebranes.py` (imports `dot_algeabra`),
  `klrw/resolution_guess.py` (calls a missing method), dead code after `return e_brane` in
  `standard_ebranes.py`, several notebooks that use removed APIs.
- Only A₁, B₂ and C₃ have ever been run in this repository. Type D and sl_n with n ≥ 3 are
  untested territory; expect to hit bugs.
- The code subclasses Sage internals (`LaurentPolynomialRing_mpair`,
  `MPolynomialRing_libsingular`, private `ETuple` fields). The notebooks were last run on
  Sage 10.0 and 10.3.

## 5. Sizes

d = number of moving strands = coefficients of λ + λ* in simple roots; each S_μ has 2^d terms.

| Type, weight | λ + λ* in simple roots | d | weights |
|---|---|---|---|
| A_n, ω₁ (n = 2…5) | (1,…,1) | n | n+1 |
| A₃, ω₂ | (1,2,1) | 4 | 6 |
| A₄, ω₂ | (1,2,2,1) | 6 | 10 |
| A₅, ω₂ | (1,2,2,2,1) | 8 | 15 |
| A₅, ω₃ | (1,2,3,2,1) | 9 | 20 |
| D₄, ω₁ | (2,2,1,1) | 6 | 8 |
| D₄, ω₃ / ω₄ | (1,2,2,1) / (1,2,1,2) | 6 | 8 |
| D₅, ω₁ | (2,2,2,1,1) | 8 | 10 |
| D₅, ω₄ / ω₅ | (1,2,3,2,2) | 10 | 16 |
| D₆, ω₁ | (2,2,2,2,1,1) | 10 | 12 |
| D₆, ω₅ / ω₆ | (1,2,3,4,2,3) up to relabelling | 15 | 32 |

## 6. Hand computation and working hypotheses

None of this has been checked by machine.

**H1 — reduction to one puncture [guess].** Left strands never interact with the right red
strand in δ₀, and horizontal composition is a functor, so S_μ should be the horizontal
product of a left stack and a right stack with differential d_L ⊗ 1 ± 1 ⊗ d_R and no cross
terms. A right stack should be a left stack rotated by 180°. `klrw/tensor_product_of_complexes.py`
exists for exactly such products.

**H2 — where δ₀² fails [hand].** In one stack, for i < j both to the right of the red
strand, the two paths I → I∖{i,j} are reduced diagrams with the same crossings; they differ
by triple-point moves on (k, i, j) with k < i, with strand i in the middle. Such a move has
a correction only if colour(k) = colour(j) and colour(i) is adjacent to it. Consequences:

- words with all colours distinct (sl_n with ω₁) need no correction at all;
- the first obstruction is at order r·t — there is no pure-u order at this stage;
- the triple (i, j, W) never contributes to δ₀² because its outer strands are W and j.

u enters at later orders through (V_i, W_i, V_i) triple points, weighted r_i·u_i.

**H3 — the toy case [hand].** Word (a, b, a) with a, b adjacent, around W_a; strands 1, 2, 3.
Cube vertices are subsets I of strands to the right of W:

```
{}      (a,b,a,W)      {1}   (a,b,W,a)     {2}   (a,a,W,b)     {3}     (b,a,W,a)
{1,2}   (a,W,a,b)      {1,3} (b,W,a,a)     {2,3} (a,W,b,a)     {1,2,3} (W,a,b,a)
```

δ₀² ≠ 0 on {2,3} → {} and on {1,2,3} → {1}. Both are cancelled by a single new term

```
δ₁ : {2,3} → {1},   ± r_a · t · [strand b crosses W],     (a,W,b,a) → (a,b,W,a)
```

and δ₁² = 0 for index reasons, so δ = δ₀ + δ₁ should be the complete answer here. In marked
terms: strand 2 makes an ordinary move, while the two a-strands 1 and 3 exchange labels.

**H4 — shape of the general formula [guess].** A sum over one-strand moves between *marked*
cube vertices, where relabelling equal-coloured strands is allowed and is paid for by a
parameter monomial fixed by the multidegree. For sl₄, ω₂, word (2,1,3,2) around W₂ the
first-order terms should be the two exchange moves for the triples (1,2,4) and (1,3,4);
second-order terms (including the first r·u terms) should come from composing these with δ₀.

**H5 — recursion [guess].**
Stack(i₁…i_m) = Cone(F_m : R_{i_m} Stack(i₁…i_{m−1}) → L_{i_m} Stack(i₁…i_{m−1})), where R/L
add a strand at the far right/left and F_m = "the new strand crosses everything" plus
homotopy corrections. A closed form would be a formula for the components of F_m, which map
I → J with |I| = |J|.

**Remark on r⁻¹ [hand].** A same-colour crossing with one dot has the multidegree of r, so
r⁻¹·(dotted crossing) can sit in the same graded piece as a bare move. This is consistent
with `r_2^-1` in the stored B₂ gluing maps, and is why the gluing step needs invertible
parameters.

## 7. Plan

### Phase 0 — environment
Check the Sage version, build the Cython extensions (`sage setup.py build_ext --inplace`),
confirm the Gurobi licence accepts large models. Smoke tests: relation checks in
`klrw/klrw_tests.py`; the B₂ run from `swipe_move.ipynb` (stored output to compare);
`standard_ebranes` for A₂, ω₁. Add a solver wrapper used only by the experiment scripts:
the original feasibility solve, an ℓ¹-minimising variant (for the fewest-terms gauge),
rounding instead of truncation, and an exact A·x = b check.

### Phase 1 — data (this directory)
- `build_stack.py`: one-puncture stack for (red colour, word), and full S_μ for (type, λ, μ).
  Reuse the δ₀ construction of `stable_envelope`, `minuscule_weights`,
  `weight_poset_in_minuscule_rep`. Drive the correction loop yourself with
  `possible_corrections`, `system_on_corrections_lhs`, `system_on_corrections_rhs`,
  `update_differential`, `min_exp_in_square`, so that each order's (A, b) is kept.
- `dump.py`: JSON with the projectives (multi-index, state, multidegree) and every entry as
  (source ε, target ε', braid word, dot exponents, parameter monomial, coefficient), by order.
- Order of cases: A₂…A₅ ω₁ (control, expect no corrections) → the toy word of H3 →
  A₃ ω₂ → A₄ ω₂, D₄ ω₁/ω₃/ω₄ → A₅ ω₂, ω₃, D₅ ω₁ → D₅ spinor, D₆ ω₁. D₆ spinor only to test a
  conjectured formula. All reduced words where affordable.

### Phase 2 — from data to formula
- Test H1, H2, H3, H5 on the data.
- Per order: affine solution space, gauge orbits [δ₀, h], uniqueness up to gauge; choose the
  fewest-terms representative, then refine globally across orders.
- Catalogue correction terms by local type: strands moved, colours, Dynkin adjacency,
  parameter monomial (u versus ħ content), dots, sign. Fit the rule.
- Implement `closed_form_differential(red, word)`. Accept a conjecture only if δ² = 0 holds
  exactly on every computed case **and** on larger cases that were not used to find it, and
  it is gauge-equivalent to the solver output wherever both exist.
- If no formula emerges, report precisely what fails (for instance exact infeasibility of an
  ansatz class, or gauge classes that do not behave uniformly), with the data.

### Phase 3 — gluing maps (stretch)
Compute the cone maps for the small cases, normalise them, look for the analogous closed
form. Cross-check the glued complex against `standard_ebranes` and against RHom to the
simple module (`klrw/simple_module.py`, `KLRWPerfectComplexToEbrane_Homset`).

### Check-ins with the project owner
After the first non-trivial data (A₃ ω₂, D₄ ω₁): show the catalogue of correction terms and
the conjectured rule. Again before spending compute on D₅/D₆, and before Phase 3. The owner
is an expert on the mathematics and wants to be asked for guidance.

### Verification
- δ² = 0 exactly over ℤ for every stored complex.
- Every solver output re-checked by A·x − b = 0 in exact arithmetic.
- Formula against solver: explicit gauge transformation on overlapping cases; formula alone
  on strictly larger ones.

## Start here

1. Check out `claude/youthful-ride-a6dudi` and read this file.
2. Ask the owner for their draft notes on the cup-brane construction (deliberately not in
   the repository); the figure of the stable envelopes and the section on setting up the
   system of equations are the relevant parts.
3. Run Phase 0. Then verify H2 and H3 on the toy word before anything larger — it is the
   cheapest test of whether the reading of the conventions in section 3 is right.
4. Subtasks can be delegated to subagents; the owner asked for Opus-class subagents.
