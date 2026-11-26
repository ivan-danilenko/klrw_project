# Gluing the stable envelopes: the full cup (I-) brane

Status marks as in `NOTES.md`: **[verified]** = exact computation, **[conjecture]** =
extrapolation. Conventions of `klrw/standard_ebranes.py`: all scalings on, parameters
invertible (Laurent ring), gluing along the linear extension of the weight poset, each step
a cone over the unique Ext class.

## 1. Data

`glue/glue.py` re-implements `standard_ebranes` step by step without Gurobi (solver patches
in `glue/patches.py`, applied by monkeypatching; nothing under `klrw/` is changed) and
records every gluing map. Glued complexes in `glue/data/`:

| case | envelopes | projectives | terms (gluing) | remarks |
|---|---|---|---|---|
| `sl_2 … sl_6`, `ω_1` | 2 … 6 | 4, 12, 32, 80, 192 | 4, 24, 96, 320, 960 | chain map unique, no homotopies |
| `sl_4`, `ω_2` | 6 | 96 | 465 (265) | dots and `r^{±1}` appear |
| `sl_5`, `ω_2` | 10 | 640 | 6294 (4238) | coefficients ±1, ±2 |
| `D_4`, `ω_1` | 8 | 512 | 4598 (2966) | coefficients ±1, ±2 |

**[verified]** for every case: the Ext group is one-dimensional at every step; the dump
rebuilds with `d² = 0`; all coefficients are integers; after rescaling each envelope by a
unit monomial all exponents are non-negative and the complex is defined over the polynomial
(non-inverted) algebra over ℤ. For `sl_2 … sl_5 ω_1` and `sl_4 ω_2` the result equals the
output of the real `standard_ebranes`.

General observations **[verified on these cases]**:

* `u` enters exactly here: `u_i` for the colours of the red strands occurs in the gluing maps
  of every case except `sl_2`, and cannot be removed by homotopies or by rescaling the
  envelopes. (Inside a stable envelope `u` never occurs.)
* No gluing component is an identity map.
* `r` and dots occur in the gluing maps only when a colour repeats (`ω_2`, `D_4`).
* For `ω_2` and `D_4` the chain map is far from unique (large spaces of null-homotopic maps),
  so a normal form has to be chosen before looking for a formula. Not done yet.

## 2. Closed formula for `sl_{n+1}`, `ω_1`

Here every colour occurs once, the stable envelopes are uncorrected cubes, and the gluing
maps are unique. The whole I-brane has `(n+1)·2ⁿ` projectives.

**Vertices.** `(k; A, C)` with `0 ≤ k ≤ n`, `A ⊆ {1,…,k}`, `C ⊆ {k+1,…,n}`; `B` = the other
colours. The idempotent is

    [A, decreasing]  W_1  [B, increasing]  W_n  [C, decreasing].

`k` labels the envelope (`k` strands around the first puncture). Homological degree
`k + #{c ≤ k : c ∈ B} + #C`. Each colour is in one of four states: `A`; `B_L` (`c ≤ k`, in
`B`); `B_R` (`c > k`, in `B`); `C`.

**Terms out of `(k; A, C)`.**

* *internal:* one colour `c ≤ k` moves `B → A`, or one colour `c > k` moves `C → B`;
* *gluing to the envelope `l = k − d`, `d ≥ 1`* — the colours `l+1, …, k` change stack:
  * (i) if they are all in `A`: all go to `C`, except `k`, which goes to `B`;
  * (ii) if `l+1 ∈ B` and the others are in `A`: all go to `C`;
  * (iii) if `l+1 ∈ B` and the others are in `A`: `l+1 → C`, `k → B`, the others `→ C`
    (for `d = 1`: `k` stays in `B`), and one further colour moves one region to the right —
    any `c ≤ l` from `A` to `B`, or any `c > k` from `B` to `C`.

Each term is the unique reduced diagram between the two idempotents, without dots.

**Coefficients.** Put `τ_x = t_{x,x−1} / t_{x,x+1}` with `t_{1,0} := u_1`, `t_{n,n+1} := u_n`.

| term | coefficient |
|---|---|
| internal, (i) | 1 |
| (ii) | `∏_{x=l+1}^{k} τ_x` |
| (iii), extra colour `c ≤ l` | `∏_{x=c}^{l} τ_x^{−1}` |
| (iii), extra colour `c > k` | `∏_{x=l+1}^{c} τ_x` |

(This is the normalisation of `standard_ebranes`; the coefficients are the unique Laurent
monomials of the right multidegree once the internal terms and the terms (i) with `d = 1`
have coefficient 1.)

**Signs.** Exterior-algebra rule: give every colour two odd generators `a_c, g_c`, ordered
`a_1, …, a_n, g_n, …, g_1`, and let the state of colour `c` be `1` (`B_R`), `a_c` (`A`),
`g_c` (`C`), `a_c g_c` (`B_L`). With contractions `ι`, wedging `ε` and `ρ_c` = "replace `a_c`
by `g_c`", acting in the order written (rightmost first):

| term | operator |
|---|---|
| internal | `ι(g_c)` |
| (i) | `(−1)^{d+1} · ι(a_k) ∏_{l<c<k} ρ_c` |
| (ii) | `∏_{l+1<c≤k} ρ_c · ι(a_{l+1})` |
| (iii), `d ≥ 2` | `± ε(g_c) ι(a_k) ∏_{l+1<x<k} ρ_x ι(a_{l+1})` |
| (iii), `d = 1` | `± ε(g_c) ι(g_k) ι(a_k)` |

with `± = (−1)^{(l+1−c)+d}` for an extra colour `c ≤ l` and `(−1)^{c−k}` for `c > k`. In this
basis the envelope `S_k` is multiplied by `(−1)^{k(k−1)/2}` relative to `standard_ebranes`.

**Evidence.**

* The formula reproduces the glued complexes for `sl_3 … sl_6` exactly — terms, signs and
  coefficients (24, 96, 320, 960 terms; `glue/omega1_formula.py`) **[verified]**.
* Built from the formula alone, the complex is homogeneous and satisfies `d² = 0` for
  `sl_7`, `sl_8`, `sl_9` (448, 1024, 2304 projectives; 2688, 7168, 18 432 terms), none of
  which were used to find it (`glue/omega1_build.py`) **[verified]**.
* That it is the cone of the unique Ext classes for `n ≥ 6`, and the formula for all `n`, is
  a **[conjecture]**.

Remarks: per colour the four states `B_L → A ⊕ C → B_R` form the three-term resolution of
the `sl_2` cup; the I-brane is the part of the tensor product of these `n` complexes in which
the colours in `{A, B_L}` precede those in `{B_R, C}`, with the higher terms (i)–(iii) as
corrections. The number of components `S_k → S_l` depends only on `n` and `d = k − l`.

## 3. `ω_2` and `D_4`: work over the polynomial algebra

Over the Laurent ring the gluing maps for `ω_2` and `D_4` are defined only up to large
spaces of homotopies. Rescaling each envelope by a unit monomial makes the glued complex
polynomial, and **over the polynomial algebra most of the freedom disappears**
(`glue/glue_poly.py`, base ℚ, parameters not inverted) **[verified]**:

| `sl_4`, `ω_2`, step (element that changes stack) | chain maps | null-homotopic |
|---|---|---|
| 1 (`p1`, colour 2) | 7 | 6 |
| 2 (`p3`, colour 3), 3 (`p2`, colour 1), 4 (`p3`) | 1 | 0 |
| 5 (`p4`, colour 2) | 7 | 6 |

(`p1 … p4` are the elements of the 2×2 minuscule poset, colours 2, 1, 3, 2.) With the
opposite sign of the shift there is no chain map at all, so the polynomial normalisation is
forced.

In poset language (`glue/poset_table.py`, `glue/poly_step.py`) every element is in one of
four states — `A`, `B_L` (in the ideal: left / right of the first red strand), `B_R`, `C`
(not in the ideal: left / right of the second red strand) — exactly as in the `ω_1` formula.
The unique maps of steps 2–4, where the element `p` that changes stack has a colour occurring
once, have the `ω_1` shape:

* (i) `p: A → B_R` and (ii) `p: B_L → C`, each with coefficient `u_2`;
* (iii) `p: B_L → B_R` together with one neighbour of `p` in the poset moving one region to
  the right (`p1: A → B_L` or `p4: B_R → C`), with coefficient `t_{2,c(p)}`.

Steps 1 and 5 (the element changing stack has colour 2, like its partner in the other stack)
contain the same three kinds of terms with coefficients `1`, `r_2² t_21 t_23`, … and, in
addition, terms of order `r_2` in which `p1` and `p4` exchange labels, some with one dot;
there a 6-dimensional family of homotopies remains and a normal form is still to be chosen.

The same computation for the larger cases (chain maps / null-homotopic maps at steps 1, 2, …,
over the polynomial algebra; Ext is one-dimensional at every step):

* `D_4`, `ω_1`: 173/172, 59/58, 1/0, 1/0, 1/0, 33/32, 228/227
* `sl_5`, `ω_2`: 208/207, 58/57, 1/0, 70/69, 79/78, 21/20, 94/93, 87/86, 230/229

So the gluing map is unique only at a few steps (3 of 7 for `D_4`, 1 of 9 for `sl_5 ω_2`);
elsewhere large families of homotopies remain even over the polynomial algebra, and a
normal form is needed.

## 4. Open

* A rule for the steps in which the element changing stack has a repeated colour, and for the
  components between non-adjacent envelopes; then the same for `sl_5 ω_2` and `D_4`.
* The memory use of the Ext computation (about 16 GB for `sl_5 ω_2`) limits the brute-force
  gluing to roughly these sizes.
