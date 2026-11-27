# 07 — Towards a proof of Conjecture H (one-step homotopy for the intertwiner comparison)

## RESUME NOTES (keep at the top; updated after every result)

* **Finished:** reading of README_FOR_PROVERS, PROOFS §1–2, NOTES §2, formula.py,
  06_main_intertwiner.md (all), 06_lib.py, 05 §§0–4, 9A, 10, 03 §2 (Thm 2.8, Cor 2.10),
  01 §B.8 (definition of β = klrw basis braid), §E (Theorem E.1).
  Environment check: `sage -python 06_check_step_formula.py A 2 1 1111 1 11111 2 1111` → all
  steps hold (so Conjecture H holds for `c^4, c^5`, red = c and red ≠ c).
* **Engine done:** `checks/07_lab.py` (labelled calculus over the fraction field; expands any
  word, all terms of (★)), `checks/07_star_check.py` ((★) expanded, per class `(I,K,τ)`:
  zero on `111, 1111, 1211, 12121, 2112`, non-zero on the cyclic `1221122` — consistent with
  06), `checks/07_tabulate.py` (prints every product in (★) by `τ`), `checks/07_theta.py`
  (local identity Θ = 0, r ≤ 4).
* **Found (see §1–§3 below):** (a) *moving-strand expansion*: a word in which one position
  moves left past strands `y_1, …, y_n` expands as a sum over hand-over sets, coefficient
  `r^{|S|} ∏ (x_{y_k} − x_{μ_k})^{-1}`; (b) the diagonal component `K = I∖m` of (★) is, for every
  chain `𝔟`, the identity `∏_{b∈𝔟}(A_b − B_b) = 0`, `A_b = x_b − x_m`, `B_b = x_b − x_{μ_b}`, and
  `A_{b_1} = B_{b_1}`; (c) **compact form of h** (key): `(x_y − x_m)ψ − r = ψ·(x_y − x_m)^{after}`,
  hence the sum of the chain terms with a fixed first element `c` is ONE diagram,
  `H_c = ε(−r) Z_c^{src} β(I;(c)) W_c^{tgt}` (dots of the strands crossed by the left slide moved
  from the source to the target); (d) with (c), the off-diagonal components `K = (I∖{m,j})∪{a}`
  vanish: for `a < j` both kinds of product are 0 by 05 Lemma 1.4 (a same-colour pair crosses
  twice), for `a > j` they cancel in pairs (symmetric polynomial slides through `X_j`).
* **Correction to (c):** the dot `x_m` in `W_c` must be read *between* the left slide `LS_c`
  and the right slide `Seg_0` (before `m` moves), not at the target; corrected form verified by
  `checks/07_compact.py` (all `(I,c)` of `11111` red 1, 2; `121121`, `112121`, `1213121`,
  `2131212`, `D_4 2132422`). Equivalently `H_c = ε(−r)·LS_c·ζ̂_c·Seg_0(m→c)`,
  `ζ̂_c = ∏_{a∈Same∖c}(x_a − x_m)` read between the two slides.
* **Group splitting (`checks/07_groups.py … adjj`):** for H1 words (one colour `c` repeated,
  all others once: `1211`, `112111`, `1121311`, `1312131`, `D_4 2132422`) (★) splits into groups
  that vanish separately: "diag" (`K = I∖m`), "same" (mover of colour `c`, `a<j` / `a>j`),
  "far" (mover of a colour not adjacent to `c`), "adj-by-j" (all terms whose prefix term has a
  mover `j` of colour adjacent to `c`, any chain start, plus `ζ·P(j; F∪(a,m))`). Only the last
  group involves trees and bigons. The bigon divided difference `Q(x_a,y) − Q(x_b,y) =
  t(x_a − x_b)` of 06 §6 is visible there (`07_tabulate.py A 2 2 1211`).
* **Doing right now:** writing up the one-colour case (Theorem H1a) completely; then the groups
  "same"/"far" with spectators, then "adj-by-j".
* **Next steps planned:** H1 with spectators; then H2 (hanging forests: the compact form must
  be extended to sums over chains *and* forests).
* **Scripts / logs:** as listed above.

---

## 0. Setting and two preliminary observations (not yet used)

Notation of `06_main_intertwiner.md`: `w = w′·c`, `m` the last slot,
`Same = {s < m : c(s) = c}`, `N = |Same|`, `ζ_m = ∏_{a ∈ Same}(x_a − x_m)` (labelled dots at the
source). For `I ∋ m`, `K ∌ m`, `|K| = |I| − 1`, (★) reads (06, `06_divide.py`)

    Φ_m(I)·[K = I∖m]  −  Σ_{J∋m} ι_R(d′)(I→J)·h(J→K)  −  Σ_{J∌m} h(I→J)·ι_L(d′)(J→K)  =  ζ_m · F(I→K),     (★)

where `ι_R(d′)` = tree differential of `w′` with the strand `m` added on the far right (the
components of the tree formula of `w` inside `C_R` are `−ι_R(d′)`), `ι_L(d′)` the same with
the strand on the far left, `F` = the tree terms of `w` from `I ∋ m` to `K ∌ m` (the move `X_m`
and the trees with mover `j < m` containing a pair `(a, m)`), and `h` the formula of
Conjecture H (06 §7.2).

**Observation 0.1 (all parameters may be set to 1 when checking a homogeneous identity).**
Let `E ∈ e A e′` be homogeneous for the grading (F2). Write `E = Σ_{α,σ} c_{α,σ} μ_{α,σ} x^α β_σ`
in the basis (F1), `μ` monomials in `r, t, u`. The degree map from monomials in `r, t, u` to the
grading group is injective (σ_c only in `r_c`; ε_{cW} only in `u_c`; ε_{cc′} gives
`m_{cc′}+m_{c′c}`; δ_c gives `Σ_{c′} m_{cc′}`; on a forest these determine the `m`, as in the
uniqueness part of `PROOFS.md` Cor. 3.1). So for fixed `(α, σ)` at most one monomial occurs, and
`E = 0` iff its specialisation at `r = t = u = 1` vanishes. (To be used only for computations.)

**Observation 0.2 (polynomial-representation picture).** In the localisation, for two strands
of equal colour, `ψ = r (x_a − x_b)^{-1}(1 − φ̂)` with `φ̂ = −φ/r`, `φ̂² = 1`; i.e. `ψ` is `r`
times a Demazure operator and `φ` is `−r` times the transposition. So `X_m` is a product of
Demazure-type operators and `Φ_m` the corresponding product of transpositions (up to `(−r)^N`
and the crossings of other colours). (06 §9 is the rigorous form of this localisation.)

## Status

Nothing proved yet in this file.
