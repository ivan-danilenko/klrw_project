# 07 — Towards a proof of Conjecture H (one-step homotopy for the intertwiner comparison)

## RESUME NOTES (keep at the top; updated after every result)

* **Finished:** reading of README_FOR_PROVERS, PROOFS §1–2, NOTES §2, formula.py,
  06_main_intertwiner.md (all), 06_lib.py, 05 §§0–4, 9A, 10, 03 §2 (Thm 2.8, Cor 2.10),
  01 §B.8 (definition of β = klrw basis braid), §E (Theorem E.1).
  Environment check: `sage -python 06_check_step_formula.py A 2 1 1111 1 11111 2 1111` → all
  steps hold (so Conjecture H holds for `c^4, c^5`, red = c and red ≠ c).
* **Doing right now:** writing `checks/07_lab.py`, a symbolic engine for the labelled
  calculus over the fraction field (elements = dict labelled target state → rational
  function in the labelled dots), able to expand any word of crossings (ψ for same colour via
  `ψ = (x_a − x_b)^{-1}(φ + r)`), and to tabulate every product occurring in (★) by
  resulting labelled permutation `τ`.
* **Next steps planned:** (1) tabulate (★) for `c^3, c^4` (stage H1, one colour) and for
  `c^N` plus single other colours; (2) find the cancellation pattern; (3) prove H1;
  (4) then H2, H3.
* **Scripts / logs:** none yet beyond the 06 scripts.

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
