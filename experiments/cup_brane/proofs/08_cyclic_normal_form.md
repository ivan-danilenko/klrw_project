# 08 — A normal form of the differential for cyclic words

## RESUME NOTES (keep at the top; updated after every result)

* **Finished:**
  * `checks/08_comb.py` (pure Python): word data, degree calculus (crossing content, Prop. 7.1 of 04),
    DFS enumeration of all dot-free homogeneous basis elements of any homological degree
    (reproduces 04 Thm 9.1(a): 2447, 1464, 204, 0, 7 for `1221122` red 2; hom. degree 0:
    290 at `r¹`, 8 at `r²`, as R04).
  * `checks/08_alg.py` (sage): maps on the cube, compose, vectorise; `d_T²` of `1221122` red 2
    in 1 s (28 coefficients at `r³`, 14 at `r⁴`, as R04).
  * `checks/08_solve.py`: complete solution set of `δ₀ + δ₁ + ε`, `ε` dot-free of `r`-degree ≥ 2,
    `d² = 0` (deferred-kernel order-by-order solver, no minimisation). Reproduces `d_F` of
    04 Thm 9.1(e) for `1221122` red 2 exactly (unique; 692 terms; 12 extra) in 5 s.
* **Doing now:** batch over all cyclic words of length 7 (A_3 classes, all reds) and length 8 (A_2).
* **Next:** classify the extra terms; conjecture a rule; test at the minuscule thresholds.
* **Where things are:** scripts `checks/08_*.py`, data `checks/08_data/`, logs `checks/08_logs/`.
