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
  * **Length 7, all 8 heap classes × all reds (28 cases), `08_batch.py len7`, log `08_logs/batch_len7.log`:**
    a dot-free completion of `δ₀+δ₁` exists (and is unique) only for `1221122`, `2112211`
    (pattern N, same-colour pair *below* the mover) and `2132213` (pattern T); it does **not**
    exist for `1122112`, `2211221`, `1221322`, `2213221` (N) and `1322132` (B): inconsistent at
    `r³`. Independent of the red colour. Same answer with two other reduced-word bases
    (`--basis left|right`, checked on `1221122`, `1322132`).
  * **Normal form `N_T`** (tree terms with ≤ 2 pairs fixed, complete dotted ansatz at `r ≥ 3`,
    `08_dotted.py`, `08_batch.py len7 --dotted`, log `08_logs/batch_len7_dotted.log`): exists and
    is **unique (kernel 0 at every order)** in all 28 cases; keeps every tree term of `r ≥ 3`;
    extra terms: 10/7/9/5/10/8/7/5 (classes in the order of the log), all at the cyclic
    configuration's own component plus a few hand-over terms. Data `08_data/*_dotted_R3.json`.
* **Doing now:** A_2 length-8 dot-free batch (`08_logs/batch_a2len8.log`); then `N_T` batch at length 8.
* **Next:** closed formula for the extra terms of `N_T`; gauge between `N_T` and the dot-free form
  where both exist; test at the minuscule thresholds.
* **Where things are:** scripts `checks/08_*.py`, data `checks/08_data/`, logs `checks/08_logs/`.
