# Brief for the proof effort

Goal: turn the computed rules for the cup-brane resolution into theorems with complete,
human-checkable proofs (eventual target: formalisation in Lean), in the greatest generality
that is actually true. Read first, in this order:

1. `../PROOFS.md` — the setting (section 1), what is already proved (section 2), what is
   only computed (3) and what is conjectured (4). Use its notation.
2. `../NOTES.md` section 2 — the tree formula. `../formula.py` is its implementation
   (`Stack(ct, colours).terms_from(I)`), and is the authoritative statement of signs.
3. `../GLUING.md` section 2 if you work on the I-brane.

## Conventions, with exact signs (all checked in the code: `checks/alg.py`)

Product `a*b` = first `a`, then `b`; a diagram is read from its *left state*. Crossings are
`ψ_i` (positions `i, i+1`, 1-based). Dots are indexed by the position in the state on the
side where they are written.

* same colour `c`: `ψ² = 0`; `x_1 ψ − ψ x_2 = r_c`, `ψ x_1 − x_2 ψ = r_c` (the dot stays on
  its strand; on the state `(c, c)`). Hence `ψ(x_1−x_2) + (x_1−x_2)ψ = 2 r_c`.
* adjacent colours: `ψ² = t_{cc'} x_c + t_{c'c} x_{c'}` (either order of the strands);
  non-adjacent different colours: `ψ² = 1`; dots slide freely through crossings of different
  colours and through red–black crossings.
* red strand of colour `k`: `ψ² = u_k x` for a black strand of colour `k`, `ψ² = 1` otherwise.
* braid relation `ψ_1ψ_2ψ_1 = ψ_2ψ_1ψ_2` holds on every state except
  * state `(c, c', c)` with `c ∼ c'`: `ψ_1ψ_2ψ_1 − ψ_2ψ_1ψ_2 = + r_c t_{cc'} · e`,
  * state `(k, W_k, k)`: `ψ_1ψ_2ψ_1 − ψ_2ψ_1ψ_2 = + r_k u_k · e`.
  (So it is exact on `(c, c, c')`, `(c', c, c)`, `(W, k, k)`, `(k, k, W)`.)
* intertwiner for two strands of the same colour: `φ = (x_1 − x_2)ψ − r_c = −ψ(x_1 − x_2) + r_c`
  satisfies `φ² = r_c²` and `x_1 φ = φ x_2`, `x_2 φ = φ x_1` (checked).
* `r_c` can be absorbed over `ℤ[r^{±1}]` by rescaling the same-colour crossings, so for
  invertible `r` this is the usual KLR / Webster algebra with `Q_{cc'}(x, x') = t_{cc'}x + t_{c'c}x'`;
  `r = 0` is the degenerate algebra in which `δ₀² = 0`.

`checks/alg.py` gives one-line access to the algebra:

    sage -python yourscript.py        # never plain python for anything that touches klrw
    from alg import Alg
    A = Alg(["A", 2], "1 W1 2 1")     # colours left to right, W<k> = red strand
    A.word("1 W1 2 1", (3, 2, 3)); A.dot("1 W1 2 1", 4); A.e(...); A.r(1); A.t(1, 2); A.u(1)

Tools for the cube: `../cupbrane.py` (`make_algebra`, `Cube`), `../fast.py` (`braid_for(cube, I, J, sigma)`
= the KLRW basis braid of a term), `../verify.py` (full `d² = 0` from the formula),
`../sample_verify.py` (`element`, `check_vertex`), `../events.py` (a term as a product of
strand slides), `../cyclefree.py` (`find_cycle`), `../degree_model.py`. Pure-Python parts
(`formula.py`, `cyclefree.py`) run with plain `python3`.

## New facts since `PROOFS.md` was written

* **The tree formula is not special to minuscule words.** On 200 random words (arbitrary
  colour sequences, also with equal neighbours; types `A_2`, `A_3`, `D_4`, up to 9 strands;
  `checks/general_words.py`) the tree formula satisfies `d² = 0` exactly when the word has no
  cyclic configuration (197 words), and fails for the 3 words that have one
  (e.g. `A_2`, red 2, word `1221122`). So the target statement is:

  > **Conjecture C1′.** For *every* word without cyclic configurations, the tree formula
  > satisfies `d² = 0`.

  Proofs should therefore not use minuscule-ness unless they must, and should say where.
* If every colour occurs at most twice, the link graph of any configuration embeds in the
  Dynkin diagram (pairs have pairwise different colours, different from the mover's), so
  there are no blocks and no cycles. This case contains `sl_n` with `ω_1, ω_2`, the vector
  representation of `so_{2n}`, and all of `D_4`.
* **Cone structure.** Let `w = w'·c` (last slot `m`, colour `c`). Vertices with `m ∉ I` form
  the cube of `w'` with an extra strand at the far left, `C_L = 1 ⊗ C'`; those with `m ∈ I`
  form `C_R = C' ⊗ 1` (extra strand at the far right; the tree differential there is `−d'`).
  The remaining terms `F : C_R → C_L` are: the elementary move `X_m` of slot `m` (it crosses
  everything; a tree with mover `m` has no pairs, because a pair linked to the mover would
  need `b_p > m`), and the trees with a mover `j < m` that contain a pair `(a, m)`.
  `d² = 0` for `w` ⟺ `d'² = 0` and `F` is a chain map (with the signs of the formula).
* **Intertwiners give a chain map for free.** Replace in `X_m` every crossing of strand `m`
  with a strand of its own colour by the intertwiner `φ`. The resulting element `Φ` should
  satisfy `(a ⊗ 1)·Φ = Φ·(1 ⊗ a)` for every element `a` of the algebra of `w'` (Kang–Kashiwara–Kim,
  renormalised R-matrices; to be checked for this presentation, including the red strand), so
  `Φ : C' ⊗ 1 → 1 ⊗ C'` is a chain map as soon as `d'² = 0`. Its degree is that of `F` plus
  `N·deg(dot)`, `N` = number of earlier slots of colour `c`, and
  `z = ∏_{y<m, c(y)=c} (x_m − x_y)` is central for `C'` and a non-zero-divisor (basis theorem).
  Heuristically `F = Φ / z`. Precisely: **if `Φ = F·z + (d h + h d)` for some `h`, then `F` is a
  chain map.** A first check by hand for `N = 1` suggests `h = ± r_c · (pure exchange of the
  labels `a` and `m`, no mover)` plus tree-decorated versions. Not yet verified by computer.

## Standards

* Write for a reader who will formalise the argument: numbered definitions, statements and
  proofs; no "similarly" for a case that is not literally the same; say which relations and
  which standing facts each step uses.
* **Check every non-trivial identity by an exact computation** in small cases before relying
  on it, including signs. Put the scripts in `proofs/checks/` (prefix them with your file's
  number) and name the script next to the claim it checks. A claim that survives no check
  must not be stated as proved.
* Be honest about the result. End your file with a section **Status** listing: proved
  completely; proved assuming (list exactly what); verified only by computation; failed
  attempts and why they failed. A clearly located gap is far more useful than an argument
  that hides it. If you find that a statement in `PROOFS.md`, `NOTES.md` or this brief is
  wrong, say so prominently.
* Standing facts you may use without proof, stating where: the basis theorem (F1) and what
  follows from it (the algebra is free as a left and as a right module over the polynomial
  ring in the dots, so multiplication by a non-zero polynomial in the dots is injective),
  the grading (F2), Matsumoto's theorem (F3).
* Write only inside `proofs/` (your own file `proofs/NN_name.md` and scripts in
  `proofs/checks/`). Do not modify anything else, in particular nothing under `klrw/`, and
  do not commit. Other agents work in the same tree at the same time on other files.
* Long computations: keep each run under a few minutes where you can; the cube of a word
  with 9–10 strands is still fast, the algebra itself is slow beyond ~12 strands.
