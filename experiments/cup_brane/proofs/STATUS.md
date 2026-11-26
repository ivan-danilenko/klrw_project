# Proof effort: state of play

Last updated 2026-11-26, shortly before a usage-limit interruption. Everything named here is
committed on the branch `claude/youthful-ride-a6dudi`. `README_FOR_PROVERS.md` is the brief
every prover and referee was given.

## Fronts

| # | file | content | state |
|---|---|---|---|
| 01 | `01_foundations.md` | presentation with exact signs; basis theorem over ℤ[r,t,u] (one red strand); torsion-freeness; intertwiners (exact braid relations, naturality); when the diagram of a tree term is independent of the reduced word; referee pass on Tier 1 of `../PROOFS.md` | finished; **not refereed** |
| 02 | `02_omega1_ibrane.md` | `ω_1` I-brane: `d² = 0` for all `n`, homogeneity, iterated cone (rank-one Hom), resolution over the Laurent ring | refereed (`R02_referee.md`: no fatal error, 3 repairable points); **author is applying the repairs** (see "REVISION NOTES" at the top of the file, if present) |
| 03 | `03_cyclic_combinatorics.md` | cyclic ⇔ one of three patterns T, B, N; type A (4,4,4,3); type D (5,4,3,1); exhaustive `E_6`, `E_7`; Deodhar comparison | refereed (`R03_referee.md`: nothing wrong); **author is applying one repair and cosmetic fixes** |
| 04 | `04_first_order.md` | `d² ≡ 0 mod (r)²` for every word; forcedness and gauge freedom at first order; the 7-strand cyclic word `1221122` | finished; **referee running** (`R04_referee.md`) |
| 05 | `05_main_sliding.md` | `d² = 0` when every colour occurs at most twice (Theorem B); one colour three times, others once (Theorem C); sliding lemma, cone decomposition, reduction lemma for arbitrary words | finished; **referee running** (`R05_referee.md`; scripts `checks/R05_*.py` of an earlier interrupted referee exist) |
| 06 | `06_main_intertwiner.md` | intertwiner cube `C_Φ`, comparison map `G` with `d_Φ G = G d` (second proof for colours ≤ 2); Conjecture H (explicit one-step homotopy for all cycle-free words, implies C1′); division in the first cyclic step | finished; **referee running** (`R06_referee.md`) |
| 07 | `07_conjecture_H.md` | proof of Conjecture H in stages H1 (chains only) → H2 → H3 (multiplicities ≤ 3, non-cyclic) → H4 (all non-cyclic words) | **prover running** |

## To do after the running agents finish

1. Apply referee repairs to 04, 05, 06 (as for 02, 03); referee 01 and 07.
2. Rewrite `../PROOFS.md` from the refereed fronts: move statements between tiers, fix the
   basis-theorem reference, state C1′ (arbitrary words avoiding T, B, N) and Conjecture H.
3. Corrections to `../NOTES.md`: §3 uniqueness holds only up to gauge for general words
   (04, Example 8.3); §4 range `4 ≤ k ≤ n−4`, pattern N, "same shape" claim; §2 the diagram of
   a term is the `klrw` basis braid (independent of the reduced word only in the cases of
   01, Theorem B.6). `../GLUING.md`: remark about the tensor product is misleading (02).
4. Open mathematics: Conjecture H for multiplicity ≥ 3 (07); the differential after the first
   cyclic step (06 §8); exactness of the cube over the base ring (06 §9); gluing maps for
   `ω_2`, `D_4`.

## How to resume an interrupted agent

Each prover/referee was told to keep a "RESUME NOTES" / "REVISION NOTES" section at the top of
its file. A fresh agent (model: Opus) given `README_FOR_PROVERS.md`, the file in question and
its `checks/NN_*.py` scripts can continue from those notes.
