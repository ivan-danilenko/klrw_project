#!/bin/zsh
# Reproduces every run quoted in proofs/05_main_sliding.md.  Usage:  zsh 05_logs/run_all.sh   (from proofs/checks)
run() { local name=$1; shift; { echo "\$ sage -python $*"; sage -python "$@"; echo "exit code $?"; } > 05_logs/$name.log 2>&1; }
(
run straighten 05_straighten.py 3000 1
run sliding 05_sliding.py 300 1
run cone_A3 05_cone.py A 3 6 40 1
run cone_D4 05_cone.py D 4 6 30 2
run reduction_A3 05_reduction.py A 3 40 6 1
run reduction_D4 05_reduction.py D 4 25 6 2
run fourterm_A3 05_fourterm.py A 3 40 7 1
run fourterm_D4 05_fourterm.py D 4 40 7 2
run T1_special 05_T1_check.py D 4 3421342 1342134 2
run T1_A3 05_T1_check.py A 3 all 2 6
run explore_stats_A3 05_explore_stats.py A 3 3 6 T1
run T1_A4 05_T1_check.py A 4 all 5 6
) &
(
run T1_D4 05_T1_check.py D 4 all 2 6
run T1_D4_long 05_T1_check.py D 4 all 7 8 60 1
run blocks_A3_N1_a1 05_blocks_general.py A 3 1 80 8 1 a1
run blocks_D4_N1_a1 05_blocks_general.py D 4 1 60 8 2 a1
run blocks_A3_N1 05_blocks_general.py A 3 1 80 8 1
) &
(
run T1_A5_long 05_T1_check.py A 5 all 7 9 40 2
run T1_D5_long 05_T1_check.py D 5 all 8 9 30 3
) &
(
run blocks_A3_N2 05_blocks_general.py A 3 2 80 8 3
run blocks_A3_N2_a1 05_blocks_general.py A 3 2 80 8 3 a1
run closure_A3_N1 05_closure.py A 3 1 60 8 dots 1
run closure_A3_N2 05_closure.py A 3 2 40 7 dots 2 shapes
run closure_A3_N2_a1 05_closure.py A 3 2 40 7 dots-a1 2
run closure_A2_N2 05_closure.py A 2 2 40 7 dots 1
run closure_A2_N2_a1 05_closure.py A 2 2 40 7 dots-a1 1
run closure_D4_N2 05_closure.py D 4 2 25 7 dots 6
run closure_D4_N2_a1 05_closure.py D 4 2 25 7 dots-a1 6
run closure_A2_N3 05_closure.py A 2 3 20 8 dots 3
run closure_A2_N3_a1 05_closure.py A 2 3 20 8 dots-a1 3
run closure_A2_N3_aN 05_closure.py A 2 3 20 8 dots-aN 3
run closure_A3_N3_a1 05_closure.py A 3 3 20 8 dots-a1 5
) &
wait
echo "all runs finished" > 05_logs/DONE
