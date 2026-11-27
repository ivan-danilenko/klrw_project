#!/bin/zsh
# referee pass 2: sampled d^2 = 0 (formula.Stack + klrw basis braid) on long T1 words
cd "$(dirname "$0")/../.."
NP=${1:-4}
s=500
for T in "A 5" "A 6" "D 5" "D 6" "E 6"; do
  set -- ${=T}
  R=$2
  for L in 9 10 11; do
    [ $L -le $((2*R)) ] || continue
    for P in "" nb red twinred; do
      s=$((s+1)); sage -python R05b_d2.py sample $1 $R 6 $L 25 $s $NP $P | grep -E "RESULT|FAIL"
    done
    if [ $1 != A ]; then s=$((s+1)); sage -python R05b_d2.py sample $1 $R 6 $L 25 $s $NP branch | grep -E "RESULT|FAIL"; fi
    if [ $1 != A ] && [ $L -ge 7 ]; then s=$((s+1)); sage -python R05b_d2.py sample $1 $R 6 $L 25 $s $NP k2 | grep -E "RESULT|FAIL"; fi
  done
done
