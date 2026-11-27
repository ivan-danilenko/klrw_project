#!/bin/zsh
cd "$(dirname "$0")/../.."
s=900
for T in "A 5 10" "A 6 11" "D 5 10" "D 6 11" "E 6 11" "E 6 12"; do
  set -- ${=T}
  for P in "" nb red twinred branch k2; do
    [ $1 = A ] && { [ "$P" = branch ] || [ "$P" = k2 ]; } && continue
    s=$((s+1)); sage -python R05b_d2.py sample $1 $2 20 $3 40 $s 4 $P | grep -E "RESULT|^FAIL"
  done
done
