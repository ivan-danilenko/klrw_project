#!/bin/zsh
# referee: sampled d^2 = 0 on long words (multiplicities <= 2)
cd "$(dirname "$0")/.."
T=$1; R=$2; S=$3
for L in 9 10 11; do
  [ $L -le $((2*R)) ] || continue
  sage -python R05_d2.py sample $T $R 12 $L 30 $((S+L))
  sage -python R05_d2.py sample $T $R 8 $L 30 $((S+L+100)) 2 nb
  sage -python R05_d2.py sample $T $R 8 $L 30 $((S+L+200)) 2 red
done
