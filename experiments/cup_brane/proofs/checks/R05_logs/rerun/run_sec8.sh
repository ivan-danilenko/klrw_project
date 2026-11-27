#!/bin/zsh
cd "$(dirname "$0")/../.."
sage -python R05b_sec8.py D 4 60 7 11 3 k2
sage -python R05b_sec8.py D 4 40 8 12 3 k2
sage -python R05b_sec8.py D 5 20 9 13 3 k2
sage -python R05b_sec8.py E 6 20 9 14 3 k2
sage -python R05b_sec8.py A 5 20 9 15 3 red
sage -python R05b_sec8.py D 5 20 9 16 3 twinred
sage -python R05b_sec8.py A 4 20 8 17 3 nb
