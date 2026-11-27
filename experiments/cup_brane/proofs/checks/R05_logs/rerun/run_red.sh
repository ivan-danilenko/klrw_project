#!/bin/zsh
cd "$(dirname "$0")/../.."
sage -python R05_reduction.py 40 7 31 T1
sage -python R05_reduction.py 40 6 32 any
sage -python R05_redlin.py A 3 1 31213 1 3
sage -python R05_redlin.py D 4 2 42142 1 4
sage -python R05_redlin.py A 2 2 21221 1 5
sage -python R05_redlin.py A 3 3 123213 0 6
sage -python R05_redlin.py D 4 3 312432 0 7
