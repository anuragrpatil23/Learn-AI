#!/bin/bash
# The settings of the paper's two demonstrations, each as a run in the tracker's project "toy-superposition".
# Usage: ./run_paper.sh [python]
PY=${1:-python}
OUT=~/run-tracker-data/runs/toy-superposition
# 5 things into 2 neurons, importance falling by 0.7 each: things always present, then rarer
for P in 1.0 0.2 0.1; do $PY toy.py --things 5 --neurons 2 --importance 0.7 --present $P --out $OUT/n5_m2_present$P | tail -2; done
# the same at its rarest without the ReLU
$PY toy.py --things 5 --neurons 2 --importance 0.7 --present 0.1 --no-relu --out $OUT/n5_m2_present0.1_linear | tail -2
# 20 things into 5 neurons, importance falling by 0.7 each, from always present to one time in a thousand
for P in 1.0 0.3 0.1 0.03 0.01 0.003 0.001; do $PY toy.py --things 20 --neurons 5 --importance 0.7 --present $P --out $OUT/n20_m5_present$P | tail -2; done
