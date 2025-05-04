#!/usr/bin/env bash
mkdir -p logs
export PYTHONWARNINGS=ignore

impls=(push_pid.py push_mpc.py)

for impl in "${impls[@]}"; do
  log="logs/${impl%.py}.csv"
  > "$log"                     # truncate
  for i in {1..60}; do
    echo "▶ Running $impl  trial $i"
    TRIAL_IDX=$i python -W ignore "$impl" >>"$log" 2>/dev/null
  done
done
