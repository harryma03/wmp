#!/usr/bin/env bash
set -euo pipefail
cd /home/aiseon/wmp
python_bin=/home/aiseon/anaconda3/envs/wmp/bin/python
export LD_LIBRARY_PATH=/home/aiseon/anaconda3/envs/wmp/lib:${LD_LIBRARY_PATH:-}
export WMP_SMALL_PHYSX=1 WMP_SKIP_EXPORT=1
for seed in 1 2 3 4 5; do
  for mode in isolated flat; do
    export WMP_GAP_AB="$mode"
    export WMP_PLAY_TRACE_PATH="/home/aiseon/wmp/logs/wmp_alignment/gap_ab/${mode}_${seed}.jsonl"
    "$python_bin" -u legged_gym/scripts/play.py --task=titatit --sim_device=cuda:0 --terrain=gap --checkpoint=21000 --num_envs=1 --headless --seed="$seed" > "logs/wmp_alignment/gap_ab/${mode}_${seed}.log" 2>&1
  done
done
python3 logs/wmp_alignment/summarize_gap_ab.py
