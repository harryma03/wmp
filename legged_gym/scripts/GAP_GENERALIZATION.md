# Independent gap generalization fine-tuning

Run from the repository root with the `wmp` environment:

```bash
LD_LIBRARY_PATH=/home/aiseon/anaconda3/envs/wmp/lib:${LD_LIBRARY_PATH:-} \
/home/aiseon/anaconda3/envs/wmp/bin/python -u legged_gym/scripts/train_gap_generalization.py \
  --task=titatit --sim_device=cuda:0 --wm_device=cuda:0 --headless \
  --num_envs=2048 --max_iterations=1000 --seed=42
```

Creates a new timestamped run; refuses to overwrite an existing directory.
Restores actor/critic, world model, depth predictor, PPO optimizer and world
model optimizer from WMP/model_21000.pt. Depth predictor optimizer and replay
buffers restart because the original checkpoint does not save them. Checkpoint
step is explicitly 21000 because older intermediate files have stale iteration
metadata. Saves every 100 iterations and at completion, with corrected metadata.
No ONNX export or deployment model replacement is performed.

Existing terrain proportions and rewards are retained. Only the gap generator
is replaced when `terrain.generalize_gap=True`: 40% isolated lanes, 40% flat
surroundings, 20% one-sided ground among the five gap columns. Lane width is
1.2–2 m; pit depth is sampled from 0.5, 1, 2 and 5 m. Forward gap position is
5.6–6.3 m within the tile. Gap widths are approximately 0.2–0.4 m by curriculum,
quantized to the existing 0.1 m heightfield grid. Half the tiles are smooth,
half have up to 2 cm floor roughness. The trench crosses the complete tile to
discourage side-floor bypass. The old isolated visual family remains, but its
dimension distribution is narrower than the original full training curriculum.

Full environment/training configuration is written to configuration.json in
the new run. This first stage tests gap surroundings generalization; it does
not yet randomize independent high-platform dimensions or certify real-robot
performance. Camera environments are reduced to 512 for GPU memory, while
the policy rollout has 2048 environments. Reward and camera noise settings
retain training defaults.

Acceptance: compare the original checkpoint and new checkpoints on held-out
isolated/flat/one-sided gaps, then verify flat walking and platforms have not
regressed. Inspect actual crossings and lateral deviation, not total reward
alone. Run ROS validation only after training-side acceptance.

For the background launch started by the assistant, progress is recorded in
logs/terrain_finetune/launch.log and its PID in logs/terrain_finetune/train.pid.
The existing deployed model remains the baseline until validation succeeds.
