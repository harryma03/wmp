# Controlled training-side gap comparison

Run from any directory:

```bash
bash /home/aiseon/wmp/logs/wmp_alignment/run_gap_ab.bash
```

Uses `/home/aiseon/anaconda3/envs/wmp/bin/python`, GPU 0, task titatit, checkpoint 21000, one environment, no window, no ONNX export. Ten sequential rollouts: five paired environment seeds, 601 observations each.

Both conditions: 20 cm nominal gap, 1.6 m lane, zero camera Euler angles, clean depth, 0.6 m/s command. Same seed gives identical terrain random draws and initialization. The original heightfield roughness is retained identically in both conditions. Only terrain outside the lane is changed: deep pit (isolated) versus flat side ground (flat). These are training heightfield/trimesh tests, not an exact reconstruction of all decorative geometry in the ROS scene.

The comparison is opt-in via `WMP_GAP_AB=isolated|flat`; ordinary play/training keeps the original terrain generation. `--seed` is explicitly applied to env_cfg.seed in this diagnostic mode because the existing task registry only applies the CLI seed to the runner, after environment creation.

Outputs: `gap_ab/{mode}_{seed}.jsonl`, process logs, and `summary.json`.

Pass criterion: before any detected reset, base passes x=6.8 m (first forward gap ends x=6.2), height >0.25 m, and remains within the 1.6 m lane. This is a first-gap crossing criterion, not certification of jump style or full-course stability. Reset detection in the summary uses >1 m state discontinuity; inspect traces for borderline cases. Only complete 601-observation runs count.
