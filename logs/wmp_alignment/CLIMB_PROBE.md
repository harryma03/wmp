# 25 cm high-platform diagnostic (2026-10-31)

Observed results: training wide plateau passed 3/3 distinct seeds; ROS wide,
original narrow, and obstacle-hidden wide each traversed in their single run.
Central plateau base-height maxima were 0.660–0.668 m in training, 0.662 m
in ROS wide, 0.660 m in ROS original, and 0.660 m in ROS hidden.
The hidden-obstacle success means 25 cm traversal alone cannot establish a
benefit from anticipatory vision. Do not extrapolate these runs to taller
platforms, stairs, or gaps. No retraining was performed.

This opt-in experiment uses checkpoint 21000 and the existing deployed ONNX
models, 0.6 m/s forward command, zero camera mounting angles, and clean depth.
It does not change training defaults, rewards, controller parameters, or the
default ROS launch scene. It does not establish performance at 30–50 cm.

## Scenes

- `scene_wmp_climb_wide.xml`: smooth floor and a 25 cm platform at x=2…3.1,
  y=-4…3.9. Robot starts at (0,0). Matches the diagnostic training plateau
  dimensions after translating training coordinates by (-4,-4).
- `scene_wmp_climb_original.xml`: entire existing ROS scene, with the robot
  starting at (-2,3.5), facing the existing 25 cm platform at x=0…1.2,
  y=3.02…3.98. Only the robot spawn is changed from scene_wmp.xml.
- `scene_wmp_climb_blind.xml`: same physics as wide; the two platforms are
  assigned geom group 2, which the existing depth camera excludes. This is an
  obstacle-visibility ablation, not a complete camera blackout.

The training probe uses a smooth fixed-height terrain instead of ordinary
play's random roughness/difficulty. Three training seeds vary initial pose and
velocity. ROS uses its usual transform_up then rl_0 initialization. Initial
states and contact solvers therefore are not identical. The original scene
also differs in surrounding geometry, so its result cannot isolate width alone.
ROS camera/control capture phases also varied: mean image age was about
105 ms in wide, 95 ms in original, and 95 ms in hidden. These are not
bit-identical initial-state or scheduling comparisons.

## Reproduce

From `/home/aiseon/wmp`, training diagnostic (never exports ONNX):

```bash
LD_LIBRARY_PATH=/home/aiseon/anaconda3/envs/wmp/lib:${LD_LIBRARY_PATH:-} \
WMP_PLAY_TRACE_PATH=/home/aiseon/wmp/logs/wmp_alignment/climb_probe/play_1.jsonl \
/home/aiseon/anaconda3/envs/wmp/bin/python logs/wmp_alignment/play_climb_probe.py \
  --task=titatit --terrain=climb --checkpoint=21000 --sim_device=cuda:0 \
  --num_envs=1 --headless --seed=1
```

Automated ROS runs (use a private ROS domain 87 and close their own simulator):

```bash
python3 logs/wmp_alignment/run_ros_climb_probe.py wide
python3 logs/wmp_alignment/run_ros_climb_probe.py original
python3 logs/wmp_alignment/run_ros_climb_probe.py blind
python3 logs/wmp_alignment/summarize_climb_probe.py
```

For manual observation, close your existing simulator, then:

```bash
bash /home/aiseon/wmp/ddt_ros2_control/ddt_ros2_control/start_wmp_sim.bash \
  scene:=scene_wmp_climb_original.xml
```

Use the existing controls to complete transform_up, enter rl_0, and command
0.6 m/s straight ahead. Use `scene_wmp_climb_wide.xml` for the wide plateau.

## Evidence

Raw traces, stdout logs, `summary.json`, and `comparison.png` are in
`logs/wmp_alignment/climb_probe/`. Traversal requires the body to reach z>0.5
in the central plateau region and then pass its far edge by 0.6 m with z>0.25
while staying inside the platform's lateral corridor. This is a body-trajectory
criterion; it does not prove all four wheels simultaneously contacted the top.
Training reset detection uses a >1 m state jump, and all training records must
contain 601 frames. Inspect process logs for timeout/early termination in ROS.

Passing with the normal camera alone is not proof of anticipatory perception;
compare against the blind scene before making that claim. Single ROS runs are
observations, not success-rate estimates.

The initial original-scene run reached 20.52 simulated seconds, then its
180-second wall-clock watchdog stopped it before the intended 22 seconds.
It had already climbed and traversed the platform. This is usable evidence
of that traversal, but not a complete 22-second rollout. The diagnostic
watchdog has since been increased to 600 seconds for these slower scenes.
