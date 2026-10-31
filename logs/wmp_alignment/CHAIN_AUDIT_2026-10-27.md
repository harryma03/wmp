# WMP training/play → ROS audit (2026-10-27)

This audit does not certify successful gap traversal. Numerical replay only verifies computation on recorded inputs.

| Stage | Training/play | ROS | Status |
|---|---|---|---|
| Task/model | titatit → TitatitWMPCfg, WMPRunner; WMP/model_21000.pt | exported actor + RSSM step | Previously checked weight/output parity |
| Observation | body angular velocity .25, gravity, joint offset 1, velocity .05, clipped action | same order; 54 values | Replay checked |
| Command | linear scale 1, yaw .25 | corrected from linear scale 2 to 1 | Fixed |
| History | 5 frames, initial zero history plus reset observation; append after physics | zero history plus current observation before action | Ordering matches; initial physical state differs |
| Action | raw 16 outputs to RSSM history; clip ±6 for physics/next observation | same; no smoothing, hip reduction 1 | Checked |
| RSSM | update every 5 policy steps; deterministic posterior; deter feature | 5-step, mode posterior, next_stoch/prev_deter handoff | Checked; image influences subsequent deter state |
| Depth projection | 64×64, 58° horizontal FOV, integer pixel rays, axial distance | equivalent square-image rays | Single-ray agrees with Warp on a triangulated ROS scene |
| Depth normalization | clamp 0..2 m, /2−.5; clean play | same | Actual input matches captured depth to float precision |
| Depth delay | previous camera frame, nominal 100 ms | previous implementation measured 150 ms | Changed to capture-time selection; runtime capture-age check passed |
| Capture clock | simulation control step | previously ROS node clock/arrival | Changed header to exact MuJoCo capture time |
| Camera pitch | randomized in [-5°,5°], fixed per environment | 0° | Not exact per-environment parity |
| PD execution | recompute every .005 s; policy .02 s; 60/1.5 legs, .5 wheel velocity gain | corresponding gains, .25 leg/5 wheel scales; 55/10 limits | Formula checked; observed applied forces close before impact |
| Joint coordinate axes | URDF joint axes | transformed CAD frames, rear signs and order remapped | World-space axes agree at nominal pose; local axes alone are misleading |
| Physics | PhysX | MuJoCo RK4 | Different solver; cannot claim identical dynamics |
| Robot asset | training primitive URDF | original CAD-based WMP variant | Not identical: compiled total mass 54.964 vs 51.381 kg, wheel radius .0948 vs .0925 m, other collision/inertial differences |
| Terrain | generated heightfield/trimesh; gap strip surrounded by deep pit | box platforms, original scene surrounded by flat ground; isolated variant removes surrounding floor | Isolated variant improves crossing but is not exact training mesh |

## Corrected diagnostic conclusions

- MuJoCo Python 3.3.0 batch-ray and scalar-ray results differ at the same pose. Raising cutoff does not remove the difference.
- Independent Warp triangle-ray test agrees with scalar depth (MAE 3.36e-8 m, max 1.88e-6 m at one test pose), but differs from batch-ray (MAE .03917 m, max .74690 m).
- Therefore successful old batch-ray rollouts are NOT a trustworthy baseline for ROS physics or control.
- A depth screenshot starting at robot x=.6 m misses the early visibility interval of the first gap. A smooth ground gradient later does not prove the camera failed.
- Replacing images continuously changes RSSM features and actions. A single-step image perturbation does not change the returned deter feature by construction; inspect the next recurrent step.

## Current timing change

WMP launcher requests 50 Hz camera capture while RSSM remains 10 Hz. The controller keeps a bounded timestamped image queue and selects the capture nearest `control_time - 0.1 s`, instead of delaying an already-old callback image once more. This targets training delay within one capture-grid half-period (10 ms); it does not claim exact phase lock. Capture timestamp is now included in inference traces. First RSSM image remains zero, matching the first empty previous-frame slot after reset.

## Remaining evidence required

Runtime test /tmp/wmp_capture_timed.jsonl measured 100 ms capture age across 87 checked updates. The robot subsequently fell off the side of the isolated lane; traversal acceptance remains FAILED. Compare against training using matched camera pitch and generated terrain, and audit primitive collision/inertia differences separately. Do not call these remaining differences the root cause without controlled experiments.
