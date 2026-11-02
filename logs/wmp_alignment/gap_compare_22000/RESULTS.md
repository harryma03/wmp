# Paired gap evaluation, 2026-11-02

Checkpoint WMP/model_21000.pt versus gap_generalization_20261031_175729/model_22000.pt.
20 complete rollouts, 601 observations each: two models, isolated/flat side ground,
seeds 1..5. Both models rerun, same initial states (maximum numerical difference 0).
20 cm nominal gap, 1.6 m lane, 0.6 m/s command, camera angles 0, clean depth.
Evaluation uses the previous diagnostic terrain, not the new training generator.

| Scene | Old | New |
|---|---|---|
| Isolated | 4/5 | 5/5 |
| Flat side ground | 0/5 | 5/5 |

Pass requires x>6.8, z>0.25, remaining within the lane, before the first actual
reset. Additional direct-crossing check requires |y-4|<0.45 throughout the
approach from x>=5.5 through clearance. The old and additional criteria give
identical results here. New flat-case lateral maxima: 0.075..0.321 m.
All new runs complete without any reset during their 12-second window.
This is a first-gap criterion, not full-course or real-robot acceptance.

Correction to the earlier gap_ab report: isolated seed 5 was incorrectly counted
as success because its reset at step 24 moved the base less than 1 m. Old and
newly recorded trajectories at steps 23/24 match, so this is a reset-detection
error, not evidence of a simulation stochastic change. New instrumentation logs
the environment done signal as episode number; only episode 0 counts.

Five seeds per condition are a small sample; 5/5 is not a guaranteed deployment
success rate. Higher/wider gaps, one-sided ground, flat walking retention and ROS
transfer require further evaluation. Deployed ONNX files were not replaced.
Raw trajectories, stdout logs, runner, summarizer and summary.json are beside
this report. Two independent play processes ran concurrently on GPU 0.
