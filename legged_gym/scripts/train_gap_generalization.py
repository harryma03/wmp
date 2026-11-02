"""Independent WMP terrain fine-tuning, preserving the deployed baseline."""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict


def main():
    args = get_args()
    args.rl_device = args.sim_device
    args.num_envs = args.num_envs or 2048
    cfg, train = task_registry.get_cfgs('titatit')
    cfg.seed = args.seed if args.seed is not None else 42
    cfg.terrain.generalize_gap = True
    cfg.depth.camera_num_envs = min(512, args.num_envs)
    cfg.sim.physx.max_gpu_contact_pairs = 2 ** 21
    cfg.sim.physx.default_buffer_size_multiplier = 2
    train.runner.resume = False
    args.resume = False
    train.runner.run_name = args.run_name or ('gap_generalization_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    train.runner.save_interval = 100
    iterations = args.max_iterations or 1000
    run_dir = ROOT / 'logs' / train.runner.experiment_name / train.runner.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / 'logs/titatit_wmp/WMP/model_21000.pt'
    (run_dir / 'configuration.json').write_text(json.dumps(dict(
        checkpoint=str(checkpoint), additional_iterations=iterations,
        environment=class_to_dict(cfg), training=class_to_dict(train)), indent=2))
    print('RUN_DIRECTORY:', run_dir, flush=True)
    env, _ = task_registry.make_env('titatit', args=args, env_cfg=cfg)
    runner, _ = task_registry.make_wmp_runner(env, args=args, train_cfg=train)
    # Strict restoration includes the auxiliary predictor used for non-camera envs.
    state = torch.load(checkpoint, map_location=runner.device)
    runner.alg.actor_critic.load_state_dict(state['model_state_dict'], strict=True)
    runner._world_model.load_state_dict(state['world_model_dict'], strict=True)
    runner.depth_predictor.load_state_dict(state['depth_predictor'], strict=True)
    runner.alg.optimizer.load_state_dict(state['optimizer_state_dict'])
    runner._world_model._model_opt._opt.load_state_dict(state['wm_optimizer_state_dict'])
    # Existing checkpoint iter metadata may lag its filename; use explicit source step.
    runner.current_learning_iteration = 21000
    original_save = runner.save
    def save(path, infos=None):
        # Store the actual numbered iteration in intermediate checkpoint metadata.
        old = runner.current_learning_iteration
        runner.current_learning_iteration = int(Path(path).stem.split('_')[-1])
        try:
            original_save(path, infos)
        finally:
            runner.current_learning_iteration = old
    runner.save = save
    print('Restored actor, critic, world model, depth predictor and both saved optimizers.', flush=True)
    print('Depth predictor optimizer and replay buffers restart (not saved by baseline).', flush=True)
    runner.learn(iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    main()
