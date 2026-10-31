"""Opt-in smooth 25 cm climb diagnostic; does not alter training defaults/export.

Run with the wmp Python environment and the usual play.py CLI arguments.
Geometry: 8 m wide platforms, front edges x=1 and x=6, length 1.1 m.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['WMP_SKIP_EXPORT'] = '1'
os.environ['WMP_SMALL_PHYSX'] = '1'
os.environ.pop('WMP_GAP_AB', None)
from legged_gym.scripts import play as playback
from legged_gym.utils.terrain import Terrain
from isaacgym import terrain_utils


def smooth_climb(self, choice, difficulty, i=0, j=0):
    terrain = terrain_utils.SubTerrain(
        'climb_probe', width=self.width_per_env_pixels,
        length=self.length_per_env_pixels,
        vertical_scale=self.cfg.vertical_scale,
        horizontal_scale=self.cfg.horizontal_scale)
    # Heightfield conversion moves the falling edge back by one grid cell.
    # Thus [60:72] becomes a plateau extending from x=6.0 to x=7.1.
    for front in (1., 6.):
        start = round(front / terrain.horizontal_scale)
        stop = start + round(1.1 / terrain.horizontal_scale) + 1
        terrain.height_field_raw[start:stop, :] = round(.25 / terrain.vertical_scale)
    return terrain


if __name__ == '__main__':
    args = playback.get_args()
    if args.task != 'titatit' or args.terrain != 'climb':
        raise ValueError('Use --task=titatit --terrain=climb')
    args.rl_device = args.sim_device
    cfg, train = playback.task_registry.get_cfgs(args.task)
    train.seed = args.seed if args.seed is not None else 1
    cfg.terrain.num_rows = 1
    cfg.terrain.border_size = 10
    cfg.depth.x_angle = cfg.depth.y_angle = cfg.depth.z_angle = [0, 0]
    Terrain.make_terrain = smooth_climb
    playback.EXPORT_POLICY = False
    playback.RECORD_FRAMES = False
    playback.MOVE_CAMERA = False
    playback.play(args)
