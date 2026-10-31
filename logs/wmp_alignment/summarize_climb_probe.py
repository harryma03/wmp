"""Summarize actual rollouts, including whether the body reached platform height."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'climb_probe'


def read(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def describe(xyz, front, length, center_y, halfwidth):
    jumps = np.flatnonzero(np.linalg.norm(np.diff(xyz, axis=0), axis=1) > 1) + 1
    if len(jumps):
        xyz = xyz[:jumps[0]]
    central = ((xyz[:, 0] > front + .4) & (xyz[:, 0] < front + length - .4)
               & (np.abs(xyz[:, 1] - center_y) < halfwidth))
    elevated = bool(np.any(central & (xyz[:, 2] > .5)))
    crossed = bool(np.any((xyz[:, 0] > front + length + .6)
                         & (xyz[:, 2] > .25)
                         & (np.abs(xyz[:, 1] - center_y) < halfwidth)))
    return dict(reached_platform_height=elevated, traversed=elevated and crossed,
                max_x=float(xyz[:, 0].max()), last_xyz=xyz[-1].tolist(),
                central_max_z=float(xyz[central, 2].max()) if central.any() else None,
                first_reset_index=int(jumps[0]) if len(jumps) else None)


fig, axes = plt.subplots(1, 2, figsize=(11, 4))
report = {}
for path in sorted(OUT.glob('play_*.jsonl')):
    rows = read(path)
    xyz = np.array([r['root'][:3] for r in rows])
    result = describe(xyz, 6, 1.1, 4, 3.95)
    result['records'] = len(rows)
    result['complete'] = len(rows) == 601
    report[path.stem] = result
    axes[0].plot(xyz[:, 0] - 6, xyz[:, 2], label=path.stem)
    axes[1].plot(xyz[:, 0] - 6, xyz[:, 1] - 4, label=path.stem)
for mode, front, length, center_y, halfwidth in (
        ('wide', 2, 1.1, -.05, 3.95), ('original', 0, 1.2, 3.5, .48),
        ('blind', 2, 1.1, -.05, 3.95)):
    path = OUT / ('ros_' + mode + '.jsonl.camera.jsonl')
    if not path.exists():
        continue
    rows = read(path)
    active = [r for r in rows if r['time'] >= 10]
    if not active:
        continue
    xyz = np.array([r['qpos'][:3] for r in active])
    result = describe(xyz, front, length, center_y, halfwidth)
    result['last_sim_time'] = rows[-1]['time']
    result['reached_22_seconds'] = rows[-1]['time'] >= 22
    controls = read(OUT / ('ros_' + mode + '.jsonl'))
    ages = [1000 * (r['time'] - r['depth_capture']) for r in controls
            if r.get('model') == 'world_model' and r.get('depth_capture', 0) > 0]
    result['depth_age_ms'] = dict(min=min(ages), mean=float(np.mean(ages)), max=max(ages)) if ages else None
    report['ros_' + mode] = result
    axes[0].plot(xyz[:, 0] - front, xyz[:, 2], label='ros_' + mode)
    axes[1].plot(xyz[:, 0] - front, xyz[:, 1] - center_y, label='ros_' + mode)
axes[0].fill_between([0, 1.1], [0, 0], [.25, .25], color='gray', alpha=.25)
axes[0].set_ylabel('Base height (m)')
axes[1].set_ylabel('Lateral offset (m)')
for ax in axes:
    ax.set_xlabel('Base x relative to front edge (m)')
    ax.set_xlim(-2.5, 3)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'comparison.png', dpi=150)
(OUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
