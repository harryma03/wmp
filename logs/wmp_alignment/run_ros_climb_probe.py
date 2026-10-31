"""Run and stop our own ROS simulation; leaves existing ROS sessions alone."""
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
mode = sys.argv[1] if len(sys.argv) > 1 else 'wide'
if mode not in ('wide', 'original', 'blind'):
    raise ValueError('Choose wide, original or blind')
out = ROOT / 'logs/wmp_alignment/climb_probe'
out.mkdir(exist_ok=True)
env = dict(os.environ, ROS_DOMAIN_ID='87',
           WMP_TRACE_PATH=str(out / ('ros_' + mode + '.jsonl')))
starter = ROOT / 'ddt_ros2_control/ddt_ros2_control/start_wmp_sim.bash'
driver = ROOT / 'logs/wmp_alignment/drive_climb_probe.py'
with (out / ('ros_' + mode + '.log')).open('w') as log:
    proc = subprocess.Popen(['bash', str(starter),
        'scene:=scene_wmp_climb_' + mode + '.xml'], env=env,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        subprocess.run(['bash', '-c',
            'source /opt/ros/humble/setup.bash\nexec /usr/bin/python3 "$1"',
            'climb-driver', str(driver)], env=env, check=True, timeout=610)
    finally:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=12)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    print('Trace:', env['WMP_TRACE_PATH'])
