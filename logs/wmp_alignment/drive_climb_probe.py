"""Simulation-only driver; requires the diagnostic's isolated ROS domain."""
import os
import time
import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

if os.getenv('ROS_DOMAIN_ID') != '87':
    raise RuntimeError('This diagnostic must use ROS_DOMAIN_ID=87')
rclpy.init()
node = rclpy.create_node('wmp_climb_probe_driver', parameter_overrides=[
    Parameter('use_sim_time', value=True)])
key = node.create_publisher(String, '/command/cmd_key',
    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
velocity = node.create_publisher(Twist, '/command/cmd_twist', 10)
deadline = time.monotonic() + 600
start = None
try:
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.02)
        now = node.get_clock().now().nanoseconds / 1e9
        if now <= 0:
            continue
        if start is None:
            start = now
        elapsed = now - start
        if elapsed >= 22:
            print('Completed 22 simulated seconds', flush=True)
            break
        key.publish(String(data='transform_up' if elapsed < 8 else 'rl_0'))
        cmd = Twist()
        cmd.linear.x = .6 if elapsed > 10 else 0.
        velocity.publish(cmd)
    else:
        raise TimeoutError('No complete rollout within 600 wall seconds')
finally:
    velocity.publish(Twist())
    node.destroy_node()
    rclpy.shutdown()
