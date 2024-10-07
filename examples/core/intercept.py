"""
The QuadX' goal is to intercept the target.
"""
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from PyFlyt.core import Aviary

# the starting position and orientations
start_pos_target = np.array([np.random.uniform(-65.0, 65.0), np.random.uniform(-65.0, 65.0), np.random.uniform(15.0, 45.0)])
start_pos_quadx = np.array([0.0, 0.0, 1.0])
start_poss = np.array([start_pos_quadx, start_pos_target])
start_orn = np.array([np.zeros(3), np.array([0.0, 50.0, 0.0])]) #np.random.uniform(0.0, 2*np.pi)])])

# individual spawn options for each drone
quadx_options = dict(use_camera=True, drone_model="cf2x", camera_angle_degrees=-90, camera_FOV_degrees=130, camera_fps=30, camera_resolution=[128, 128])
target_options = dict(drone_model="primitive_drone")

# environment setup
env = Aviary(
    start_pos=start_poss,
    start_orn=start_orn,
    render=True,
    drone_type=["quadx", "quadx"],
    drone_options=[quadx_options, target_options],
)

# set quadx to velocity control and target as position
env.set_mode([6, 7])
#env.set_setpoint(1, np.concatenate([env.state(1)[-1], 0], axis=0))
quadx_target_ms_vel = 5.0
# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in tqdm.tqdm(range(2000)):
    env.step()
    target_pos = env.state(1)[-1]
    quadx_pos = env.state(0)[-1]
    # calculate the velocity vector
    vel = target_pos - quadx_pos
    # normalize the velocity vector
    vel = vel / np.linalg.norm(vel)
    # set the velocity vector
    env.set_setpoint(0, np.array([vel[0]*quadx_target_ms_vel, vel[1]*quadx_target_ms_vel, 0.0, vel[2]*quadx_target_ms_vel]))
    # reset the debugVisualizerCamera to look at the quadx
    env.resetDebugVisualizerCamera(3, 20, 10, env.state(1)[-1])
    if (np.any(env.contact_array[1:2])):
        print("Contact!")
del env