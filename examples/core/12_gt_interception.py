"""Spawns a fixedwing and a quadx drone. The fixedwing spawns somewhere in the sky,
    while the quadx spawns somewhere in the ground. The quadx' goal is to intercept.
"""
import time
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from PyFlyt.core import Aviary

# the starting position and orientations
start_pos_fixedwing = np.array([np.random.uniform(-65.0, -25.0), np.random.uniform(-5.0, 5.0), np.random.uniform(35.0, 45.0)])
start_pos_quadx = np.array([0.0, 0.0, 1.0])
start_poss = np.array([start_pos_quadx, start_pos_fixedwing])
start_orn = np.array([np.zeros(3), np.array([0.0, 50.0, 0.0])]) #np.random.uniform(0.0, 2*np.pi)])])

# individual spawn options for each drone
quadx_options = dict(use_camera=True, drone_model="cf2x", camera_angle_degrees=-90, camera_FOV_degrees=130, camera_fps=30, camera_resolution=[128, 128])
fixedwing_options = dict(starting_velocity=np.array([20., 0., 1.]), drone_model="fixedwing")

# environment setup
env = Aviary(
    start_pos=start_poss,
    start_orn=start_orn,
    render=True,
    drone_type=["quadx", "fixedwing"],
    drone_options=[quadx_options, fixedwing_options],
)

# set quadx to velocity control and fixedwing as pitch, roll, yaw and thrust
env.set_mode([6, 0])
quadx_target_ms_vel = 16.0
# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in tqdm.tqdm(range(1800)):
    env.step()
    fixedwing_pos = env.state(1)[-1]
    quadx_pos = env.state(0)[-1]
    # calculate the velocity vector
    vel = fixedwing_pos - quadx_pos
    # normalize the velocity vector
    vel = vel / np.linalg.norm(vel)
    # set the velocity vector
    roll = pitch = 0
    env.set_setpoint(0, np.array([vel[0]*quadx_target_ms_vel, vel[1]*quadx_target_ms_vel, 0.0, vel[2]*quadx_target_ms_vel]))
    env.set_setpoint(1, np.array([0., -0.1, 0.0, 0.3]))
    #time.sleep(0.05)

del env