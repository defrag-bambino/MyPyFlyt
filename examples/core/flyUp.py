import numpy as np
import tqdm
import matplotlib.pyplot as plt

from PyFlyt.core import Aviary

# the starting position and orientation
start_pos_quadx = np.array([0.0, 0.0, 1.0])
start_poss = np.array([start_pos_quadx])
start_orn = np.array([np.zeros(3)])

# individual spawn options
quadx_options = dict(use_camera=True, drone_model="cf2x", camera_angle_degrees=-90, camera_FOV_degrees=130, camera_fps=30)

# environment setup
env = Aviary(
    start_pos=start_poss,
    start_orn=start_orn,
    render=True,
    drone_type=["quadx"],
    drone_options=[quadx_options],
)

# set quadx to angular velocity control
env.set_mode([0])

# prepare a window for the camera
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_title("QUADX")
plt.ioff()
fig.tight_layout()

# Initialize the image
img_quadx = ax.imshow(env.drones[0].rgbaImg, animated=True)
img_w, img_h = env.drones[0].rgbaImg.shape[0], env.drones[0].rgbaImg.shape[0] # 128, 128

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in tqdm.tqdm(range(250)):
    env.step()
    env.set_setpoint(0, np.array([0.0, 0.0, 0.0, 0.5]))
    # img_quadx.set_data(env.drones[0].rgbaImg)
    # fig.canvas.draw_idle()
    # plt.pause(0.0001)

del env