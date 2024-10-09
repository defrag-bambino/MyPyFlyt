"""
    Spawns two fixedwings. One of them goes straight.
    The other one is controlled manually via keyboard and chases.
"""
import numpy as np
import pygame
import pygame.locals
import cv2
from PyFlyt.core import Aviary





# the starting position and orientations
start_poss = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([np.zeros(3)])

# individual spawn options for each drone
drone_ops = dict(use_camera=True, drone_model="primitive_drone", camera_angle_degrees=-0, camera_FOV_degrees=90, camera_fps=40, camera_resolution=[512, 512])
# environment setup
env = Aviary(
    start_pos=start_poss,
    start_orn=start_orn,
    render=False,
    drone_type=["quadx"],
    drone_options=[drone_ops],
)

# set fixedwings to manual control
env.set_mode([0])  # Both fixedwings in manual control mode

# initialize pygame
pygame.init()

# set up the window
window_size = (512, 512)
screen = pygame.display.set_mode(window_size)

# set up the clock
clock = pygame.time.Clock()

# control variables
pitch = 0.0
roll = 0.0
yaw = 0.0
throttle = 0.75

# main loop
done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    keys = pygame.key.get_pressed()
    pitch = 1 if keys[pygame.K_w] else -1 if keys[pygame.K_s] else 0
    roll = 1 if keys[pygame.K_d] else -1 if keys[pygame.K_a] else 0
    yaw = 1 if keys[pygame.K_e] else -1 if keys[pygame.K_q] else 0
    throttle = 1 if keys[pygame.K_UP] else -1 if keys[pygame.K_DOWN] else 0

    # ensure the control inputs are within the valid range -1 to 1
    pitch = np.clip(pitch, -1, 1)
    roll = np.clip(roll, -1, 1)
    yaw = np.clip(yaw, -1, 1)
    throttle = np.clip(throttle, 0, 1)

    # step the environment
    env.step()

    # Get the position of the second drone (index 1)
    fixedwing_pos = env.state(0)[-1]

    # set the control inputs for the target fixedwing
    env.set_setpoint(0, np.array([roll, pitch, yaw, throttle]))

    # get the camera image from the first fixedwing
    camera_img = env.drones[0].rgbaImg
    # convert the image to a format pygame can use
    camera_img = cv2.cvtColor(camera_img, cv2.COLOR_RGBA2RGB)
    camera_img = pygame.surfarray.make_surface(camera_img.swapaxes(0, 1))

    # display the image
    screen.blit(camera_img, (0, 0))
    pygame.display.flip()

    # cap the frame rate
    #clock.tick(100)

pygame.quit()
del env