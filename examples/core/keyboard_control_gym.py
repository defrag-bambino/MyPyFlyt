"""
    Spawns two fixedwings. One of them goes straight.
    The other one is controlled manually via keyboard and chases.
"""
import numpy as np
import pygame
import pygame.locals
import cv2

import gymnasium
import PyFlyt.gym_envs

env = gymnasium.make("PyFlyt/QuadX-Hover-v3", render_mode="human", flight_dome_size=50.0)
opts = {"drone_model": "primitive_drone"}
obs, _ = env.reset(options=opts)

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

term, trunc = False, False
while not (term or trunc):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    keys = pygame.key.get_pressed()
    pitch = 1*np.pi if keys[pygame.K_w] else -1*np.pi if keys[pygame.K_s] else 0
    roll = 1*np.pi if keys[pygame.K_d] else -1*np.pi if keys[pygame.K_a] else 0
    yaw = 1*np.pi if keys[pygame.K_q] else -1*np.pi if keys[pygame.K_e] else 0
    throttle = 1 if keys[pygame.K_UP] else -1 if keys[pygame.K_DOWN] else 0.05

    # ensure the control inputs are within the valid range -1 to 1
    pitch = np.clip(pitch, -1, 1)
    roll = np.clip(roll, -1, 1)
    yaw = np.clip(yaw, -1, 1)
    throttle = np.clip(throttle, 0, 1)

    # step the environment
    obs, rew, term, trunc, _ = env.step(np.array([roll, pitch, yaw, throttle]))

    print("ang vels:", obs[0:3])
    print("pos:", obs[3:6])
    print("vel:", obs[6:9])
    print("pos:", obs[9:12])

    pygame.display.flip()

    # cap the frame rate
    clock.tick(30)

pygame.quit()
del env