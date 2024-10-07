"""
    Spawns two fixedwings. One of them goes straight.
    The other one is controlled manually via keyboard and chases.
"""
import numpy as np
import pygame
import pygame.locals
import cv2
from PyFlyt.core import Aviary



def detect_and_draw_box(img):
    RGBA_img_fixedwing = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(RGBA_img_fixedwing, 250, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 1:
        # Find the smallest contour by area
        smallest_contour = min(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(smallest_contour)
        frame_out = cv2.rectangle(img, (x, y), (x+w, y+h), (200, 0, 0), 1)
        target_pos = np.array([x+w/2, y+h/2])
    else:
        frame_out = img
        target_pos = np.array([0, 0])
    
    return frame_out, target_pos

# the starting position and orientations
start_pos_fixedwing1 = np.array([np.random.uniform(-65.0, -25.0), np.random.uniform(-5.0, 5.0), np.random.uniform(35.0, 45.0)])
start_pos_fixedwing2 = np.array([0.0, 0.0, 50.0]) # the target
start_poss = np.array([start_pos_fixedwing1, start_pos_fixedwing2])
start_orn = np.array([np.zeros(3), np.array([0.0, 0.0, 0.0])])

# individual spawn options for each drone
fixedwing_options1 = dict(use_camera=True, drone_model="fixedwing", camera_angle_degrees=-0, camera_FOV_degrees=90, camera_fps=40, camera_resolution=[512, 512])
fixedwing_options2 = dict(drone_model="fixedwing")

# environment setup
env = Aviary(
    start_pos=start_poss,
    start_orn=start_orn,
    render=True,
    drone_type=["fixedwing", "fixedwing"],
    drone_options=[fixedwing_options1, fixedwing_options2],
)
env.removeBody(env.planeId)
# set fixedwings to manual control
env.set_mode([0, 0])  # Both fixedwings in manual control mode

# initialize pygame
pygame.init()

# set up the window
window_size = (800, 800)
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
    fixedwing_pos = env.state(1)[-1]

    # set the control inputs for the target fixedwing
    env.set_setpoint(1, np.array([0., -0.01, 0., 0.75]))
    env.set_setpoint(0, np.array([roll, pitch, yaw, throttle]))

    # get the camera image from the first fixedwing
    camera_img = env.drones[0].rgbaImg
    camera_img = cv2.cvtColor(camera_img, cv2.COLOR_RGBA2RGB)
    camera_img, target_pos = detect_and_draw_box(camera_img.copy())
    camera_img = cv2.resize(camera_img, window_size)

    # convert the image to a format pygame can use
    camera_img = pygame.surfarray.make_surface(camera_img.swapaxes(0, 1))

    # display the image
    screen.blit(camera_img, (0, 0))
    pygame.display.flip()

    # cap the frame rate
    #clock.tick(100)

pygame.quit()
del env