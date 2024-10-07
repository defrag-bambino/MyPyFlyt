"""Spawns a fixedwing and a quadx drone. The fixedwing spawns somewhere in the sky,
    while the quadx spawns somewhere in the ground. The quadx' goal is to intercept.
"""
import time
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import cv2

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
env.removeBody(env.planeId)

# set quadx to angular velocity control and fixedwing as pitch, roll, yaw and thrust
env.set_mode([0, 0])

# prepare a window for the cameras
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title("Fixedwing")
ax[1].set_title("QuadX")
fig.axes[0].axis("off")
fig.axes[1].axis("off")
plt.ioff()
fig.tight_layout()

# Initialize the images
img_quadx = ax[0].imshow(env.drones[0].rgbaImg, animated=True)
img_w, img_h = env.drones[0].rgbaImg.shape[1], env.drones[0].rgbaImg.shape[0] # 128, 128

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in tqdm.tqdm(range(800)):
    env.step()
    # invert the image
    RGBA_img_fixedwing = cv2.cvtColor(env.drones[0].rgbaImg.copy(), cv2.COLOR_RGB2GRAY)
    # background pixels are 255, while objects are less than 255
    ret, thresh = cv2.threshold(RGBA_img_fixedwing, 200, 255, cv2.THRESH_BINARY_INV)
    # apply some dilation
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_ct = cv2.drawContours(RGBA_img_fixedwing, contours, -1, (0, 0, 200), 2)
    if len(contours) > 0:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame_out = cv2.rectangle(env.drones[0].rgbaImg.copy(), (x, y), (x+w, y+h), (200, 0, 0), 1)
            target_pos = np.array([x+w/2, y+h/2])
            pitch = (target_pos[1] - img_h/2) / img_h
            roll = (target_pos[0] - img_w/2) / img_w
    else:
        frame_out = env.drones[0].rgbaImg.copy()
        pitch = 0.0
        roll = 0.0
    #print(f"Roll: {roll}, Pitch: {pitch}")
    env.set_setpoint(0, np.array([roll*2, pitch*4, 0.0, 0.5]))
    env.set_setpoint(1, np.array([0., -0.1, 0.0, 0.3]))    

    # draw a arrowed line at the four edges of the image, pointing to the outside
    # the tip of the arrow should be at the edge of the image
    # the tail of the arrow should start only a few pixels inside the image, e.g. 15 pixels
    max_tail = 15
    # top
    frame_out = cv2.arrowedLine(frame_out, (img_w//2, 15), (img_w//2, 0), (0, 0, 255), 1, tipLength=0.5)
    # bottom
    frame_out = cv2.arrowedLine(frame_out, (img_w//2, img_h-16), (img_w//2, img_h-1), (0, 0, 255), 1, tipLength=0.5)
    # left
    frame_out = cv2.arrowedLine(frame_out, (15, img_h//2), (0, img_h//2), (0, 0, 255), 1, tipLength=0.5)
    # right
    frame_out = cv2.arrowedLine(frame_out, (img_w-16, img_h//2), (img_w-1, img_h//2), (0, 0, 255), 1, tipLength=0.5)

    # on the right side of the image, write a text with the roll amount
    # on the top side of the image, write a text with the pitch amount
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor = (0, 0, 255)
    frame_out = cv2.putText(frame_out, f"Roll: {roll:.2f}", (img_w-100, img_h-10), font, fontScale, fontColor, 1, cv2.LINE_AA)
    frame_out = cv2.putText(frame_out, f"Pitch: {pitch:.2f}", (img_w-100, 10), font, fontScale, fontColor, 1, cv2.LINE_AA)





    RGBA_img_quadx = env.drones[0].rgbaImg
    # update the images
    img_quadx.set_data(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))
    #img_quadx.set_data(RGBA_img_quadx)
    fig.canvas.draw_idle()
    plt.pause(0.001)
    time.sleep(0.02)

del env