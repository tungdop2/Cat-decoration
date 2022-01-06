import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from math import atan2, degrees

eyes = (0, 1)
nose = 2
# overlay function
def overlay_transparent(background, overlay, center_x, center_y, overlay_size=None):
    b = background.copy()
    b = cv2.cvtColor(b, cv2.COLOR_BGR2BGRA)

    o = overlay.copy()
    o = cv2.resize(o, (overlay_size[0], overlay_size[1]))
    
    w, h = b.shape[:2]
    # print(o.shape)
    # print(center_x, center_y)
    # print(b.shape)
    for i in range(0, o.shape[0]):
        for j in range(0, o.shape[1]):
            if o[i, j][3] != 0:
                b[center_y + i - o.shape[0] // 2, center_x + j - o.shape[1] // 2] = o[i, j]

    b = cv2.cvtColor(b, cv2.COLOR_BGRA2BGR)
    return b

# angle rotation function
def angle_between(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff))

def add_glasses(img, landmarks):

    # get the center of the glasses
    glass_center = (landmarks.reshape((-1, 2))[eyes[0]] + landmarks.reshape((-1, 2))[eyes[1]]) / 2
    # glass_center = landmarks.reshape((-1, 2))[nose] * 0.3 + glass_center * 0.7

    # get the angle of the glasses
    angle = angle_between(landmarks.reshape((-1, 2))[eyes[0]], landmarks.reshape((-1, 2))[eyes[1]])

    # rotate
    glasses = cv2.imread('resources/glasses.png', cv2.IMREAD_UNCHANGED)
    gl = Image.fromarray(np.uint8(glasses))
    gl = gl.rotate(-angle, expand=True)
    glasses = np.array(gl)
    if landmarks.reshape((-1, 2))[eyes[0]][0] > landmarks.reshape((-1, 2))[eyes[1]][0]:
        glass_size = np.linalg.norm(landmarks.reshape((-1, 2))[eyes[0]] - landmarks.reshape((-1, 2))[eyes[1]])
    else:
        glass_size = np.linalg.norm(landmarks.reshape((-1, 2))[eyes[1]] - landmarks.reshape((-1, 2))[eyes[0]])
    glass_size = int(glass_size * 1.5)
    glass_size = (glass_size, glass_size * glasses.shape[0] // glasses.shape[1])

    img_result = overlay_transparent(img, glasses, int(glass_center[0]), int(glass_center[1]), glass_size)

    return img_result
