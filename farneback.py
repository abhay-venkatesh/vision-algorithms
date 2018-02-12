"""
References:
"""

import cv2
import numpy as np

frame1 = cv2.imread('./images/pic0.png')
frame2 = cv2.imread('./images/pic1.png')
old_gt = cv2.imread('./images/seg0.png')
old_gt = cv2.cvtColor(old_gt,cv2.COLOR_BGR2GRAY)
previous_frame = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
next_frame = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
flow = cv2.calcOpticalFlowFarneback(previous_frame, next_frame, 0.5, 3, 15, 3, 5, 1.2, 0)

height = flow.shape[0]
width = flow.shape[1]
R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))
pixel_map = R2 + flow

pixel_map_x = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        pixel_map_x[i][j] = pixel_map[i][j][0]

pixel_map_y = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        pixel_map_y[i][j] = pixel_map[i][j][1]

pixel_map_x_32 = pixel_map_x.astype('float32')
pixel_map_y_32 = pixel_map_y.astype('float32')

new_gt = cv2.remap(old_gt, pixel_map_x_32, pixel_map_y_32, cv2.INTER_NEAREST)
cv2.imwrite('seg1approx.png', new_gt)