"""
Generate an optical flow between two images using the Lucas-Kanade
optical flow algorithm. 

References:
[1] - https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_lucas_kanade.html
"""

import numpy as np 
import cv2

frame1 = cv2.imread('./images/pic0.png')
frame2 = cv2.imread('./images/pic2.png')

# params for ShiTomasi corner detection
feature_params = dict(maxCorners = 100,
                      qualityLevel = 0.3,
                      minDistance = 7,
                      blockSize = 7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize  = (15,15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | 
                 			 cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Generate some random colors
color_B = np.random.randint(0,1,(100))
color_G = np.random.randint(0,1,(100))
color_R = np.random.randint(254,255,(100))
color = np.stack((color_B, color_G, color_R), axis=1)
print(color.shape)

# Find corners in first frame
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(frame1_gray, mask = None, **feature_params)

# Create a mask for drawing purposes
mask = np.zeros_like(frame1)

frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Calculate the optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, p0, 
									   None, **lk_params)

# Select good points
good_new = p1[st==1]
good_old = p0[st==1]

# draw the tracks
for i,(new,old) in enumerate(zip(good_new,good_old)):
	a,b = new.ravel()
	c,d = old.ravel()
	mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
	frame2 = cv2.circle(frame2,(a,b),5,color[i].tolist(),-1)


img = cv2.add(frame2, mask)
img = cv2.resize(img, (480, 320))
cv2.imwrite("test.png", img)
