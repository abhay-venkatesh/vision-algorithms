"""
References:
[1] - https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_lucas_kanade.html
"""

import cv2
import numpy as np

frame1 = cv2.imread('./images/pic0.png')
frame2 = cv2.imread('./images/pic1.png')
previous_frame = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
next_frame = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)
hsv[...,1] = 255

i = 0
while i < 4:
    next_frame = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(previous_frame, next_frame, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalmyhsv.pgm',rgb)

    i += 1

cap.release()
cv2.destroyAllWindows()