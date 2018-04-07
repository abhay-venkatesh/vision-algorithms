from utils.DataPostprocessor import DataPostprocessor

import numpy as np 
import cv2

frame1 = cv2.imread('./images/pic39-0.png')
frame2 = cv2.imread('./images/seg39.png')
frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
frame2 = frame2/8
dp = DataPostprocessor()
dp.write_out(0, frame1, frame2, 0)