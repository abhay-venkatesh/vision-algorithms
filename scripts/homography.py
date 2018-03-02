"""
Given two matrices A,B calculate the homography H between them.
"""

import cv2
import numpy as np
 
if __name__ == '__main__' :
 
    # Read source image.
    im_src = cv2.imread('./images/pic0.png')
    # print(im_src.shape)
    # A = im_src
    # Four corners of the book in source image
    # corners = A[::A.shape[0]-1, ::A.shape[1]-1]
   	# corners = A[tuple(slice(None, None, j-1) for j in A.shape)]
    # pts_src = np.array(corners.reshape(4,2))
    # print(pts_src)
    pts_src = np.array([[863, 140], [916, 136], [865, 185], [892, 160]])
 
 
    # Read destination image.
    im_dst = cv2.imread('./images/pic1.png')
    # B = im_dst
    # Four corners of the book in destination image.
    # corners = B[::B.shape[0]-1, ::B.shape[1]-1]
    # corners = A[tuple(slice(None, None, j-1) for j in A.shape)]
    # pts_src = np.array(corners.reshape(4,2))
    pts_dst = np.array([[863, 131], [921, 123], [870, 167], [900, 146]])
 
    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
     
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
     
    # Display images
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Source Image", im_out)
 
    cv2.waitKey(0)