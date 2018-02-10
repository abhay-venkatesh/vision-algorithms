"""
Given two images A and B, perform:
    1. Compute the optical flow between images A and B
    2. From the optical flow, get a set of four points
    3. Using the set of four points, compute a homography H between A and B
    4. Use the homography H to generate C such that AH = C

"""
import numpy as np 
import cv2
import itertools
from matplotlib import pyplot as plt

class OpticalFlowTransformer:
    def __init__(self):
        pass

    def drawMatches(self, img1, kp1, img2, kp2, matches):
        """
        My own implementation of cv2.drawMatches as OpenCV 2.4.9
        does not have this function available but it's supported in
        OpenCV 3.0.0

        This function takes in two images with their associated 
        keypoints, as well as a list of DMatch data structure (matches) 
        that contains which keypoints matched in which images.

        An image will be produced where a montage is shown with
        the first image followed by the second image beside it.

        Keypoints are delineated with circles, while lines are connected
        between matching keypoints.

        img1,img2 - Grayscale images
        kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
                  detection algorithms
        matches - A list of matches of corresponding keypoints through any
                  OpenCV keypoint matching algorithm
        """

        # Create a new output image that concatenates the two images together
        # (a.k.a) a montage
        rows1 = img1.shape[0]
        cols1 = img1.shape[1]
        rows2 = img2.shape[0]
        cols2 = img2.shape[1]

        out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

        # Place the first image to the left
        out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

        # Place the next image to the right of it
        out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

        # For each pair of points we have between both images
        # draw circles, then connect a line between them
        for mat in matches:

            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            # x - columns
            # y - rows
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt

            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
            cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

        # Show the image
        # cv2.imshow('Matched Features', out)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def hamming_homography(self, A_path, B_path):
        """
            https://stackoverflow.com/questions/11114349/how-to-visualize-descriptor-matching-using-opencv-module-in-python
        """
        img1 = cv2.imread(A_path) # Original image, queryImage
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(B_path) # Rotated image, trainImage
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # img2 = cv2.imread(A_path) # Original image, queryImage
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # img1 = cv2.imread(B_path) # Rotated image, trainImage
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Create ORB detector with 1000 keypoints with a scaling pyramid factor
        # of 1.2
        orb = cv2.ORB(1000, 1.2)

        # Detect keypoints of original image
        (kp1,des1) = orb.detectAndCompute(img1, None)

        # Detect keypoints of rotated image
        (kp2,des2) = orb.detectAndCompute(img2, None)

        # Create matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Do matching
        matches = bf.match(des1,des2)

        # Sort the matches based on distance.  Least distance
        # is better
        matches = sorted(matches, key=lambda val: val.distance)

        # Show only the top 10 matches
        self.drawMatches(img1, kp1, img2, kp2, matches[:10])

        img1_matches = []
        img2_matches = []
        for match in matches[:250]:
            img1_matches.append(kp1[match.queryIdx].pt)
            img2_matches.append(kp2[match.trainIdx].pt)

        img1_matches = np.array(img1_matches)
        img2_matches = np.array(img2_matches)

        self.H, status = cv2.findHomography(img1_matches, img2_matches)


    def flann_homography(self, A_path, B_path):
        MIN_MATCH_COUNT = 10
        # img1 = cv2.imread(A_path, 0)
        # img2 = cv2.imread(B_path, 0)
        img1 = cv2.imread(A_path, 0)
        img2 = cv2.imread(B_path, 0)

        # Initiate SIFT detector
        sift = cv2.SIFT()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            self.H = M
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.CV_AA)

        else:
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

        # plt.imshow(img3, 'gray'),plt.show()


    def SIFT_for_interest_points(self, img, template, distance=5):
        detector = cv2.FeatureDetector_create("SIFT")
        descriptor = cv2.DescriptorExtractor_create("SIFT")
        skp = detector.detect(img)
        skp, sd = descriptor.compute(img, skp)

        tkp = detector.detect(template)
        tkp, td = descriptor.compute(template, tkp)

        flann_params = dict(algorithm=1, trees=4)
        flann = cv2.flann_Index(sd, flann_params)
        idx, dist = flann.knnSearch(td, 1, params={})
        del flann

        dist = dist[:,0]/2500.0
        dist = dist.reshape(-1,).tolist()
        idx = idx.reshape(-1).tolist()
        indices = range(len(dist))
        indices.sort(key=lambda i: dist[i])
        dist = [dist[i] for i in indices]
        idx = [idx[i] for i in indices]
        skp_final = []
        for i, dis in itertools.izip(idx, dist):
            if dis < distance:
                skp_final.append(skp[i])

        flann = cv2.flann_Index(td, flann_params)
        idx, dist = flann.knnSearch(sd, 1, params={})
        del flann

        dist = dist[:,0]/2500.0
        dist = dist.reshape(-1,).tolist()
        idx = idx.reshape(-1).tolist()
        indices = range(len(dist))
        indices.sort(key=lambda i: dist[i])
        dist = [dist[i] for i in indices]
        idx = [idx[i] for i in indices]
        tkp_final = []
        for i, dis in itertools.izip(idx, dist):
            if dis < distance:
                tkp_final.append(tkp[i])

        return skp_final, tkp_final

    def compute_homography_using_SIFT(self, A_path, B_path):
        """

            Args:
                A_path: Path to image A
                B_path: Path to image B

            Returns:
                None
        """
        frame1 = cv2.imread(A_path)
        frame2 = cv2.imread(B_path)

        frame1_key_points, frame2_key_points = self.SIFT_for_interest_points(frame1, frame2)

        frame1_interest_points = []
        for point in frame1_key_points:
            frame1_interest_points.append([point.pt[0], point.pt[1]])
        frame1_interest_points = np.array(frame1_interest_points)

        frame2_interest_points = []
        for point in frame2_key_points:
            frame2_interest_points.append([point.pt[0], point.pt[1]])
        frame2_interest_points = np.array(frame2_interest_points)
        
        print(len(frame1_interest_points))
        print(len(frame2_interest_points))
        maxlen = max(len(frame1_interest_points), len(frame2_interest_points))
        self.H, status = cv2.findHomography(frame1_interest_points, 
                                            frame2_interest_points[:len(frame1_interest_points)])

    def compute_homography_from_optical_flow(self, A_path, B_path):
        """
            Computes a homography H from two images A and B by computing
            the optical flow between them using the Lucas-Kanade algorithm.

            Args:
                A_path: Path to image A
                B_path: Path to image B


            Returns:
                None
        """
        frame1 = cv2.imread(A_path)
        frame2 = cv2.imread(B_path)

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
        color = np.random.randint(0,255,(100,3))

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

        pts_dst = []
        pts_src = []
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            pts_dst.append([a,b])
            pts_src.append([c,d])

        pts_dst = np.array(pts_dst)
        pts_src = np.array(pts_src)
        
        self.H, status = cv2.findHomography(pts_src, pts_dst)

    def apply_homography(self, A_path):
        im_src = cv2.imread(A_path)
        im_out = cv2.warpPerspective(im_src, self.H, 
                                     (im_src.shape[1],im_src.shape[0]))
        cv2.imwrite('warped.png', im_out)
        pass


def main():
    oft = OpticalFlowTransformer()
    # oft.compute_homography_using_SIFT('./images/pic0.png', './images/pic1.png')
    oft.hamming_homography('./images/pic0.png', './images/pic1.png')
    # oft.flann_homography('./images/pic0.png', './images/pic1.png')
    oft.apply_homography('./images/pic0.png')

if __name__ == '__main__':
    main()


