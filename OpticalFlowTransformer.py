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

class OpticalFlowTransformer:
	def __init__(self):
		pass

	def SIFT_for_interest_points(self, img, template, distance=200):
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
		self.H, status = cv2.findHomography(frame1_interest_points[:len(frame2_interest_points)], frame2_interest_points)

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
	oft.compute_homography_using_SIFT('./images/pic0.png', './images/pic1.png')
	oft.apply_homography('./images/pic0.png')

if __name__ == '__main__':
	main()


