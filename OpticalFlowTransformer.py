"""
Given two images A and B, perform:
	1. Compute the optical flow between images A and B
	2. From the optical flow, get a set of four points
	3. Using the set of four points, compute a homography H between A and B
	4. Use the homography H to generate C such that AH = C

"""

class OpticalFlowTransformer:
	def __init__(self):
		pass

	def learn_homography(self, A, B):
		pass

	def apply_homography(self, A):
		pass

def main():
	oft = OpticalFlowTransformer()

if __name__ == '__main__':
	main()


