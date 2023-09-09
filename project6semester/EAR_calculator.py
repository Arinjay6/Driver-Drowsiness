# Define the function for calculating the Eye Aspect Ratio(EAR)
from scipy.spatial import distance as dist 
def eye_aspect_ratio(eye):
	# Vertical eye landmarks
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# Horizontal eye landmarks 
	C = dist.euclidean(eye[0], eye[3])

	# The EAR Equation 
	EAR = (A + B) / (2.0 * C)
	return EAR

def mouth_aspect_ratio(mouth): 
    # Width of the mouth at the centre
	A = dist.euclidean(mouth[13], mouth[19])
	# B reprsents the width of the mouth slightly near the corners
	B = dist.euclidean(mouth[14], mouth[18])
	# C represents the width of the mouth at corners
	C = dist.euclidean(mouth[15], mouth[17])

	MAR = (A + B + C) / 3.0
	return MAR