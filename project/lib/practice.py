import cv2
import numpy as np

def find_centroid(src):
    """
    This method finds the centroid via moments analysis.
    Information on moments and centroids can be found here:
    http://en.wikipedia.org/wiki/Image_moment
    @param src	the source image
    @return		the centroid point
    """
    M = cv2.moments(src)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx, cy
