import cv2
from matplotlib import pyplot as plt
import numpy as np


# hsv_color_range -- tuple of np arrays (lower color bound, upper color bound)
_ROPE_THRESHOLD = [np.array([ 4, 31, 98]), np.array([8, 225, 238])]

def mask_rope(frame):
    """Convert frame to HSV and and threshold image to given color range in HSV.
    Arguments:
    frame -- image to generate mask in np array format
    """

    blur = cv2.blur(frame, (5, 5))
    blur0 = cv2.medianBlur(blur, 5)
    blur1 = cv2.GaussianBlur(blur0, (5, 5), 0)
    blur2 = cv2.bilateralFilter(blur1, 9, 75, 75)
    hsvFrame = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsvFrame,*_ROPE_THRESHOLD)
    out = cv2.bitwise_and(frame,frame,mask=mask)
    out[out != 0] = 255
    return out 

if __name__ == '__main__':
    img = cv2.imread('0_1.png') 
    plt.imshow(img)
    plt.show()
    mask = mask_rope(img)
    plt.imshow(mask)
    plt.show()
