import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageSequence

# hsv_color_range -- tuple of np arrays (lower color bound, upper color bound)
_ROPE_THRESHOLD = [np.array([ 4, 31, 98]), np.array([8, 225, 238])]

def mask_rope(frame):
    """Convert numpy RGB frame to HSV and and threshold image to given color range in HSV.
    Arguments:
    frame -- image to generate mask in np array format
    """

    frame = frame[:,:,::-1] # to BGR
    blur = cv2.blur(frame, (5, 5))
    blur0 = cv2.medianBlur(blur, 5)
    blur1 = cv2.GaussianBlur(blur0, (5, 5), 0)
    blur2 = cv2.bilateralFilter(blur1, 9, 75, 75)
    hsvFrame = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsvFrame,*_ROPE_THRESHOLD)
    out = cv2.bitwise_and(frame,frame,mask=mask)
    out[out != 0] = 255
    return out 

class RopeColorThreshold(object):
    """Threshold image to rope color"""

    def __call__(self, sample):
        sample = sample[:,:,::-1] # to BGR
        blur = cv2.blur(sample, (5, 5))
        blur0 = cv2.medianBlur(blur, 5)
        blur1 = cv2.GaussianBlur(blur0, (5, 5), 0)
        blur2 = cv2.bilateralFilter(blur1, 9, 75, 75)
        hsvFrame = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsvFrame,*_ROPE_THRESHOLD)
        out = cv2.bitwise_and(sample,sample,mask=mask)
        out[out != 0] = 255
        return Image.array(out)

if __name__ == '__main__':
    img = cv2.imread('test_im.png') 
    print(img[0])
    plt.imshow(img)
    plt.show()
    mask = mask_rope(img)
    plt.imsave("test.png",mask)
    plt.show()
