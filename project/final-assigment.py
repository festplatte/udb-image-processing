import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# load images
def load_images():
    imgL = cv.imread('../img/allee_l.png', 1)
    imgR = cv.imread('../img/allee_r.png', 1)
    return imgL, imgR


# gererate dept map
def build_dept_map(imgL, imgR):
    stereo = cv.StereoBM_create()
    stereo.setBlockSize(19)  # default: 21
    stereo.setDisp12MaxDiff(-1)  # default: -1
    stereo.setMinDisparity(0)  # default: 0
    stereo.setNumDisparities(64)  # default: 64
    stereo.setSpeckleRange(0)  # default: 0
    stereo.setSpeckleWindowSize(0)  # default: 0
    stereo.setPreFilterCap(31)  # default: 31
    stereo.setPreFilterSize(9)  # default: 9
    stereo.setPreFilterType(1)  # default: 1
    stereo.setSmallerBlockSize(0)  # default: 0
    stereo.setTextureThreshold(10)  # default: 10
    stereo.setUniquenessRatio(12)  # default: 15
    return stereo.compute(imgL, imgR)


# generate filter
# threshold
def build_filter(img):
    ret, thresh = cv.threshold(img, 440, 255, cv.THRESH_TOZERO)
    ret, thresh = cv.threshold(thresh, 455, 255, cv.THRESH_TOZERO_INV)
    ret, thresh = cv.threshold(thresh, 1, 1, cv.THRESH_BINARY)
    return thresh


if __name__ == '__main__':
    imgL, imgR = load_images()

    imgL_gray = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    imgR_gray = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    dept_map = build_dept_map(imgL_gray, imgR_gray)

    thresh = build_filter(dept_map)

    segmented_img = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    segmented_img[thresh == 0] = [0, 0, 0]

    plt.imshow(segmented_img)
    plt.show()
