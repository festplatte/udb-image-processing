import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# load images
def load_images():
    imgL = cv.imread('../img/cat_l.jpg', 1)
    imgR = cv.imread('../img/cat_r.jpg', 1)
    return imgL, imgR


# gererate dept map
def build_dept_map(imgL, imgR):
    stereo = cv.StereoBM_create()
    stereo.setBlockSize(37)  # default: 21
    stereo.setDisp12MaxDiff(-1)  # default: -1
    stereo.setMinDisparity(0)  # default: 0
    stereo.setNumDisparities(96)  # default: 64
    stereo.setSpeckleRange(0)  # default: 0
    stereo.setSpeckleWindowSize(0)  # default: 0
    stereo.setPreFilterCap(31)  # default: 31
    stereo.setPreFilterSize(9)  # default: 9
    stereo.setPreFilterType(1)  # default: 1
    stereo.setSmallerBlockSize(0)  # default: 0
    stereo.setTextureThreshold(10)  # default: 10
    stereo.setUniquenessRatio(13)  # default: 15
    return stereo.compute(imgL, imgR)


# generate filter
# threshold
def build_filter(img):
    ret, thresh = cv.threshold(img, 440, 255, cv.THRESH_TOZERO)
    ret, thresh = cv.threshold(thresh, 455, 255, cv.THRESH_TOZERO_INV)
    ret, thresh = cv.threshold(thresh, 1, 1, cv.THRESH_BINARY)
    return thresh


def watershed(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(
        dist_transform, 0.7*dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv.watershed(img, markers)
    return markers


if __name__ == '__main__':
    imgL, imgR = load_images()

    # markers = watershed(imgL)
    # imgL[markers == -1] = [255, 0, 0]

    imgL_gray = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    imgR_gray = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    dept_map = build_dept_map(imgL_gray, imgR_gray)

    # thresh = build_filter(dept_map)

    # segmented_img = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    # segmented_img[thresh == 0] = [0, 0, 0]

    plt.imshow(dept_map, 'gray')
    plt.show()
