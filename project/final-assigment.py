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


def watershed(img, thresh):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ret, thresh = cv.threshold(
    #     gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # noise removal
    kernel = np.ones((10, 10), np.uint8)
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


def dilation_erode(img, kernel_size=10):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_dilation = cv.dilate(img, kernel, iterations=1)
    img_erode = cv.erode(img_dilation, kernel, iterations=1)
    return img_erode


class ImageDisplay:
    def __init__(self):
        self.images = []
        self.labels = []
        self.colors = []

    def add_image(self, img, label, color='gray'):
        self.images.append(img)
        self.labels.append(label)
        self.colors.append(color)

    def show(self):
        img_len = len(self.images)
        sqrt = np.ceil(np.sqrt(img_len))
        # x = np.ceil(sqrt)
        # y = np.floor(sqrt)
        for i in range(img_len):
            plt.subplot(sqrt, sqrt, i +
                        1), plt.imshow(self.images[i], self.colors[i])
            plt.title(self.labels[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


if __name__ == '__main__':
    img_display = ImageDisplay()

    imgL, imgR = load_images()

    imgL_rgb = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    imgL_gray = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    imgR_gray = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    img_display.add_image(imgL_gray, 'original')

    dept_map = build_dept_map(imgL_gray, imgR_gray)
    dept_map = dept_map + np.absolute(np.amin(dept_map))
    dept_map = np.uint8(dept_map / np.amax(dept_map) * 255)
    dept_map = cv.medianBlur(dept_map, 7)
    img_display.add_image(dept_map, 'dept map')

    # build filter for the whole picture
    final_filter = np.full(np.shape(imgL), [0, 0, 255])

    # extract cat
    # dept_map = dilation_erode(dept_map)
    ret, cat_filter1 = cv.threshold(
        dept_map, 100, 255, cv.THRESH_TOZERO_INV)
    ret, cat_filter1 = cv.threshold(
        cat_filter1, 75, 255, cv.THRESH_BINARY)
    # kernel = np.ones((8, 8), np.uint8)
    # cat_filter = cv.dilate(cat_filter, kernel, iterations=3)
    # cat_filter = dilation_erode(cat_filter)
    img_display.add_image(cat_filter1, 'cat filter 1')

    # imgL_rgb = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    # markers = watershed(imgL_rgb, cat_filter)
    # imgL_rgb[markers == -1] = [255, 0, 0]

    cat_filter2 = cv.inRange(imgL_rgb, np.array(
        [30, 30, 30]), np.array([150, 150, 80]))
    cat_filter2 = dilation_erode(cat_filter2, 20)
    cat_filter2 = cv.bitwise_not(cat_filter2)
    img_display.add_image(cat_filter2, 'cat filter 2')
    cat_filter = cv.bitwise_and(cat_filter1, cat_filter2)
    cat_filter = dilation_erode(cat_filter)
    img_display.add_image(cat_filter, 'cat filter')

    final_filter[cat_filter == 255] = [255, 0, 0]
    cat = cv.bitwise_and(imgL_rgb, imgL_rgb, mask=cat_filter)
    img_display.add_image(cat, 'cat', 'viridis')

    # extract bench
    ret, bench_filter = cv.threshold(
        dept_map, 230, 255, cv.THRESH_TOZERO_INV)
    ret, bench_filter = cv.threshold(
        bench_filter, 190, 255, cv.THRESH_BINARY)
    img_display.add_image(bench_filter, 'bench filter')

    final_filter[bench_filter == 255] = [0, 255, 0]

    # add final filter to be displayed
    img_display.add_image(final_filter, 'final filter', 'viridis')

    # thresh = cv.adaptiveThreshold(
    #     dept_map, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 501, 10)
    # th = 0

    # ret, thresh1 = cv.threshold(
    #     imgL_gray, th, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # ret, thresh2 = cv.threshold(
    #     imgL_gray, th, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # ret, thresh3 = cv.threshold(
    #     imgL_gray, th, 255, cv.THRESH_TOZERO + cv.THRESH_OTSU)
    # ret, thresh4 = cv.threshold(
    #     imgL_gray, th, 255, cv.THRESH_TOZERO_INV + cv.THRESH_OTSU)
    # ret, thresh5 = cv.threshold(
    #     imgL_gray, th, 255, cv.THRESH_TRUNC + cv.THRESH_OTSU)

    # thresh2 = dilation_erode(thresh2)

    # kernel = np.ones((15, 15), np.uint8)
    # img_dilation = cv.dilate(dept_map, kernel, iterations=1)
    # img_erode = cv.erode(img_dilation, kernel, iterations=1)
    # dept_map_bgr = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    # markers = watershed(dept_map_bgr)
    # dept_map_bgr[markers == -1] = [255, 0, 0]

    # thresh = build_filter(dept_map)

    # segmented_img = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    # segmented_img[thresh == 0] = [0, 0, 0]

    img_display.show()
