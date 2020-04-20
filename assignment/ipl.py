#!/usr/bin/env python

'''
This module (image-processing-library) contains functions and classes used for image processing.
'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# class to collect images with labels and display them all at once
class ImageDisplay:
    def __init__(self):
        self.images = []
        self.labels = []
        self.colors = []

    # adds an image to the collection
    def add_image(self, img, label, color='gray'):
        self.images.append(img)
        self.labels.append(label)
        self.colors.append(color)

    # displays all added images
    def show(self):
        img_len = len(self.images)
        sqrt = np.ceil(np.sqrt(img_len))
        for i in range(img_len):
            plt.subplot(sqrt, sqrt, i +
                        1), plt.imshow(self.images[i], self.colors[i])
            plt.title(self.labels[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


# class to produce a segmentation filter for an image
class ImageSegmentation:
    # saves the image to segment and the initial filter
    def __init__(self, img, bg_color=[0, 0, 255]):
        self.filter = np.full(np.shape(img), bg_color)
        self.img = img

    # add a mask to the filter
    def add_mask(self, mask, color):
        self.filter[mask == 255] = color

    # applies a mask to the image and returns the result
    def apply_mask(self, mask):
        return cv.bitwise_and(self.img, self.img, mask=mask)

    # returns the current filter
    def get_filter(self):
        return self.filter


# loads images and converts them to rgb
def load_images():
    imgL = cv.imread('images/cat_l.jpg', 1)
    imgR = cv.imread('images/cat_r.jpg', 1)
    imgL = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    imgR = cv.cvtColor(imgR, cv.COLOR_BGR2RGB)
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


# applies dilation and erode to the given image with the given kernel size
def dilation_erode(img, kernel_size=10):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_dilation = cv.dilate(img, kernel, iterations=1)
    img_erode = cv.erode(img_dilation, kernel, iterations=1)
    return img_erode


# applies erode and dilation to the given image with the given kernel size
def erode_dilation(img, kernel_size=10):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_erode = cv.erode(img, kernel, iterations=1)
    img_dilation = cv.dilate(img_erode, kernel, iterations=1)
    return img_dilation
