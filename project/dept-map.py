import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# load images
imgL = cv.imread('../img/allee_l.png', 1)
imgR = cv.imread('../img/allee_r.png', 1)

imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
# plt.imshow(imgL, 'gray')
# plt.show()

# flags = [i for i in dir(cv) if i.startswith('COLOR_')]
# print(flags)

# gererate dept map
stereo = cv.StereoBM_create()
stereo.setBlockSize(19) # default: 21
stereo.setDisp12MaxDiff(-1) # default: -1
stereo.setMinDisparity(0) # default: 0
stereo.setNumDisparities(64) # default: 64
stereo.setSpeckleRange(0) # default: 0
stereo.setSpeckleWindowSize(0) # default: 0
stereo.setPreFilterCap(31) # default: 31
stereo.setPreFilterSize(9) # default: 9
stereo.setPreFilterType(1) # default: 1
stereo.setSmallerBlockSize(0) # default: 0
stereo.setTextureThreshold(10) # default: 10
stereo.setUniquenessRatio(12) # default: 15
dept_map = stereo.compute(imgL, imgR)

dept_map = cv.GaussianBlur(dept_map,(5,5),0)

print(dept_map.shape)
print(dept_map[260])
plt.imshow(dept_map, 'gray')
plt.show()
