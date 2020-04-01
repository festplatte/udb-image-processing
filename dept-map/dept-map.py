import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread('../3d-images/img/room_l.jpg', 1)
imgR = cv.imread('../3d-images/img/room_r.jpg', 1)

imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
# plt.imshow(imgL, 'gray')
# plt.show()

# flags = [i for i in dir(cv) if i.startswith('COLOR_')]
# print(flags)

stereo = cv.StereoBM_create(numDisparities=128, blockSize=17)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()
