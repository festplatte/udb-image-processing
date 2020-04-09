import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# config
IMG1_PATH = '../img/room_l.jpg'
IMG2_PATH = '../img/room_r.jpg'
DEST_IMG_PATH = '../img/side-by-side1.jpg'

# load images
img1 = cv.imread(IMG1_PATH, 1)
img2 = cv.imread(IMG2_PATH, 1)

# copy images
dest_img = cv.hconcat([img1, img2])

# save dest image
cv.imwrite(DEST_IMG_PATH, dest_img)

# show dest image
plt.imshow(dest_img[..., ::-1])
plt.show()
