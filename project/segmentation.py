import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# load images
img = cv.imread('../img/allee_l.png', 1)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# generate mask to segment the man in the middle of the picture
# threshold
ret,thresh = cv.threshold(img,40,1,cv.THRESH_BINARY_INV)
thresh = cv.medianBlur(thresh,5)

# gradients
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
sobel_added = np.absolute(sobelx) + np.absolute(sobely)

# laplacian = laplacian + (-1* np.amin(laplacian))
# laplacian = laplacian / np.amax(laplacian)
# print(laplacian[340])

# watershed
ret, thresh1 = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh1,cv.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
# closing = cv.morphologyEx(thresh1,cv.MORPH_CLOSE,kernel, iterations = 2)
# sure_fg = cv.dilate(closing,kernel,iterations=3)
# print(sure_fg[340])

dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
# unknown = opening - closing
unknown = cv.subtract(sure_bg,sure_fg)
unknown = np.uint8(unknown)

# print(closing[340])
# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# print(markers[340])

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

# markers = cv.watershed(img,markers)
# img[markers == -1] = [255,0,0]

# print(unknown[340])

# display results
titles = ['Original Image','thresh','laplacian','sobelx','sobely', 'sobel added', 'sure_bg', 'sure_fg', 'watershed']
images = [img, thresh, laplacian, sobelx, sobely, sobel_added, sure_bg, sure_fg, unknown]
for i in range(len(images)):
    plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
