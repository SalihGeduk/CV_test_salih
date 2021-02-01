import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as img
from scipy import signal  
import cv2 as cv



## ---------- For non rotated image ------------
# I have used template match algorithm which is built in
# OpenCV. 
# import images 

im_big = img.imread("./StarMap.png")

img_big = np.array(np.float32(im_big))

im_small = img.imread("./Small_area.png")

img_small = np.array(np.float32(im_small))

# get the image sizes
m,n,k = im_big.shape
print(m,n,k)
m_s,n_s,k_s = im_small.shape


# get the only three part of the images
img_ = img_big[:,:,0:3].copy()
template = img_small[:,:,0:3].copy()
meth = 'cv.TM_CCOEFF'
method = eval(meth)
# Apply template Matching
res = cv.matchTemplate(img_,template,method)
# get the location 
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + m_s, top_left[1] + n_s)

# to draw the result on the figure
cv.rectangle(img_,top_left, bottom_right, 10.0, 2)

#
plt.figure()
plt.imshow(res,cmap = 'gray')
plt.title('Matching Result of non rotated image')
plt.xticks([])
plt.yticks([])

plt.figure()
plt.imshow(img_,cmap = 'gray')
plt.title('Detected Point on the figure for non rotated image')
plt.xticks([])
plt.yticks([])


# ---    For rotated image ------------
# Here to obtain a rotation invariant match algorithm
# I have decided to use first some key detectors (SIFT, Blob, ORB etc.)
# I have implemented the key and description detector
# then used match Brute Force match to match the descriptions in both
# images, However I was not succesfully find the match
# the next task must be to find the locations where more matches occurs
# and then find the map according to the matched descriptors

im_small = img.imread("./Small_area_rotated.png")

img_small = np.array(np.float32(im_small))
plt.figure()
plt.imshow(img_small)

img2 = np.array(img_big[:,:,0:3].copy()*255.0 ,dtype=np.uint8)       # queryImage
img1 = np.array(img_small[:,:,0:3].copy()*255.0,dtype=np.uint8) # trainImage


# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# I did check the results however, matching was not succesfull,
# therefore could go after that
# Draw first 100 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:100],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure()
plt.imshow(img3)
plt.show()




