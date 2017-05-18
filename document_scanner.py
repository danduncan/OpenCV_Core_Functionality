#--------------------------------------------------------------------
# Author: Dan Duncan
# Date created: 4/28/2017
#
# Takes photo of document and returns the scanned document
#
# Note: This is intended as a test of the image perspective
# transformations, and is based on an example from PyImageSearch blog.
#
#--------------------------------------------------------------------

from imutils.perspective import four_point_transform
import imutils
from skimage.filters import threshold_adaptive
import numpy as np
import cv2 as cv

#--------------------------------------------------------------------
# HELPER FUNCTIONS

# Quick image visualization
def visualize(image,boxName="image"):
    cv.imshow(boxName, image)
    cv.waitKey(0)  # Wait for user to press any key before continuing
    cv.destroyAllWindows()
    cv.waitKey(1)  # Due to bug in OpenCV, this line required for cv to work on Mac
    return None

# Combine two images side-by-side, with a black bar in the middle
def side_by_side(image1,image2,barwidth=10):
    # Make copies to prevent modifying the originals
    img1 = image1
    img2 = image2

    # Assumes images are of size (h,w,3) or (h,w)
    dim1 = img1.shape
    dim2 = img2.shape

    # If either image has only 1 color channel, expand to three channels
    if len(dim1) == 2:
        img1 = cv.cvtColor(img1, cv.COLOR_GRAY2RGB)
        dim1 = img1.shape
    if len(dim2) == 2:
        img2 = cv.cvtColor(img2, cv.COLOR_GRAY2RGB)
        dim2 = img2.shape

    # Break into separate components
    h1, w1, d1 = dim1
    h2, w2, d2 = dim2

    # Calculate output frame height and width
    height = max(h1,h2)
    width = w1 + w2 + barwidth

    # Create new array
    output = np.zeros((height,width,3)).astype('uint8')

    # Add first image to new array
    output[0:h1,0:w1,0:3] = img1

    # Calculate second image position and add to array
    x2 = w1 + barwidth
    output[0:h2,x2:,:] = img2

    # Return the final image
    return output


#--------------------------------------------------------------------
# START SCRIPT
# PART 1: Noise reduction and Canny edge detection

input_path = "input/receipt.jpg"

# Load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
# We do our edge detection on the resized image, but we
# retain the ratio in order to do our extraction on the
# original image
image = cv.imread(input_path)
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# Convert the image to grayscale, blur it, and find edges
# in the image
# Blurring removes high frequency noise and aids contour detection
# using the Canny edge detector
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(gray, 75, 200)

# Show the original image and the edge detected image
print "STEP 1: Edge Detection"
output = side_by_side(orig,edged)
visualize(side_by_side(image,edged))

#--------------------------------------------------------------------
# PART 2: Find the contours in the edged image
# Sort contours by descending size
# The largest contour with exactly 4 edges is assumed to be the piece of paper
# Retain only the 5 largest contours for checking

cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]

# Loop over the largest contours
for c in cnts:
    # approximate the contour
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)

    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

# show the contour (outline) of the piece of paper
print "STEP 2: Find contours of paper"
cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
visualize(image)
output = side_by_side(output,image)

#--------------------------------------------------------------------
# PART 3: Transform Image
# Apply the four point transform to obtain a top-down
# view of the original image
# This is a custom wrapper function that uses
# cv.getPerspectiveTransform and a set of 4 points to
# return the top-down view of those 4 points.
# More documentation here:
# www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# Convert the warped image to grayscale, then threshold it
# to enhance black/white contrast
warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
warped = threshold_adaptive(warped, 251, offset=10)
warped = warped.astype("uint8") * 255

# Show the original and scanned images
print "STEP 3: Apply perspective transform"
output = side_by_side(output,warped)
visualize(output)
visualize(side_by_side(orig,warped))
