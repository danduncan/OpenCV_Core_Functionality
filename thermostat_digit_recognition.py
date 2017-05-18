#--------------------------------------------------------------------
# Author: Dan Duncan
# Date created: 4/27/2017
#
# Note: This is intended as a test of the Canny edge detection
# algorithm, and is based on an example from the PyImageSearch blog.
#
#--------------------------------------------------------------------

# Take a photo of a thermostat and read the screen

from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2 as cv
import numpy as np

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
    output[0:,x2:,:] = img2

    # Return the final image
    return output

# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

input_path = "input/thermostat.jpg"


# START SCRIPT #

# Load the example image
image = cv.imread(input_path)

# Pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0) # Reduces high-frequency noise
edged = cv.Canny(blurred, 50, 200, 255) # Canny edge detector
#visualize(side_by_side(gray,edged))


# Now that "edged" provides a simple edge map,
# find contours and sort them by size in descending order.
cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv.contourArea, reverse=True)
displayCnt = None

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)

    # if the contour has four vertices, then we have found
    # the thermostat display
    if len(approx) == 4:
        displayCnt = approx
        break

# extract the thermostat display, apply a perspective transform
# to it
warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(image, displayCnt.reshape(4, 2))
lcd = output.copy()

# Threshold the warped image
thresh = cv.threshold(warped, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
lcd = side_by_side(lcd,thresh)

# Apply morphological operations to clean up the thresholded image
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 5))
thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
lcd = side_by_side(lcd,thresh)
#visualize(lcd)

# PART 2: Find the Digits
# Find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv.findContours(thresh.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
digitCnts = []

# Loop over the digit area candidates
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv.boundingRect(c)

    # if the contour is sufficiently large, it must be a digit
    if w >= 15 and (h >= 30 and h <= 40):
        digitCnts.append(c)

# Sort the contours from left-to-right, then initialize the
# actual digits themselves
digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]
digits = []

# Extract the value of each digit
# Loop over each of the digits
for c in digitCnts:
    # Extract the digit ROI
    (x, y, w, h) = cv.boundingRect(c)
    roi = thresh[y:y + h, x:x + w]

    # compute the width and height of each of the 7 segments
    # we are going to examine
    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dHC = int(roiH * 0.05)

    # define the set of 7 segments
    segments = [
        ((0, 0), (w, dH)),  # top
        ((0, 0), (dW, h // 2)),  # top-left
        ((w - dW, 0), (w, h // 2)),  # top-right
        ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
        ((0, h // 2), (dW, h)),  # bottom-left
        ((w - dW, h // 2), (w, h)),  # bottom-right
        ((0, h - dH), (w, h))  # bottom
    ]
    on = [0] * len(segments)

    # loop over the segments
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # extract the segment ROI, count the total number of
        # thresholded pixels in the segment, and then compute
        # the area of the segment
        segROI = roi[yA:yB, xA:xB]
        total = cv.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)

        # if the total number of non-zero pixels is greater than
        # 50% of the area, mark the segment as "on"
        if total / float(area) > 0.5:
            on[i] = 1

    # lookup the digit and draw it on the image
    digit = DIGITS_LOOKUP[tuple(on)]
    digits.append(digit)
    cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv.putText(output, str(digit), (x - 10, y - 10),cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

print(u"{}{}.{} \u00b0C".format(*digits))
visualize(output)