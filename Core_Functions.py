#--------------------------------------------------------------------
# Author: Dan Duncan
# Date created: 4/19/2017
#
# Test of the core functionalities of OpenCV on still images.
# Includes file I/O, displaying images, and benchmarking of
# color space transformations.
#
# Timing metrics
#               0.2 MPixels   10 MPixels
# BGR2LAB       1.2 ms        17.6 ms
# BGR2HSV       1.4 ms        28.3 ms
# BGR2HSV_FULL  1.1 ms        25.4 ms
# BGR2RGB       0.6 ms        7.9 ms
# BGR2GRAY      0.2 ms        7.0 ms
# BGR2YCrCb     0.4 ms        18.6 ms
# BGR2XYZ       0.4 ms        20.2 ms
#
#--------------------------------------------------------------------

import cv2 as cv
from matplotlib import pyplot as plt
import time

#--------------------------------------------------------------------
# USER OPTIONS:

# Select modules to enable
module0 = False  # Display images using opencv
module1 = False  # Display images using matplotlib
module2 = True  # Convert images between different color spaces

# Load a test image with three different color formats
inputFile = "input/face0.jpg"  # 533 x 400 pixels
#inputFile = "input/face_large.jpg" # 3500 x 2789 pixels


#--------------------------------------------------------------------
# START SCRIPT

img_color = cv.imread(inputFile,1) # Color, but discard transparency (default)
img_gray = cv.imread(inputFile,0) # Grayscale
img_full = cv.imread(inputFile,-1) # Color, including transparency


# Module 0: Read, display, and write images
if module0 == True:
    # Display the images in sequence
    cv.imshow('Color',img_color)
    cv.waitKey(0) # Wait for user to press any key before continuing
    cv.destroyAllWindows()
    cv.waitKey(1) # Required extra line for cv to work on Mac

    cv.imshow('Gray',img_gray)
    k = cv.waitKey(0) # Save user's input key to k
    cv.destroyAllWindows()
    cv.waitKey(1) # Required extra line for cv to work on Mac

    cv.imshow('Color with Transparency',img_full)
    cv.waitKey(0) # Wait for user to press any key before continuing
    cv.destroyAllWindows()
    cv.waitKey(1) # Required extra line for cv to work on Mac

    # Display image in resized window
    # 1/2 size: 267 x 200
    # Note that img.shape returns (height, width)
    # OpenCV functions take arguments as (width, height)
    h, w = img_color.shape[:2]
    interp = cv.INTER_AREA  # Use AREA for shrinking, LINEAR or CUBIC for zooming
    img_small = cv.resize(img_color, (200, 267), interpolation=interp)
    #cv.namedWindow('smallframe', cv.WINDOW_NORMAL)
    cv.imshow('smallframe',img_small)
    #cv.resizeWindow('smallframe', 200, 267)
    cv.waitKey(0)  # Wait for user to press any key before continuing
    cv.destroyAllWindows()
    cv.waitKey(1)  # Required extra line for cv to work on Mac

    # Allow user to save grayscale image by typing an 's'
    if k == ord('s'):
        cv.imwrite('input/face0_gray.jpg', img_gray)
        print("Grayscale image saved.")
    else:
        print("Image not saved.")

# Module 1: Display images using matplotlib
if module1 == True:

    # Display grayscale image
    plt.imshow(img_gray, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    # Display color image
    # This is harder, because OpenCV is BGR, while matplotlib is RGB
    b,g,r = cv.split(img_color)
    img_color_rgb = cv.merge([r,g,b])
    plt.subplot(121)
    plt.imshow(img_color)  # Distorted color
    plt.subplot(122)
    plt.imshow(img_color_rgb)  # Correct color
    plt.show()

    # Two other ways to get the correct color
    img_color_rgb_2 = img_color[:, :, ::-1] # Uses numpy indexing
    img_color_rgb_3 = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)

# Module 2: Convert images to different color spaces
if module2 == True:

    # Display all possible color conversion flags
    allFlags = [i for i in dir(cv) if i.startswith('COLOR_BGR2')]
    print(allFlags)

    # Important color conversion flags:
    # COLOR_BGR2GRAY
    # COLOR_BGR2HSV, COLOR_BGR2HSV_FULL
    # COLOR_BGR2LAB, COLOR_BGR2Lab  <-- These are equivalent
    # COLOR_BGR2XYZ
    # COLOR_BGR2YCR_CB, COLOR_BGR2YCrCb <-- These are equivalent
    flags = ["COLOR_BGR2RGB",
             "COLOR_BGR2GRAY",
             "COLOR_BGR2LAB",
             "COLOR_BGR2HSV",
             "COLOR_BGR2HSV_FULL",
             "COLOR_BGR2YCrCb",
             "COLOR_BGR2XYZ"
             ]

    # Print image resolution
    resMP = (img_color.shape[0] * img_color.shape[1]) / 1e6
    print("Image Resolution: %.4f MP" % resMP)

    # Do conversions for all flags
    print("Conversion time:")
    for flag in flags:
        tstart = time.time()
        img_new = cv.cvtColor(img_color,getattr(cv,flag))
        t = time.time() - tstart
        print("\t" + flag + "\t\t%.3f ms" % (1000*t))

    # Display image and check display time
    # Note that cv.imshow only accepts images in BGR format
    showImage = False
    if showImage:
        t0 = time.time()
        cv.imshow('Color',img_color)
        tdisp = time.time() - t0
        print("Display time:\t%.3f ms" % (1000*tdisp))
        cv.waitKey(0) # Wait for user to press any key before continuing
        cv.destroyAllWindows()
        cv.waitKey(1) # Required extra line for cv to work on Mac

print("\nAll done!")