import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Select modules to enable
module0 = True
module1 = False

# Load a test image with three different color formats
# Test image is 533 x 400
img_color = cv.imread('input/face0.jpg',1) # Color, but discard transparency (default)
img_gray = cv.imread('input/face0.jpg',0) # Grayscale
img_full = cv.imread('input/face0.jpg',-1) # Color, including transparency


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


print("\nAll done!")