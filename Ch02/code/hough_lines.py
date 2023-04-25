import numpy as np
from skimage.transform import hough_line, probabilistic_hough_line
from skimage.feature import canny
from skimage import io, color

image = io.imread('image.png')

# Check if the image has an alpha channel (transparency layer) and convert to RGB if necessary
if image.shape[2] == 4:  # check if the image has an alpha channel
    image = color.rgba2rgb(image)  # convert to RGB format

# Convert the RGB image to grayscale
gray_image = color.rgb2gray(image)

# Apply Canny edge detection on the grayscale image
edges = canny(gray_image, sigma=2, low_threshold=1, high_threshold=25)
lines = hough_line(gray_image)
probabilistic_lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)






