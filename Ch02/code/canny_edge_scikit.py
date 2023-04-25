# Import necessary libraries from scikit-image
from skimage import io
from skimage import feature
from skimage import color

# Load the image and convert it to grayscale
img = io.imread("image.jpg")
img = color.rgb2gray(img)

# Detect edges using the Canny edge detection algorithm with a sigma value of 3
edge = feature.canny(img, 3)

# Display the resulting image of edges and save it as a new image
io.imshow(edge)
io.imsave("new_img.jpg", edge)
io.show()
