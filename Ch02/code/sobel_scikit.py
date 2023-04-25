from skimage import io
from skimage import filters
from skimage import color

img = io.imread("image.jpg")

# Convert the image to grayscale
img = color.rgb2gray(img)

# Apply Sobel edge detection on the grayscale image
edge = filters.sobel(img)
io.imshow(edge)
io.imsave("sobel.jpg", edge)
io.show()