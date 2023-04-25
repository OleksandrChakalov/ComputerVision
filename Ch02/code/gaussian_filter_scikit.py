from skimage import io
from skimage import filters

img = io.imread("image.jpg")

# This code applies a Gaussian filter with a sigma of 5 to the input image using the 'filters.gaussian' function and stores the result in the 'out' variable.
# The Gaussian filter is a linear filter used to smooth images and reduce noise.
out = filters.gaussian(img, sigma=5)
io.imsave("gaussian_filter_scikit.jpg", out)
io.imshow(out)
io.show()