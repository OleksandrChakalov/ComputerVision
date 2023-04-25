from matplotlib import pyplot as plt

from skimage import data, io
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.feature import structure_tensor

image = io.imread('image.png', as_gray=True)

# This code computes the structure tensor elements Axx, Axy, and Ayy of the input image using the 'structure_tensor' function from the 'skimage.feature' module and stores them in the corresponding variables.
# The structure tensor is a matrix that summarizes local image structure at each pixel.
Axx, Axy, Ayy = structure_tensor(image, sigma=1)

# This code computes the corners in the input image using the Harris corner detector with the computed structure tensor elements and stores the (y,x) coordinates of the detected corners in the 'coords' variable.
# The minimum distance between detected corners is set to 5 pixels.
coords = corner_peaks(corner_harris(Axx, Axy, Ayy), min_distance=5)

# This code refines the (sub-pixel) corner positions by fitting a quadratic polynomial to the local image intensity around each corner using the 'corner_subpix' function from the 'skimage.feature' module and stores the refined (y,x) coordinates of the corners in the 'coords_subpix' variable.
# The window size for fitting the polynomial is set to 13 pixels.
coords_subpix = corner_subpix(image, coords, window_size=13)

# This code creates a new figure and axes object for displaying the input image and detected corners.
fig, ax = plt.subplots()

# This code displays the input image in the axes object using the 'imshow' function from the 'pyplot' module and sets the colormap to grayscale.
ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)


# This code plots the detected corners in blue dots using the (y,x) coordinates from the 'coords' variable.
ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)

# This code plots the refined corner positions in red crosses using the (y,x) coordinates from the 'coords_subpix' variable.
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)

# This code sets the x and y limits of the axes object to (0,350) and (350,0), respectively.
ax.axis((0, 350, 350, 0))

plt.show()





