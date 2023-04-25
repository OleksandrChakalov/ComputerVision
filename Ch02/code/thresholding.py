import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import threshold_otsu, threshold_local
from skimage.io import imread
from skimage.color import rgb2gray

# Load input image and convert to grayscale
image = imread('image.jpg')
image = rgb2gray(image)

# Apply global thresholding using Otsu's method
global_thresh = threshold_otsu(image)
binary_global = image > global_thresh

# Apply adaptive thresholding using a local threshold based on the mean of a local neighborhood
block_size = 35
binary_adaptive = threshold_local(image, block_size, offset=10)

# Display the original image, the binary image obtained using global thresholding, and the binary image obtained using adaptive thresholding
fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image')

ax1.imshow(binary_global)
ax1.set_title('Global thresholding')

ax2.imshow(binary_adaptive)
ax2.set_title('Adaptive thresholding')

for ax in axes:
    ax.axis('off')

plt.show()

