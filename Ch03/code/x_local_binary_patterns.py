from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage.io import imread
from skimage.color import label2rgb
import numpy as np

# settings for LBP
radius = 3
n_points = 8 * radius

# Get three different images to test the algorithm with
brick = imread('goldengate1.png', as_gray=True)
grass = imread('goldengate2.png', as_gray=True)
wall = imread('test.png', as_gray=True)

# Calculate the LBP features for all the three images
brick_lbp = local_binary_pattern(brick, n_points, radius, 'uniform')
grass_lbp = local_binary_pattern(grass, n_points, radius, 'uniform')
wall_lbp = local_binary_pattern(wall, n_points, radius, 'uniform')

# Next we will augment these images by rotating the images by 22 degrees
brick_rot = rotate(brick, angle=22, resize=False)
grass_rot = rotate(grass, angle=22, resize=False)
wall_rot = rotate(wall, angle=22, resize=False)

# Let us calculate the LBP features for all the rotated images
brick_rot_lbp = local_binary_pattern(brick_rot, n_points, radius, 'uniform')
grass_rot_lbp = local_binary_pattern(grass_rot, n_points, radius, 'uniform')
wall_rot_lbp = local_binary_pattern(wall_rot, n_points, radius, 'uniform')

# We will pick any one image say brick image and try to find
# its best match among the rotated images
# Create a list with LBP features of all three images

bins_num = int(brick_lbp.max() + 1)
brick_hist, _ = np.histogram(brick_lbp, normed=True, bins=bins_num, range=(0, bins_num))

lbp_features = [brick_rot_lbp, grass_rot_lbp, wall_rot_lbp]
min_score = 1000 # Set a very large best score value initially
idx = 0 # To keep track of the winner

for feature in lbp_features:
    histogram, _ = np.histogram(feature, normed=True, bins=bins_num, range=(0, bins_num))
    p = np.asarray(brick_hist)
    q = np.asarray(histogram)
    filter_idx = np.logical_and(p != 0, q != 0)
    score = np.sum(p[filter_idx] * np.log2(p[filter_idx] / q[filter_idx]))
    if score < min_score:
        min_score = score
        winner = idx
    idx = idx + 1

if winner == 0:
    print('Brick matched with Brick Rotated')
elif winner == 1:
    print('Brick matched with Grass Rotated')
elif winner == 2:
    print('Brick matched with Wall Rotated')
