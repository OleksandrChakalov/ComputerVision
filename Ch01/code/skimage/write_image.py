# write an image

from skimage import io 


img = io.imread('image.png')


io.imsave('new_image.png', img)
