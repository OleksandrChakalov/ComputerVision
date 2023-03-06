# read an image using skimage

from skimage import io 

img = io.imread('image.png')
io.imshow(img)
io.show()

