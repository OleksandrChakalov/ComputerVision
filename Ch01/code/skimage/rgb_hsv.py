# convert a color image to grayscale image

from skimage import color, io


img = io.imread('image.png')


img = color.rgba2rgb(img)
hsv_image = color.rgb2hsv(img)


io.imshow(hsv_image)
io.show()

