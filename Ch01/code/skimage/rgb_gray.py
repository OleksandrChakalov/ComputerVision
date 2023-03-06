# convert a color image to grayscale image

from skimage import color, io 


img = io.imread('image.png')

if img.shape[-1] == 4:
    img = img[..., :3]


gray_image = color.rgb2gray(img)

io.imshow(gray_image)
io.show()
