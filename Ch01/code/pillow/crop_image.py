# crop an image

from PIL import Image


img = Image.open('../image.png')


dim = (100, 100, 400, 400)
crop_img = img.crop(dim)

crop_img.show()
