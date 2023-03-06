# rotate an image

from PIL import Image


img = Image.open('../image.png')


rotated_img = img.rotate(90)

rotated_img.show()
