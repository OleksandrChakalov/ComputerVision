# convert RGB images to Gray scale images

from PIL import Image


img = Image.open('../image.png')


gray_image = img.convert("L")

gray_image.show()
