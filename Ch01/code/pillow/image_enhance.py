# enhance an image

from PIL import Image
from PIL import ImageEnhance

img = Image.open('../image.png')

enhancer = ImageEnhance.Brightness(img)
bright_img = enhancer.enhance(2)

bright_img.show()
