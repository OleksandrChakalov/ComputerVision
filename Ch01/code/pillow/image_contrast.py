# # change the brightness
#
# from PIL import Image
# from PIL import ImageEnhancer
#
# img = Image.open('image.png')
#
# enhancer = ImageEnhancer.Contrast(img)
#
# new_img = enhancer.enhance(2)
#
# new_img.show()


# change the brightness

from PIL import Image, ImageEnhance


img = Image.open('../image.png')


enhancer = ImageEnhance.Contrast(img)

new_img = enhancer.enhance(2)

new_img.show()
