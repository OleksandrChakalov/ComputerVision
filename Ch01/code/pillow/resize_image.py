# resize a given image

from PIL import Image


img = Image.open('../image.png')


resize_img = img.resize((200, 200))

resize_img.show()
