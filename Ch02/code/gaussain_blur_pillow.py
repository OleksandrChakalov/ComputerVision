from PIL import Image
from PIL import ImageFilter

img = Image.open("image.jpg")

# This code applies a Gaussian blur filter to the image with a radius of 5 pixels.
blur_img = img.filter(ImageFilter.GaussianBlur(5))

blur_img.save("new_blur.jpg")
blur_img.show()