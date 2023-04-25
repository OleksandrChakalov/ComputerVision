from PIL import Image
from PIL import ImageFilter

# The code opens an image named "image.jpg" and converts it to grayscale (L).
img = Image.open("image.jpg")
img = img.convert("L")

# This code creates a new image by applying a custom kernel filter to the grayscale image.
# The kernel is a 3x3 matrix with the following values:
#  1  0 -1
#  5  0 -5
#  1  0  1
# The filter is applied to each pixel in the image to produce a new value.
# The new value is then used to replace the original value in the output image.
new_img = img.filter(ImageFilter.Kernel((3,3),[1,0,-1,5,0,-5,1,0,1]))

new_img.save("new_img.jpg")
new_img.show()
