# get default images in scikit-image

from skimage import data
from skimage import io 


img_camera = data.camera()


img_text = data.text()

io.imshow(img_text)
io.show()
