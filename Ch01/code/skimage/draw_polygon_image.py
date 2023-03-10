# draw a polygon on an image

import numpy as np
from skimage import io, draw


img = np.zeros((100, 100), dtype=np.uint8)
r = np.array([10, 25, 80, 50])
c = np.array([10, 60, 40, 10])
x, y = draw.polygon(r, c)
img[x, y] = 1


io.imshow(img)
io.show()
