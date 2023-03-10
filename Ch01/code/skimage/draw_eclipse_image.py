# draw an eclipse on a black image

import numpy as np 
from skimage import io, draw



img = np.zeros((100, 100), dtype=np.uint8)


x, y = draw.ellipse(50, 50, 10, 20)
img[x, y] = 1

io.imshow(img)
io.show()
