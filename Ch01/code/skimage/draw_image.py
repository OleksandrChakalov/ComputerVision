# draw a circle on a black image

import numpy as np 
from skimage import io, draw 



img = np.zeros((100, 100), dtype=np.uint8)


rr, cc = draw.disk((50, 50), radius=10)
img[rr, cc] = 1

io.imshow(img)
io.show()
