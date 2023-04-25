import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# define the mean (mu) and standard deviation (sigma) of a normal (Gaussian) distribution.
mu = 0
sigma = 0.1

# create an array of 100 evenly-spaced points between -3 and 3 using the 'linspace' function from the 'numpy' library.
x = np.linspace(-3, 3, 100)

plt.plot(x, norm.pdf(x, mu, sigma))

plt.show()