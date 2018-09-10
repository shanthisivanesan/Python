# slide 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Basic plotting 
# set the seed so you can reproduce the results
np.random.seed(123)
x = np.arange(1,7)
y = np.random.randn(6)
# by default the plot is a line graph
plt.plot(x,y)

# add red dots
plt.plot(x,y,"ro")
plt.show()