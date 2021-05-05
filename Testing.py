import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(10,5)
y = range(10,15)
plt.boxplot(x, positions=y, vert=False)
plt.show()